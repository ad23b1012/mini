"""
Curated Explanation Pipeline — Thesis-Ready XAI Outputs.

Scans the test set for high-quality face crops, selects the best
samples per emotion class, and generates publication-quality
explanations with anatomically grounded Grad-CAM heatmaps.

This script ensures that only samples with:
  - Verified frontal face crops (MediaPipe FaceMesh validated)
  - Sufficient model confidence (≥15% by default)
  - Clear facial feature activation in Grad-CAM

are included in the final explanation output.

Usage:
    uv run python scripts/explain_curated.py \
        --checkpoint results/checkpoints/best_model.pt \
        --samples-per-class 3 \
        --output-dir results/explanations_curated
"""

import argparse
from datetime import datetime, timezone
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.meld_dataset import MELDDataset
from data.affectnet_dataset import AffectNetDataset
from data.transforms import get_denormalize_transform
from models.multimodal_model import build_model
from explainers.gradcam import GradCAMExplainer
from explainers.shap_text import SHAPTextExplainer
from explainers.faithfulness import CrossModalFaithfulness
from explainers.nlg_report import NLGReportGenerator
from utils.helpers import set_seed, load_config, get_device
from utils.visualization import (
    plot_gradcam_overlay,
    plot_shap_tokens,
    plot_combined_explanation,
)
from utils.face_quality import check_face_quality, get_face_quality_summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate curated, thesis-ready XAI explanations"
    )
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, choices=["meld", "affectnet"])
    parser.add_argument(
        "--samples-per-class", type=int, default=3,
        help="Number of best samples to explain per emotion class",
    )
    parser.add_argument(
        "--output-dir", type=str, default="./results/explanations_curated",
    )
    parser.add_argument("--nlg-mode", type=str, default="template",
                        choices=["template", "llm"])
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducible scanning and selection",
    )
    parser.add_argument(
        "--min-confidence", type=float, default=0.15,
        help="Minimum model confidence to include a sample",
    )
    parser.add_argument(
        "--max-scan", type=int, default=0,
        help="Maximum number of samples to scan for quality (0 = entire split)",
    )
    return parser.parse_args()


def denormalize_image(image_tensor: torch.Tensor) -> np.ndarray:
    """Convert normalized tensor back to displayable numpy array."""
    denorm = get_denormalize_transform()
    img = denorm(image_tensor.cpu())
    img = img.permute(1, 2, 0).numpy()
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img


def sanitize_for_json(value):
    """Recursively replace NaN/Inf values with None before saving."""
    if isinstance(value, dict):
        return {k: sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize_for_json(v) for v in value]
    if isinstance(value, tuple):
        return [sanitize_for_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return sanitize_for_json(value.tolist())
    if isinstance(value, bool):
        return value
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def build_run_metadata(args, dataset_name: str, class_names: list, seed: int) -> dict:
    """Describe how this curated run was produced."""
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": os.path.abspath(__file__),
        "dataset": dataset_name,
        "checkpoint": os.path.abspath(args.checkpoint),
        "config": os.path.abspath(args.config),
        "output_dir": os.path.abspath(args.output_dir),
        "seed": int(seed),
        "samples_per_class": int(args.samples_per_class),
        "nlg_mode": args.nlg_mode,
        "min_confidence": float(args.min_confidence),
        "max_scan": int(args.max_scan),
        "class_names": class_names,
    }


def scan_for_quality(dataset, class_names, max_scan=500):
    """
    Scan dataset samples for face quality, returning sorted candidates
    per emotion class.

    Returns:
        dict mapping class_idx -> list of (dataset_idx, quality_score) 
        sorted by quality descending.
    """
    num_classes = len(class_names)
    candidates = {i: [] for i in range(num_classes)}

    scan_count = len(dataset) if max_scan <= 0 else min(max_scan, len(dataset))
    scope_text = (
        f"entire split ({scan_count} samples)"
        if max_scan <= 0
        else f"up to {scan_count} samples"
    )

    print(f"\n{'=' * 60}")
    print(f"Scanning {scope_text} for face quality...")
    print(f"{'=' * 60}")

    indices = np.random.choice(len(dataset), scan_count, replace=False)

    valid_count = 0
    invalid_count = 0

    for i, idx in enumerate(indices):
        if (i + 1) % 50 == 0:
            print(f"  Scanned {i + 1}/{scan_count} "
                  f"(valid: {valid_count}, invalid: {invalid_count})")

        sample = dataset[int(idx)]
        label = sample["label"]

        # Denormalize and check quality
        img = denormalize_image(sample["image"])
        quality = check_face_quality(img)

        if quality["is_valid"]:
            candidates[label].append((int(idx), quality["quality_score"]))
            valid_count += 1
        else:
            invalid_count += 1

    # Sort each class by quality score (best first)
    for cls_idx in candidates:
        candidates[cls_idx].sort(key=lambda x: x[1], reverse=True)

    print(f"\nScan complete: {valid_count} valid, {invalid_count} invalid "
          f"({valid_count / scan_count * 100:.1f}% pass rate)")

    for cls_idx in range(num_classes):
        n = len(candidates[cls_idx])
        name = class_names[cls_idx]
        top_score = candidates[cls_idx][0][1] if n > 0 else 0
        print(f"  {name}: {n} valid candidates (best quality: {top_score:.3f})")

    return candidates


def predict_sample(model, sample, device):
    """Run a single forward pass and return predicted class + confidence."""
    image = sample["image"].unsqueeze(0).to(device)
    input_ids = sample.get("input_ids", None)
    attention_mask = sample.get("attention_mask", None)

    if input_ids is not None:
        input_ids = input_ids.unsqueeze(0).to(device)
        attention_mask = attention_mask.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(
            image=image,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        probs = torch.softmax(outputs["logits"], dim=1)[0]

    pred_class = int(torch.argmax(probs).item())
    confidence = float(probs[pred_class].item())
    return pred_class, confidence


def explain_single_curated(
    model,
    sample,
    gradcam_explainer,
    shap_explainer,
    faithfulness,
    nlg_generator,
    class_names,
    sample_dir,
    sample_id,
    device,
    min_confidence,
):
    """Run full explanation on a single curated sample."""
    image = sample["image"].unsqueeze(0).to(device)
    label = sample["label"]
    emotion_name = class_names[label] if label < len(class_names) else str(label)

    is_multimodal = "input_ids" in sample
    input_ids = sample.get("input_ids", None)
    attention_mask = sample.get("attention_mask", None)
    utterance = sample.get("utterance", "")
    text_input = sample.get("text_input", utterance)
    speaker = sample.get("speaker", "Unknown")

    if input_ids is not None:
        input_ids = input_ids.unsqueeze(0).to(device)
        attention_mask = attention_mask.unsqueeze(0).to(device)

    original_img = denormalize_image(sample["image"])

    # Double-check quality (should pass since we pre-filtered)
    quality = check_face_quality(original_img)
    if not quality["is_valid"]:
        return None

    os.makedirs(sample_dir, exist_ok=True)

    # ---- Grad-CAM ----
    # Pass target_class=None so Grad-CAM explains the model's *predicted* class
    # (argmax of logits), not the ground-truth label.
    gradcam_result = gradcam_explainer.generate(
        image=image,
        target_class=None,
        input_ids=input_ids,
        attention_mask=attention_mask,
        original_image=original_img,
    )

    confidence = gradcam_result["confidence"]
    if confidence < min_confidence:
        print(f"    Skipped: confidence {confidence*100:.1f}% < {min_confidence*100:.0f}%")
        return None

    pred_class = gradcam_result["predicted_class"]
    vis_heatmap = gradcam_result.get("masked_heatmap", gradcam_result["heatmap"])

    print(f"    Pred: {class_names[pred_class]} ({confidence*100:.1f}%) | "
          f"True: {emotion_name}")

    # Save Grad-CAM
    plot_gradcam_overlay(
        original_img, vis_heatmap,
        emotion_name=class_names[pred_class],
        confidence=confidence,
        save_path=os.path.join(sample_dir, "gradcam_overlay.png"),
    )

    # ---- SHAP ----
    shap_result = None
    if is_multimodal and utterance:
        # Use the model's predicted class (not the ground-truth label) so
        # SHAP token importance reflects the actual decision being explained.
        shap_result = shap_explainer.explain(
            text=text_input,
            image=image,
            target_class=pred_class,
            emotion_names=class_names,
        )

        plot_shap_tokens(
            shap_result["tokens"],
            shap_result["shap_values"],
            emotion_name=class_names[shap_result["predicted_class"]],
            save_path=os.path.join(sample_dir, "shap_tokens.png"),
        )

    # ---- Faithfulness ----
    faithfulness_metrics = None
    if is_multimodal and shap_result is not None:
        try:
            faithfulness_metrics = faithfulness.compute_all_metrics(
                image=image,
                input_ids=input_ids,
                attention_mask=attention_mask,
                gradcam_heatmap=gradcam_result["heatmap"],
                shap_values=shap_result["shap_values"],
                model_token_scores=shap_result["model_token_scores"],
                special_token_mask=shap_result["special_token_mask"],
                tokens=shap_result["tokens"],
                target_class=pred_class,  # explain the predicted class, not ground truth
            )
        except Exception as e:
            print(f"    Warning: Faithfulness computation failed: {e}")

    # ---- NLG Report ----
    token_importance = shap_result["token_importance"] if shap_result else {}
    report = nlg_generator.generate_report(
        emotion_name=class_names[pred_class],
        confidence=confidence,
        region_scores=gradcam_result["region_scores"],
        token_importance=token_importance,
        faithfulness_metrics=faithfulness_metrics,
        utterance=utterance,
        speaker=speaker,
    )

    with open(os.path.join(sample_dir, "explanation_report.txt"), "w") as f:
        f.write(report)

    # ---- Combined Visualization ----
    if shap_result is not None:
        plot_combined_explanation(
            original_image=original_img,
            heatmap=vis_heatmap,
            tokens=shap_result["tokens"],
            shap_values=shap_result["shap_values"],
            region_scores=gradcam_result["region_scores"],
            emotion_name=class_names[pred_class],
            confidence=confidence,
            utterance=utterance,
            faithfulness_metrics=faithfulness_metrics,
            save_path=os.path.join(sample_dir, "combined_explanation.png"),
        )

    # ---- Save results JSON ----
    results = {
        "sample_id": sample_id,
        "true_emotion": emotion_name,
        "predicted_emotion": class_names[pred_class],
        "confidence": confidence,
        "region_scores": gradcam_result["region_scores"],
        "utterance": utterance,
        "model_text_input": text_input,
        "speaker": speaker,
        "face_quality_score": quality["quality_score"],
    }
    if token_importance:
        results["top_tokens"] = dict(list(token_importance.items())[:10])
    if faithfulness_metrics:
        results["faithfulness"] = faithfulness_metrics

    with open(os.path.join(sample_dir, "results.json"), "w") as f:
        json.dump(sanitize_for_json(results), f, indent=2, default=str)

    return results


def main():
    args = parse_args()
    config = load_config(args.config)
    device = get_device()
    seed = args.seed if args.seed is not None else config.get("project", {}).get("seed", 42)
    set_seed(seed)

    dataset_name = args.dataset or config["dataset"]["name"]
    dataset_cfg = config["dataset"].get(dataset_name, {})
    num_classes = dataset_cfg.get("num_classes", 7)
    class_names = dataset_cfg.get("class_names", [str(i) for i in range(num_classes)])

    # Load dataset
    print(f"\nLoading {dataset_name} test set...")
    if dataset_name == "meld":
        dataset = MELDDataset(
            root_dir=dataset_cfg.get("root_dir", "./datasets/meld"),
            split="test",
            image_size=dataset_cfg.get("image_size", 260),
            text_model_name=config["model"]["text"]["backbone"],
        )
    else:
        dataset = AffectNetDataset(
            root_dir=dataset_cfg.get("root_dir", "./datasets/affectnet"),
            split="val",
        )

    # Load model
    print("\nLoading model...")
    model = build_model(config).to(device)
    checkpoint = torch.load(
        args.checkpoint, map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Initialize explainers
    print("Initializing explainers...")

    target_layer = model.vision_encoder.get_gradcam_target_layer()
    gradcam = GradCAMExplainer(
        model=model, target_layer=target_layer, device=device
    )

    xai_cfg = config.get("xai", {})
    shap_explainer = SHAPTextExplainer(
        model=model,
        tokenizer_name=config["model"]["text"]["backbone"],
        device=device,
        method=xai_cfg.get("shap", {}).get("method", "partition"),
        max_evals=xai_cfg.get("shap", {}).get("max_evals", 500),
    )

    faithfulness = CrossModalFaithfulness(
        model=model,
        device=device,
        num_perturbation_steps=xai_cfg.get("faithfulness", {}).get(
            "num_perturbation_steps", 10
        ),
        top_k_features=xai_cfg.get("faithfulness", {}).get("top_k_features", 5),
    )

    nlg_cfg = config.get("nlg", {})
    nlg_generator = NLGReportGenerator(
        mode=args.nlg_mode,
        min_importance=nlg_cfg.get("template", {}).get("min_importance", 0.05),
        max_features=nlg_cfg.get("template", {}).get("max_features", 5),
    )

    # ---- Phase 1: Scan for quality ----
    candidates = scan_for_quality(dataset, class_names, max_scan=args.max_scan)

    # ---- Phase 2: Select best correct samples per class ----
    selected = []
    for cls_idx in range(num_classes):
        cls_candidates = candidates[cls_idx]
        if not cls_candidates:
            print(f"\nWARNING: No valid face crops found for {class_names[cls_idx]}")
            continue

        print(
            f"\nSelecting up to {args.samples_per_class} high-quality, "
            f"correctly predicted samples for {class_names[cls_idx]}..."
        )
        selected_for_class = []
        evaluated = 0

        for dataset_idx, quality_score in cls_candidates:
            sample = dataset[dataset_idx]
            pred_class, confidence = predict_sample(model, sample, device)
            evaluated += 1

            if pred_class != cls_idx:
                continue
            if confidence < args.min_confidence:
                continue

            selected_for_class.append({
                "dataset_idx": dataset_idx,
                "class_idx": cls_idx,
                "class_name": class_names[cls_idx],
                "quality_score": quality_score,
                "predicted_class": pred_class,
                "confidence": confidence,
            })

            print(
                f"  Accepted sample {dataset_idx} "
                f"(quality={quality_score:.3f}, confidence={confidence:.3f})"
            )

            if len(selected_for_class) >= args.samples_per_class:
                break

            if evaluated % 25 == 0:
                print(
                    f"  Checked {evaluated} candidates, "
                    f"accepted {len(selected_for_class)}"
                )

        if len(selected_for_class) < args.samples_per_class:
            print(
                f"WARNING: Only found {len(selected_for_class)}/"
                f"{args.samples_per_class} correctly predicted samples "
                f"for {class_names[cls_idx]}"
            )

        selected.extend(selected_for_class)

    print(
        f"\nSelected {len(selected)} curated samples for explanation "
        f"(target: {args.samples_per_class} per class)"
    )

    # ---- Phase 3: Run explanations ----
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    run_metadata = build_run_metadata(
        args=args,
        dataset_name=dataset_name,
        class_names=class_names,
        seed=seed,
    )

    all_results = []
    for i, sel in enumerate(selected):
        idx = sel["dataset_idx"]
        cls_name = sel["class_name"]
        print(f"\n[{i+1}/{len(selected)}] Sample {idx} ({cls_name}, "
              f"quality={sel['quality_score']:.3f}, "
              f"confidence={sel['confidence']:.3f})")

        sample = dataset[idx]
        sample_dir = os.path.join(output_dir, f"sample_{idx}")

        result = explain_single_curated(
            model, sample, gradcam, shap_explainer, faithfulness,
            nlg_generator, class_names, sample_dir, idx, device,
            min_confidence=args.min_confidence,
        )

        if result is not None:
            all_results.append(result)
        else:
            print(f"    Skipped during explanation")

    # ---- Phase 4: Save summary ----
    summary = {
        "total_selected": len(selected),
        "total_explained": len(all_results),
        "samples_per_class": args.samples_per_class,
        "min_confidence": args.min_confidence,
        "results": all_results,
    }

    summary_path = os.path.join(output_dir, "curated_summary.json")
    with open(summary_path, "w") as f:
        json.dump(sanitize_for_json(summary), f, indent=2, default=str)

    metadata_path = os.path.join(output_dir, "run_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(
            sanitize_for_json(
                {
                    **run_metadata,
                    "total_selected": len(selected),
                    "total_explained": len(all_results),
                    "selected_sample_ids": [
                        item["dataset_idx"] for item in selected
                    ],
                    "explained_sample_ids": [
                        result["sample_id"] for result in all_results
                    ],
                }
            ),
            f,
            indent=2,
            default=str,
        )

    print(f"\n{'=' * 60}")
    print(f"Curated explanations complete!")
    print(f"  Total explained: {len(all_results)}/{len(selected)}")
    print(f"  Output directory: {output_dir}")
    print(f"{'=' * 60}")

    # Print per-class breakdown
    from collections import Counter
    class_counts = Counter(r["true_emotion"] for r in all_results)
    print("\nPer-class breakdown:")
    for cls_name in class_names:
        count = class_counts.get(cls_name, 0)
        status = "[OK]" if count >= args.samples_per_class else "[!!]"
        print(f"  {status} {cls_name}: {count}/{args.samples_per_class}")

    # Cleanup
    gradcam.remove_hooks()


if __name__ == "__main__":
    main()
