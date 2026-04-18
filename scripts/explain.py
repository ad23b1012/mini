"""
Explainability Pipeline Script.

Runs the full XAI pipeline on selected samples:
  1. Load trained model
  2. Run Grad-CAM on face image
  3. Run SHAP on text
  4. Compute cross-modal faithfulness metrics (CMFS)
  5. Generate NLG report
  6. Save visualizations

Usage:
    # Explain specific samples from test set
    python scripts/explain.py --config config/config.yaml \
        --checkpoint results/checkpoints/best_model.pt \
        --num-samples 10

    # Explain a single sample by index
    python scripts/explain.py --checkpoint results/checkpoints/best_model.pt \
        --sample-idx 42
"""

import argparse
from datetime import datetime, timezone
import json
import sys
import os

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
    plot_perturbation_curves,
)
from utils.face_quality import check_face_quality, get_face_quality_summary


def parse_args():
    parser = argparse.ArgumentParser(description="Run XAI Explanation Pipeline")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, choices=["meld", "affectnet"])
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--sample-idx", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="./results/explanations")
    parser.add_argument("--nlg-mode", type=str, default="template",
                        choices=["template", "llm"])
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducible sample selection")
    parser.add_argument("--quality-filter", action="store_true", default=True,
                        help="Filter out bad face crops before explaining")
    parser.add_argument("--no-quality-filter", dest="quality_filter",
                        action="store_false")
    parser.add_argument("--min-confidence", type=float, default=0.10,
                        help="Skip predictions below this confidence")
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
    """Describe how this explanation run was produced."""
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": os.path.abspath(__file__),
        "dataset": dataset_name,
        "checkpoint": os.path.abspath(args.checkpoint),
        "config": os.path.abspath(args.config),
        "output_dir": os.path.abspath(args.output_dir),
        "seed": int(seed),
        "sample_idx": args.sample_idx,
        "num_samples_requested": int(args.num_samples),
        "nlg_mode": args.nlg_mode,
        "quality_filter": bool(args.quality_filter),
        "min_confidence": float(args.min_confidence),
        "class_names": class_names,
    }


def explain_single_sample(
    model,
    sample: dict,
    gradcam_explainer: GradCAMExplainer,
    shap_explainer: SHAPTextExplainer,
    faithfulness,
    nlg_generator: NLGReportGenerator,
    class_names: list,
    output_dir: str,
    sample_id: int,
    device: str,
    quality_filter: bool = True,
    min_confidence: float = 0.10,
):
    """Run the full explanation pipeline on a single sample."""
    print(f"\n{'-' * 50}")
    print(f"Explaining sample {sample_id}")
    print(f"{'-' * 50}")

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

    # ---- Face Quality Check ----
    original_img = denormalize_image(sample["image"])

    if quality_filter:
        quality = check_face_quality(original_img)
        quality_summary = get_face_quality_summary(quality)
        print(f"  Face quality: {quality_summary}")

        if not quality["is_valid"]:
            print(f"  SKIPPED: Bad face crop ({quality['reason']})")
            return {"sample_id": sample_id, "skipped": True,
                    "skip_reason": quality["reason"],
                    "true_emotion": emotion_name}
    else:
        quality = {"quality_score": -1}
        original_img = denormalize_image(sample["image"])

    print(f"  True emotion: {emotion_name}")
    if utterance:
        print(f'  Utterance: "{utterance[:80]}"')

    # Create sample output directory
    sample_dir = os.path.join(output_dir, f"sample_{sample_id}")
    os.makedirs(sample_dir, exist_ok=True)

    # ---- 1. Grad-CAM ----
    print("  Running Grad-CAM...")

    # Pass target_class=None: Grad-CAM will explain the model's *predicted*
    # class (argmax of logits) rather than the ground-truth label.
    gradcam_result = gradcam_explainer.generate(
        image=image,
        target_class=None,
        input_ids=input_ids,
        attention_mask=attention_mask,
        original_image=original_img,
    )

    pred_confidence = gradcam_result['confidence']
    print(f"  Predicted: {class_names[gradcam_result['predicted_class']]} "
          f"({pred_confidence*100:.1f}%)")
    print(f"  Top face regions: {gradcam_result['region_scores']}")

    # Skip low-confidence predictions if quality filter is on
    if quality_filter and pred_confidence < min_confidence:
        print(f"  SKIPPED: Confidence too low ({pred_confidence*100:.1f}% < {min_confidence*100:.0f}%)")
        return {"sample_id": sample_id, "skipped": True,
                "skip_reason": f"low_confidence ({pred_confidence:.3f})",
                "true_emotion": emotion_name}

    # Use face-masked heatmap for visualization (eliminates background noise)
    vis_heatmap = gradcam_result.get("masked_heatmap", gradcam_result["heatmap"])

    # Save Grad-CAM visualization
    plot_gradcam_overlay(
        original_img,
        vis_heatmap,
        emotion_name=class_names[gradcam_result["predicted_class"]],
        confidence=gradcam_result["confidence"],
        save_path=os.path.join(sample_dir, "gradcam_overlay.png"),
    )


    # ---- 2. SHAP (text) ----
    shap_result = None
    if is_multimodal and utterance:
        print("  Running SHAP on text...")
        # Explain the model's predicted class, not the ground-truth label.
        # Pass image= so SHAP evaluates in genuine multimodal context.
        shap_result = shap_explainer.explain(
            text=text_input,
            image=image,
            target_class=gradcam_result["predicted_class"],
            emotion_names=class_names,
        )

        top_features = shap_explainer.get_top_features(shap_result, k=5)
        print(f"  Top positive tokens: {top_features['positive'][:3]}")

        # Save SHAP plot
        plot_shap_tokens(
            shap_result["tokens"],
            shap_result["shap_values"],
            emotion_name=class_names[shap_result["predicted_class"]],
            save_path=os.path.join(sample_dir, "shap_tokens.png"),
        )

    # ---- 3. Faithfulness Metrics ----
    faithfulness_metrics = None
    if is_multimodal and shap_result is not None:
        print("  Computing faithfulness metrics...")
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
                target_class=gradcam_result["predicted_class"],  # predicted, not ground truth
            )
            print(f"  CMFS Score: {faithfulness_metrics['cmfs_score']:.3f}")
            print(f"  Cross-Modal Agreement: {faithfulness_metrics['cross_modal_agreement']:.3f}")
        except Exception as e:
            print(f"  Warning: Faithfulness computation failed: {e}")

    # ---- 4. NLG Report ----
    print("  Generating NLG report...")
    token_importance = shap_result["token_importance"] if shap_result else {}

    report = nlg_generator.generate_report(
        emotion_name=class_names[gradcam_result["predicted_class"]],
        confidence=gradcam_result["confidence"],
        region_scores=gradcam_result["region_scores"],
        token_importance=token_importance,
        faithfulness_metrics=faithfulness_metrics,
        utterance=utterance,
        speaker=speaker,
    )

    print(f"\n{report}\n")

    # Save report
    report_path = os.path.join(sample_dir, "explanation_report.txt")
    with open(report_path, "w") as f:
        f.write(report)

    # ---- 5. Combined Visualization ----
    if shap_result is not None:
        plot_combined_explanation(
            original_image=original_img,
            heatmap=vis_heatmap,
            tokens=shap_result["tokens"],
            shap_values=shap_result["shap_values"],
            region_scores=gradcam_result["region_scores"],
            emotion_name=class_names[gradcam_result["predicted_class"]],
            confidence=gradcam_result["confidence"],
            utterance=utterance,
            faithfulness_metrics=faithfulness_metrics,
            save_path=os.path.join(sample_dir, "combined_explanation.png"),
        )

    # Save raw results as JSON
    results = {
        "sample_id": sample_id,
        "true_emotion": emotion_name,
        "predicted_emotion": class_names[gradcam_result["predicted_class"]],
        "confidence": gradcam_result["confidence"],
        "region_scores": gradcam_result["region_scores"],
        "utterance": utterance,
        "model_text_input": text_input,
        "speaker": speaker,
    }
    if token_importance:
        results["top_tokens"] = dict(list(token_importance.items())[:10])
    if faithfulness_metrics:
        results["faithfulness"] = faithfulness_metrics
    if quality_filter:
        results["face_quality_score"] = quality.get("quality_score", -1)

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

    # Grad-CAM
    target_layer = model.vision_encoder.get_gradcam_target_layer()
    gradcam = GradCAMExplainer(
        model=model, target_layer=target_layer, device=device
    )

    # SHAP
    xai_cfg = config.get("xai", {})
    shap_explainer = SHAPTextExplainer(
        model=model,
        tokenizer_name=config["model"]["text"]["backbone"],
        device=device,
        method=xai_cfg.get("shap", {}).get("method", "partition"),
        max_evals=xai_cfg.get("shap", {}).get("max_evals", 500),
    )

    # Faithfulness
    faithfulness = CrossModalFaithfulness(
        model=model,
        device=device,
        num_perturbation_steps=xai_cfg.get("faithfulness", {}).get(
            "num_perturbation_steps", 10
        ),
        top_k_features=xai_cfg.get("faithfulness", {}).get("top_k_features", 5),
    )

    # NLG
    nlg_cfg = config.get("nlg", {})
    nlg_generator = NLGReportGenerator(
        mode=args.nlg_mode,
        min_importance=nlg_cfg.get("template", {}).get("min_importance", 0.05),
        max_features=nlg_cfg.get("template", {}).get("max_features", 5),
    )

    # Run explanations
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    all_results = []
    run_metadata = build_run_metadata(
        args=args,
        dataset_name=dataset_name,
        class_names=class_names,
        seed=seed,
    )

    if args.sample_idx is not None:
        # Explain single sample
        sample = dataset[args.sample_idx]
        result = explain_single_sample(
            model, sample, gradcam, shap_explainer, faithfulness,
            nlg_generator, class_names, output_dir, args.sample_idx, device,
        )
        all_results.append(result)
    else:
        # Explain N random samples, trying extra to fill quota after filtering
        max_attempts = min(args.num_samples * 5, len(dataset))
        indices = np.random.choice(
            len(dataset), max_attempts, replace=False
        )
        explained_count = 0
        skipped_count = 0
        for idx in indices:
            if explained_count >= args.num_samples:
                break
            sample = dataset[int(idx)]
            result = explain_single_sample(
                model, sample, gradcam, shap_explainer, faithfulness,
                nlg_generator, class_names, output_dir, int(idx), device,
                quality_filter=args.quality_filter,
                min_confidence=args.min_confidence,
            )
            if result.get("skipped", False):
                skipped_count += 1
            else:
                explained_count += 1
            all_results.append(result)

        print(f"\n  Explained: {explained_count}, Skipped: {skipped_count}")

    # Save summary
    summary_path = os.path.join(output_dir, "explanation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(sanitize_for_json(all_results), f, indent=2, default=str)

    explained_results = [
        result for result in all_results if not result.get("skipped", False)
    ]
    skipped_results = [
        result for result in all_results if result.get("skipped", False)
    ]

    metadata_path = os.path.join(output_dir, "run_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(
            sanitize_for_json(
                {
                    **run_metadata,
                    "num_results_written": len(all_results),
                    "num_explained": len(explained_results),
                    "num_skipped": len(skipped_results),
                    "explained_sample_ids": [
                        result["sample_id"] for result in explained_results
                    ],
                    "skipped_sample_ids": [
                        result["sample_id"] for result in skipped_results
                    ],
                }
            ),
            f,
            indent=2,
            default=str,
        )

    print(f"\n{'=' * 60}")
    print(
        "Explanations generated for "
        f"{len(explained_results)} valid samples "
        f"(processed {len(all_results)}, skipped {len(skipped_results)})"
    )
    print(f"Output directory: {output_dir}")
    print(f"{'=' * 60}")

    # Cleanup
    gradcam.remove_hooks()


if __name__ == "__main__":
    main()
