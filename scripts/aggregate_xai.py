"""
Aggregate XAI Statistics — Per-Emotion Face Region & Text Token Analysis.

Runs Grad-CAM on the ENTIRE test set and aggregates:
  1. Per-emotion mean region scores (which face part matters most for each emotion?)
  2. Per-emotion top text tokens (via SHAP on a stratified sample)
  3. Summary tables for the paper (LaTeX + JSON)

This is required for the publication — reviewers need aggregate statistics,
not just cherry-picked example heatmaps.

Usage:
    python scripts/aggregate_xai.py \
        --config config/config.yaml \
        --checkpoint results/checkpoints/meld_multimodal_best.pt \
        --output-dir results/aggregate_xai
"""

import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.meld_dataset import MELDDataset
from explainers.gradcam import GradCAMExplainer
from models.multimodal_model import build_model
from scripts.evaluate import (
    _build_meld_dataset_kwargs,
    _build_model_from_checkpoint,
    _resolve_checkpoint_mode,
)
from utils.helpers import get_device, load_config, set_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate XAI statistics across entire test set"
    )
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="meld")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--output-dir", type=str, default="./results/aggregate_xai")
    parser.add_argument(
        "--shap-samples-per-class", type=int, default=15,
        help="Number of samples per class for (expensive) SHAP analysis",
    )
    parser.add_argument(
        "--skip-shap", action="store_true",
        help="Skip SHAP analysis (only run Grad-CAM aggregation)",
    )
    return parser.parse_args()


def _get_target_layer(model):
    """Resolve the Grad-CAM target layer using the vision encoder's built-in method."""
    return model.vision_encoder.get_gradcam_target_layer()


def _unnormalize_image(image_tensor):
    """Convert normalized tensor back to uint8 RGB for MediaPipe."""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = image_tensor.cpu().numpy().transpose(1, 2, 0)  # [H, W, 3]
    img = (img * std + mean) * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def aggregate_gradcam(model, dataloader, config, device, class_names):
    """Run Grad-CAM on every test sample and aggregate region scores by emotion."""
    target_layer = _get_target_layer(model)
    explainer = GradCAMExplainer(
        model=model,
        target_layer=target_layer,
        device=device,
        use_gradcam_pp=True,
    )

    # Accumulators: {emotion_name: {region_name: [scores]}}
    region_accum = defaultdict(lambda: defaultdict(list))
    confidence_accum = defaultdict(list)
    correct_accum = defaultdict(lambda: {"correct": 0, "total": 0})

    # All known regions
    all_regions = set()

    print("\n[Aggregate XAI] Running Grad-CAM on full test set...")
    for batch in tqdm(dataloader, desc="Grad-CAM scan"):
        images = batch["image"]
        labels = batch["label"]
        batch_size = images.shape[0]

        # Prepare multimodal inputs
        input_ids = batch.get("input_ids")
        attention_mask = batch.get("attention_mask")

        for i in range(batch_size):
            img_tensor = images[i].unsqueeze(0)
            label_idx = int(labels[i].item())
            emotion = class_names[label_idx]

            # Get original image for landmark detection
            original_image = _unnormalize_image(images[i])

            # Build forward kwargs
            ids = input_ids[i].unsqueeze(0) if input_ids is not None else None
            mask = attention_mask[i].unsqueeze(0) if attention_mask is not None else None

            try:
                result = explainer.generate(
                    image=img_tensor,
                    target_class=None,  # Use model's predicted class
                    input_ids=ids,
                    attention_mask=mask,
                    original_image=original_image,
                )
            except Exception:
                continue

            pred_class = result["predicted_class"]
            confidence = result["confidence"]
            region_scores = result.get("region_scores", {})

            # Track per ground-truth emotion
            confidence_accum[emotion].append(confidence)
            correct_accum[emotion]["total"] += 1
            if pred_class == label_idx:
                correct_accum[emotion]["correct"] += 1

            # Accumulate region scores (keyed by PREDICTED emotion for XAI faithfulness)
            pred_emotion = class_names[pred_class]
            for region_name, score in region_scores.items():
                if region_name not in ("total_activation",):
                    region_accum[pred_emotion][region_name].append(score)
                    all_regions.add(region_name)

    explainer.remove_hooks()

    # Compute statistics
    region_stats = {}
    for emotion in class_names:
        region_stats[emotion] = {}
        for region in sorted(all_regions):
            scores = region_accum[emotion].get(region, [])
            if scores:
                region_stats[emotion][region] = {
                    "mean": float(np.mean(scores)),
                    "std": float(np.std(scores)),
                    "n": len(scores),
                }
            else:
                region_stats[emotion][region] = {
                    "mean": 0.0,
                    "std": 0.0,
                    "n": 0,
                }

    # Summary: top region per emotion
    top_regions = {}
    for emotion in class_names:
        if region_stats[emotion]:
            sorted_regions = sorted(
                region_stats[emotion].items(),
                key=lambda x: x[1]["mean"],
                reverse=True,
            )
            if sorted_regions:
                top_regions[emotion] = {
                    "region": sorted_regions[0][0],
                    "mean_score": sorted_regions[0][1]["mean"],
                    "ranking": [
                        {"region": r, "mean": s["mean"], "std": s["std"]}
                        for r, s in sorted_regions
                    ],
                }

    return {
        "region_stats": region_stats,
        "top_regions": top_regions,
        "confidence_per_emotion": {
            e: {
                "mean": float(np.mean(v)),
                "std": float(np.std(v)),
                "n": len(v),
            }
            for e, v in confidence_accum.items()
        },
        "accuracy_per_emotion": {
            e: {
                "accuracy": v["correct"] / max(v["total"], 1),
                "correct": v["correct"],
                "total": v["total"],
            }
            for e, v in correct_accum.items()
        },
    }


def aggregate_shap(model, dataset, config, device, class_names, samples_per_class):
    """Run SHAP on a stratified sample and aggregate top tokens per emotion."""
    from explainers.shap_text import SHAPTextExplainer

    shap_explainer = SHAPTextExplainer(model=model, device=device)

    # Stratified sample selection
    class_indices = defaultdict(list)
    for idx in range(len(dataset)):
        label = dataset[idx]["label"]
        if isinstance(label, torch.Tensor):
            label = label.item()
        class_indices[int(label)].append(idx)

    selected_indices = []
    for class_idx, indices in class_indices.items():
        np.random.shuffle(indices)
        selected_indices.extend(indices[:samples_per_class])

    # Token accumulators: {emotion: {token: [importance_scores]}}
    token_accum = defaultdict(lambda: defaultdict(list))

    print(f"\n[Aggregate XAI] Running SHAP on {len(selected_indices)} stratified samples...")
    for sample_idx in tqdm(selected_indices, desc="SHAP scan"):
        sample = dataset[sample_idx]
        image = sample["image"].unsqueeze(0)
        label_idx = int(sample["label"].item()) if isinstance(sample["label"], torch.Tensor) else int(sample["label"])
        text_input = sample.get("text_input", "")

        try:
            shap_result = shap_explainer.explain(
                text=text_input,
                image=image,
                target_class=None,  # Use predicted class
            )
        except Exception:
            continue

        pred_class = shap_result.get("predicted_class", label_idx)
        pred_emotion = class_names[pred_class]
        tokens = shap_result.get("tokens", [])
        values = shap_result.get("shap_values", [])

        if len(tokens) != len(values):
            continue

        for token, value in zip(tokens, values):
            # Skip special tokens and very short tokens
            clean = token.strip().lower().replace("▁", "").replace("Ġ", "")
            if len(clean) < 2 or clean in ("[cls]", "[sep]", "[pad]", "<s>", "</s>"):
                continue
            token_accum[pred_emotion][clean].append(abs(float(value)))

    # Compute top tokens per emotion
    top_tokens = {}
    for emotion in class_names:
        token_scores = {}
        for token, scores in token_accum.get(emotion, {}).items():
            token_scores[token] = {
                "mean_importance": float(np.mean(scores)),
                "frequency": len(scores),
                "total_importance": float(np.sum(scores)),
            }
        sorted_tokens = sorted(
            token_scores.items(),
            key=lambda x: x[1]["mean_importance"],
            reverse=True,
        )
        top_tokens[emotion] = [
            {"token": t, **s} for t, s in sorted_tokens[:20]
        ]

    return {"top_tokens_per_emotion": top_tokens}


def generate_latex_table(gradcam_data, shap_data, class_names, output_path):
    """Generate LaTeX table for paper: per-emotion top region + top tokens."""
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Aggregate XAI analysis: dominant facial region and most influential text tokens per emotion class, computed across the full MELD test set.}",
        "\\label{tab:aggregate_xai}",
        "\\begin{tabular}{lllcl}",
        "\\toprule",
        "Emotion & Top Face Region & Contribution & Accuracy & Top Text Tokens \\\\",
        "\\midrule",
    ]

    top_regions = gradcam_data.get("top_regions", {})
    accuracy_data = gradcam_data.get("accuracy_per_emotion", {})
    top_tokens = shap_data.get("top_tokens_per_emotion", {}) if shap_data else {}

    for emotion in class_names:
        region_info = top_regions.get(emotion, {})
        region_name = region_info.get("region", "N/A").replace("_", "\\_")
        region_score = region_info.get("mean_score", 0)
        acc = accuracy_data.get(emotion, {}).get("accuracy", 0)
        tokens = top_tokens.get(emotion, [])
        token_str = ", ".join(
            f"\\textit{{{t['token']}}}" for t in tokens[:5]
        ) if tokens else "---"
        lines.append(
            f"{emotion} & {region_name} & {region_score:.1%} & {acc:.1%} & {token_str} \\\\"
        )

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table*}",
    ])

    latex = "\n".join(lines)
    with open(output_path, "w") as f:
        f.write(latex)
    return latex


def main():
    args = parse_args()
    config = load_config(args.config)
    device = get_device()
    set_seed(config.get("project", {}).get("seed", 42))
    os.makedirs(args.output_dir, exist_ok=True)

    dataset_name = args.dataset or config["dataset"]["name"]
    dataset_cfg = config["dataset"].get(dataset_name, {})
    class_names = dataset_cfg.get("class_names", [str(i) for i in range(7)])

    # Load checkpoint
    print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = _build_model_from_checkpoint(config, checkpoint).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"  Mode: {_resolve_checkpoint_mode(checkpoint, config)}")
    print(f"  Epoch: {checkpoint.get('epoch', '?')}")

    # Build dataset + loader
    print(f"\nLoading {dataset_name} {args.split} set...")
    dataset = MELDDataset(**_build_meld_dataset_kwargs(config, split=args.split))
    dataloader = DataLoader(
        dataset,
        batch_size=1,  # Process one at a time for Grad-CAM
        shuffle=False,
        num_workers=0,
    )

    # === Part 1: Grad-CAM aggregate (runs on full test set) ===
    gradcam_data = aggregate_gradcam(model, dataloader, config, device, class_names)
    gradcam_path = os.path.join(args.output_dir, "gradcam_aggregate.json")
    with open(gradcam_path, "w") as f:
        json.dump(gradcam_data, f, indent=2)
    print(f"\nGrad-CAM aggregate saved to {gradcam_path}")

    # Print summary table
    print(f"\n{'Emotion':<12} {'Top Region':<15} {'Score':>8} {'Accuracy':>10}")
    print("-" * 48)
    for emotion in class_names:
        info = gradcam_data["top_regions"].get(emotion, {})
        acc = gradcam_data["accuracy_per_emotion"].get(emotion, {}).get("accuracy", 0)
        print(
            f"{emotion:<12} {info.get('region', 'N/A'):<15} "
            f"{info.get('mean_score', 0):>7.1%} {acc:>9.1%}"
        )

    # === Part 2: SHAP aggregate (runs on stratified sample) ===
    shap_data = None
    if not args.skip_shap:
        mode = _resolve_checkpoint_mode(checkpoint, config)
        if mode in ("multimodal", "text_only"):
            shap_data = aggregate_shap(
                model, dataset, config, device, class_names,
                samples_per_class=args.shap_samples_per_class,
            )
            shap_path = os.path.join(args.output_dir, "shap_aggregate.json")
            with open(shap_path, "w") as f:
                json.dump(shap_data, f, indent=2)
            print(f"\nSHAP aggregate saved to {shap_path}")

            # Print top tokens
            print(f"\n{'Emotion':<12} {'Top 5 Tokens'}")
            print("-" * 60)
            for emotion in class_names:
                tokens = shap_data["top_tokens_per_emotion"].get(emotion, [])
                token_str = ", ".join(t["token"] for t in tokens[:5])
                print(f"{emotion:<12} {token_str}")
        else:
            print("\n[SHAP] Skipped — checkpoint is vision-only (no text encoder)")

    # === Part 3: LaTeX table ===
    latex_path = os.path.join(args.output_dir, "aggregate_xai_table.tex")
    latex = generate_latex_table(gradcam_data, shap_data, class_names, latex_path)
    print(f"\nLaTeX table saved to {latex_path}")

    print(f"\n{'=' * 60}")
    print("AGGREGATE XAI ANALYSIS COMPLETE")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
