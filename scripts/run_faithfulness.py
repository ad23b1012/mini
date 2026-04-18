"""
Run Faithfulness Metrics at Scale.

Computes sufficiency, comprehensiveness, and cross-modal agreement
on a stratified sample of the test set. Outputs aggregate statistics
and per-sample details for the paper.

Usage:
    python scripts/run_faithfulness.py \
        --config config/config.yaml \
        --checkpoint results/checkpoints/meld_multimodal_best.pt \
        --output-dir results/faithfulness
"""

import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from tqdm import tqdm

from data.meld_dataset import MELDDataset
from explainers.gradcam import GradCAMExplainer
from explainers.shap_text import SHAPTextExplainer
from explainers.faithfulness import CrossModalFaithfulness
from scripts.evaluate import (
    _build_meld_dataset_kwargs,
    _build_model_from_checkpoint,
    _resolve_checkpoint_mode,
)
from utils.helpers import get_device, load_config, set_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run faithfulness metrics on test samples"
    )
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="meld")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument(
        "--samples-per-class", type=int, default=10,
        help="Number of samples per emotion class to evaluate",
    )
    parser.add_argument("--output-dir", type=str, default="./results/faithfulness")
    return parser.parse_args()


def _get_target_layer(model):
    """Resolve the Grad-CAM target layer using the vision encoder's built-in method."""
    return model.vision_encoder.get_gradcam_target_layer()


def _unnormalize_image(image_tensor):
    """Convert normalized tensor back to uint8 RGB."""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = image_tensor.cpu().numpy().transpose(1, 2, 0)
    img = (img * std + mean) * 255.0
    return np.clip(img, 0, 255).astype(np.uint8)


def main():
    args = parse_args()
    config = load_config(args.config)
    device = get_device()
    set_seed(config.get("project", {}).get("seed", 42))
    os.makedirs(args.output_dir, exist_ok=True)

    dataset_name = args.dataset or config["dataset"]["name"]
    dataset_cfg = config["dataset"].get(dataset_name, {})
    class_names = dataset_cfg.get("class_names", [str(i) for i in range(7)])
    num_classes = len(class_names)

    # Load model
    print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = _build_model_from_checkpoint(config, checkpoint).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    mode = _resolve_checkpoint_mode(checkpoint, config)
    print(f"  Mode: {mode}")
    if mode == "vision_only":
        print("  WARNING: Faithfulness metrics need both modalities. Skipping text metrics.")

    # Build dataset
    print(f"\nLoading {dataset_name} {args.split}...")
    dataset = MELDDataset(**_build_meld_dataset_kwargs(config, split=args.split))

    # Stratified sampling
    class_indices = defaultdict(list)
    for idx in range(len(dataset)):
        label = dataset[idx]["label"]
        label_int = label.item() if isinstance(label, torch.Tensor) else int(label)
        class_indices[label_int].append(idx)

    selected = []
    for cls_idx, indices in sorted(class_indices.items()):
        np.random.shuffle(indices)
        selected.extend(indices[: args.samples_per_class])
    print(f"  Selected {len(selected)} samples ({args.samples_per_class}/class)")

    # Build explainers
    target_layer = _get_target_layer(model)
    gradcam = GradCAMExplainer(
        model=model, target_layer=target_layer,
        device=device, use_gradcam_pp=True,
    )
    shap_explainer = None
    if mode in ("multimodal", "text_only"):
        shap_explainer = SHAPTextExplainer(model=model, device=device)

    faithfulness = CrossModalFaithfulness(
        model=model, device=device,
        num_perturbation_steps=config.get("xai", {}).get(
            "faithfulness", {}
        ).get("num_perturbation_steps", 10),
        top_k_features=config.get("xai", {}).get(
            "faithfulness", {}
        ).get("top_k_features", 5),
    )

    # Run faithfulness on each sample
    all_metrics = []
    per_class_metrics = defaultdict(list)

    print(f"\nRunning faithfulness evaluation on {len(selected)} samples...")
    for sample_idx in tqdm(selected, desc="Faithfulness"):
        sample = dataset[sample_idx]
        image = sample["image"].unsqueeze(0).to(device)
        label_idx = sample["label"].item() if isinstance(sample["label"], torch.Tensor) else int(sample["label"])
        emotion = class_names[label_idx]

        input_ids = sample.get("input_ids")
        attention_mask = sample.get("attention_mask")
        if input_ids is not None:
            input_ids = input_ids.unsqueeze(0).to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(0).to(device)

        original_image = _unnormalize_image(sample["image"])

        try:
            # Grad-CAM
            gc_result = gradcam.generate(
                image=image, target_class=None,
                input_ids=input_ids, attention_mask=attention_mask,
                original_image=original_image,
            )
            heatmap = gc_result["heatmap"]
            pred_class = gc_result["predicted_class"]

            # SHAP (if multimodal)
            shap_values = np.zeros(1)
            model_token_scores = np.zeros(1)
            special_token_mask = np.zeros(1, dtype=bool)
            tokens = []
            text_input = sample.get("text_input", "")

            if shap_explainer is not None and text_input:
                shap_result = shap_explainer.explain(
                    text=text_input,
                    image=image,
                    target_class=pred_class,
                )
                shap_values = shap_result.get("shap_values", np.zeros(1))
                tokens = shap_result.get("tokens", [])
                model_token_scores = shap_result.get("model_token_scores", np.zeros(1))
                special_token_mask = shap_result.get("special_token_mask", np.zeros(1, dtype=bool))

            # Faithfulness metrics
            if input_ids is not None and shap_explainer is not None:
                metrics = faithfulness.compute_all_metrics(
                    image=image,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    gradcam_heatmap=heatmap,
                    shap_values=shap_values,
                    model_token_scores=model_token_scores,
                    special_token_mask=special_token_mask,
                    tokens=tokens,
                    target_class=pred_class,
                )
            else:
                # Vision-only: only vision metrics
                metrics = {
                    "vision_sufficiency": faithfulness._vision_sufficiency(
                        image, input_ids, attention_mask, heatmap, pred_class
                    ),
                    "vision_comprehensiveness": faithfulness._vision_comprehensiveness(
                        image, input_ids, attention_mask, heatmap, pred_class
                    ),
                }

            # Store
            record = {
                "sample_idx": sample_idx,
                "true_emotion": emotion,
                "predicted_class": pred_class,
                "predicted_emotion": class_names[pred_class],
                "correct": pred_class == label_idx,
                **{k: float(v) if np.isfinite(v) else None for k, v in metrics.items()},
            }
            all_metrics.append(record)
            per_class_metrics[emotion].append(metrics)

        except Exception as e:
            print(f"  Skip sample {sample_idx}: {e}")
            continue

    gradcam.remove_hooks()

    # Aggregate statistics
    metric_names = set()
    for records in per_class_metrics.values():
        for r in records:
            metric_names.update(r.keys())

    aggregate = {}
    for metric_name in sorted(metric_names):
        all_values = []
        for records in per_class_metrics.values():
            for r in records:
                v = r.get(metric_name, float("nan"))
                if np.isfinite(v):
                    all_values.append(v)
        if all_values:
            aggregate[metric_name] = {
                "mean": float(np.mean(all_values)),
                "std": float(np.std(all_values)),
                "median": float(np.median(all_values)),
                "n": len(all_values),
            }

    per_class_summary = {}
    for emotion in class_names:
        per_class_summary[emotion] = {}
        for metric_name in sorted(metric_names):
            values = [
                r.get(metric_name, float("nan"))
                for r in per_class_metrics.get(emotion, [])
            ]
            finite = [v for v in values if np.isfinite(v)]
            if finite:
                per_class_summary[emotion][metric_name] = {
                    "mean": float(np.mean(finite)),
                    "std": float(np.std(finite)),
                    "n": len(finite),
                }

    # Save results
    output = {
        "checkpoint": args.checkpoint,
        "mode": mode,
        "samples_per_class": args.samples_per_class,
        "total_evaluated": len(all_metrics),
        "aggregate": aggregate,
        "per_class": per_class_summary,
        "per_sample": all_metrics,
    }

    results_path = os.path.join(args.output_dir, "faithfulness_results.json")
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    # Generate LaTeX table
    latex_lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{Faithfulness metrics for multimodal explanations (mean $\\pm$ std across MELD test samples).}",
        "\\label{tab:faithfulness}",
        "\\begin{tabular}{lcc}",
        "\\toprule",
        "Metric & Mean & Std \\\\",
        "\\midrule",
    ]
    display_names = {
        "vision_sufficiency": "Vision Sufficiency",
        "vision_comprehensiveness": "Vision Comprehensiveness",
        "text_sufficiency": "Text Sufficiency",
        "text_comprehensiveness": "Text Comprehensiveness",
        "cross_modal_agreement": "Cross-Modal Agreement",
        "cmfs_score": "\\textbf{CMFS Score}",
        "vision_auc_fidelity": "Vision AUC Fidelity",
        "text_auc_fidelity": "Text AUC Fidelity",
    }
    for key, display in display_names.items():
        if key in aggregate:
            m = aggregate[key]["mean"]
            s = aggregate[key]["std"]
            latex_lines.append(f"{display} & {m:.4f} & {s:.4f} \\\\")

    latex_lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    latex = "\n".join(latex_lines)

    latex_path = os.path.join(args.output_dir, "faithfulness_table.tex")
    with open(latex_path, "w") as f:
        f.write(latex)

    # Print summary
    print(f"\n{'=' * 60}")
    print("FAITHFULNESS METRICS SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Samples evaluated: {len(all_metrics)}")
    for key, display in display_names.items():
        if key in aggregate:
            m = aggregate[key]
            print(f"  {display.replace(chr(92), ''):<30} {m['mean']:.4f} ± {m['std']:.4f}")
    print(f"\n  Results: {results_path}")
    print(f"  LaTeX:   {latex_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
