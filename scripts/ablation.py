"""
Ablation Study Script.

Runs systematic experiments comparing:
  1. Vision-only (EfficientNet-B2 alone)
  2. Text-only (DeBERTa-v3-base alone)
  3. Multimodal + Concat fusion
  4. Multimodal + Gated fusion
  5. Multimodal + Cross-Attention fusion (proposed)

Outputs a LaTeX comparison table for the paper.

Usage:
    python scripts/ablation.py --config config/config.yaml
    python scripts/ablation.py --config config/config.yaml --quick  # Fewer epochs
"""

import argparse
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.meld_dataset import MELDDataset
from data.affectnet_dataset import AffectNetDataset
from models.multimodal_model import build_model

from training.trainer import Trainer
from training.metrics import EmotionMetrics
from utils.helpers import set_seed, load_config, get_device


def parse_args():
    parser = argparse.ArgumentParser(description="Run Ablation Studies")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--quick", action="store_true",
                        help="Run with fewer epochs for quick iteration")
    parser.add_argument("--output-dir", type=str, default="./results/ablation")
    return parser.parse_args()


def build_ablation_model(config: dict, mode: str, fusion: str):
    """Build a model variant for ablation using the standard factory."""
    from copy import deepcopy
    abl_config = deepcopy(config)
    abl_config.setdefault("model", {})["mode"] = mode
    abl_config["model"].setdefault("fusion", {})["strategy"] = fusion
    if mode == "vision_only":
        abl_config["model"]["vision"]["freeze_layers"] = 2
    return build_model(abl_config, mode=mode)


def main():
    args = parse_args()
    config = load_config(args.config)
    device = get_device()
    set_seed(config.get("project", {}).get("seed", 42))

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    dataset_name = config["dataset"]["name"]
    dataset_cfg = config["dataset"].get(dataset_name, {})
    num_classes = dataset_cfg.get("num_classes", 7)
    class_names = dataset_cfg.get("class_names", [str(i) for i in range(num_classes)])

    # Override epochs for ablation
    ablation_cfg = config.get("ablation", {})
    if args.quick:
        epochs = ablation_cfg.get("quick_epochs", 10)
    else:
        epochs = config["training"]["epochs"]
    config["training"]["epochs"] = epochs

    # Define ablation experiments
    experiments = [
        {"name": "Vision-Only (EfficientNet-B2)", "mode": "vision_only", "fusion": "concat"},
        {"name": "Text-Only (DeBERTa-v3-base)", "mode": "text_only", "fusion": "concat"},
        {"name": "Multimodal + Concat", "mode": "multimodal", "fusion": "concat"},
        {"name": "Multimodal + Gated", "mode": "multimodal", "fusion": "gated"},
        {"name": "Multimodal + Cross-Attention (Ours)", "mode": "multimodal", "fusion": "cross_attention"},
    ]

    # Build dataloaders (shared across experiments)
    print(f"\nLoading {dataset_name} dataset...")
    text_backbone = config["model"]["text"]["backbone"]
    batch_size = config["training"]["batch_size"]

    if dataset_name == "meld":
        train_dataset = MELDDataset(
            root_dir=dataset_cfg.get("root_dir", "./datasets/meld"),
            split="train",
            text_model_name=text_backbone,
        )
        val_dataset = MELDDataset(
            root_dir=dataset_cfg.get("root_dir", "./datasets/meld"),
            split="dev",
            text_model_name=text_backbone,
        )
        test_dataset = MELDDataset(
            root_dir=dataset_cfg.get("root_dir", "./datasets/meld"),
            split="test",
            text_model_name=text_backbone,
        )
    else:
        train_dataset = AffectNetDataset(
            root_dir=dataset_cfg.get("root_dir", "./datasets/affectnet"),
            split="train", num_classes=num_classes,
        )
        val_dataset = AffectNetDataset(
            root_dir=dataset_cfg.get("root_dir", "./datasets/affectnet"),
            split="val", num_classes=num_classes,
        )
        test_dataset = val_dataset  # AffectNet doesn't have a separate test split

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4,
    )

    # Run experiments
    all_results = []

    for exp in experiments:
        print(f"\n{'=' * 60}")
        print(f"ABLATION: {exp['name']}")
        print(f"{'=' * 60}")

        # Reset seed for each experiment (fair comparison)
        set_seed(config.get("project", {}).get("seed", 42))

        # Build model
        model = build_ablation_model(config, exp["mode"], exp["fusion"])

        # Update checkpoint dir for this experiment
        exp_ckpt_dir = os.path.join(output_dir, exp["name"].replace(" ", "_").replace("(", "").replace(")", ""))
        config["project"]["checkpoint_dir"] = exp_ckpt_dir
        config["project"]["log_dir"] = os.path.join(exp_ckpt_dir, "logs")

        # Train
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device,
        )
        trainer.model_mode = exp["mode"]
        trainer.fusion_strategy = exp["fusion"]
        trainer.train_class_distribution = dict(train_dataset.get_class_distribution())

        best_val_metrics = trainer.train()

        # Evaluate on test set
        print("\nEvaluating on test set...")
        model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in test_loader:
                image = batch["image"].to(device)
                forward_kwargs = {"image": image}
                if "input_ids" in batch and exp["mode"] != "vision_only":
                    forward_kwargs["input_ids"] = batch["input_ids"].to(device)
                    forward_kwargs["attention_mask"] = batch["attention_mask"].to(device)

                output = model(**forward_kwargs)
                preds = output["logits"].argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(batch["label"].numpy())

        metrics = EmotionMetrics(num_classes, class_names)
        test_results = metrics.compute(np.array(all_preds), np.array(all_labels))

        result = {
            "name": exp["name"],
            "mode": exp["mode"],
            "fusion": exp["fusion"],
            "val_metrics": best_val_metrics,
            "test_metrics": test_results,
        }
        all_results.append(result)

        print(f"\n  Test results for {exp['name']}:")
        print(f"    Accuracy: {test_results['accuracy']:.4f}")
        print(f"    F1 (weighted): {test_results['f1_weighted']:.4f}")
        print(f"    F1 (macro): {test_results['f1_macro']:.4f}")

    # Generate comparison table
    print(f"\n{'=' * 60}")
    print("ABLATION STUDY RESULTS")
    print(f"{'=' * 60}\n")

    header = f"{'Model':<45} {'Accuracy':>10} {'F1-W':>8} {'F1-M':>8}"
    print(header)
    print("-" * len(header))
    for r in all_results:
        t = r["test_metrics"]
        print(
            f"{r['name']:<45} {t['accuracy']:>10.4f} "
            f"{t['f1_weighted']:>8.4f} {t['f1_macro']:>8.4f}"
        )

    # Generate LaTeX table
    latex_lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{Ablation study on model components and fusion strategies.}",
        "\\label{tab:ablation}",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Model & Accuracy & F1 (Weighted) & F1 (Macro) \\\\",
        "\\midrule",
    ]

    for r in all_results:
        t = r["test_metrics"]
        is_ours = "Ours" in r["name"]
        name = r["name"]
        if is_ours:
            latex_lines.append(
                f"\\textbf{{{name}}} & \\textbf{{{t['accuracy']:.4f}}} "
                f"& \\textbf{{{t['f1_weighted']:.4f}}} "
                f"& \\textbf{{{t['f1_macro']:.4f}}} \\\\"
            )
        else:
            latex_lines.append(
                f"{name} & {t['accuracy']:.4f} & {t['f1_weighted']:.4f} "
                f"& {t['f1_macro']:.4f} \\\\"
            )

    latex_lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])

    latex_table = "\n".join(latex_lines)
    print(f"\nLaTeX Table:\n{latex_table}")

    # Save everything
    results_path = os.path.join(output_dir, "ablation_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    latex_path = os.path.join(output_dir, "ablation_table.tex")
    with open(latex_path, "w") as f:
        f.write(latex_table)

    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()
