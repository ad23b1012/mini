"""
Evaluation Script.

Loads a trained checkpoint and runs comprehensive evaluation on the test set.
Generates metrics, confusion matrix, classification report, and LaTeX tables.

Usage:
    python scripts/evaluate.py --config config/config.yaml \
        --checkpoint results/checkpoints/best_model.pt \
        --dataset meld --split test
"""

import argparse
from copy import deepcopy
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.meld_dataset import MELDDataset
from data.affectnet_dataset import AffectNetDataset
from models.multimodal_model import build_model
from training.metrics import EmotionMetrics
from utils.helpers import set_seed, load_config, get_device
from utils.visualization import plot_confusion_matrix


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, choices=["meld", "affectnet"])
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-dir", type=str, default="./results/evaluation")
    return parser.parse_args()


def _build_meld_dataset_kwargs(config: dict, split: str) -> dict:
    dataset_cfg = config.get("dataset", {}).get("meld", {})
    model_cfg = config.get("model", {})
    text_backbone = model_cfg.get("text", {}).get(
        "backbone", "microsoft/deberta-v3-base"
    )
    context_cfg = dataset_cfg.get("text_context", {})
    quality_cfg = dataset_cfg.get("face_quality", {})

    quality_filter = quality_cfg.get("filter_eval", False)
    if split == "train":
        quality_filter = quality_cfg.get("filter_train", False)

    split_name = split
    if split in {"val", "dev"}:
        split_name = "dev"

    return {
        "root_dir": dataset_cfg.get("root_dir", "./datasets/meld"),
        "split": split_name,
        "image_size": dataset_cfg.get("image_size", 260),
        "max_text_length": dataset_cfg.get("max_text_length", 128),
        "text_model_name": text_backbone,
        "use_dialogue_history": context_cfg.get("use_dialogue_history", False),
        "history_window": context_cfg.get("history_window", 0),
        "include_speaker_in_text": context_cfg.get("include_speaker", True),
        "context_separator": context_cfg.get("separator", " [SEP] "),
        "quality_filter": quality_filter,
        "min_face_quality_score": quality_cfg.get("min_quality_score", 0.0),
        "repair_invalid_faces": False,
        "refresh_quality_cache": False,
    }


def _resolve_checkpoint_mode(checkpoint: dict, config: dict) -> str:
    """Infer the model mode recorded in a checkpoint."""
    return (
        checkpoint.get("model_mode")
        or checkpoint.get("config", {}).get("model", {}).get("mode")
        or config.get("model", {}).get("mode")
        or "multimodal"
    )


def _resolve_checkpoint_fusion(checkpoint: dict, config: dict) -> str:
    """Infer the fusion strategy recorded in a checkpoint."""
    return (
        checkpoint.get("fusion_strategy")
        or checkpoint.get("config", {}).get("model", {}).get("fusion", {}).get("strategy")
        or config.get("model", {}).get("fusion", {}).get("strategy")
        or "cross_attention"
    )


def _build_model_from_checkpoint(config: dict, checkpoint: dict):
    """Recreate the architecture needed for a checkpoint."""
    model_config = deepcopy(checkpoint.get("config") or config)
    dataset_name = checkpoint.get("dataset_name") or model_config.get("dataset", {}).get("name", "meld")

    model_config.setdefault("dataset", {})
    model_config["dataset"]["name"] = dataset_name
    model_config["dataset"].setdefault(dataset_name, {})
    if "num_classes" in checkpoint:
        model_config["dataset"][dataset_name]["num_classes"] = checkpoint["num_classes"]
    if "class_names" in checkpoint:
        model_config["dataset"][dataset_name]["class_names"] = checkpoint["class_names"]

    model_config.setdefault("model", {})
    model_config["model"]["mode"] = _resolve_checkpoint_mode(checkpoint, model_config)
    model_config["model"].setdefault("fusion", {})
    model_config["model"]["fusion"]["strategy"] = _resolve_checkpoint_fusion(
        checkpoint, model_config
    )

    return build_model(model_config, mode=model_config["model"]["mode"])


def main():
    args = parse_args()
    config = load_config(args.config)
    device = get_device()
    set_seed(config.get("project", {}).get("seed", 42))

    dataset_name = args.dataset or config["dataset"]["name"]
    dataset_cfg = config["dataset"].get(dataset_name, {})
    num_classes = dataset_cfg.get("num_classes", 7)
    class_names = dataset_cfg.get("class_names", [str(i) for i in range(num_classes)])

    # Build dataset
    print(f"\nLoading {dataset_name} {args.split} set...")
    if dataset_name == "meld":
        dataset = MELDDataset(**_build_meld_dataset_kwargs(config, split=args.split if args.split != "test" else "test"))
    else:
        split_map = {"test": "val", "val": "val", "train": "train"}
        dataset = AffectNetDataset(
            root_dir=dataset_cfg.get("root_dir", "./datasets/affectnet"),
            split=split_map.get(args.split, "val"),
            num_classes=num_classes,
        )

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    num_workers = config.get("training", {}).get("num_workers", 0)
    pin_memory = config.get("training", {}).get("pin_memory", False)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # Build and load model
    print("\nLoading model from checkpoint...")
    model = _build_model_from_checkpoint(config, checkpoint).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"  Loaded from epoch {checkpoint.get('epoch', '?')}")
    print(f"  Model mode: {_resolve_checkpoint_mode(checkpoint, config)}")
    print(f"  Fusion strategy: {_resolve_checkpoint_fusion(checkpoint, config)}")

    # Run evaluation
    print("\nRunning evaluation...")
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            image = batch["image"].to(device)
            label = batch["label"]

            forward_kwargs = {"image": image}
            if "input_ids" in batch:
                forward_kwargs["input_ids"] = batch["input_ids"].to(device)
                forward_kwargs["attention_mask"] = batch["attention_mask"].to(device)

            output = model(**forward_kwargs)
            logits = output["logits"]
            probs = torch.softmax(logits, dim=1)

            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_labels.extend(label.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Compute metrics
    metrics = EmotionMetrics(num_classes=num_classes, class_names=class_names)
    results = metrics.compute(all_preds, all_labels)

    # Create output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    for key, value in results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
    print("=" * 60)

    # Classification report
    report = metrics.classification_report_str(all_preds, all_labels)
    print("\nClassification Report:")
    print(report)

    # Save results
    results_path = os.path.join(output_dir, "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    # Confusion matrix
    cm = metrics.confusion_matrix(all_preds, all_labels)
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plot_confusion_matrix(cm, class_names, save_path=cm_path)

    # LaTeX table
    latex = metrics.generate_latex_table(all_preds, all_labels)
    latex_path = os.path.join(output_dir, "results_table.tex")
    with open(latex_path, "w") as f:
        f.write(latex)
    print(f"LaTeX table saved to {latex_path}")

    # Save classification report
    report_path = os.path.join(output_dir, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)

    print(f"\nAll evaluation outputs saved to {output_dir}/")


if __name__ == "__main__":
    main()
