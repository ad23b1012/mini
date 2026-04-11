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
        dataset = MELDDataset(
            root_dir=dataset_cfg.get("root_dir", "./datasets/meld"),
            split=args.split if args.split != "test" else "test",
            image_size=dataset_cfg.get("image_size", 260),
            text_model_name=config["model"]["text"]["backbone"],
        )
    else:
        split_map = {"test": "val", "val": "val", "train": "train"}
        dataset = AffectNetDataset(
            root_dir=dataset_cfg.get("root_dir", "./datasets/affectnet"),
            split=split_map.get(args.split, "val"),
            num_classes=num_classes,
        )

    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    # Build and load model
    print("\nLoading model from checkpoint...")
    model = build_model(config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"  Loaded from epoch {checkpoint.get('epoch', '?')}")

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
