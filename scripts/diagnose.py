"""
Training setup diagnostic script.

Audits:
  - MELD class imbalance after any quality filtering
  - resolved focal alpha weights
  - sampler behavior across a few mini-batches
  - optional transfer-checkpoint compatibility
"""

import argparse
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from data.meld_dataset import MELDDataset
from models.multimodal_model import build_model
from scripts.train import _build_meld_dataset_kwargs, resolve_auto_focal_alpha
from utils.helpers import get_device, load_config, load_transfer_weights, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Audit training data and transfer setup")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--sample-batches", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--transfer-checkpoint", type=str, default=None)
    parser.add_argument(
        "--transfer-component",
        type=str,
        default="vision_encoder",
        choices=["full_model", "vision_encoder", "text_encoder", "encoders"],
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    set_seed(config.get("project", {}).get("seed", 42))

    print("=" * 70)
    print("TRAINING SETUP AUDIT")
    print("=" * 70)

    train_dataset = MELDDataset(**_build_meld_dataset_kwargs(config, split="train"))
    print("\nFiltered train distribution:")
    for emotion, count in train_dataset.get_class_distribution().items():
        print(f"  {emotion}: {count}")

    resolve_auto_focal_alpha(config, train_dataset)
    loss_cfg = config.get("training", {}).get("loss", {})
    print("\nResolved loss configuration:")
    print(f"  loss name: {loss_cfg.get('name', 'focal')}")
    print(f"  focal gamma: {loss_cfg.get('focal_gamma', 2.0)}")
    print(f"  focal alpha: {loss_cfg.get('focal_alpha')}")

    print("\nSampler sanity check:")
    sampler = WeightedRandomSampler(
        weights=train_dataset.sample_weights,
        num_samples=len(train_dataset),
        replacement=True,
    )
    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=0,
    )

    batch_counter = Counter()
    for batch_idx, batch in enumerate(loader):
        labels = batch["label"].tolist()
        batch_counter.update(labels)
        print(f"  Batch {batch_idx + 1}: labels={labels}")
        if batch_idx + 1 >= args.sample_batches:
            break

    print("\nSampled label counts across inspected batches:")
    class_names = list(train_dataset.get_class_distribution().keys())
    for idx, name in enumerate(class_names):
        print(f"  {name}: {batch_counter.get(idx, 0)}")

    if args.transfer_checkpoint:
        print("\nTransfer compatibility check:")
        device = get_device()
        model = build_model(config).to(device)
        load_transfer_weights(
            model=model,
            checkpoint_path=args.transfer_checkpoint,
            component=args.transfer_component,
            map_location=device,
        )

    print("\nAudit complete.")


if __name__ == "__main__":
    main()
