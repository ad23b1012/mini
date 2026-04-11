"""
Main Training Script.

Usage:
    # Train on MELD (multimodal — face + text)
    python scripts/train.py --config config/config.yaml --dataset meld

    # Train on AffectNet (vision-only)
    python scripts/train.py --config config/config.yaml --dataset affectnet --mode vision_only

    # Quick debug run
    python scripts/train.py --config config/config.yaml --debug

    # Resume from checkpoint
    python scripts/train.py --config config/config.yaml --resume results/checkpoints/best_model.pt
"""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from data.meld_dataset import MELDDataset
from data.affectnet_dataset import AffectNetDataset
from models.multimodal_model import build_model, MultimodalEmotionModel
from training.trainer import Trainer
from utils.helpers import set_seed, load_config, get_device


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Multimodal Emotion Recognition Model"
    )
    parser.add_argument(
        "--config", type=str, default="config/config.yaml",
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--dataset", type=str, choices=["meld", "affectnet"],
        help="Override dataset from config",
    )
    parser.add_argument(
        "--mode", type=str,
        choices=["multimodal", "vision_only", "text_only"],
        help="Override model mode",
    )
    parser.add_argument(
        "--epochs", type=int, help="Override number of epochs",
    )
    parser.add_argument(
        "--batch-size", type=int, help="Override batch size",
    )
    parser.add_argument(
        "--lr", type=float, help="Override learning rate",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume training from",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Quick debug run (1 epoch, small batch)",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Override device (cuda/cpu/mps)",
    )
    return parser.parse_args()


def build_dataloaders(config: dict, dataset_name: str, mode: str):
    """Build train and validation data loaders."""
    train_cfg = config.get("training", {})
    dataset_cfg = config.get("dataset", {}).get(dataset_name, {})
    model_cfg = config.get("model", {})

    batch_size = train_cfg.get("batch_size", 32)
    num_workers = train_cfg.get("num_workers", 4)
    pin_memory = train_cfg.get("pin_memory", True)
    text_backbone = model_cfg.get("text", {}).get(
        "backbone", "microsoft/deberta-v3-base"
    )

    if dataset_name == "meld":
        train_dataset = MELDDataset(
            root_dir=dataset_cfg.get("root_dir", "./datasets/meld"),
            split="train",
            image_size=dataset_cfg.get("image_size", 260),
            max_text_length=dataset_cfg.get("max_text_length", 128),
            text_model_name=text_backbone,
        )
        val_dataset = MELDDataset(
            root_dir=dataset_cfg.get("root_dir", "./datasets/meld"),
            split="dev",
            image_size=dataset_cfg.get("image_size", 260),
            max_text_length=dataset_cfg.get("max_text_length", 128),
            text_model_name=text_backbone,
        )
    else:  # affectnet
        train_dataset = AffectNetDataset(
            root_dir=dataset_cfg.get("root_dir", "./datasets/affectnet"),
            split="train",
            num_classes=dataset_cfg.get("num_classes", 8),
            image_size=dataset_cfg.get("image_size", 260),
        )
        val_dataset = AffectNetDataset(
            root_dir=dataset_cfg.get("root_dir", "./datasets/affectnet"),
            split="val",
            num_classes=dataset_cfg.get("num_classes", 8),
            image_size=dataset_cfg.get("image_size", 260),
        )

    # Print class distribution
    print("\nClass distribution:")
    train_dist = train_dataset.get_class_distribution()
    for emotion, count in train_dist.items():
        print(f"  {emotion}: {count}")

    # Balanced sampling for training
    sampler = None
    shuffle = True
    if hasattr(train_dataset, "sample_weights"):
        sampler = WeightedRandomSampler(
            weights=train_dataset.sample_weights,
            num_samples=len(train_dataset),
            replacement=True,
        )
        shuffle = False
        print("\nUsing balanced sampling with WeightedRandomSampler")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader


def main():
    args = parse_args()

    # Load config
    config = load_config(args.config)

    # Apply CLI overrides
    if args.dataset:
        config["dataset"]["name"] = args.dataset
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    if args.lr:
        config["training"]["optimizer"]["lr"] = args.lr

    # Debug mode
    if args.debug:
        config["training"]["epochs"] = 1
        config["training"]["batch_size"] = 4
        config["training"]["log_every_n_steps"] = 5
        print("⚡ Debug mode: 1 epoch, batch_size=4")

    # Set seed
    project_cfg = config.get("project", {})
    set_seed(project_cfg.get("seed", 42))

    # Device
    device = args.device or get_device()

    # Dataset
    dataset_name = config["dataset"]["name"]
    mode = args.mode or "multimodal"

    # Build dataloaders
    print(f"\nLoading {dataset_name} dataset...")
    train_loader, val_loader = build_dataloaders(config, dataset_name, mode)

    # Build model
    print(f"\nBuilding model (mode: {mode})...")

    # Override mode in model building
    model = build_model(config)
    if mode != "multimodal":
        # Rebuild with different mode
        model = MultimodalEmotionModel(
            num_classes=config["dataset"][dataset_name]["num_classes"],
            mode=mode,
            vision_backbone=config["model"]["vision"]["backbone"],
            vision_pretrained=config["model"]["vision"]["pretrained"],
            text_backbone=config["model"]["text"]["backbone"],
            text_pretrained=config["model"]["text"]["pretrained"],
            fusion_strategy=config["model"]["fusion"]["strategy"],
            fusion_hidden_dim=config["model"]["fusion"]["hidden_dim"],
        )

    # Build trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
    )

    # Resume from checkpoint if specified
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Train
    best_metrics = trainer.train()

    # Print final results
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    for key, value in best_metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
