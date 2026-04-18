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
from models.multimodal_model import build_model
from training.trainer import Trainer
from utils.helpers import (
    get_device,
    load_config,
    load_transfer_weights,
    set_seed,
)


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
        "--init-checkpoint", type=str, default=None,
        help="Initialize model weights from another checkpoint without loading optimizer state",
    )
    parser.add_argument(
        "--transfer-component",
        type=str,
        default=None,
        choices=["full_model", "vision_encoder", "text_encoder", "encoders"],
        help="Subset of weights to load from --init-checkpoint",
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


def _build_meld_dataset_kwargs(config: dict, split: str) -> dict:
    """Build MELD dataset arguments from config."""
    dataset_cfg = config.get("dataset", {}).get("meld", {})
    model_cfg = config.get("model", {})
    text_backbone = model_cfg.get("text", {}).get(
        "backbone", "microsoft/deberta-v3-base"
    )
    context_cfg = dataset_cfg.get("text_context", {})
    quality_cfg = dataset_cfg.get("face_quality", {})

    quality_filter = quality_cfg.get("filter_train", False)
    if split != "train":
        quality_filter = quality_cfg.get("filter_eval", False)

    return {
        "root_dir": dataset_cfg.get("root_dir", "./datasets/meld"),
        "split": split,
        "image_size": dataset_cfg.get("image_size", 260),
        "max_text_length": dataset_cfg.get("max_text_length", 128),
        "text_model_name": text_backbone,
        "use_dialogue_history": context_cfg.get("use_dialogue_history", False),
        "history_window": context_cfg.get("history_window", 0),
        "include_speaker_in_text": context_cfg.get("include_speaker", True),
        "context_separator": context_cfg.get("separator", " [SEP] "),
        "quality_filter": quality_filter,
        "min_face_quality_score": quality_cfg.get("min_quality_score", 0.0),
        "repair_invalid_faces": quality_cfg.get("repair_invalid", False) and split == "train",
        "refresh_quality_cache": quality_cfg.get("refresh_cache", False) and split == "train",
    }


def resolve_auto_focal_alpha(config: dict, train_dataset) -> None:
    """Populate focal alpha from the current train distribution when requested."""
    loss_cfg = config.get("training", {}).get("loss", {})
    if loss_cfg.get("name", "focal") != "focal":
        return

    focal_alpha = loss_cfg.get("focal_alpha", None)
    if focal_alpha not in (None, "auto"):
        return

    strategy = loss_cfg.get("class_weight_strategy", "effective_num")
    beta = loss_cfg.get("effective_num_beta", 0.999)

    if hasattr(train_dataset, "get_class_weights"):
        weights = train_dataset.get_class_weights(strategy=strategy, beta=beta)
    elif hasattr(train_dataset, "class_weights"):
        weights = train_dataset.class_weights
    else:
        return

    loss_cfg["focal_alpha"] = [float(value) for value in weights.tolist()]
    print("\nResolved focal_alpha from train distribution:")
    for emotion, weight in zip(train_dataset.get_class_distribution().keys(), loss_cfg["focal_alpha"]):
        print(f"  {emotion}: {weight:.4f}")


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
        train_dataset = MELDDataset(**_build_meld_dataset_kwargs(config, split="train"))
        val_dataset = MELDDataset(**_build_meld_dataset_kwargs(config, split="dev"))
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

    resolve_auto_focal_alpha(config, train_dataset)

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

    return train_loader, val_loader, train_dataset, val_dataset


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
    if args.resume and args.init_checkpoint:
        raise ValueError("Use either --resume or --init-checkpoint, not both.")

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
    config.setdefault("model", {})["mode"] = mode

    # Build dataloaders
    print(f"\nLoading {dataset_name} dataset...")
    train_loader, val_loader, train_dataset, _ = build_dataloaders(config, dataset_name, mode)

    # Build model
    print(f"\nBuilding model (mode: {mode})...")

    # For vision-only: unfreeze more layers (text signal is absent, vision must
    # carry everything) and extend patience so the model has time to converge.
    vision_freeze_layers = config["model"]["vision"]["freeze_layers"]
    es_patience = config.get("training", {}).get("early_stopping", {}).get("patience", 10)
    if mode == "vision_only":
        vision_freeze_layers = 2  # unfreeze blocks 2-7 (was 5)
        es_patience = 15           # more time without text signal
        config.setdefault("training", {}).setdefault("early_stopping", {})["patience"] = es_patience
        config["model"]["vision"]["freeze_layers"] = vision_freeze_layers
        print(f"  [vision_only] vision_freeze_layers overridden to {vision_freeze_layers}")
        print(f"  [vision_only] early_stopping patience overridden to {es_patience}")

    # Build model once with the correct settings
    model = build_model(config, mode=mode)

    if args.init_checkpoint:
        transfer_component = args.transfer_component or "vision_encoder"
        transfer_report = load_transfer_weights(
            model=model,
            checkpoint_path=args.init_checkpoint,
            component=transfer_component,
            map_location=device,
        )
        config.setdefault("training", {})["transfer"] = {
            "init_checkpoint": args.init_checkpoint,
            "component": transfer_component,
            "report": transfer_report,
        }

    # Build trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
    )

    # Stamp metadata so checkpoints are self-describing
    trainer.model_mode = mode
    trainer.fusion_strategy = config["model"]["fusion"]["strategy"]
    trainer.train_class_distribution = dict(train_dataset.get_class_distribution())

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
