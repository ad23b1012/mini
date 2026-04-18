"""Helper utilities."""

import os
import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import yaml


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic operations (may slow down training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def count_parameters(model: torch.nn.Module) -> dict:
    """
    Count model parameters.

    Returns:
        dict with total, trainable, and frozen parameter counts.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
        "total_mb": total * 4 / (1024 ** 2),  # Approx memory in MB (float32)
    }


def get_device(prefer_cuda: bool = True) -> str:
    """Get the best available device."""
    if prefer_cuda and torch.cuda.is_available():
        device = "cuda"
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        print("Using Apple MPS (Metal Performance Shaders)")
    else:
        device = "cpu"
        print("Using CPU")
    return device


def load_transfer_weights(
    model: torch.nn.Module,
    checkpoint_path: str,
    component: str = "vision_encoder",
    map_location: str = "cpu",
) -> Dict[str, object]:
    """
    Load a compatible subset of weights from another checkpoint.

    This is designed for transfer learning, such as:
      - AffectNet vision-only pretraining -> MELD multimodal fine-tuning

    Args:
        model: Destination model.
        checkpoint_path: Path to a saved checkpoint or raw state dict.
        component: One of "full_model", "vision_encoder", "text_encoder", "encoders".
        map_location: Device map location for torch.load.

    Returns:
        dict summarizing loaded and skipped parameters.
    """
    checkpoint = torch.load(
        checkpoint_path,
        map_location=map_location,
        weights_only=False,
    )
    source_state = checkpoint.get("model_state_dict", checkpoint)
    target_state = model.state_dict()

    compatible_state = {}
    skipped = []

    for key, value in source_state.items():
        target_key = key

        if component == "vision_encoder":
            if not key.startswith("vision_encoder."):
                continue
        elif component == "text_encoder":
            if not key.startswith("text_encoder."):
                continue
        elif component == "encoders":
            if not (
                key.startswith("vision_encoder.") or key.startswith("text_encoder.")
            ):
                continue
        elif component != "full_model":
            raise ValueError(f"Unsupported transfer component: {component}")

        if target_key not in target_state:
            skipped.append((key, "missing_in_target"))
            continue

        if tuple(target_state[target_key].shape) != tuple(value.shape):
            skipped.append(
                (
                    key,
                    f"shape_mismatch source={tuple(value.shape)} target={tuple(target_state[target_key].shape)}",
                )
            )
            continue

        compatible_state[target_key] = value

    missing_keys, unexpected_keys = model.load_state_dict(compatible_state, strict=False)

    report = {
        "checkpoint_path": str(checkpoint_path),
        "component": component,
        "loaded_tensors": len(compatible_state),
        "missing_keys": list(missing_keys),
        "unexpected_keys": list(unexpected_keys),
        "skipped": skipped,
        "source_metrics": checkpoint.get("metrics", {}),
        "source_config": checkpoint.get("config", {}),
    }

    print(f"\nLoaded transfer weights from {checkpoint_path}")
    print(f"  Component: {component}")
    print(f"  Compatible tensors loaded: {report['loaded_tensors']}")
    print(f"  Missing keys after load: {len(report['missing_keys'])}")
    print(f"  Unexpected keys after load: {len(report['unexpected_keys'])}")
    print(f"  Skipped tensors: {len(report['skipped'])}")

    return report
