"""Helper utilities."""

import os
import random
from pathlib import Path

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
            f"  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB"
        )
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        print("Using Apple MPS (Metal Performance Shaders)")
    else:
        device = "cpu"
        print("Using CPU")
    return device
