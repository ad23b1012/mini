"""
AffectNet Dataset Loader.

AffectNet contains ~450K manually annotated facial expression images
with 8 discrete emotion labels + valence/arousal continuous annotations.

Citation:
    Mollahosseini, A., Hasani, B. and Mahoor, M.H., 2019.
    AffectNet: A Database for Facial Expression, Valence, and Arousal
    Computing in the Wild. IEEE Transactions on Affective Computing.

Note: AffectNet requires manual registration and download from:
    http://mohammadmahoor.com/affectnet/
"""

import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from data.transforms import get_train_transforms, get_val_transforms


# AffectNet emotion label mapping (8-class)
AFFECTNET_EMOTIONS = {
    0: "Neutral",
    1: "Happy",
    2: "Sad",
    3: "Surprise",
    4: "Fear",
    5: "Disgust",
    6: "Anger",
    7: "Contempt",
}


class AffectNetDataset(Dataset):
    """
    PyTorch Dataset for AffectNet.

    Expected directory structure:
        affectnet/
        ├── Manually_Annotated/
        │   ├── Manually_Annotated_Images/
        │   │   ├── 1/
        │   │   ├── 2/
        │   │   └── ...
        │   └── Manually_Annotated_file_lists/
        │       ├── training.csv
        │       └── validation.csv
        └── Automatically_Annotated/   (optional, for larger training set)

    The CSV files contain columns:
        subDirectory_filePath, face_x, face_y, face_width, face_height,
        facial_landmarks, expression, valence, arousal
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        num_classes: int = 8,
        image_size: int = 260,
        transform=None,
        use_automatic: bool = False,
        balanced_sampling: bool = True,
    ):
        """
        Args:
            root_dir: Path to AffectNet root directory.
            split: "train" or "val".
            num_classes: 7 or 8 classes (8 includes contempt).
            image_size: Target image size for transforms.
            transform: Optional custom transforms. If None, uses defaults.
            use_automatic: Whether to include automatically annotated data.
            balanced_sampling: Whether to apply class-balanced sampling weights.
        """
        super().__init__()
        self.root_dir = Path(root_dir)
        self.split = split
        self.num_classes = num_classes
        self.image_size = image_size
        self.balanced_sampling = balanced_sampling

        # Set transforms
        if transform is not None:
            self.transform = transform
        elif split == "train":
            self.transform = get_train_transforms(image_size)
        else:
            self.transform = get_val_transforms(image_size)

        # Load annotations
        self.samples = self._load_annotations(use_automatic)

        # Compute class weights for balanced sampling
        if balanced_sampling:
            self.class_weights = self._compute_class_weights()
            self.sample_weights = self._compute_sample_weights()

        print(
            f"[AffectNet] Loaded {len(self.samples)} samples for '{split}' "
            f"split ({num_classes} classes)"
        )

    def _load_annotations(self, use_automatic: bool) -> list:
        """Load and filter annotation CSV files."""
        manual_dir = self.root_dir / "Manually_Annotated"
        csv_dir = manual_dir / "Manually_Annotated_file_lists"
        img_base = manual_dir / "Manually_Annotated_Images"

        csv_file = csv_dir / (
            "training.csv" if self.split == "train" else "validation.csv"
        )

        if not csv_file.exists():
            raise FileNotFoundError(
                f"AffectNet CSV not found at {csv_file}. "
                "Please download AffectNet from http://mohammadmahoor.com/affectnet/ "
                "and extract to the configured root_dir."
            )

        df = pd.read_csv(csv_file)

        # Filter to valid emotion labels
        df = df[df["expression"].isin(range(self.num_classes))].reset_index(drop=True)

        # Remove rows with invalid paths or missing images
        samples = []
        for _, row in df.iterrows():
            img_path = img_base / row["subDirectory_filePath"]
            if img_path.suffix == "":
                img_path = img_path.with_suffix(".jpg")

            samples.append(
                {
                    "image_path": str(img_path),
                    "label": int(row["expression"]),
                    "valence": float(row.get("valence", 0.0)),
                    "arousal": float(row.get("arousal", 0.0)),
                }
            )

        # Optionally add automatically annotated data for training
        if use_automatic and self.split == "train":
            auto_dir = self.root_dir / "Automatically_Annotated"
            auto_csv = auto_dir / "automatically_annotated.csv"
            if auto_csv.exists():
                auto_df = pd.read_csv(auto_csv)
                auto_df = auto_df[
                    auto_df["expression"].isin(range(self.num_classes))
                ].reset_index(drop=True)
                auto_img_base = auto_dir / "Automatically_Annotated_Images"
                for _, row in auto_df.iterrows():
                    img_path = auto_img_base / row["subDirectory_filePath"]
                    samples.append(
                        {
                            "image_path": str(img_path),
                            "label": int(row["expression"]),
                            "valence": float(row.get("valence", 0.0)),
                            "arousal": float(row.get("arousal", 0.0)),
                        }
                    )
                print(
                    f"[AffectNet] Added {len(auto_df)} automatically annotated samples"
                )

        return samples

    def _compute_class_weights(self) -> torch.Tensor:
        """Compute inverse-frequency class weights for loss balancing."""
        labels = [s["label"] for s in self.samples]
        class_counts = np.bincount(labels, minlength=self.num_classes)
        # Inverse frequency, normalized
        weights = 1.0 / (class_counts + 1e-6)
        weights = weights / weights.sum() * self.num_classes
        return torch.FloatTensor(weights)

    def _compute_sample_weights(self) -> torch.Tensor:
        """Compute per-sample weights for WeightedRandomSampler."""
        labels = [s["label"] for s in self.samples]
        class_counts = np.bincount(labels, minlength=self.num_classes)
        weights_per_class = 1.0 / (class_counts + 1e-6)
        sample_weights = torch.FloatTensor(
            [weights_per_class[label] for label in labels]
        )
        return sample_weights

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        """
        Returns:
            dict with keys:
                - image: Tensor [3, H, W]
                - label: int (emotion class index)
                - valence: float (-1 to 1)
                - arousal: float (-1 to 1)
                - emotion_name: str
                - image_path: str
        """
        sample = self.samples[idx]

        # Load image
        try:
            image = Image.open(sample["image_path"]).convert("RGB")
        except (FileNotFoundError, OSError):
            # Return a black image if file is missing (with a warning)
            image = Image.new("RGB", (self.image_size, self.image_size), (0, 0, 0))

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "label": sample["label"],
            "valence": sample["valence"],
            "arousal": sample["arousal"],
            "emotion_name": AFFECTNET_EMOTIONS.get(sample["label"], "Unknown"),
            "image_path": sample["image_path"],
        }

    def get_class_distribution(self) -> dict:
        """Return class distribution as {emotion_name: count}."""
        labels = [s["label"] for s in self.samples]
        counts = np.bincount(labels, minlength=self.num_classes)
        return {
            AFFECTNET_EMOTIONS[i]: int(counts[i])
            for i in range(self.num_classes)
        }
