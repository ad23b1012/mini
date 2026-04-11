"""
Image transforms for training and evaluation.

Provides augmentation pipelines tuned for facial expression recognition:
- Training: aggressive augmentation to improve generalization
- Validation/Test: minimal transforms for consistent evaluation
"""

from torchvision import transforms


def get_train_transforms(image_size: int = 260) -> transforms.Compose:
    """
    Training augmentation pipeline for face images.

    Includes:
        - RandomResizedCrop: scale variation + slight translation
        - RandomHorizontalFlip: faces are roughly symmetric
        - ColorJitter: lighting robustness
        - RandomRotation: slight head tilt variation
        - RandomGrayscale: color invariance
        - RandomErasing: occlusion robustness (cutout-style)
        - Normalize: ImageNet statistics (since we use pretrained backbone)
    """
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                image_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05
            ),
            transforms.RandomRotation(degrees=10),
            transforms.RandomGrayscale(p=0.05),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],   # ImageNet stats
                std=[0.229, 0.224, 0.225],
            ),
            transforms.RandomErasing(
                p=0.2, scale=(0.02, 0.15), ratio=(0.3, 3.3)
            ),
        ]
    )


def get_val_transforms(image_size: int = 260) -> transforms.Compose:
    """
    Validation / test transform pipeline.

    Minimal processing: resize, center crop, normalize.
    """
    return transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.1)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def get_denormalize_transform() -> transforms.Normalize:
    """
    Inverse normalization for visualizing images.
    Use this before displaying Grad-CAM overlays.
    """
    return transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
    )
