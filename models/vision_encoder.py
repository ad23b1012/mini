"""
Vision Encoder — EfficientNet-B2 backbone for facial feature extraction.

Uses timm (PyTorch Image Models) for a pretrained EfficientNet-B2:
  - Input:  [B, 3, 260, 260] face images
  - Output: spatial features [B, 1408, H, W] for Grad-CAM
            pooled features [B, 1408] for fusion

EfficientNet-B2 was chosen for:
  - Good accuracy/compute trade-off for face analysis
  - Native 260x260 input resolution (good for face crops)
  - 1408-dim features rich enough for emotion cues
"""

import torch
import torch.nn as nn
import timm


class VisionEncoder(nn.Module):
    """
    EfficientNet-B2 vision encoder.

    Extracts spatial feature maps (for Grad-CAM explanations) and
    pooled feature vectors (for multimodal fusion).
    """

    def __init__(
        self,
        backbone: str = "efficientnet_b2",
        pretrained: bool = True,
        feature_dim: int = 1408,
        dropout: float = 0.3,
        freeze_layers: int = 0,
    ):
        """
        Args:
            backbone: timm model name.
            pretrained: Whether to load ImageNet pretrained weights.
            feature_dim: Expected feature dimension (1408 for EfficientNet-B2).
            dropout: Dropout rate after pooling.
            freeze_layers: Number of initial blocks to freeze.
        """
        super().__init__()
        self.feature_dim = feature_dim

        # Load EfficientNet-B2 without classification head
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,         # Remove classifier
            global_pool="",        # Remove global pooling (we do it manually)
        )

        # Adaptive pooling for variable input sizes
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout)

        # Optional projection to a different dim
        self.projector = None

        # Freeze early layers if requested
        if freeze_layers > 0:
            self._freeze_layers(freeze_layers)

        # Verify feature dim
        self._verify_feature_dim()

    def _freeze_layers(self, num_blocks: int):
        """Freeze the first N blocks of EfficientNet."""
        # EfficientNet blocks are in self.backbone.blocks
        blocks = list(self.backbone.blocks.children())
        for i, block in enumerate(blocks):
            if i < num_blocks:
                for param in block.parameters():
                    param.requires_grad = False
        print(f"[VisionEncoder] Froze first {num_blocks} blocks")

    def _verify_feature_dim(self):
        """Verify the output feature dimension with a dummy forward pass."""
        with torch.no_grad():
            dummy = torch.randn(1, 3, 260, 260)
            features = self.backbone(dummy)
            actual_dim = features.shape[1]
            if actual_dim != self.feature_dim:
                print(
                    f"[VisionEncoder] Warning: expected feature_dim={self.feature_dim}, "
                    f"got {actual_dim}. Adjusting."
                )
                self.feature_dim = actual_dim

    def forward(
        self, x: torch.Tensor, return_spatial: bool = False
    ) -> dict:
        """
        Forward pass.

        Args:
            x: Input images [B, 3, H, W].
            return_spatial: If True, also return spatial feature maps.

        Returns:
            dict with:
                - features: Pooled features [B, feature_dim]
                - spatial_features: Spatial feature maps [B, C, H', W'] (if requested)
        """
        # Extract spatial feature maps from backbone
        spatial = self.backbone(x)  # [B, 1408, 9, 9] for 260x260 input

        # Global average pooling
        pooled = self.pool(spatial)  # [B, 1408, 1, 1]
        pooled = pooled.flatten(1)   # [B, 1408]
        pooled = self.dropout(pooled)

        result = {"features": pooled}

        if return_spatial:
            result["spatial_features"] = spatial

        return result

    def get_gradcam_target_layer(self):
        """
        Return the target layer for Grad-CAM.

        For EfficientNet, this is typically the last convolutional block.
        The spatial features from this layer tell us which face regions
        the model focuses on.
        """
        # The last block in EfficientNet's feature extractor
        return self.backbone.blocks[-1]

    def get_feature_dim(self) -> int:
        """Return the output feature dimension."""
        return self.feature_dim
