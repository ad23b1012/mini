"""
Full Multimodal Emotion Recognition Model.

Assembles:
  - VisionEncoder (EfficientNet-B2)
  - TextEncoder (DeBERTa-v3-base)
  - Fusion module (CrossAttention / Gated / Concat)
  - Classification head

Supports three operating modes for ablation:
  - "multimodal": vision + text (full model)
  - "vision_only": face image only
  - "text_only": transcript text only
"""

from typing import Optional

import torch
import torch.nn as nn
import yaml

from models.vision_encoder import VisionEncoder
from models.text_encoder import TextEncoder
from models.fusion import CrossAttentionFusion, GatedFusion, ConcatFusion


FUSION_REGISTRY = {
    "cross_attention": CrossAttentionFusion,
    "gated": GatedFusion,
    "concat": ConcatFusion,
}


class ClassificationHead(nn.Module):
    """
    MLP classification head with configurable depth.

    Includes batch normalization, GELU activation, and dropout
    for robust emotion classification.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: list = None,
        dropout: float = 0.3,
        activation: str = "gelu",
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]

        act_fn = nn.GELU() if activation == "gelu" else nn.ReLU()

        layers = []
        prev_dim = input_dim
        for hdim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hdim),
                    nn.BatchNorm1d(hdim),
                    act_fn,
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hdim

        # Final classification layer
        layers.append(nn.Linear(prev_dim, num_classes))

        self.head = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class MultimodalEmotionModel(nn.Module):
    """
    Full multimodal emotion recognition model.

    Combines vision encoder, text encoder, fusion module, and classifier
    into an end-to-end differentiable architecture.
    """

    def __init__(
        self,
        num_classes: int = 7,
        mode: str = "multimodal",
        # Vision encoder config
        vision_backbone: str = "efficientnet_b2",
        vision_pretrained: bool = True,
        vision_feature_dim: int = 1408,
        vision_dropout: float = 0.3,
        vision_freeze_layers: int = 0,
        # Text encoder config
        text_backbone: str = "microsoft/deberta-v3-base",
        text_pretrained: bool = True,
        text_feature_dim: int = 768,
        text_dropout: float = 0.1,
        text_freeze_layers: int = 0,
        text_max_length: int = 128,
        # Fusion config
        fusion_strategy: str = "cross_attention",
        fusion_hidden_dim: int = 512,
        fusion_num_heads: int = 8,
        fusion_dropout: float = 0.2,
        fusion_num_layers: int = 2,
        # Classifier config
        classifier_hidden_dims: list = None,
        classifier_dropout: float = 0.3,
        classifier_activation: str = "gelu",
    ):
        """
        Args:
            num_classes: Number of emotion categories.
            mode: "multimodal", "vision_only", or "text_only".
            ... (see individual module docs for param details)
        """
        super().__init__()
        self.mode = mode
        self.num_classes = num_classes

        if classifier_hidden_dims is None:
            classifier_hidden_dims = [256, 128]

        # ---- Vision Encoder ----
        if mode in ("multimodal", "vision_only"):
            self.vision_encoder = VisionEncoder(
                backbone=vision_backbone,
                pretrained=vision_pretrained,
                feature_dim=vision_feature_dim,
                dropout=vision_dropout,
                freeze_layers=vision_freeze_layers,
            )
            actual_vision_dim = self.vision_encoder.get_feature_dim()
        else:
            self.vision_encoder = None
            actual_vision_dim = vision_feature_dim

        # ---- Text Encoder ----
        if mode in ("multimodal", "text_only"):
            self.text_encoder = TextEncoder(
                backbone=text_backbone,
                pretrained=text_pretrained,
                feature_dim=text_feature_dim,
                dropout=text_dropout,
                freeze_layers=text_freeze_layers,
                max_length=text_max_length,
            )
            actual_text_dim = self.text_encoder.get_feature_dim()
        else:
            self.text_encoder = None
            actual_text_dim = text_feature_dim

        # ---- Fusion Module ----
        if mode == "multimodal":
            FusionClass = FUSION_REGISTRY.get(fusion_strategy, CrossAttentionFusion)
            fusion_kwargs = {
                "vision_dim": actual_vision_dim,
                "text_dim": actual_text_dim,
                "hidden_dim": fusion_hidden_dim,
                "dropout": fusion_dropout,
            }
            # Cross-attention has extra params
            if fusion_strategy == "cross_attention":
                fusion_kwargs["num_heads"] = fusion_num_heads
                fusion_kwargs["num_layers"] = fusion_num_layers

            self.fusion = FusionClass(**fusion_kwargs)
            classifier_input_dim = self.fusion.get_output_dim()
        else:
            self.fusion = None
            classifier_input_dim = (
                actual_vision_dim if mode == "vision_only" else actual_text_dim
            )

        # ---- Classification Head ----
        self.classifier = ClassificationHead(
            input_dim=classifier_input_dim,
            num_classes=num_classes,
            hidden_dims=classifier_hidden_dims,
            dropout=classifier_dropout,
            activation=classifier_activation,
        )

        # Print model summary
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\n[MultimodalEmotionModel] Mode: {mode}")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        if mode == "multimodal":
            print(f"  Fusion strategy: {fusion_strategy}")
        print()

    def forward(
        self,
        image: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_features: bool = False,
    ) -> dict:
        """
        Forward pass.

        Args:
            image: Face image tensor [B, 3, H, W] (required for vision/multimodal)
            input_ids: Token IDs [B, seq_len] (required for text/multimodal)
            attention_mask: Attention mask [B, seq_len]
            return_features: If True, return intermediate features for XAI.

        Returns:
            dict with:
                - logits: [B, num_classes]
                - features: intermediate representations (if return_features)
                - fusion_info: fusion gate values, attention weights (if multimodal)
        """
        result = {}
        features_dict = {}

        # ---- Vision forward ----
        if self.vision_encoder is not None and image is not None:
            vision_out = self.vision_encoder(
                image, return_spatial=return_features
            )
            vision_features = vision_out["features"]
            features_dict["vision"] = vision_out
        else:
            vision_features = None

        # ---- Text forward ----
        if self.text_encoder is not None and input_ids is not None:
            text_out = self.text_encoder(
                input_ids,
                attention_mask,
                return_all_tokens=return_features,
            )
            text_features = text_out["features"]
            features_dict["text"] = text_out
        else:
            text_features = None

        # ---- Fusion ----
        if self.mode == "multimodal":
            fusion_kwargs = {
                "vision_features": vision_features,
                "text_features": text_features,
            }
            # Pass spatial features for cross-attention
            if (
                return_features
                and "vision" in features_dict
                and "spatial_features" in features_dict["vision"]
            ):
                fusion_kwargs["vision_spatial"] = features_dict["vision"][
                    "spatial_features"
                ]

            fusion_out = self.fusion(**fusion_kwargs)
            fused = fusion_out["fused_features"]
            result["fusion_info"] = fusion_out
        elif self.mode == "vision_only":
            fused = vision_features
        else:  # text_only
            fused = text_features

        # ---- Classification ----
        logits = self.classifier(fused)

        result["logits"] = logits

        if return_features:
            result["features"] = features_dict
            result["fused_features"] = fused

        return result


def build_model(config: dict) -> MultimodalEmotionModel:
    """
    Factory function to build model from config dictionary.

    Args:
        config: Parsed YAML config dict (or subset under 'model' key).

    Returns:
        Configured MultimodalEmotionModel instance.
    """
    model_cfg = config.get("model", config)
    dataset_cfg = config.get("dataset", {})

    # Determine number of classes from dataset config
    dataset_name = dataset_cfg.get("name", "meld")
    num_classes = dataset_cfg.get(dataset_name, {}).get("num_classes", 7)

    # Build model
    vision_cfg = model_cfg.get("vision", {})
    text_cfg = model_cfg.get("text", {})
    fusion_cfg = model_cfg.get("fusion", {})
    classifier_cfg = model_cfg.get("classifier", {})

    model = MultimodalEmotionModel(
        num_classes=num_classes,
        mode="multimodal",
        # Vision
        vision_backbone=vision_cfg.get("backbone", "efficientnet_b2"),
        vision_pretrained=vision_cfg.get("pretrained", True),
        vision_feature_dim=vision_cfg.get("feature_dim", 1408),
        vision_dropout=vision_cfg.get("dropout", 0.3),
        vision_freeze_layers=vision_cfg.get("freeze_layers", 0),
        # Text
        text_backbone=text_cfg.get("backbone", "microsoft/deberta-v3-base"),
        text_pretrained=text_cfg.get("pretrained", True),
        text_feature_dim=text_cfg.get("feature_dim", 768),
        text_dropout=text_cfg.get("dropout", 0.1),
        text_freeze_layers=text_cfg.get("freeze_layers", 0),
        text_max_length=text_cfg.get("max_length", 128),
        # Fusion
        fusion_strategy=fusion_cfg.get("strategy", "cross_attention"),
        fusion_hidden_dim=fusion_cfg.get("hidden_dim", 512),
        fusion_num_heads=fusion_cfg.get("num_heads", 8),
        fusion_dropout=fusion_cfg.get("dropout", 0.2),
        fusion_num_layers=fusion_cfg.get("num_layers", 2),
        # Classifier
        classifier_hidden_dims=classifier_cfg.get("hidden_dims", [256, 128]),
        classifier_dropout=classifier_cfg.get("dropout", 0.3),
        classifier_activation=classifier_cfg.get("activation", "gelu"),
    )

    return model


def build_model_from_yaml(config_path: str) -> MultimodalEmotionModel:
    """Build model from a YAML config file path."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return build_model(config)
