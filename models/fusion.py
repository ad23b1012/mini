"""
Multimodal Fusion Modules.

Implements three fusion strategies for combining vision and text features:
  1. CrossAttentionFusion — text attends to visual regions, vision attends to text tokens
  2. GatedFusion —  learnable gate controlling per-modality contribution
  3. ConcatFusion — simple concatenation baseline (for ablation)

The cross-attention fusion is the primary contribution — it allows the model
to learn which face regions correspond to which words, enabling richer
cross-modal explanations.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Cross-Attention Fusion (Primary — best for XAI)
# ============================================================


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention fusion between vision spatial features and text token embeddings.

    Architecture:
        1. Project vision features (1408-d) and text features (768-d) to shared dim
        2. Cross-attention: text tokens attend to visual spatial positions
        3. Cross-attention: visual positions attend to text tokens
        4. Combine both attended representations via learned gating

    This design enables:
        - Rich cross-modal interactions
        - Interpretable attention maps (which face region relates to which word)
        - Better Grad-CAM/SHAP alignment for faithfulness metric
    """

    def __init__(
        self,
        vision_dim: int = 1408,
        text_dim: int = 768,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Project both modalities to shared dimension
        self.vision_proj = nn.Sequential(
            nn.Linear(vision_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Cross-attention layers: text → vision
        self.text_to_vision_layers = nn.ModuleList(
            [
                CrossAttentionBlock(hidden_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )

        # Cross-attention layers: vision → text
        self.vision_to_text_layers = nn.ModuleList(
            [
                CrossAttentionBlock(hidden_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )

        # Gated combination of both directions
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )

        # Final fusion layer norm
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        vision_features: torch.Tensor,
        text_features: torch.Tensor,
        vision_spatial: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Args:
            vision_features: Pooled vision features [B, vision_dim]
            text_features: Pooled text features [B, text_dim]
            vision_spatial: Spatial vision features [B, C, H, W] (optional, for cross-attn)
            text_mask: Text attention mask [B, seq_len] (optional)

        Returns:
            dict with:
                - fused_features: [B, hidden_dim]
                - cross_attention_weights: attention maps for interpretability
        """
        # If we have spatial features, reshape for cross-attention
        if vision_spatial is not None:
            B, C, H, W = vision_spatial.shape
            vision_seq = vision_spatial.flatten(2).transpose(1, 2)  # [B, H*W, C]
            vision_proj = self.vision_proj(vision_seq)  # [B, H*W, hidden_dim]
        else:
            # Use pooled features as single-token sequence
            vision_proj = self.vision_proj(vision_features).unsqueeze(1)  # [B, 1, hidden_dim]

        # Project text (if we have token-level features, they'd come from encoder)
        text_proj = self.text_proj(text_features)
        if text_proj.dim() == 2:
            text_proj = text_proj.unsqueeze(1)  # [B, 1, hidden_dim]

        cross_attn_weights = []

        # Text attending to vision regions
        text_attended = text_proj
        for layer in self.text_to_vision_layers:
            text_attended, attn_w = layer(
                query=text_attended, key=vision_proj, value=vision_proj
            )
            cross_attn_weights.append(attn_w)

        # Vision attending to text tokens
        vision_attended = vision_proj
        for layer in self.vision_to_text_layers:
            vision_attended, attn_w = layer(
                query=vision_attended, key=text_proj, value=text_proj
            )
            cross_attn_weights.append(attn_w)

        # Pool both attended sequences
        text_pooled = text_attended.mean(dim=1)      # [B, hidden_dim]
        vision_pooled = vision_attended.mean(dim=1)  # [B, hidden_dim]

        # Gated combination
        combined = torch.cat([text_pooled, vision_pooled], dim=-1)  # [B, hidden_dim*2]
        gate_values = self.gate(combined)  # [B, hidden_dim]

        fused = gate_values * text_pooled + (1 - gate_values) * vision_pooled
        fused = self.output_norm(fused)

        return {
            "fused_features": fused,
            "cross_attention_weights": cross_attn_weights,
            "gate_values": gate_values,
        }

    def get_output_dim(self) -> int:
        return self.hidden_dim


class CrossAttentionBlock(nn.Module):
    """Single cross-attention block with residual connection and FFN."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            output: [B, seq_len, hidden_dim]
            attn_weights: [B, query_len, key_len]
        """
        # Cross-attention with residual
        attended, attn_weights = self.cross_attn(
            query, key, value, key_padding_mask=key_padding_mask
        )
        query = self.norm1(query + attended)

        # FFN with residual
        query = self.norm2(query + self.ffn(query))

        return query, attn_weights


# ============================================================
# Gated Fusion (Alternative — simpler but effective)
# ============================================================


class GatedFusion(nn.Module):
    """
    Gated fusion with learned modality weighting.

    Uses a sigmoid gate to dynamically weight visual vs. textual features.
    Simpler than cross-attention but surprisingly competitive.
    """

    def __init__(
        self,
        vision_dim: int = 1408,
        text_dim: int = 768,
        hidden_dim: int = 512,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Project to shared space
        self.vision_proj = nn.Sequential(
            nn.Linear(vision_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Learnable gate
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        vision_features: torch.Tensor,
        text_features: torch.Tensor,
        **kwargs,
    ) -> dict:
        """
        Args:
            vision_features: [B, vision_dim]
            text_features: [B, text_dim]

        Returns:
            dict with fused_features [B, hidden_dim] and gate_values [B, 1]
        """
        v = self.vision_proj(vision_features)
        t = self.text_proj(text_features)

        gate_input = torch.cat([v, t], dim=-1)
        alpha = self.gate_net(gate_input)  # [B, 1] — how much vision vs text

        fused = alpha * v + (1 - alpha) * t
        fused = self.output_norm(fused)

        return {
            "fused_features": fused,
            "gate_values": alpha,
        }

    def get_output_dim(self) -> int:
        return self.hidden_dim


# ============================================================
# Concat Fusion (Ablation baseline)
# ============================================================


class ConcatFusion(nn.Module):
    """
    Simple concatenation fusion (ablation baseline).

    Concatenates projected vision and text features, then applies MLP.
    """

    def __init__(
        self,
        vision_dim: int = 1408,
        text_dim: int = 768,
        hidden_dim: int = 512,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.fusion_mlp = nn.Sequential(
            nn.Linear(vision_dim + text_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        vision_features: torch.Tensor,
        text_features: torch.Tensor,
        **kwargs,
    ) -> dict:
        combined = torch.cat([vision_features, text_features], dim=-1)
        fused = self.fusion_mlp(combined)
        return {"fused_features": fused}

    def get_output_dim(self) -> int:
        return self.hidden_dim
