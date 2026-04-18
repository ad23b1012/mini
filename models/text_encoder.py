"""
Text Encoder — DeBERTa-v3-base for contextual text understanding.

Uses HuggingFace Transformers for a pretrained DeBERTa-v3-base:
  - Input:  tokenized text (input_ids + attention_mask)
  - Output: [CLS] embedding [B, 768] for fusion
            all token embeddings [B, seq_len, 768] for SHAP

DeBERTa-v3-base was chosen over BERT because:
  - Disentangled attention + enhanced mask decoder
  - State-of-the-art on SuperGLUE at base model size
  - Smaller (86M vs 110M) — faster training on RTX
  - Better contextual representations for emotion-laden text
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class TextEncoder(nn.Module):
    """
    DeBERTa-v3-base text encoder.

    Extracts [CLS] representation and per-token embeddings for
    downstream fusion and SHAP explainability.
    """

    def __init__(
        self,
        backbone: str = "microsoft/deberta-v3-base",
        pretrained: bool = True,
        feature_dim: int = 768,
        dropout: float = 0.1,
        freeze_layers: int = 0,
        max_length: int = 128,
    ):
        """
        Args:
            backbone: HuggingFace model name or path.
            pretrained: Whether to load pretrained weights.
            feature_dim: Hidden size of the model (768 for deberta-v3-base).
            dropout: Dropout after [CLS] pooling.
            freeze_layers: Number of transformer layers to freeze from bottom.
            max_length: Maximum token sequence length.
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.max_length = max_length

        # Load DeBERTa model (force float32 for consistent dtype with pooling layers)
        if pretrained:
            self.backbone = AutoModel.from_pretrained(backbone, torch_dtype=torch.float32)
        else:
            config = AutoConfig.from_pretrained(backbone)
            self.backbone = AutoModel.from_config(config)

        # Update feature dim from actual model
        self.feature_dim = self.backbone.config.hidden_size

        # [CLS] pooling + dropout
        self.dropout = nn.Dropout(p=dropout)

        # Optional: learned pooling instead of [CLS] token
        self.attention_pool = AttentionPooling(self.feature_dim)

        # Freeze layers if requested
        if freeze_layers > 0:
            self._freeze_layers(freeze_layers)

        # Enable gradient checkpointing to save VRAM
        self.backbone.gradient_checkpointing_enable()

    def _freeze_layers(self, num_layers: int):
        """Freeze the embeddings and first N transformer layers."""
        # Freeze embeddings
        for param in self.backbone.embeddings.parameters():
            param.requires_grad = False

        # Freeze encoder layers
        if hasattr(self.backbone, "encoder"):
            layers = self.backbone.encoder.layer
            for i in range(min(num_layers, len(layers))):
                for param in layers[i].parameters():
                    param.requires_grad = False

        print(f"[TextEncoder] Froze embeddings + first {num_layers} layers")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_all_tokens: bool = False,
    ) -> dict:
        """
        Forward pass.

        Args:
            input_ids: Token IDs [B, seq_len].
            attention_mask: Attention mask [B, seq_len].
            return_all_tokens: If True, also return all token embeddings.

        Returns:
            dict with:
                - features: Pooled [CLS] features [B, feature_dim]
                - token_embeddings: All token embeddings [B, seq_len, feature_dim]
                  (only if return_all_tokens=True)
        """
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        # All token embeddings
        token_embeddings = outputs.last_hidden_state  # [B, seq_len, 768]

        # Pooled representation via attention pooling
        pooled = self.attention_pool(token_embeddings, attention_mask)
        pooled = self.dropout(pooled)

        result = {"features": pooled}

        if return_all_tokens:
            result["token_embeddings"] = token_embeddings

        return result

    def get_feature_dim(self) -> int:
        """Return the output feature dimension."""
        return self.feature_dim


class AttentionPooling(nn.Module):
    """
    Attention-weighted pooling over token embeddings.

    Instead of using just the [CLS] token (which can lose information),
    this learns a weighted combination of all tokens. This gives a
    richer representation for emotion detection where multiple words
    contribute to the emotional signal.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            token_embeddings: [B, seq_len, hidden_dim]
            attention_mask: [B, seq_len]

        Returns:
            pooled: [B, hidden_dim]
        """
        # Compute attention scores
        scores = self.attention(token_embeddings).squeeze(-1)  # [B, seq_len]

        # Mask padding tokens
        scores = scores.masked_fill(attention_mask == 0, float("-inf"))

        # Softmax over valid tokens
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)  # [B, seq_len, 1]

        # Weighted sum
        pooled = (token_embeddings * weights).sum(dim=1)  # [B, hidden_dim]
        return pooled
