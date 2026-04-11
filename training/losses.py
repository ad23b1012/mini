"""
Loss Functions for Emotion Recognition.

  - FocalLoss: handles severe class imbalance (common in emotion datasets)
  - LabelSmoothingCE: prevents overconfident predictions
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification.

    Addresses class imbalance by down-weighting well-classified examples
    and focusing training on hard, misclassified samples.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Reference:
        Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ):
        """
        Args:
            gamma: Focusing parameter. Higher = more focus on hard examples.
                   gamma=0 reduces to standard CE. Typical: 1.0-3.0.
            alpha: Per-class weight tensor [num_classes]. If None, uniform.
            reduction: "mean", "sum", or "none".
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            logits: [B, C] raw logits.
            targets: [B] class indices.
        """
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        p_t = torch.exp(-ce_loss)  # p_t = probability of correct class

        focal_weight = (1 - p_t) ** self.gamma

        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            alpha_t = alpha[targets]
            focal_weight = alpha_t * focal_weight

        loss = focal_weight * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class LabelSmoothingCE(nn.Module):
    """
    Cross-entropy loss with label smoothing.

    Prevents the model from becoming overconfident by softening the
    one-hot target distribution.

    For target class c:
        smoothed_label[c] = 1 - smoothing + smoothing / num_classes
        smoothed_label[i≠c] = smoothing / num_classes
    """

    def __init__(
        self,
        smoothing: float = 0.1,
        num_classes: int = 7,
        reduction: str = "mean",
    ):
        """
        Args:
            smoothing: Label smoothing factor (0 = no smoothing).
            num_classes: Number of classes.
            reduction: "mean" or "sum".
        """
        super().__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes
        self.reduction = reduction

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            logits: [B, C] raw logits.
            targets: [B] class indices.
        """
        log_probs = F.log_softmax(logits, dim=1)

        # Create smoothed labels
        with torch.no_grad():
            smooth_targets = torch.full_like(
                log_probs, self.smoothing / self.num_classes
            )
            smooth_targets.scatter_(
                1,
                targets.unsqueeze(1),
                1.0 - self.smoothing + self.smoothing / self.num_classes,
            )

        loss = -(smooth_targets * log_probs).sum(dim=1)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
