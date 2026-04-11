"""
Visualization utilities for XAI outputs.

Generates publication-quality figures for:
  - Grad-CAM heatmap overlays on face images
  - SHAP waterfall / bar plots for text
  - Combined multimodal explanation panels
  - Confusion matrix heatmaps
  - Perturbation fidelity curves
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.gridspec import GridSpec


# Use non-interactive backend for server/script usage
matplotlib.use("Agg")

# Publication-quality defaults
plt.rcParams.update(
    {
        "font.size": 12,
        "font.family": "serif",
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)


def plot_gradcam_overlay(
    original_image: np.ndarray,
    heatmap: np.ndarray,
    emotion_name: str,
    confidence: float,
    save_path: Optional[str] = None,
    alpha: float = 0.45,
) -> plt.Figure:
    """
    Create a publication-quality Grad-CAM overlay figure.

    Shows: original image | heatmap | overlay

    Args:
        original_image: RGB image [H, W, 3] in [0, 255].
        heatmap: Grad-CAM heatmap [H, W] in [0, 1].
        emotion_name: Predicted emotion.
        confidence: Model confidence.
        save_path: Optional path to save the figure.
        alpha: Overlay blending factor.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    axes[0].imshow(original_image.astype(np.uint8))
    axes[0].set_title("Input Face")
    axes[0].axis("off")

    # Heatmap
    axes[1].imshow(heatmap, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Grad-CAM Activation")
    axes[1].axis("off")

    # Overlay
    heatmap_color = cv2.applyColorMap(
        np.uint8(255 * heatmap), cv2.COLORMAP_JET
    )
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    if heatmap_color.shape[:2] != original_image.shape[:2]:
        heatmap_color = cv2.resize(
            heatmap_color,
            (original_image.shape[1], original_image.shape[0]),
        )

    overlay = np.uint8(
        alpha * heatmap_color + (1 - alpha) * original_image
    )
    axes[2].imshow(overlay)
    axes[2].set_title(f"Prediction: {emotion_name} ({confidence*100:.1f}%)")
    axes[2].axis("off")

    plt.suptitle("Visual Explanation (Grad-CAM)", fontsize=16, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"  Saved Grad-CAM overlay to {save_path}")

    return fig


def plot_shap_tokens(
    tokens: List[str],
    shap_values: np.ndarray,
    emotion_name: str,
    save_path: Optional[str] = None,
    top_k: int = 15,
) -> plt.Figure:
    """
    Create a horizontal bar chart of SHAP token importances.

    Args:
        tokens: List of word tokens.
        shap_values: SHAP values per token.
        emotion_name: Predicted emotion.
        save_path: Optional path to save.
        top_k: Number of top features to show.
    """
    # Pair tokens with values and sort by absolute value
    pairs = list(zip(tokens, shap_values))
    pairs = [(t, v) for t, v in pairs if t.strip()]  # Remove empty tokens
    pairs.sort(key=lambda x: abs(x[1]), reverse=True)
    pairs = pairs[:top_k]
    pairs.reverse()  # For bottom-to-top display

    token_labels = [p[0] for p in pairs]
    values = [p[1] for p in pairs]
    colors = ["#e74c3c" if v > 0 else "#3498db" for v in values]

    fig, ax = plt.subplots(figsize=(10, max(4, len(pairs) * 0.4)))
    bars = ax.barh(range(len(pairs)), values, color=colors, edgecolor="white")

    ax.set_yticks(range(len(pairs)))
    ax.set_yticklabels(token_labels)
    ax.set_xlabel("SHAP Value (contribution to prediction)")
    ax.set_title(
        f"Text Feature Importance for '{emotion_name}'",
        fontweight="bold",
    )
    ax.axvline(x=0, color="black", linewidth=0.5)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#e74c3c", label="Pushes toward prediction"),
        Patch(facecolor="#3498db", label="Pushes against prediction"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"  Saved SHAP plot to {save_path}")

    return fig


def plot_region_importance(
    region_scores: Dict[str, float],
    emotion_name: str,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot facial region importance as a radar/bar chart."""
    regions = list(region_scores.keys())
    scores = list(region_scores.values())

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.YlOrRd(np.array(scores) / max(max(scores), 0.01))

    bars = ax.barh(regions, scores, color=colors, edgecolor="white")
    ax.set_xlabel("Importance Score")
    ax.set_title(
        f"Facial Region Importance for '{emotion_name}'",
        fontweight="bold",
    )

    # Add value labels
    for bar, score in zip(bars, scores):
        ax.text(
            bar.get_width() + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{score:.3f}",
            va="center",
            fontsize=10,
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
    title: str = "Confusion Matrix",
    normalize: bool = True,
) -> plt.Figure:
    """
    Plot a publication-quality confusion matrix.

    Args:
        cm: Confusion matrix [C, C].
        class_names: List of class names.
        save_path: Optional save path.
        title: Figure title.
        normalize: Whether to normalize by row (for recall).
    """
    if normalize:
        cm_display = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)
        fmt = ".2f"
    else:
        cm_display = cm
        fmt = "d"

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar_kws={"label": "Proportion" if normalize else "Count"},
    )
    ax.set_xlabel("Predicted", fontsize=14)
    ax.set_ylabel("True", fontsize=14)
    ax.set_title(title, fontsize=16, fontweight="bold")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"  Saved confusion matrix to {save_path}")

    return fig


def plot_combined_explanation(
    original_image: np.ndarray,
    heatmap: np.ndarray,
    tokens: List[str],
    shap_values: np.ndarray,
    region_scores: Dict[str, float],
    emotion_name: str,
    confidence: float,
    utterance: str = "",
    faithfulness_metrics: Optional[Dict[str, float]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Create a comprehensive multimodal explanation panel.

    This is the main visualization for the paper — combines all
    explanation components into a single figure.

    Layout:
        ┌──────────────┬──────────────┐
        │  Face + CAM   │  SHAP tokens │
        ├──────────────┼──────────────┤
        │  Region bars  │  Metrics     │
        └──────────────┴──────────────┘
    """
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, hspace=0.35, wspace=0.3)

    # ---- Top Left: Face + Grad-CAM Overlay ----
    ax1 = fig.add_subplot(gs[0, 0])
    heatmap_color = cv2.applyColorMap(
        np.uint8(255 * heatmap), cv2.COLORMAP_JET
    )
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    if heatmap_color.shape[:2] != original_image.shape[:2]:
        heatmap_color = cv2.resize(
            heatmap_color,
            (original_image.shape[1], original_image.shape[0]),
        )
    overlay = np.uint8(0.45 * heatmap_color + 0.55 * original_image)
    ax1.imshow(overlay)
    ax1.set_title(
        f"Grad-CAM: {emotion_name} ({confidence*100:.1f}%)",
        fontweight="bold",
    )
    ax1.axis("off")

    # ---- Top Right: SHAP Token Importance ----
    ax2 = fig.add_subplot(gs[0, 1])
    pairs = [(t, v) for t, v in zip(tokens, shap_values) if t.strip()]
    pairs.sort(key=lambda x: abs(x[1]), reverse=True)
    pairs = pairs[:10]
    pairs.reverse()

    token_labels = [p[0] for p in pairs]
    values = [p[1] for p in pairs]
    colors = ["#e74c3c" if v > 0 else "#3498db" for v in values]

    ax2.barh(range(len(pairs)), values, color=colors, edgecolor="white")
    ax2.set_yticks(range(len(pairs)))
    ax2.set_yticklabels(token_labels, fontsize=10)
    ax2.set_xlabel("SHAP Value")
    ax2.set_title("Text Feature Importance", fontweight="bold")
    ax2.axvline(x=0, color="black", linewidth=0.5)

    # ---- Bottom Left: Region Importance ----
    ax3 = fig.add_subplot(gs[1, 0])
    regions = list(region_scores.keys())
    scores = list(region_scores.values())
    bar_colors = plt.cm.YlOrRd(np.array(scores) / max(max(scores), 0.01))
    ax3.barh(regions, scores, color=bar_colors, edgecolor="white")
    ax3.set_xlabel("Importance Score")
    ax3.set_title("Facial Region Importance", fontweight="bold")

    # ---- Bottom Right: Metrics & Info ----
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")

    info_lines = [
        f"Predicted Emotion: {emotion_name}",
        f"Confidence: {confidence*100:.1f}%",
    ]
    if utterance:
        # Truncate long utterances
        display_text = (
            utterance[:80] + "..." if len(utterance) > 80 else utterance
        )
        info_lines.append(f'Utterance: "{display_text}"')

    if faithfulness_metrics:
        info_lines.append("")
        info_lines.append("── Faithfulness Metrics ──")
        info_lines.append(
            f"CMFS Score: {faithfulness_metrics.get('cmfs_score', 0):.3f}"
        )
        info_lines.append(
            f"Cross-Modal Agreement: {faithfulness_metrics.get('cross_modal_agreement', 0):.3f}"
        )
        info_lines.append(
            f"Vision Sufficiency: {faithfulness_metrics.get('vision_sufficiency', 0):.3f}"
        )
        info_lines.append(
            f"Text Sufficiency: {faithfulness_metrics.get('text_sufficiency', 0):.3f}"
        )

    info_text = "\n".join(info_lines)
    ax4.text(
        0.05,
        0.95,
        info_text,
        transform=ax4.transAxes,
        fontsize=12,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0", alpha=0.8),
    )

    plt.suptitle(
        "Multimodal Emotion Analysis — Explanation Report",
        fontsize=18,
        fontweight="bold",
        y=1.02,
    )

    if save_path:
        plt.savefig(save_path)
        print(f"  Saved combined explanation to {save_path}")

    return fig


def plot_perturbation_curves(
    vision_curve: np.ndarray,
    text_curve: np.ndarray,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot perturbation fidelity curves for both modalities."""
    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.linspace(0, 100, len(vision_curve))

    ax.plot(x, vision_curve, "b-o", label="Vision (Grad-CAM)", markersize=4)
    ax.plot(x, text_curve, "r-s", label="Text (SHAP)", markersize=4)
    ax.set_xlabel("% Features Removed (most important first)")
    ax.set_ylabel("Prediction Confidence")
    ax.set_title("Perturbation Fidelity Curves", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add AUC annotations
    vision_auc = np.trapz(vision_curve, x / 100)
    text_auc = np.trapz(text_curve, x / 100)
    ax.text(
        0.95,
        0.95,
        f"Vision AUC: {vision_auc:.3f}\nText AUC: {text_auc:.3f}",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    return fig
