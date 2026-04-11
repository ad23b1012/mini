"""
Grad-CAM Explainer for Vision Modality.

Generates class-discriminative localization maps highlighting which
facial regions (eyes, mouth, forehead, etc.) contributed most to the
emotion prediction.

The implementation supports:
  - Standard Grad-CAM
  - Grad-CAM++ (weighted version for better localization)
  - Region-level importance scoring (maps heatmap to facial AU regions)

References:
    Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
    via Gradient-based Localization", ICCV 2017.
"""

from typing import Optional, Tuple, List

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class GradCAMExplainer:
    """
    Grad-CAM explainer for the vision encoder.

    Hooks into a target convolutional layer to compute gradient-weighted
    activation maps, showing which spatial regions of the face image
    influenced the emotion prediction.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        target_layer: torch.nn.Module,
        device: str = "cuda",
        use_gradcam_pp: bool = False,
    ):
        """
        Args:
            model: The full multimodal model (or just vision encoder).
            target_layer: The convolutional layer to extract Grad-CAM from.
            device: "cuda" or "cpu".
            use_gradcam_pp: If True, use Grad-CAM++ for better localization.
        """
        self.model = model
        self.target_layer = target_layer
        self.device = device
        self.use_gradcam_pp = use_gradcam_pp

        # Storage for hooks
        self.activations = None
        self.gradients = None

        # Register hooks
        self._register_hooks()

    def _register_hooks(self):
        """Register forward and backward hooks on the target layer."""

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.forward_handle = self.target_layer.register_forward_hook(forward_hook)
        self.backward_handle = self.target_layer.register_full_backward_hook(
            backward_hook
        )

    def remove_hooks(self):
        """Remove registered hooks (call when done)."""
        self.forward_handle.remove()
        self.backward_handle.remove()

    def generate(
        self,
        image: torch.Tensor,
        target_class: Optional[int] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Generate Grad-CAM heatmap for a single image.

        Args:
            image: Input image tensor [1, 3, H, W] or [3, H, W].
            target_class: Target class index. If None, uses predicted class.
            input_ids: Optional text tokens (for multimodal forward pass).
            attention_mask: Optional text attention mask.

        Returns:
            dict with:
                - heatmap: numpy array [H, W] in [0, 1] range
                - predicted_class: int
                - confidence: float (softmax probability)
                - region_scores: dict mapping facial regions to importance
        """
        self.model.eval()

        # Ensure batch dimension
        if image.dim() == 3:
            image = image.unsqueeze(0)
        image = image.to(self.device).requires_grad_(True)

        # Forward pass
        forward_kwargs = {"image": image, "return_features": True}
        if input_ids is not None:
            forward_kwargs["input_ids"] = input_ids.to(self.device)
            forward_kwargs["attention_mask"] = attention_mask.to(self.device)

        output = self.model(**forward_kwargs)
        logits = output["logits"]

        # Determine target class
        if target_class is None:
            target_class = logits.argmax(dim=1).item()

        # Get confidence
        probs = F.softmax(logits, dim=1)
        confidence = probs[0, target_class].item()

        # Backward pass for target class
        self.model.zero_grad()
        logits[0, target_class].backward(retain_graph=True)

        # Get activations and gradients
        activations = self.activations  # [1, C, H', W']
        gradients = self.gradients      # [1, C, H', W']

        if activations is None or gradients is None:
            raise RuntimeError(
                "Failed to capture activations/gradients. "
                "Check that target_layer is correct."
            )

        # Compute Grad-CAM weights
        if self.use_gradcam_pp:
            # Grad-CAM++: uses second and third derivatives
            grad_2 = gradients ** 2
            grad_3 = gradients ** 3
            sum_activations = activations.sum(dim=(2, 3), keepdim=True)
            alpha = grad_2 / (
                2 * grad_2 + sum_activations * grad_3 + 1e-8
            )
            weights = (alpha * F.relu(gradients)).sum(dim=(2, 3), keepdim=True)
        else:
            # Standard Grad-CAM: global average pooling of gradients
            weights = gradients.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]

        # Weighted combination of activation maps
        cam = (weights * activations).sum(dim=1, keepdim=True)  # [1, 1, H', W']
        cam = F.relu(cam)  # Keep only positive contributions

        # Normalize to [0, 1]
        cam = cam.squeeze()
        if cam.max() > 0:
            cam = cam / cam.max()

        # Resize to input image size
        cam_np = cam.cpu().numpy()
        input_h, input_w = image.shape[2], image.shape[3]
        heatmap = cv2.resize(cam_np, (input_w, input_h))

        # Compute region-level scores
        region_scores = self._compute_region_scores(heatmap)

        return {
            "heatmap": heatmap,
            "predicted_class": target_class,
            "confidence": confidence,
            "region_scores": region_scores,
        }

    def _compute_region_scores(self, heatmap: np.ndarray) -> dict:
        """
        Map Grad-CAM heatmap to facial region importance scores.

        Divides the face into anatomical regions (approximate) and
        computes average importance for each region.

        Regions are based on typical face proportions:
            - Forehead: top 25%
            - Eyes: 25-45% vertically, split L/R
            - Nose: center 35-60%
            - Mouth: 60-80%
            - Jaw/Chin: bottom 20%
        """
        h, w = heatmap.shape

        regions = {
            "forehead": heatmap[: int(0.25 * h), :],
            "left_eye": heatmap[int(0.25 * h): int(0.45 * h), : w // 2],
            "right_eye": heatmap[int(0.25 * h): int(0.45 * h), w // 2:],
            "nose": heatmap[int(0.35 * h): int(0.60 * h), int(0.3 * w): int(0.7 * w)],
            "mouth": heatmap[int(0.60 * h): int(0.80 * h), int(0.2 * w): int(0.8 * w)],
            "jaw_chin": heatmap[int(0.80 * h):, :],
        }

        scores = {}
        for region_name, region_map in regions.items():
            if region_map.size > 0:
                scores[region_name] = float(region_map.mean())
            else:
                scores[region_name] = 0.0

        # Normalize scores to sum to 1
        total = sum(scores.values()) + 1e-8
        scores = {k: v / total for k, v in scores.items()}

        return scores

    def generate_overlay(
        self,
        image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.5,
        colormap: int = cv2.COLORMAP_JET,
    ) -> np.ndarray:
        """
        Overlay Grad-CAM heatmap on the original face image.

        Args:
            image: Original image as numpy array [H, W, 3] (RGB, 0-255).
            heatmap: Grad-CAM heatmap [H, W] in [0, 1].
            alpha: Blending factor.
            colormap: OpenCV colormap.

        Returns:
            Blended overlay image [H, W, 3] (RGB, 0-255).
        """
        # Convert heatmap to colormap
        heatmap_uint8 = np.uint8(255 * heatmap)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

        # Resize to match image
        if heatmap_color.shape[:2] != image.shape[:2]:
            heatmap_color = cv2.resize(
                heatmap_color, (image.shape[1], image.shape[0])
            )

        # Blend
        overlay = np.uint8(alpha * heatmap_color + (1 - alpha) * image)
        return overlay

    def batch_generate(
        self,
        images: torch.Tensor,
        target_classes: Optional[List[int]] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> List[dict]:
        """Generate Grad-CAM for a batch of images."""
        results = []
        N = images.shape[0]

        for i in range(N):
            img = images[i].unsqueeze(0)
            target = target_classes[i] if target_classes else None

            ids = input_ids[i].unsqueeze(0) if input_ids is not None else None
            mask = (
                attention_mask[i].unsqueeze(0)
                if attention_mask is not None
                else None
            )

            result = self.generate(
                image=img,
                target_class=target,
                input_ids=ids,
                attention_mask=mask,
            )
            results.append(result)

        return results

    def __del__(self):
        """Clean up hooks on deletion."""
        try:
            self.remove_hooks()
        except Exception:
            pass
