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
        use_gradcam_pp: bool = True,
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
        original_image: Optional[np.ndarray] = None,
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
        # Keep logits identical to normal inference. The hook captures
        # activations directly from the target layer, so we do not need to
        # enable the alternate feature-return path here.
        forward_kwargs = {"image": image, "return_features": False}
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

        # Gaussian smoothing to fill sparse activations and produce
        # smoother, more anatomically meaningful heatmaps
        heatmap = cv2.GaussianBlur(heatmap, (0, 0), sigmaX=4.0)

        # Remove noise floor (pixels below threshold)
        heatmap[heatmap < 0.05] = 0.0

        # Re-normalize to [0, 1] after smoothing
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()

        # Compute region-level scores (use original image for landmark detection)
        self._last_masked_heatmap = None  # Reset before computation
        self._last_face_mask = None
        region_scores = self._compute_region_scores(heatmap, original_image)

        # Use face-masked heatmap for visualization if available
        masked_heatmap = getattr(self, '_last_masked_heatmap', None)
        if masked_heatmap is None:
            masked_heatmap = heatmap  # Fallback to raw heatmap

        return {
            "heatmap": heatmap,
            "masked_heatmap": masked_heatmap,
            "predicted_class": target_class,
            "confidence": confidence,
            "region_scores": region_scores,
        }

    def _compute_region_scores(
        self, heatmap: np.ndarray, original_image: np.ndarray = None
    ) -> dict:
        """
        Map Grad-CAM heatmap to facial region importance scores.

        Uses MediaPipe FaceMesh to detect 468 facial landmarks and maps
        heatmap importance to actual anatomical regions. Falls back to
        static grid if landmarks cannot be detected.

        Args:
            heatmap: Grad-CAM heatmap [H, W] in [0, 1].
            original_image: Original face image [H, W, 3] (RGB, 0-255).
                            Required for landmark detection.
        """
        h, w = heatmap.shape

        # Try landmark-based scoring first
        if original_image is not None:
            landmark_scores = self._compute_landmark_region_scores(
                heatmap, original_image
            )
            if landmark_scores is not None:
                return landmark_scores

        # Fallback: static grid (only when landmarks fail)
        return self._compute_grid_region_scores(heatmap)

    def _create_face_mask(
        self, landmarks, h: int, w: int
    ) -> np.ndarray:
        """
        Create a binary face mask using the FaceMesh face oval contour.

        This eliminates all background pixels from the heatmap, ensuring
        only face-region activations contribute to region scoring.

        Returns:
            Binary mask [H, W] with 1 inside the face, 0 outside.
        """
        # MediaPipe FACEMESH_FACE_OVAL indices (ordered contour of the face)
        face_oval_indices = [
            10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
            397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
            172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
        ]

        contour_pts = []
        for idx in face_oval_indices:
            lm = landmarks.landmark[idx]
            px = int(lm.x * w)
            py = int(lm.y * h)
            contour_pts.append([px, py])

        contour_pts = np.array(contour_pts, dtype=np.int32)

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, contour_pts, 1)

        return mask.astype(np.float32)

    def _get_landmark_points(
        self, landmarks, indices: list, h: int, w: int
    ) -> np.ndarray:
        """Convert landmark indices to pixel coordinates [N, 2]."""
        pts = []
        for idx in indices:
            if idx >= len(landmarks.landmark):
                continue
            lm = landmarks.landmark[idx]
            px = max(0, min(w - 1, int(lm.x * w)))
            py = max(0, min(h - 1, int(lm.y * h)))
            pts.append([px, py])
        return np.array(pts, dtype=np.int32)

    def _polygon_region_score(
        self, heatmap: np.ndarray, points: np.ndarray, pad: int = 8,
        use_sum: bool = False, total_activation: float = 1.0,
    ) -> float:
        """
        Compute heatmap importance inside a convex hull polygon region.

        Uses cv2.fillConvexPoly to create a filled mask from the landmark
        points. Supports two scoring modes:
          - mean: average heatmap value inside the polygon
          - sum (area-weighted): proportion of total activation in this region

        The sum mode prevents large regions (like jaw_chin) from dominating
        scores just because they cover more pixels.

        Args:
            heatmap: Grad-CAM heatmap [H, W] in [0, 1].
            points: Landmark coordinates [N, 2].
            pad: Pixel padding to expand the polygon.
            use_sum: If True, return sum/total_activation instead of mean.
            total_activation: Total heatmap activation (for normalization).

        Returns:
            Region importance score.
        """
        if len(points) < 3:
            return 0.0

        h, w = heatmap.shape

        # Expand polygon slightly outward from centroid for better coverage
        if pad > 0:
            centroid = points.mean(axis=0)
            direction = points.astype(np.float64) - centroid
            norms = np.linalg.norm(direction, axis=1, keepdims=True)
            norms = np.clip(norms, 1e-6, None)
            direction = direction / norms
            points = (points.astype(np.float64) + direction * pad).astype(np.int32)
            # Clamp to image bounds
            points[:, 0] = np.clip(points[:, 0], 0, w - 1)
            points[:, 1] = np.clip(points[:, 1], 0, h - 1)

        # Create filled polygon mask
        mask = np.zeros((h, w), dtype=np.uint8)
        hull = cv2.convexHull(points)
        cv2.fillConvexPoly(mask, hull, 1)

        # Compute heatmap importance inside the polygon
        masked_values = heatmap[mask > 0]
        if masked_values.size == 0:
            return 0.0

        if use_sum:
            return float(masked_values.sum() / total_activation)
        return float(masked_values.mean())

    def _compute_landmark_region_scores(
        self, heatmap: np.ndarray, original_image: np.ndarray
    ) -> dict:
        """
        Compute region scores using MediaPipe FaceMesh landmarks.

        Pipeline:
          1. Detect 468 facial landmarks via FaceMesh
          2. Create face-contour mask to eliminate background
          3. Apply mask to heatmap (zero out non-face areas)
          4. Score each anatomical region using convex-hull polygons

        Returns:
            dict mapping region names to normalized importance scores,
            or None if face landmarks cannot be detected (triggers fallback).
        """
        import mediapipe as mp

        h, w = heatmap.shape

        # Initialize FaceMesh
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.3,
        )

        # Ensure image is uint8 RGB
        if original_image.dtype != np.uint8:
            img = np.uint8(np.clip(original_image, 0, 255))
        else:
            img = original_image.copy()

        # Resize image to match heatmap if needed
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h))

        results = face_mesh.process(img)
        face_mesh.close()

        if not results.multi_face_landmarks:
            return None  # Fall back to grid

        landmarks = results.multi_face_landmarks[0]

        # ---- Step 1: Create face mask and apply to heatmap ----
        face_mask = self._create_face_mask(landmarks, h, w)
        masked_heatmap = heatmap * face_mask

        # Re-normalize masked heatmap to [0, 1]
        if masked_heatmap.max() > 0:
            masked_heatmap = masked_heatmap / masked_heatmap.max()

        # Store masked heatmap for visualization (set as instance var)
        self._last_masked_heatmap = masked_heatmap
        self._last_face_mask = face_mask

        # ---- Step 2: Define anatomical region landmark groups ----
        # Each group uses distinct, anatomically accurate FaceMesh indices
        region_indices = {
            "left_eyebrow": [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],
            "right_eyebrow": [300, 293, 334, 296, 336, 285, 295, 282, 283, 276],
            "left_eye": [
                33, 7, 163, 144, 145, 153, 154, 155, 133, 173,
                157, 158, 159, 160, 161, 246,
            ],
            "right_eye": [
                362, 382, 381, 380, 374, 373, 390, 249, 263, 466,
                388, 387, 386, 385, 384, 398,
            ],
            "nose": [
                1, 2, 98, 327, 168, 6, 197, 195, 5, 4,
                19, 94, 370, 114, 217, 126, 209, 49, 131, 134,
                51, 281, 363, 360, 279, 429, 437, 343,
            ],
            "mouth": [
                61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
                291, 308, 324, 318, 402, 317, 14, 87, 178, 88,
                95, 78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
            ],
            "jaw_chin": [
                # Only the chin/lower jaw area (below the mouth, not full contour)
                152, 377, 400, 378, 379, 365,  # right lower chin
                148, 176, 149, 150, 136, 172,  # left lower chin
            ],
        }

        # ---- Step 3: Compute area-weighted scores for each region ----
        # Use proportion of total activation (not mean) so large regions like
        # jaw_chin don't dominate just because their polygon is bigger.
        total_activation = float(masked_heatmap.sum()) + 1e-8
        scores = {}
        for region_name, indices in region_indices.items():
            pts = self._get_landmark_points(landmarks, indices, h, w)
            if len(pts) < 3:
                scores[region_name] = 0.0
                continue
            scores[region_name] = self._polygon_region_score(
                masked_heatmap, pts, pad=8, use_sum=True,
                total_activation=total_activation,
            )

        # ---- Step 4: Forehead — special handling ----
        # Forehead = area above eyebrows, bounded by face contour width
        # Get eyebrow top boundary
        eyebrow_all = region_indices["left_eyebrow"] + region_indices["right_eyebrow"]
        eyebrow_pts = self._get_landmark_points(landmarks, eyebrow_all, h, w)

        if len(eyebrow_pts) > 0:
            min_eyebrow_y = int(eyebrow_pts[:, 1].min())
            min_face_x = int(eyebrow_pts[:, 0].min())
            max_face_x = int(eyebrow_pts[:, 0].max())

            # Also get the top of the face contour for better x-bounds
            face_oval_top = [10, 338, 297, 109, 67, 103, 54, 21, 162]
            oval_pts = self._get_landmark_points(landmarks, face_oval_top, h, w)
            if len(oval_pts) > 0:
                min_face_x = min(min_face_x, int(oval_pts[:, 0].min()))
                max_face_x = max(max_face_x, int(oval_pts[:, 0].max()))

            # Clamp boundaries
            min_eyebrow_y = max(1, min_eyebrow_y)
            min_face_x = max(0, min_face_x)
            max_face_x = min(w, max_face_x)

            # Extract forehead region (only within face contour width)
            forehead_region = masked_heatmap[:min_eyebrow_y, min_face_x:max_face_x]
            if forehead_region.size > 0:
                scores["forehead"] = float(forehead_region.sum() / total_activation)
            else:
                scores["forehead"] = 0.0
        else:
            scores["forehead"] = 0.0

        # ---- Step 5: Normalize scores to sum to 1 ----
        total = sum(scores.values()) + 1e-8
        scores = {k: v / total for k, v in scores.items()}

        return scores

    def _compute_grid_region_scores(self, heatmap: np.ndarray) -> dict:
        """
        Fallback: Map Grad-CAM heatmap to facial region importance scores
        using a static grid when landmarks cannot be detected.
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
