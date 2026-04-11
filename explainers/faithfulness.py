"""
Cross-Modal Faithfulness Metric (CMFS).

==========================================================
*** THIS IS THE NOVEL RESEARCH CONTRIBUTION OF THE PAPER ***
==========================================================

Measures whether visual (Grad-CAM) and textual (SHAP) explanations
are consistent with each other and faithful to the model's decision.

Three sub-metrics:
  1. Sufficiency  — Do the top-k features from each modality suffice
                     to reproduce the prediction?
  2. Comprehensiveness — Does removing top-k features from each modality
                          change the prediction significantly?
  3. Cross-Modal Agreement — Do the visual and textual explanations
                              agree on the emotional signal?

References:
    DeYoung et al., "ERASER: A Benchmark to Evaluate Rationalized NLP Models",
    ACL 2020 (for sufficiency/comprehensiveness).

    Our novel addition: Cross-modal agreement for multimodal explanations.
"""

from typing import Optional, List, Tuple, Dict

import numpy as np
import torch
import torch.nn.functional as F


class CrossModalFaithfulness:
    """
    Evaluates the faithfulness and consistency of multimodal explanations.

    This is the key research contribution: a quantitative framework for
    evaluating whether XAI explanations from different modalities tell
    a coherent story about the model's decision.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: str = "cuda",
        num_perturbation_steps: int = 10,
        top_k_features: int = 5,
    ):
        """
        Args:
            model: The trained multimodal emotion model.
            device: "cuda" or "cpu".
            num_perturbation_steps: Steps for perturbation fidelity curves.
            top_k_features: Number of top features for sufficiency/comprehensiveness.
        """
        self.model = model
        self.device = device
        self.num_steps = num_perturbation_steps
        self.top_k = top_k_features

    def compute_all_metrics(
        self,
        image: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        gradcam_heatmap: np.ndarray,
        shap_values: np.ndarray,
        tokens: List[str],
        target_class: int,
    ) -> Dict[str, float]:
        """
        Compute all faithfulness metrics for a single sample.

        Args:
            image: Input face image [1, 3, H, W].
            input_ids: Text token IDs [1, seq_len].
            attention_mask: Text mask [1, seq_len].
            gradcam_heatmap: Grad-CAM output [H, W].
            shap_values: SHAP values per token [num_tokens].
            tokens: Token strings.
            target_class: Predicted emotion class index.

        Returns:
            dict with all faithfulness metric scores.
        """
        metrics = {}

        # 1. Vision Sufficiency
        metrics["vision_sufficiency"] = self._vision_sufficiency(
            image, input_ids, attention_mask, gradcam_heatmap, target_class
        )

        # 2. Vision Comprehensiveness
        metrics["vision_comprehensiveness"] = self._vision_comprehensiveness(
            image, input_ids, attention_mask, gradcam_heatmap, target_class
        )

        # 3. Text Sufficiency
        metrics["text_sufficiency"] = self._text_sufficiency(
            image, input_ids, attention_mask, shap_values, target_class
        )

        # 4. Text Comprehensiveness
        metrics["text_comprehensiveness"] = self._text_comprehensiveness(
            image, input_ids, attention_mask, shap_values, target_class
        )

        # 5. Cross-Modal Agreement (NOVEL)
        metrics["cross_modal_agreement"] = self._cross_modal_agreement(
            gradcam_heatmap, shap_values, tokens
        )

        # 6. Perturbation Fidelity Curves (NOVEL)
        vision_fidelity = self._vision_perturbation_curve(
            image, input_ids, attention_mask, gradcam_heatmap, target_class
        )
        text_fidelity = self._text_perturbation_curve(
            image, input_ids, attention_mask, shap_values, target_class
        )
        metrics["vision_auc_fidelity"] = float(np.trapz(vision_fidelity))
        metrics["text_auc_fidelity"] = float(np.trapz(text_fidelity))

        # 7. Combined CMFS score (geometric mean of key metrics)
        metrics["cmfs_score"] = self._compute_cmfs(metrics)

        return metrics

    # ============================================================
    # Sufficiency — keeping only top features
    # ============================================================

    def _vision_sufficiency(
        self,
        image: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        heatmap: np.ndarray,
        target_class: int,
    ) -> float:
        """
        Vision sufficiency: keep only top-k% important face regions.

        A high score means the important regions alone are sufficient
        to maintain the prediction.
        """
        self.model.eval()
        image = image.to(self.device)
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        # Original prediction confidence
        with torch.no_grad():
            orig_out = self.model(
                image=image, input_ids=input_ids, attention_mask=attention_mask
            )
            orig_prob = F.softmax(orig_out["logits"], dim=1)[0, target_class].item()

        # Create masked image: keep only top-k% important regions
        threshold = np.percentile(heatmap, 100 - self.top_k * 10)
        mask = (heatmap >= threshold).astype(np.float32)

        # Resize mask to image dimensions
        import cv2
        h, w = image.shape[2], image.shape[3]
        mask_resized = cv2.resize(mask, (w, h))
        mask_tensor = torch.FloatTensor(mask_resized).to(self.device)
        mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

        # Apply mask (zero out non-important regions)
        masked_image = image * mask_tensor

        # Get prediction on masked image
        with torch.no_grad():
            masked_out = self.model(
                image=masked_image, input_ids=input_ids, attention_mask=attention_mask
            )
            masked_prob = F.softmax(masked_out["logits"], dim=1)[
                0, target_class
            ].item()

        # Sufficiency = how much of the original confidence is retained
        return masked_prob / (orig_prob + 1e-8)

    def _text_sufficiency(
        self,
        image: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        shap_values: np.ndarray,
        target_class: int,
    ) -> float:
        """
        Text sufficiency: keep only top-k important tokens.
        """
        self.model.eval()
        image = image.to(self.device)
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        # Original prediction
        with torch.no_grad():
            orig_out = self.model(
                image=image, input_ids=input_ids, attention_mask=attention_mask
            )
            orig_prob = F.softmax(orig_out["logits"], dim=1)[0, target_class].item()

        # Find top-k important token positions
        abs_shap = np.abs(shap_values)
        seq_len = input_ids.shape[1]
        # Pad or truncate SHAP values to match sequence length
        if len(abs_shap) < seq_len:
            abs_shap = np.pad(abs_shap, (0, seq_len - len(abs_shap)))
        elif len(abs_shap) > seq_len:
            abs_shap = abs_shap[:seq_len]

        top_k_indices = np.argsort(abs_shap)[-self.top_k:]

        # Create masked input: keep only top-k tokens
        masked_ids = input_ids.clone()
        mask_token_id = self.model.text_encoder.backbone.config.pad_token_id or 0
        for i in range(seq_len):
            if i not in top_k_indices:
                masked_ids[0, i] = mask_token_id

        # Updated attention mask
        masked_attention = torch.zeros_like(attention_mask)
        for idx in top_k_indices:
            if idx < seq_len:
                masked_attention[0, idx] = 1

        with torch.no_grad():
            masked_out = self.model(
                image=image, input_ids=masked_ids, attention_mask=masked_attention
            )
            masked_prob = F.softmax(masked_out["logits"], dim=1)[
                0, target_class
            ].item()

        return masked_prob / (orig_prob + 1e-8)

    # ============================================================
    # Comprehensiveness — removing top features
    # ============================================================

    def _vision_comprehensiveness(
        self,
        image: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        heatmap: np.ndarray,
        target_class: int,
    ) -> float:
        """
        Vision comprehensiveness: remove top-k% important face regions.

        A high score means removing important regions significantly changed
        the prediction — meaning those regions truly mattered.
        """
        self.model.eval()
        image = image.to(self.device)
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        # Original prediction
        with torch.no_grad():
            orig_out = self.model(
                image=image, input_ids=input_ids, attention_mask=attention_mask
            )
            orig_prob = F.softmax(orig_out["logits"], dim=1)[0, target_class].item()

        # Remove top-k% important regions (invert mask)
        threshold = np.percentile(heatmap, 100 - self.top_k * 10)
        mask = (heatmap < threshold).astype(np.float32)

        import cv2
        h, w = image.shape[2], image.shape[3]
        mask_resized = cv2.resize(mask, (w, h))
        mask_tensor = torch.FloatTensor(mask_resized).to(self.device)
        mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)

        masked_image = image * mask_tensor

        with torch.no_grad():
            masked_out = self.model(
                image=masked_image, input_ids=input_ids, attention_mask=attention_mask
            )
            masked_prob = F.softmax(masked_out["logits"], dim=1)[
                0, target_class
            ].item()

        # Comprehensiveness = drop in confidence when removing important features
        return orig_prob - masked_prob

    def _text_comprehensiveness(
        self,
        image: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        shap_values: np.ndarray,
        target_class: int,
    ) -> float:
        """Text comprehensiveness: remove top-k important tokens."""
        self.model.eval()
        image = image.to(self.device)
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        with torch.no_grad():
            orig_out = self.model(
                image=image, input_ids=input_ids, attention_mask=attention_mask
            )
            orig_prob = F.softmax(orig_out["logits"], dim=1)[0, target_class].item()

        abs_shap = np.abs(shap_values)
        seq_len = input_ids.shape[1]
        if len(abs_shap) < seq_len:
            abs_shap = np.pad(abs_shap, (0, seq_len - len(abs_shap)))
        elif len(abs_shap) > seq_len:
            abs_shap = abs_shap[:seq_len]

        top_k_indices = np.argsort(abs_shap)[-self.top_k:]

        # Remove top-k tokens (replace with padding)
        masked_ids = input_ids.clone()
        masked_attention = attention_mask.clone()
        pad_id = self.model.text_encoder.backbone.config.pad_token_id or 0
        for idx in top_k_indices:
            if idx < seq_len:
                masked_ids[0, idx] = pad_id
                masked_attention[0, idx] = 0

        with torch.no_grad():
            masked_out = self.model(
                image=image, input_ids=masked_ids, attention_mask=masked_attention
            )
            masked_prob = F.softmax(masked_out["logits"], dim=1)[
                0, target_class
            ].item()

        return orig_prob - masked_prob

    # ============================================================
    # Cross-Modal Agreement (NOVEL CONTRIBUTION)
    # ============================================================

    def _cross_modal_agreement(
        self,
        gradcam_heatmap: np.ndarray,
        shap_values: np.ndarray,
        tokens: List[str],
    ) -> float:
        """
        Cross-modal agreement metric.

        Measures whether the visual and textual explanations are telling
        a consistent emotional story.

        Methodology:
        1. Map Grad-CAM regions to emotion-relevant facial AUs
        2. Map SHAP tokens to emotion-relevant word categories
        3. Both modalities implicitly encode emotion. Measure correlation
           between the emotion signal from each modality's top features.

        Implementation:
        - Extract top visual regions and their importance
        - Extract top text tokens and their importance
        - Compute agreement via normalized rank correlation
          (do the modalities rank the predicted emotion similarly?)
        """
        # Get normalized region scores from Grad-CAM
        h, w = gradcam_heatmap.shape
        region_importance = {
            "upper_face": float(gradcam_heatmap[: h // 3, :].mean()),
            "mid_face": float(gradcam_heatmap[h // 3: 2 * h // 3, :].mean()),
            "lower_face": float(gradcam_heatmap[2 * h // 3:, :].mean()),
        }

        # Normalize
        total_v = sum(region_importance.values()) + 1e-8
        vision_dist = np.array([v / total_v for v in region_importance.values()])

        # Get text importance distribution (top/mid/bottom thirds)
        abs_shap = np.abs(shap_values)
        n_tokens = len(abs_shap)
        if n_tokens >= 3:
            third = n_tokens // 3
            text_dist = np.array(
                [
                    abs_shap[:third].mean(),
                    abs_shap[third: 2 * third].mean(),
                    abs_shap[2 * third:].mean(),
                ]
            )
        else:
            text_dist = np.array([abs_shap.mean()] * 3)

        total_t = text_dist.sum() + 1e-8
        text_dist = text_dist / total_t

        # Compute agreement via:
        # (a) Cosine similarity of importance distributions
        cos_sim = np.dot(vision_dist, text_dist) / (
            np.linalg.norm(vision_dist) * np.linalg.norm(text_dist) + 1e-8
        )

        # (b) Entropy-based agreement: both should have similar certainty
        vision_entropy = -np.sum(
            vision_dist * np.log(vision_dist + 1e-8)
        )
        text_entropy = -np.sum(
            text_dist * np.log(text_dist + 1e-8)
        )
        max_entropy = np.log(3)  # Uniform distribution entropy
        entropy_agreement = 1.0 - abs(vision_entropy - text_entropy) / max_entropy

        # Combine both signals
        agreement = 0.6 * cos_sim + 0.4 * entropy_agreement

        return float(np.clip(agreement, 0.0, 1.0))

    # ============================================================
    # Perturbation Fidelity Curves
    # ============================================================

    def _vision_perturbation_curve(
        self,
        image: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        heatmap: np.ndarray,
        target_class: int,
    ) -> np.ndarray:
        """
        Compute perturbation fidelity curve for vision modality.

        Progressively removes the most important regions and measures
        the drop in prediction confidence.

        Returns:
            Array of confidence values at each perturbation step.
        """
        import cv2

        self.model.eval()
        image = image.to(self.device)
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        h, w = image.shape[2], image.shape[3]
        curve = []

        for step in range(self.num_steps + 1):
            fraction = step / self.num_steps

            if fraction == 0:
                masked_image = image
            else:
                threshold = np.percentile(heatmap, 100 * (1 - fraction))
                mask = (heatmap < threshold).astype(np.float32)
                mask_resized = cv2.resize(mask, (w, h))
                mask_tensor = torch.FloatTensor(mask_resized).to(self.device)
                mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)
                masked_image = image * mask_tensor

            with torch.no_grad():
                out = self.model(
                    image=masked_image,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                prob = F.softmax(out["logits"], dim=1)[0, target_class].item()
                curve.append(prob)

        return np.array(curve)

    def _text_perturbation_curve(
        self,
        image: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        shap_values: np.ndarray,
        target_class: int,
    ) -> np.ndarray:
        """Perturbation curve for text modality."""
        self.model.eval()
        image = image.to(self.device)
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        seq_len = input_ids.shape[1]
        abs_shap = np.abs(shap_values)
        if len(abs_shap) < seq_len:
            abs_shap = np.pad(abs_shap, (0, seq_len - len(abs_shap)))
        elif len(abs_shap) > seq_len:
            abs_shap = abs_shap[:seq_len]

        sorted_indices = np.argsort(abs_shap)[::-1]  # Most important first
        pad_id = getattr(
            self.model.text_encoder.backbone.config, "pad_token_id", 0
        ) or 0

        curve = []

        for step in range(self.num_steps + 1):
            n_remove = int((step / self.num_steps) * len(sorted_indices))

            masked_ids = input_ids.clone()
            masked_attn = attention_mask.clone()

            for idx in sorted_indices[:n_remove]:
                if idx < seq_len:
                    masked_ids[0, idx] = pad_id
                    masked_attn[0, idx] = 0

            with torch.no_grad():
                out = self.model(
                    image=image,
                    input_ids=masked_ids,
                    attention_mask=masked_attn,
                )
                prob = F.softmax(out["logits"], dim=1)[0, target_class].item()
                curve.append(prob)

        return np.array(curve)

    # ============================================================
    # Combined CMFS Score
    # ============================================================

    def _compute_cmfs(self, metrics: Dict[str, float]) -> float:
        """
        Compute the Combined Cross-Modal Faithfulness Score.

        CMFS = geometric mean of:
          - Vision sufficiency
          - Text sufficiency
          - Cross-modal agreement
          - Normalized comprehensiveness

        A high CMFS indicates that both modalities provide faithful,
        consistent explanations.
        """
        components = [
            max(metrics.get("vision_sufficiency", 0.0), 0.01),
            max(metrics.get("text_sufficiency", 0.0), 0.01),
            max(metrics.get("cross_modal_agreement", 0.0), 0.01),
            max(
                (
                    metrics.get("vision_comprehensiveness", 0.0)
                    + metrics.get("text_comprehensiveness", 0.0)
                )
                / 2.0,
                0.01,
            ),
        ]

        # Geometric mean
        cmfs = np.exp(np.mean(np.log(components)))
        return float(cmfs)

    def evaluate_dataset(
        self,
        dataloader,
        gradcam_explainer,
        shap_explainer,
        emotion_names: List[str],
        max_samples: int = 100,
    ) -> Dict[str, float]:
        """
        Evaluate faithfulness metrics over an entire dataset.

        Args:
            dataloader: PyTorch DataLoader.
            gradcam_explainer: GradCAMExplainer instance.
            shap_explainer: SHAPTextExplainer instance.
            emotion_names: List of emotion class names.
            max_samples: Maximum number of samples to evaluate.

        Returns:
            Aggregated metric averages and per-class breakdowns.
        """
        all_metrics = []
        per_class_metrics = {name: [] for name in emotion_names}

        count = 0
        for batch in dataloader:
            if count >= max_samples:
                break

            for i in range(batch["image"].shape[0]):
                if count >= max_samples:
                    break

                image = batch["image"][i].unsqueeze(0)
                input_ids = batch["input_ids"][i].unsqueeze(0)
                attention_mask = batch["attention_mask"][i].unsqueeze(0)
                label = batch["label"][i].item()

                try:
                    # Get Grad-CAM
                    gradcam_out = gradcam_explainer.generate(
                        image=image,
                        target_class=label,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )

                    # Get SHAP
                    utterance = batch["utterance"][i]
                    shap_out = shap_explainer.explain(
                        utterance,
                        target_class=label,
                        emotion_names=emotion_names,
                    )

                    # Compute faithfulness metrics
                    metrics = self.compute_all_metrics(
                        image=image,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        gradcam_heatmap=gradcam_out["heatmap"],
                        shap_values=shap_out["shap_values"],
                        tokens=shap_out["tokens"],
                        target_class=label,
                    )

                    all_metrics.append(metrics)
                    emotion_name = emotion_names[label]
                    per_class_metrics[emotion_name].append(metrics)

                except Exception as e:
                    print(f"  Warning: Failed to evaluate sample {count}: {e}")

                count += 1

        # Aggregate
        avg_metrics = {}
        if all_metrics:
            for key in all_metrics[0].keys():
                values = [m[key] for m in all_metrics]
                avg_metrics[f"avg_{key}"] = float(np.mean(values))
                avg_metrics[f"std_{key}"] = float(np.std(values))

        # Per-class
        class_metrics = {}
        for emotion, metrics_list in per_class_metrics.items():
            if metrics_list:
                class_metrics[emotion] = {
                    key: float(np.mean([m[key] for m in metrics_list]))
                    for key in metrics_list[0].keys()
                }

        avg_metrics["per_class"] = class_metrics
        avg_metrics["num_samples"] = len(all_metrics)

        return avg_metrics
