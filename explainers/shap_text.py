"""
SHAP Text Explainer for the Text Modality.

Computes per-token importance scores showing which words/phrases in the
transcript most influenced the emotion prediction.

Supports:
  - Partition SHAP (fast, tree-based partitioning of text)
  - Kernel SHAP (model-agnostic, slower but more reliable)

References:
    Lundberg & Lee, "A Unified Approach to Interpreting Model Predictions",
    NeurIPS 2017.
"""

from typing import Optional, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import shap
from transformers import AutoTokenizer


class SHAPTextExplainer:
    """
    SHAP-based explainer for the text modality.

    Computes how each word/token in the input text contributes to
    the emotion prediction.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer_name: str = "microsoft/deberta-v3-base",
        device: str = "cuda",
        method: str = "partition",
        max_evals: int = 500,
        batch_size: int = 16,
    ):
        """
        Args:
            model: The full multimodal model or text-only model.
            tokenizer_name: HuggingFace tokenizer.
            device: "cuda" or "cpu".
            method: "partition" or "kernel".
            max_evals: Maximum SHAP evaluations per sample.
            batch_size: Batch size for SHAP evaluation.
        """
        self.model = model
        self.device = device
        self.method = method
        self.max_evals = max_evals
        self.batch_size = batch_size

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        # Create SHAP masker and explainer
        self.masker = shap.maskers.Text(self.tokenizer)

        # Wrap model for SHAP
        self._model_fn = self._make_model_fn()

        # Create explainer
        if method == "partition":
            self.explainer = shap.Explainer(
                self._model_fn,
                self.masker,
                output_names=None,  # Will be set per-call
            )
        else:
            self.explainer = shap.KernelExplainer(
                self._model_fn,
                shap.maskers.Text(self.tokenizer),
            )

    def _make_model_fn(self):
        """
        Create a callable that SHAP can use.

        Takes raw text strings and returns prediction probabilities.
        """
        model = self.model
        device = self.device
        tokenizer = self.tokenizer

        def predict(texts):
            """Predict emotion probabilities from text strings."""
            model.eval()
            all_probs = []

            # Process in batches
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i: i + self.batch_size]

                # Tokenize
                encoded = tokenizer(
                    list(batch_texts),
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors="pt",
                ).to(device)

                with torch.no_grad():
                    # Create dummy image (zeros) for multimodal mode
                    dummy_image = torch.zeros(
                        len(batch_texts), 3, 260, 260, device=device
                    )

                    output = model(
                        image=dummy_image,
                        input_ids=encoded["input_ids"],
                        attention_mask=encoded["attention_mask"],
                    )

                logits = output["logits"]
                probs = F.softmax(logits, dim=1).cpu().numpy()
                all_probs.append(probs)

            return np.concatenate(all_probs, axis=0)

        return predict

    def explain(
        self,
        text: str,
        target_class: Optional[int] = None,
        emotion_names: Optional[List[str]] = None,
    ) -> dict:
        """
        Generate SHAP explanation for a single text input.

        Args:
            text: Input text (utterance/transcript).
            target_class: Target emotion class. If None, uses predicted class.
            emotion_names: List of emotion class names for output.

        Returns:
            dict with:
                - shap_values: np.ndarray [num_tokens, num_classes] or [num_tokens]
                - tokens: List[str] — tokenized words
                - token_importance: dict mapping token -> importance for target class
                - predicted_class: int
                - base_value: float (expected value)
        """
        # Run SHAP
        shap_values = self.explainer(
            [text], max_evals=self.max_evals
        )

        # Get prediction
        probs = self._model_fn(np.array([text]))
        if target_class is None:
            target_class = int(probs.argmax(axis=1)[0])
        confidence = float(probs[0, target_class])

        # Extract values for the target class
        if hasattr(shap_values, "values"):
            values = shap_values.values[0]  # [num_tokens, num_classes] or [num_tokens]
            if values.ndim == 2:
                target_values = values[:, target_class]
            else:
                target_values = values
            base_value = float(shap_values.base_values[0])
        else:
            target_values = np.zeros(1)
            base_value = 0.0

        # Get word-level tokens
        if hasattr(shap_values, "data"):
            tokens = list(shap_values.data[0])
        else:
            tokens = text.split()

        # Build token importance dict
        token_importance = {}
        for token, value in zip(tokens, target_values):
            token_str = str(token).strip()
            if token_str:
                token_importance[token_str] = float(value)

        # Sort by absolute importance
        token_importance = dict(
            sorted(
                token_importance.items(),
                key=lambda x: abs(x[1]),
                reverse=True,
            )
        )

        return {
            "shap_values": target_values,
            "tokens": tokens,
            "token_importance": token_importance,
            "predicted_class": target_class,
            "confidence": confidence,
            "base_value": base_value,
            "emotion_name": (
                emotion_names[target_class]
                if emotion_names and target_class < len(emotion_names)
                else str(target_class)
            ),
        }

    def explain_batch(
        self,
        texts: List[str],
        target_classes: Optional[List[int]] = None,
        emotion_names: Optional[List[str]] = None,
    ) -> List[dict]:
        """Generate SHAP explanations for multiple texts."""
        results = []
        for i, text in enumerate(texts):
            target = target_classes[i] if target_classes else None
            result = self.explain(text, target_class=target, emotion_names=emotion_names)
            results.append(result)
        return results

    def get_top_features(
        self,
        explanation: dict,
        k: int = 5,
        include_negative: bool = True,
    ) -> dict:
        """
        Get the top-k most important tokens from an explanation.

        Args:
            explanation: Output from self.explain().
            k: Number of top features.
            include_negative: Include tokens that pushed away from prediction.

        Returns:
            dict with:
                - positive: list of (token, score) pushing toward prediction
                - negative: list of (token, score) pushing away (if requested)
        """
        importance = explanation["token_importance"]

        # Split positive and negative contributions
        positive = [(t, s) for t, s in importance.items() if s > 0]
        negative = [(t, s) for t, s in importance.items() if s < 0]

        # Sort by magnitude
        positive.sort(key=lambda x: x[1], reverse=True)
        negative.sort(key=lambda x: abs(x[1]), reverse=True)

        result = {"positive": positive[:k]}
        if include_negative:
            result["negative"] = negative[:k]

        return result
