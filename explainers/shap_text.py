"""
SHAP Text Explainer for the Text Modality.

Computes per-token importance scores showing which words/phrases in the
transcript most influenced the emotion prediction.

Supports:
  - Partition SHAP (fast, tree-based partitioning of text)
  - Kernel SHAP (approximated here with a permutation-style fallback)

References:
    Lundberg & Lee, "A Unified Approach to Interpreting Model Predictions",
    NeurIPS 2017.
"""

from collections import defaultdict
from typing import Callable, List, Optional

import numpy as np
import shap
import torch
import torch.nn.functional as F
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
        self.max_length = getattr(
            getattr(self.model, "text_encoder", None), "max_length", 128
        )

        self.tokenizer = self._load_tokenizer(tokenizer_name)
        self.masker = shap.maskers.Text(self.tokenizer)

    @staticmethod
    def _load_tokenizer(tokenizer_name: str):
        """Load tokenizer with a safe fallback for regex patching."""
        try:
            return AutoTokenizer.from_pretrained(
                tokenizer_name,
                fix_mistral_regex=True,
            )
        except (AttributeError, TypeError):
            return AutoTokenizer.from_pretrained(tokenizer_name)

    def _make_model_fn(
        self,
        image: Optional[torch.Tensor] = None,
    ) -> Callable[[List[str]], np.ndarray]:
        """
        Create a callable that SHAP can use.

        The callable is bound to the current sample's image so text SHAP values
        are conditioned on the real multimodal context instead of a dummy frame.
        """
        model = self.model
        device = self.device
        tokenizer = self.tokenizer
        fixed_image = None

        if image is not None:
            fixed_image = image.detach()
            if fixed_image.dim() == 3:
                fixed_image = fixed_image.unsqueeze(0)
            fixed_image = fixed_image.to(device)

        def predict(texts):
            model.eval()
            all_probs = []
            text_list = list(texts)

            for i in range(0, len(text_list), self.batch_size):
                batch_texts = text_list[i: i + self.batch_size]
                encoded = tokenizer(
                    list(batch_texts),
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(device)

                with torch.no_grad():
                    if fixed_image is None:
                        batch_image = torch.zeros(
                            len(batch_texts), 3, 260, 260, device=device
                        )
                    else:
                        batch_image = fixed_image.expand(
                            len(batch_texts), -1, -1, -1
                        )

                    output = model(
                        image=batch_image,
                        input_ids=encoded["input_ids"],
                        attention_mask=encoded["attention_mask"],
                    )

                probs = F.softmax(output["logits"], dim=1).cpu().numpy()
                all_probs.append(probs)

            return np.concatenate(all_probs, axis=0)

        return predict

    def _build_explainer(
        self,
        model_fn: Callable[[List[str]], np.ndarray],
    ):
        """Build a fresh explainer for the current sample context."""
        if self.method == "partition":
            return shap.Explainer(
                model_fn,
                self.masker,
                output_names=None,
            )

        # The codebase only uses partition in practice; for "kernel" requests
        # fall back to a permutation-style explainer over the same text masker.
        return shap.Explainer(
            model_fn,
            self.masker,
            algorithm="permutation",
            output_names=None,
        )

    @staticmethod
    def _clean_token(token: str) -> str:
        """Format tokenizer pieces into something readable for plots/reports."""
        cleaned = (
            str(token)
            .replace("▁", " ")
            .replace("Ġ", " ")
            .replace("</w>", "")
            .strip()
        )
        return cleaned if cleaned else "<space>"

    def _build_model_token_scores(
        self,
        text: str,
        shap_values: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Align SHAP scores to the padded model input sequence.

        SHAP returns scores for content pieces only, while the model uses a
        padded sequence with special tokens. This alignment lets the downstream
        faithfulness metrics perturb the exact token positions seen by the model.
        """
        encoded = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"][0]
        attention_mask = encoded["attention_mask"][0]
        special_token_mask = np.array(
            self.tokenizer.get_special_tokens_mask(
                input_ids.tolist(),
                already_has_special_tokens=True,
            ),
            dtype=bool,
        )

        model_token_scores = np.zeros(len(input_ids), dtype=np.float32)
        active_content_positions = [
            idx
            for idx, (attn, is_special) in enumerate(
                zip(attention_mask.tolist(), special_token_mask.tolist())
            )
            if attn == 1 and not is_special
        ]

        for position, value in zip(active_content_positions, shap_values):
            if np.isfinite(value):
                model_token_scores[position] = float(value)

        return model_token_scores, special_token_mask

    def _aggregate_token_importance(
        self,
        tokens: List[str],
        shap_values: np.ndarray,
    ) -> dict:
        """Aggregate repeated token pieces instead of overwriting them."""
        aggregated = defaultdict(float)

        for token, value in zip(tokens, shap_values):
            if not np.isfinite(value):
                continue
            aggregated[self._clean_token(token)] += float(value)

        return dict(
            sorted(
                aggregated.items(),
                key=lambda item: abs(item[1]),
                reverse=True,
            )
        )

    def explain(
        self,
        text: str,
        image: Optional[torch.Tensor] = None,
        target_class: Optional[int] = None,
        emotion_names: Optional[List[str]] = None,
    ) -> dict:
        """
        Generate SHAP explanation for a single text input.

        Args:
            text: Input text (utterance/transcript).
            image: Optional image tensor used to condition multimodal predictions.
            target_class: Target emotion class. If None, uses predicted class.
            emotion_names: List of emotion class names for output.

        Returns:
            dict with:
                - shap_values: np.ndarray [num_tokens, num_classes] or [num_tokens]
                - tokens: List[str] for display
                - raw_tokens: Original SHAP token pieces
                - token_importance: dict mapping token -> importance for target class
                - model_token_scores: np.ndarray aligned to padded model input_ids
                - special_token_mask: np.ndarray flagging special tokens
                - predicted_class: int
                - base_value: float (expected value)
        """
        model_fn = self._make_model_fn(image=image)
        explainer = self._build_explainer(model_fn)

        shap_values = explainer([text], max_evals=self.max_evals)

        probs = model_fn(np.array([text], dtype=object))
        if target_class is None:
            target_class = int(probs.argmax(axis=1)[0])
        confidence = float(probs[0, target_class])

        if hasattr(shap_values, "values"):
            values = shap_values.values[0]
            if values.ndim == 2:
                target_values = values[:, target_class]
            else:
                target_values = values

            base_values = shap_values.base_values[0]
            if isinstance(base_values, np.ndarray) and base_values.size > 1:
                base_value = float(base_values[target_class])
            else:
                base_value = float(base_values)
        else:
            target_values = np.zeros(1, dtype=np.float32)
            base_value = 0.0

        if hasattr(shap_values, "data"):
            raw_tokens = list(shap_values.data[0])
        else:
            raw_tokens = text.split()

        tokens = [self._clean_token(token) for token in raw_tokens]
        token_importance = self._aggregate_token_importance(
            raw_tokens,
            target_values,
        )
        model_token_scores, special_token_mask = self._build_model_token_scores(
            text=text,
            shap_values=target_values,
        )

        return {
            "shap_values": np.asarray(target_values, dtype=np.float32),
            "tokens": tokens,
            "raw_tokens": raw_tokens,
            "token_importance": token_importance,
            "model_token_scores": model_token_scores,
            "special_token_mask": special_token_mask,
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
        images: Optional[List[torch.Tensor]] = None,
        target_classes: Optional[List[int]] = None,
        emotion_names: Optional[List[str]] = None,
    ) -> List[dict]:
        """Generate SHAP explanations for multiple texts."""
        results = []
        for i, text in enumerate(texts):
            target = target_classes[i] if target_classes else None
            image = images[i] if images else None
            result = self.explain(
                text,
                image=image,
                target_class=target,
                emotion_names=emotion_names,
            )
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

        positive = [(token, score) for token, score in importance.items() if score > 0]
        negative = [(token, score) for token, score in importance.items() if score < 0]

        positive.sort(key=lambda item: item[1], reverse=True)
        negative.sort(key=lambda item: abs(item[1]), reverse=True)

        result = {"positive": positive[:k]}
        if include_negative:
            result["negative"] = negative[:k]

        return result
