"""
Natural Language Generation (NLG) Report Generator.

Converts structured XAI outputs (Grad-CAM regions, SHAP tokens,
faithfulness metrics) into human-readable plain-English emotional
analysis reports.

Supports two modes:
  1. Template-based — deterministic, reproducible, no external deps
  2. LLM-powered — uses a local LLM (via Ollama) for fluent narration

Example output:
    "The model detected **sadness** (confidence: 87.3%) in this utterance.

    Visual cues: The face analysis focused primarily on the mouth region
    (importance: 0.42) showing a downturned expression, along with the
    eye region (0.31) suggesting tearfulness.

    Textual cues: The phrase 'I can't do this anymore' contributed most
    strongly (+0.65), followed by the word 'alone' (+0.38).

    Cross-modal consistency: 0.89 — both modalities strongly agree on
    the sadness signal."
"""

import json
import math
import re
import subprocess
from typing import Dict, List, Optional


# Emotion descriptions for richer NLG
EMOTION_DESCRIPTIONS = {
    "Neutral": {
        "face_cues": "minimal muscle movement, relaxed expression",
        "text_cues": "matter-of-fact language, neutral tone",
        "description": "absence of strong emotional signals",
    },
    "Surprise": {
        "face_cues": "widened eyes, raised eyebrows, open mouth",
        "text_cues": "exclamatory phrases, unexpected content",
        "description": "unexpected event or revelation",
    },
    "Fear": {
        "face_cues": "widened eyes, tense forehead, slightly open mouth",
        "text_cues": "anxiety-related words, threat language",
        "description": "perceived danger or anxiety",
    },
    "Sadness": {
        "face_cues": "downturned mouth corners, drooping eyelids, furrowed brow",
        "text_cues": "loss-related words, expressions of pain or longing",
        "description": "feelings of loss, grief, or disappointment",
    },
    "Joy": {
        "face_cues": "raised cheeks, crow's feet wrinkles, Duchenne smile",
        "text_cues": "positive language, laughter indicators, upbeat expressions",
        "description": "happiness, amusement, or positive excitement",
    },
    "Disgust": {
        "face_cues": "wrinkled nose, raised upper lip, narrowed eyes",
        "text_cues": "aversion language, negative evaluation words",
        "description": "aversion or strong dislike",
    },
    "Anger": {
        "face_cues": "furrowed brows, tightened jaw, narrowed eyes, pressed lips",
        "text_cues": "hostile language, confrontational expressions",
        "description": "hostility, frustration, or indignation",
    },
    "Contempt": {
        "face_cues": "asymmetric smile, raised lip corner on one side",
        "text_cues": "dismissive language, sarcasm markers",
        "description": "superiority or disdain",
    },
    "Happy": {
        "face_cues": "raised cheeks, crow's feet wrinkles, Duchenne smile",
        "text_cues": "positive language, laughter indicators, upbeat expressions",
        "description": "happiness, amusement, or positive excitement",
    },
    "Sad": {
        "face_cues": "downturned mouth corners, drooping eyelids, furrowed brow",
        "text_cues": "loss-related words, expressions of pain or longing",
        "description": "feelings of loss, grief, or disappointment",
    },
}

# Facial region descriptions
REGION_DESCRIPTIONS = {
    "forehead": "forehead area (associated with brow furrows and surprise signals)",
    "left_eye": "left eye region (gaze, tear ducts, crow's feet)",
    "right_eye": "right eye region (gaze, widening, narrowing)",
    "nose": "nose bridge and nostrils (wrinkle indicators for disgust)",
    "mouth": "mouth region (smile, frown, lip tension)",
    "jaw_chin": "jaw and chin area (tension, clenching indicators)",
    "upper_face": "upper face (forehead and eyes — surprise, fear signals)",
    "mid_face": "mid-face (nose and cheeks — disgust, contempt cues)",
    "lower_face": "lower face (mouth and jaw — smile, anger signals)",
}


class NLGReportGenerator:
    """
    Generates natural language explanations from XAI outputs.

    Converts Grad-CAM region scores, SHAP token importance, and
    faithfulness metrics into readable emotional analysis reports.
    """

    def __init__(
        self,
        mode: str = "template",
        min_importance: float = 0.05,
        max_features: int = 5,
        llm_model: str = "llama3.1:8b",
        llm_provider: str = "ollama",
    ):
        """
        Args:
            mode: "template" or "llm".
            min_importance: Minimum feature importance to include in report.
            max_features: Maximum features per modality in report.
            llm_model: Model name for LLM narration.
            llm_provider: "ollama" or "openai".
        """
        self.mode = mode
        self.min_importance = min_importance
        self.max_features = max_features
        self.llm_model = llm_model
        self.llm_provider = llm_provider

    @staticmethod
    def _is_finite(value) -> bool:
        return isinstance(value, (int, float)) and math.isfinite(float(value))

    @staticmethod
    def _is_word_like(token: str) -> bool:
        return bool(re.search(r"[A-Za-z0-9]", token))

    def _select_report_tokens(
        self,
        token_importance: Dict[str, float],
        positive: bool,
        limit: int,
    ) -> List[tuple[str, float]]:
        """Prefer actual word pieces over bare punctuation in the narrative."""
        candidates = [
            (token, score)
            for token, score in token_importance.items()
            if (score > self.min_importance if positive else score < -self.min_importance)
        ]
        if not candidates:
            return []

        primary = [item for item in candidates if self._is_word_like(item[0])]
        fallback = [item for item in candidates if not self._is_word_like(item[0])]
        ordered = primary + fallback
        return ordered[:limit]

    def _describe_sufficiency(self, value: float, modality_name: str) -> str:
        """Turn sufficiency into calibrated English instead of overclaiming."""
        if not self._is_finite(value):
            return f"{modality_name} sufficiency is unavailable for this sample."
        if value > 1.05:
            return (
                f"{modality_name} sufficiency: {value:.3f} — masking down to the "
                f"highlighted {modality_name.lower()} features increased confidence, "
                "so this estimate is unstable rather than strongly faithful."
            )
        if value >= 0.75:
            return (
                f"{modality_name} sufficiency: {value:.3f} — the highlighted "
                f"{modality_name.lower()} features retain most of the signal."
            )
        if value >= 0.40:
            return (
                f"{modality_name} sufficiency: {value:.3f} — the highlighted "
                f"{modality_name.lower()} features capture part of the signal, "
                "but surrounding context still matters."
            )
        return (
            f"{modality_name} sufficiency: {value:.3f} — the prediction depends on "
            f"broader {modality_name.lower()} context beyond the highlighted features."
        )

    def generate_report(
        self,
        emotion_name: str,
        confidence: float,
        region_scores: Dict[str, float],
        token_importance: Dict[str, float],
        faithfulness_metrics: Optional[Dict[str, float]] = None,
        utterance: str = "",
        speaker: str = "Unknown",
    ) -> str:
        """
        Generate a complete explanation report.

        Args:
            emotion_name: Predicted emotion (e.g., "Sadness").
            confidence: Model confidence (0-1).
            region_scores: Grad-CAM facial region importance dict.
            token_importance: SHAP token -> importance dict.
            faithfulness_metrics: Optional CMFS metrics dict.
            utterance: Original text input.
            speaker: Speaker name (for MELD).

        Returns:
            Human-readable explanation string.
        """
        if self.mode == "template":
            return self._template_report(
                emotion_name,
                confidence,
                region_scores,
                token_importance,
                faithfulness_metrics,
                utterance,
                speaker,
            )
        else:
            return self._llm_report(
                emotion_name,
                confidence,
                region_scores,
                token_importance,
                faithfulness_metrics,
                utterance,
                speaker,
            )

    def _template_report(
        self,
        emotion_name: str,
        confidence: float,
        region_scores: Dict[str, float],
        token_importance: Dict[str, float],
        faithfulness_metrics: Optional[Dict[str, float]],
        utterance: str,
        speaker: str,
    ) -> str:
        """Generate report using templates (deterministic, reproducible)."""
        lines = []

        # ---- Header ----
        lines.append("=" * 60)
        lines.append("MULTIMODAL EMOTION ANALYSIS REPORT")
        lines.append("=" * 60)
        lines.append("")

        # ---- Prediction Summary ----
        lines.append(f"Predicted Emotion: **{emotion_name}**")
        lines.append(f"Confidence: {confidence * 100:.1f}%")
        if speaker != "Unknown":
            lines.append(f"Speaker: {speaker}")
        if utterance:
            lines.append(f'Utterance: "{utterance}"')
        lines.append("")

        if confidence < 0.30:
            lines.append(
                "This is a low-confidence prediction, so the explanation should be "
                "read as tentative."
            )
            lines.append("")

        # Get emotion description
        emo_info = EMOTION_DESCRIPTIONS.get(emotion_name, {})
        if emo_info:
            lines.append(
                f"This prediction indicates {emo_info.get('description', 'an emotional response')}."
            )
            lines.append("")

        # ---- Visual Explanation ----
        lines.append("--- Visual Cues (Face Analysis) ---")
        lines.append("")

        # Sort regions by importance
        sorted_regions = sorted(
            region_scores.items(), key=lambda x: x[1], reverse=True
        )
        top_regions = [
            (r, s) for r, s in sorted_regions
            if s >= self.min_importance
        ][: self.max_features]

        if top_regions:
            primary_region, primary_score = top_regions[0]
            region_desc = REGION_DESCRIPTIONS.get(
                primary_region, primary_region
            )
            lines.append(
                f"The model focused primarily on the {region_desc} "
                f"(importance: {primary_score:.2f})."
            )

            if len(top_regions) > 1:
                secondary = ", ".join(
                    [
                        f"{REGION_DESCRIPTIONS.get(r, r)} ({s:.2f})"
                        for r, s in top_regions[1:]
                    ]
                )
                lines.append(f"Secondary attention areas: {secondary}.")

            if emo_info.get("face_cues"):
                lines.append(
                    f"Typical facial cues for {emotion_name.lower()}: "
                    f"{emo_info['face_cues']}."
                )
        else:
            lines.append("No significant facial region activations detected.")

        lines.append("")

        # ---- Textual Explanation ----
        lines.append("--- Textual Cues (Language Analysis) ---")
        lines.append("")

        # Get top positive and negative tokens
        positive_tokens = self._select_report_tokens(
            token_importance,
            positive=True,
            limit=self.max_features,
        )
        negative_tokens = self._select_report_tokens(
            token_importance,
            positive=False,
            limit=3,
        )

        if positive_tokens:
            primary_token, primary_shap = positive_tokens[0]
            lines.append(
                f'The word/phrase "{primary_token}" contributed most strongly '
                f"toward {emotion_name.lower()} (SHAP: +{primary_shap:.3f})."
            )

            if len(positive_tokens) > 1:
                other_tokens = ", ".join(
                    [f'"{t}" (+{s:.3f})' for t, s in positive_tokens[1:]]
                )
                lines.append(f"Other supporting words: {other_tokens}.")

            if emo_info.get("text_cues"):
                lines.append(
                    f"This aligns with typical {emotion_name.lower()} "
                    f"text patterns: {emo_info['text_cues']}."
                )
        else:
            lines.append(
                "No individual words showed strong contribution to this prediction."
            )

        if negative_tokens:
            counter_tokens = ", ".join(
                [f'"{t}" ({s:.3f})' for t, s in negative_tokens]
            )
            lines.append(
                f"\nWords pushing against {emotion_name.lower()}: {counter_tokens}."
            )

        lines.append("")

        # ---- Faithfulness Metrics ----
        if faithfulness_metrics:
            lines.append("--- Explanation Quality Metrics ---")
            lines.append("")

            cmfs = faithfulness_metrics.get("cmfs_score", float("nan"))
            if self._is_finite(cmfs):
                lines.append(
                    f"Cross-Modal Faithfulness Score (CMFS): {cmfs:.3f}"
                )
            else:
                lines.append(
                    "Cross-Modal Faithfulness Score (CMFS): unavailable for this sample."
                )

            agreement = faithfulness_metrics.get("cross_modal_agreement", float("nan"))
            if self._is_finite(agreement):
                if agreement > 0.7:
                    agreement_desc = "strong agreement"
                elif agreement > 0.4:
                    agreement_desc = "moderate agreement"
                else:
                    agreement_desc = "weak agreement"

                lines.append(
                    f"Cross-Modal Agreement: {agreement:.3f} ({agreement_desc}) — "
                    f"visual and textual explanations {'consistently support' if agreement > 0.5 else 'show noticeable divergence in'} "
                    f"the {emotion_name.lower()} prediction."
                )
            else:
                lines.append(
                    "Cross-Modal Agreement: unavailable for this sample."
                )

            lines.append(
                self._describe_sufficiency(
                    faithfulness_metrics.get("vision_sufficiency", float("nan")),
                    "Vision",
                )
            )
            lines.append(
                self._describe_sufficiency(
                    faithfulness_metrics.get("text_sufficiency", float("nan")),
                    "Text",
                )
            )

            lines.append("")

        # ---- Footer ----
        lines.append("=" * 60)

        return "\n".join(lines)

    def _llm_report(
        self,
        emotion_name: str,
        confidence: float,
        region_scores: Dict[str, float],
        token_importance: Dict[str, float],
        faithfulness_metrics: Optional[Dict[str, float]],
        utterance: str,
        speaker: str,
    ) -> str:
        """
        Generate report using a local LLM for more fluent narration.

        Falls back to template mode if LLM is unavailable.
        """
        # Prepare structured data for the LLM
        xai_data = {
            "predicted_emotion": emotion_name,
            "confidence": f"{confidence * 100:.1f}%",
            "speaker": speaker,
            "utterance": utterance,
            "top_facial_regions": {
                k: f"{v:.3f}"
                for k, v in sorted(
                    region_scores.items(), key=lambda x: x[1], reverse=True
                )[:4]
            },
            "top_text_features": {
                k: f"{v:+.3f}"
                for k, v in list(token_importance.items())[:5]
            },
        }

        if faithfulness_metrics:
            agreement = faithfulness_metrics.get("cross_modal_agreement", float("nan"))
            cmfs = faithfulness_metrics.get("cmfs_score", float("nan"))
            if self._is_finite(agreement):
                xai_data["cross_modal_agreement"] = f"{agreement:.3f}"
            if self._is_finite(cmfs):
                xai_data["cmfs_score"] = f"{cmfs:.3f}"

        prompt = f"""You are an AI emotion analysis expert. Given the following XAI 
(Explainable AI) data from a multimodal emotion recognition model, write a clear, 
concise, and informative plain-English explanation for a non-technical reader.

XAI Data:
{json.dumps(xai_data, indent=2)}

Guidelines:
- Explain what emotion was detected and why
- Describe which parts of the face and which words were most relevant
- Mention whether the visual and text signals agree
- Keep it to 3-4 short paragraphs
- Use simple, accessible language — avoid jargon
- Do NOT make up information not present in the data

Write the explanation:"""

        try:
            if self.llm_provider == "ollama":
                result = subprocess.run(
                    ["ollama", "run", self.llm_model, prompt],
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                if result.returncode == 0 and result.stdout.strip():
                    return result.stdout.strip()
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            print(f"[NLG] LLM unavailable ({e}), falling back to template mode")

        # Fallback to template
        return self._template_report(
            emotion_name,
            confidence,
            region_scores,
            token_importance,
            faithfulness_metrics,
            utterance,
            speaker,
        )

    def generate_comparative_report(
        self,
        reports: List[Dict],
        emotion_names: List[str],
    ) -> str:
        """
        Generate a comparative report across multiple samples.

        Useful for paper results sections — summarizes patterns across
        many predictions.
        """
        lines = []
        lines.append("=" * 60)
        lines.append("COMPARATIVE ANALYSIS REPORT")
        lines.append("=" * 60)
        lines.append(f"\nTotal samples analyzed: {len(reports)}\n")

        # Aggregate per emotion
        emotion_stats = {}
        for report in reports:
            emo = report.get("emotion_name", "Unknown")
            if emo not in emotion_stats:
                emotion_stats[emo] = {
                    "count": 0,
                    "avg_confidence": 0,
                    "avg_agreement": 0,
                    "top_regions": {},
                    "top_tokens": {},
                }
            stats = emotion_stats[emo]
            stats["count"] += 1
            stats["avg_confidence"] += report.get("confidence", 0)
            stats["avg_agreement"] += report.get(
                "cross_modal_agreement", 0
            )

            # Track region frequencies
            for region, score in report.get("region_scores", {}).items():
                if region not in stats["top_regions"]:
                    stats["top_regions"][region] = 0
                stats["top_regions"][region] += score

        for emo, stats in emotion_stats.items():
            n = stats["count"]
            lines.append(f"\n--- {emo} ({n} samples) ---")
            lines.append(
                f"  Average confidence: {stats['avg_confidence'] / n * 100:.1f}%"
            )
            lines.append(
                f"  Average cross-modal agreement: {stats['avg_agreement'] / n:.3f}"
            )

            # Top regions for this emotion
            sorted_regions = sorted(
                stats["top_regions"].items(),
                key=lambda x: x[1],
                reverse=True,
            )[:3]
            region_strs = [f"{r} ({s / n:.2f})" for r, s in sorted_regions]
            lines.append(f"  Key facial regions: {', '.join(region_strs)}")

        return "\n".join(lines)
