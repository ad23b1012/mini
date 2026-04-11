"""
Evaluation Metrics for Emotion Recognition.

Computes:
  - Per-class accuracy
  - Weighted F1, Macro F1
  - Confusion matrix
  - Classification report
"""

from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)


class EmotionMetrics:
    """
    Tracks and computes classification metrics for emotion recognition.
    """

    def __init__(
        self,
        num_classes: int = 7,
        class_names: Optional[List[str]] = None,
    ):
        self.num_classes = num_classes
        self.class_names = class_names or [
            str(i) for i in range(num_classes)
        ]

    def compute(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
    ) -> Dict[str, float]:
        """
        Compute all metrics from predictions and ground truth labels.

        Args:
            predictions: [N] predicted class indices.
            labels: [N] ground truth class indices.

        Returns:
            dict with accuracy, F1 scores, precision, recall.
        """
        metrics = {}

        # Overall accuracy
        metrics["accuracy"] = float(accuracy_score(labels, predictions))

        # F1 scores
        metrics["f1_weighted"] = float(
            f1_score(
                labels, predictions, average="weighted", zero_division=0
            )
        )
        metrics["f1_macro"] = float(
            f1_score(
                labels, predictions, average="macro", zero_division=0
            )
        )

        # Precision and recall
        metrics["precision_weighted"] = float(
            precision_score(
                labels, predictions, average="weighted", zero_division=0
            )
        )
        metrics["recall_weighted"] = float(
            recall_score(
                labels, predictions, average="weighted", zero_division=0
            )
        )

        # Per-class F1
        per_class_f1 = f1_score(
            labels,
            predictions,
            average=None,
            labels=range(self.num_classes),
            zero_division=0,
        )
        for i, name in enumerate(self.class_names):
            if i < len(per_class_f1):
                metrics[f"f1_{name}"] = float(per_class_f1[i])

        return metrics

    def confusion_matrix(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
    ) -> np.ndarray:
        """Compute confusion matrix."""
        return confusion_matrix(
            labels,
            predictions,
            labels=range(self.num_classes),
        )

    def classification_report_str(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
    ) -> str:
        """Generate a full classification report string."""
        return classification_report(
            labels,
            predictions,
            target_names=self.class_names,
            labels=range(self.num_classes),
            zero_division=0,
        )

    def generate_latex_table(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        model_name: str = "MMER-XAI",
    ) -> str:
        """
        Generate a LaTeX-formatted results table for the paper.

        Produces a table with per-class precision, recall, F1
        plus weighted and macro averages.
        """
        report = classification_report(
            labels,
            predictions,
            target_names=self.class_names,
            labels=range(self.num_classes),
            output_dict=True,
            zero_division=0,
        )

        lines = []
        lines.append("\\begin{table}[h]")
        lines.append("\\centering")
        lines.append(
            f"\\caption{{Per-class results for {model_name} on emotion recognition}}"
        )
        lines.append("\\begin{tabular}{lcccc}")
        lines.append("\\toprule")
        lines.append("Emotion & Precision & Recall & F1-Score & Support \\\\")
        lines.append("\\midrule")

        for name in self.class_names:
            if name in report:
                r = report[name]
                lines.append(
                    f"{name} & {r['precision']:.3f} & {r['recall']:.3f} "
                    f"& {r['f1-score']:.3f} & {int(r['support'])} \\\\"
                )

        lines.append("\\midrule")

        # Averages
        for avg_name in ["weighted avg", "macro avg"]:
            if avg_name in report:
                r = report[avg_name]
                display_name = avg_name.replace(" avg", "").capitalize() + " Avg"
                lines.append(
                    f"\\textbf{{{display_name}}} & {r['precision']:.3f} "
                    f"& {r['recall']:.3f} & {r['f1-score']:.3f} "
                    f"& {int(r['support'])} \\\\"
                )

        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}")

        return "\n".join(lines)
