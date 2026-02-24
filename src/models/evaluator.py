"""Classification evaluation suite with per-language metrics.

Tracks accuracy, F1 scores, confusion matrices, and cross-lingual
transfer performance for comprehensive model assessment.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

logger = logging.getLogger(__name__)


class ClassificationEvaluator:
    """Evaluates classification performance across languages and intents.

    Computes standard metrics (accuracy, F1, precision, recall) with
    breakdowns per language and per intent category, plus cross-lingual
    transfer analysis.

    Args:
        intent_categories: List of valid intent category labels.
    """

    def __init__(self, intent_categories: list[str] | None = None) -> None:
        from src.utils.config import load_config

        config = load_config()
        self.intent_categories = (
            intent_categories or config.classification.intent_categories
        )
        self.results_history: list[dict] = []

    def evaluate(
        self,
        y_true: list[str],
        y_pred: list[str],
        languages: list[str],
        confidences: list[float] | None = None,
    ) -> dict:
        """Run comprehensive evaluation with per-language breakdown.

        Args:
            y_true: Ground truth labels.
            y_pred: Predicted labels.
            languages: Language code for each sample.
            confidences: Optional confidence scores for each prediction.

        Returns:
            Dictionary containing overall metrics, per-language metrics,
            confusion matrix, cross-lingual analysis, and low-confidence stats.
        """
        df = pd.DataFrame(
            {
                "true": y_true,
                "pred": y_pred,
                "language": languages,
                "confidence": confidences or [1.0] * len(y_true),
            }
        )

        results: dict = {
            "overall": self._compute_metrics(y_true, y_pred),
            "per_language": {},
            "confusion_matrix": self._compute_confusion_matrix(y_true, y_pred),
            "cross_lingual_transfer": {},
            "low_confidence_analysis": {},
        }

        # Per-language metrics
        for lang in sorted(df["language"].unique()):
            lang_df = df[df["language"] == lang]
            lang_metrics = self._compute_metrics(
                lang_df["true"].tolist(), lang_df["pred"].tolist()
            )
            lang_metrics["sample_count"] = len(lang_df)
            results["per_language"][lang] = lang_metrics

        # Cross-lingual transfer analysis
        results["cross_lingual_transfer"] = self._analyze_cross_lingual(df)

        # Low confidence analysis
        if confidences:
            low_conf = df[df["confidence"] < 0.7]
            if len(low_conf) > 0:
                results["low_confidence_analysis"] = {
                    "count": len(low_conf),
                    "accuracy": float(
                        accuracy_score(low_conf["true"], low_conf["pred"])
                    ),
                }
            else:
                results["low_confidence_analysis"] = {
                    "count": 0,
                    "accuracy": None,
                }

        self.results_history.append(results)
        logger.info(
            "Evaluation complete: overall accuracy=%.4f, f1_macro=%.4f",
            results["overall"]["accuracy"],
            results["overall"]["f1_macro"],
        )
        return results

    def _compute_metrics(self, y_true: list, y_pred: list) -> dict:
        """Compute standard classification metrics.

        Args:
            y_true: Ground truth labels.
            y_pred: Predicted labels.

        Returns:
            Dictionary with accuracy, F1, precision, and recall scores.
        """
        labels = sorted(set(y_true) | set(y_pred))
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1_macro": float(
                f1_score(y_true, y_pred, average="macro", zero_division=0)
            ),
            "f1_weighted": float(
                f1_score(y_true, y_pred, average="weighted", zero_division=0)
            ),
            "precision_macro": float(
                precision_score(y_true, y_pred, average="macro", zero_division=0)
            ),
            "recall_macro": float(
                recall_score(y_true, y_pred, average="macro", zero_division=0)
            ),
            "classification_report": classification_report(
                y_true, y_pred, labels=labels, output_dict=True, zero_division=0
            ),
        }

    def _compute_confusion_matrix(self, y_true: list, y_pred: list) -> dict[str, list]:
        """Compute confusion matrix as a serializable dictionary.

        Args:
            y_true: Ground truth labels.
            y_pred: Predicted labels.

        Returns:
            Dictionary with 'labels' and 'matrix' keys.
        """
        labels = sorted(set(y_true) | set(y_pred))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        return {
            "labels": labels,
            "matrix": cm.tolist(),
        }

    def _analyze_cross_lingual(self, df: pd.DataFrame) -> dict:
        """Analyze cross-lingual transfer performance.

        Computes per-language accuracy relative to the overall accuracy
        to identify languages where the model transfers well or poorly.

        Args:
            df: DataFrame with true, pred, language, confidence columns.

        Returns:
            Dictionary with per-language transfer gaps.
        """
        if len(df) == 0:
            return {}

        overall_acc = float(accuracy_score(df["true"], df["pred"]))
        transfer_gaps: dict[str, float] = {}

        for lang in sorted(df["language"].unique()):
            lang_df = df[df["language"] == lang]
            lang_acc = float(accuracy_score(lang_df["true"], lang_df["pred"]))
            transfer_gaps[lang] = round(lang_acc - overall_acc, 4)

        return {
            "overall_accuracy": round(overall_acc, 4),
            "language_gaps": transfer_gaps,
        }

    def save_results(self, output_path: str = "data/evaluation_results.json") -> Path:
        """Save evaluation results history to JSON.

        Args:
            output_path: Path for the output JSON file.

        Returns:
            Path to the saved results file.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        serializable = []
        for result in self.results_history:
            entry = json.loads(json.dumps(result, default=_json_serializer))
            serializable.append(entry)

        with open(path, "w") as f:
            json.dump(serializable, f, indent=2)

        logger.info("Saved %d evaluation results to %s", len(serializable), path)
        return path


def _json_serializer(obj: object) -> object:
    """JSON serializer for numpy types.

    Args:
        obj: Object to serialize.

    Returns:
        JSON-serializable representation.

    Raises:
        TypeError: If the object type is not supported.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Type {type(obj)} not serializable")
