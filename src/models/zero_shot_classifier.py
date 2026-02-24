"""Zero-shot text classification using transformer models.

Wraps the HuggingFace zero-shot-classification pipeline with
configurable intent categories, confidence scoring, and batch support.
"""

import logging

import torch
from transformers import pipeline

from src.utils.config import load_config

logger = logging.getLogger(__name__)


class ZeroShotClassifier:
    """Zero-shot text classifier using transformer-based NLI models.

    Uses HuggingFace's zero-shot-classification pipeline to classify
    text into configurable intent categories without task-specific
    fine-tuning.

    Args:
        model_name: HuggingFace model identifier for zero-shot classification.
        device: Device to run inference on ('auto', 'cpu', 'cuda').
        candidate_labels: List of intent category labels.
        confidence_threshold: Minimum confidence for a prediction to be trusted.
    """

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        candidate_labels: list[str] | None = None,
        confidence_threshold: float | None = None,
    ) -> None:
        config = load_config()
        self.model_name = model_name or config.model.zero_shot_model
        self.device = self._resolve_device(device or config.model.device)
        self.candidate_labels = (
            candidate_labels or config.classification.intent_categories
        )
        self.confidence_threshold = (
            confidence_threshold or config.classification.confidence_threshold
        )
        self.classifier = self._load_model()

    @staticmethod
    def _resolve_device(device: str) -> str:
        """Resolve the 'auto' device setting to an actual device.

        Args:
            device: Device string ('auto', 'cpu', 'cuda').

        Returns:
            Resolved device string.
        """
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def _load_model(self):
        """Load the zero-shot classification pipeline.

        Returns:
            HuggingFace pipeline instance.
        """
        logger.info("Loading model %s on %s", self.model_name, self.device)
        return pipeline(
            "zero-shot-classification",
            model=self.model_name,
            device=0 if self.device == "cuda" else -1,
        )

    def classify(
        self,
        text: str,
        multi_label: bool = True,
        top_k: int = 2,
    ) -> dict:
        """Classify a single text into intent categories.

        Args:
            text: Input text to classify.
            multi_label: Whether to allow multiple labels.
            top_k: Number of top predictions to include.

        Returns:
            Dictionary with keys: primary_intent, primary_confidence,
            secondary_intent, secondary_confidence, all_scores,
            needs_human_review.
        """
        result = self.classifier(
            text,
            self.candidate_labels,
            multi_label=multi_label,
        )

        scores = dict(zip(result["labels"], result["scores"]))
        primary_intent = result["labels"][0]
        primary_conf = result["scores"][0]

        secondary_intent = result["labels"][1] if len(result["labels"]) > 1 else None
        secondary_conf = result["scores"][1] if len(result["scores"]) > 1 else None

        return {
            "primary_intent": primary_intent,
            "primary_confidence": round(primary_conf, 4),
            "secondary_intent": secondary_intent,
            "secondary_confidence": (
                round(secondary_conf, 4) if secondary_conf else None
            ),
            "all_scores": {k: round(v, 4) for k, v in scores.items()},
            "needs_human_review": primary_conf < self.confidence_threshold,
        }

    def classify_batch(
        self,
        texts: list[str],
        multi_label: bool = True,
        batch_size: int = 8,
    ) -> list[dict]:
        """Classify a batch of texts efficiently.

        Args:
            texts: List of input texts.
            multi_label: Whether to allow multiple labels per text.
            batch_size: Number of texts per inference batch.

        Returns:
            List of classification result dictionaries.
        """
        results = self.classifier(
            texts,
            self.candidate_labels,
            multi_label=multi_label,
            batch_size=batch_size,
        )

        if isinstance(results, dict):
            results = [results]

        return [self._format_result(r) for r in results]

    def _format_result(self, result: dict) -> dict:
        """Format a raw pipeline result into a standardized dictionary.

        Args:
            result: Raw result from the HuggingFace pipeline.

        Returns:
            Standardized classification result dictionary.
        """
        scores = dict(zip(result["labels"], result["scores"]))
        primary_intent = result["labels"][0]
        primary_conf = result["scores"][0]

        secondary_intent = result["labels"][1] if len(result["labels"]) > 1 else None
        secondary_conf = result["scores"][1] if len(result["scores"]) > 1 else None

        return {
            "primary_intent": primary_intent,
            "primary_confidence": round(primary_conf, 4),
            "secondary_intent": secondary_intent,
            "secondary_confidence": (
                round(secondary_conf, 4) if secondary_conf else None
            ),
            "all_scores": {k: round(v, 4) for k, v in scores.items()},
            "needs_human_review": primary_conf < self.confidence_threshold,
        }
