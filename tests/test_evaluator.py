"""Tests for the classification evaluation suite."""

import json
from pathlib import Path

import pytest

from src.models.evaluator import ClassificationEvaluator
from src.utils.config import reset_config_cache


@pytest.fixture(autouse=True)
def _clear_cache():
    reset_config_cache()
    yield
    reset_config_cache()


@pytest.fixture()
def evaluator() -> ClassificationEvaluator:
    """Create an evaluator with default categories."""
    return ClassificationEvaluator()


@pytest.fixture()
def sample_predictions() -> dict:
    """Create sample prediction data for testing."""
    return {
        "y_true": [
            "billing",
            "technical_support",
            "account",
            "billing",
            "complaint",
            "feedback",
            "billing",
            "technical_support",
            "account",
            "billing",
            "complaint",
            "feedback",
        ],
        "y_pred": [
            "billing",
            "technical_support",
            "account",
            "billing",
            "complaint",
            "feedback",
            "complaint",
            "technical_support",
            "billing",
            "billing",
            "complaint",
            "general_inquiry",
        ],
        "languages": [
            "en",
            "en",
            "en",
            "en",
            "es",
            "es",
            "es",
            "es",
            "fr",
            "fr",
            "fr",
            "fr",
        ],
        "confidences": [
            0.95,
            0.88,
            0.75,
            0.91,
            0.82,
            0.65,
            0.55,
            0.79,
            0.60,
            0.85,
            0.90,
            0.40,
        ],
    }


class TestComputeMetrics:
    """Tests for metric computation."""

    def test_perfect_predictions(self, evaluator: ClassificationEvaluator) -> None:
        """Perfect predictions should yield accuracy and F1 of 1.0."""
        labels = ["billing", "account", "billing"]
        metrics = evaluator._compute_metrics(labels, labels)
        assert metrics["accuracy"] == 1.0
        assert metrics["f1_macro"] == 1.0

    def test_imperfect_predictions(self, evaluator: ClassificationEvaluator) -> None:
        """Imperfect predictions should yield intermediate scores."""
        y_true = ["billing", "account", "billing", "account"]
        y_pred = ["billing", "billing", "billing", "account"]
        metrics = evaluator._compute_metrics(y_true, y_pred)
        assert 0.0 < metrics["accuracy"] < 1.0

    def test_all_metric_keys_present(self, evaluator: ClassificationEvaluator) -> None:
        """All expected metric keys should be present."""
        y_true = ["billing", "account"]
        y_pred = ["billing", "account"]
        metrics = evaluator._compute_metrics(y_true, y_pred)
        expected_keys = {
            "accuracy",
            "f1_macro",
            "f1_weighted",
            "precision_macro",
            "recall_macro",
            "classification_report",
        }
        assert set(metrics.keys()) == expected_keys


class TestConfusionMatrix:
    """Tests for confusion matrix computation."""

    def test_confusion_matrix_shape(self, evaluator: ClassificationEvaluator) -> None:
        """Confusion matrix should be square with correct dimensions."""
        y_true = ["billing", "account", "billing"]
        y_pred = ["billing", "account", "account"]
        cm = evaluator._compute_confusion_matrix(y_true, y_pred)
        assert "labels" in cm
        assert "matrix" in cm
        n_labels = len(cm["labels"])
        assert len(cm["matrix"]) == n_labels
        assert all(len(row) == n_labels for row in cm["matrix"])

    def test_perfect_confusion_matrix(self, evaluator: ClassificationEvaluator) -> None:
        """Perfect predictions should have values only on diagonal."""
        labels = ["billing", "account", "billing", "account"]
        cm = evaluator._compute_confusion_matrix(labels, labels)
        matrix = cm["matrix"]
        for i, row in enumerate(matrix):
            for j, val in enumerate(row):
                if i == j:
                    assert val > 0
                else:
                    assert val == 0


class TestEvaluate:
    """Tests for the full evaluation method."""

    def test_evaluate_returns_all_sections(
        self, evaluator: ClassificationEvaluator, sample_predictions: dict
    ) -> None:
        """Evaluate should return all expected sections."""
        results = evaluator.evaluate(**sample_predictions)
        expected_sections = {
            "overall",
            "per_language",
            "confusion_matrix",
            "cross_lingual_transfer",
            "low_confidence_analysis",
        }
        assert set(results.keys()) == expected_sections

    def test_per_language_breakdown(
        self, evaluator: ClassificationEvaluator, sample_predictions: dict
    ) -> None:
        """Should have metrics for each language in the data."""
        results = evaluator.evaluate(**sample_predictions)
        assert "en" in results["per_language"]
        assert "es" in results["per_language"]
        assert "fr" in results["per_language"]

    def test_per_language_sample_count(
        self, evaluator: ClassificationEvaluator, sample_predictions: dict
    ) -> None:
        """Each language should report correct sample count."""
        results = evaluator.evaluate(**sample_predictions)
        assert results["per_language"]["en"]["sample_count"] == 4
        assert results["per_language"]["es"]["sample_count"] == 4

    def test_low_confidence_analysis(
        self, evaluator: ClassificationEvaluator, sample_predictions: dict
    ) -> None:
        """Low confidence analysis should count low-confidence predictions."""
        results = evaluator.evaluate(**sample_predictions)
        low_conf = results["low_confidence_analysis"]
        assert "count" in low_conf
        assert low_conf["count"] > 0

    def test_cross_lingual_transfer(
        self, evaluator: ClassificationEvaluator, sample_predictions: dict
    ) -> None:
        """Cross-lingual transfer should contain language gaps."""
        results = evaluator.evaluate(**sample_predictions)
        transfer = results["cross_lingual_transfer"]
        assert "overall_accuracy" in transfer
        assert "language_gaps" in transfer

    def test_results_history(
        self, evaluator: ClassificationEvaluator, sample_predictions: dict
    ) -> None:
        """Results should be appended to history."""
        evaluator.evaluate(**sample_predictions)
        evaluator.evaluate(**sample_predictions)
        assert len(evaluator.results_history) == 2

    def test_evaluate_without_confidences(
        self, evaluator: ClassificationEvaluator, sample_predictions: dict
    ) -> None:
        """Should work without confidence scores."""
        del sample_predictions["confidences"]
        results = evaluator.evaluate(**sample_predictions)
        assert results["overall"]["accuracy"] > 0


class TestSaveResults:
    """Tests for saving evaluation results."""

    def test_save_results_creates_file(
        self,
        evaluator: ClassificationEvaluator,
        sample_predictions: dict,
        tmp_path: Path,
    ) -> None:
        """Should save results to a JSON file."""
        evaluator.evaluate(**sample_predictions)
        output = tmp_path / "results.json"
        path = evaluator.save_results(str(output))
        assert path.exists()

        with open(path) as f:
            data = json.load(f)
        assert len(data) == 1
        assert "overall" in data[0]

    def test_save_multiple_evaluations(
        self,
        evaluator: ClassificationEvaluator,
        sample_predictions: dict,
        tmp_path: Path,
    ) -> None:
        """Should save all evaluation results."""
        evaluator.evaluate(**sample_predictions)
        evaluator.evaluate(**sample_predictions)
        output = tmp_path / "results.json"
        evaluator.save_results(str(output))

        with open(output) as f:
            data = json.load(f)
        assert len(data) == 2
