"""Tests for the zero-shot classification pipeline."""

from unittest.mock import MagicMock, patch

import pytest

from src.utils.config import reset_config_cache


@pytest.fixture(autouse=True)
def _clear_cache():
    reset_config_cache()
    yield
    reset_config_cache()


MOCK_PIPELINE_RESULT = {
    "sequence": "I was charged twice",
    "labels": [
        "billing",
        "complaint",
        "technical_support",
        "account",
        "general_inquiry",
        "feedback",
    ],
    "scores": [0.85, 0.45, 0.12, 0.08, 0.05, 0.03],
}


@pytest.fixture()
def mock_pipeline():
    """Mock the transformers pipeline to avoid model downloads."""
    with patch("src.models.zero_shot_classifier.pipeline") as mock_pl:
        mock_classifier = MagicMock()
        mock_classifier.return_value = MOCK_PIPELINE_RESULT
        mock_pl.return_value = mock_classifier
        yield mock_classifier


@pytest.fixture()
def classifier(mock_pipeline):
    """Create a ZeroShotClassifier with mocked pipeline."""
    from src.models.zero_shot_classifier import ZeroShotClassifier

    return ZeroShotClassifier(
        model_name="mock-model",
        device="cpu",
        candidate_labels=[
            "billing",
            "complaint",
            "technical_support",
            "account",
            "general_inquiry",
            "feedback",
        ],
    )


class TestClassifierInit:
    """Tests for ZeroShotClassifier initialization."""

    def test_init_with_defaults(self, mock_pipeline) -> None:
        """Should initialize with config defaults."""
        from src.models.zero_shot_classifier import ZeroShotClassifier

        clf = ZeroShotClassifier(model_name="mock-model", device="cpu")
        assert clf.model_name == "mock-model"
        assert clf.device == "cpu"

    def test_resolve_device_cpu(self, mock_pipeline) -> None:
        """CPU device should remain as-is."""
        from src.models.zero_shot_classifier import ZeroShotClassifier

        assert ZeroShotClassifier._resolve_device("cpu") == "cpu"

    def test_resolve_device_auto(self, mock_pipeline) -> None:
        """Auto device should resolve to cpu or cuda."""
        from src.models.zero_shot_classifier import ZeroShotClassifier

        result = ZeroShotClassifier._resolve_device("auto")
        assert result in ("cpu", "cuda")


class TestClassifySingle:
    """Tests for single text classification."""

    def test_classify_returns_expected_keys(self, classifier) -> None:
        """Classify result should have all expected keys."""
        result = classifier.classify("I was charged twice")
        expected_keys = {
            "primary_intent",
            "primary_confidence",
            "secondary_intent",
            "secondary_confidence",
            "all_scores",
            "needs_human_review",
        }
        assert set(result.keys()) == expected_keys

    def test_classify_primary_intent(self, classifier) -> None:
        """Primary intent should be the highest-scoring label."""
        result = classifier.classify("I was charged twice")
        assert result["primary_intent"] == "billing"

    def test_classify_primary_confidence(self, classifier) -> None:
        """Primary confidence should be a float between 0 and 1."""
        result = classifier.classify("I was charged twice")
        assert 0.0 <= result["primary_confidence"] <= 1.0

    def test_classify_secondary_intent(self, classifier) -> None:
        """Secondary intent should be the second-highest label."""
        result = classifier.classify("I was charged twice")
        assert result["secondary_intent"] == "complaint"

    def test_classify_all_scores(self, classifier) -> None:
        """all_scores should contain all candidate labels."""
        result = classifier.classify("I was charged twice")
        assert isinstance(result["all_scores"], dict)
        assert "billing" in result["all_scores"]

    def test_needs_human_review_false(self, classifier) -> None:
        """High-confidence prediction should not need review."""
        result = classifier.classify("I was charged twice")
        assert result["needs_human_review"] is False

    def test_needs_human_review_true(self, classifier, mock_pipeline) -> None:
        """Low-confidence prediction should need review."""
        mock_pipeline.return_value = {
            "labels": ["billing", "complaint"],
            "scores": [0.3, 0.2],
        }
        result = classifier.classify("ambiguous text")
        assert result["needs_human_review"] is True


class TestClassifyBatch:
    """Tests for batch classification."""

    def test_batch_single_item(self, classifier, mock_pipeline) -> None:
        """Single-item batch should return a list with one result."""
        mock_pipeline.return_value = MOCK_PIPELINE_RESULT
        results = classifier.classify_batch(["test text"])
        assert isinstance(results, list)
        assert len(results) == 1

    def test_batch_multiple_items(self, classifier, mock_pipeline) -> None:
        """Multi-item batch should return matching number of results."""
        mock_pipeline.return_value = [MOCK_PIPELINE_RESULT, MOCK_PIPELINE_RESULT]
        results = classifier.classify_batch(["text 1", "text 2"])
        assert len(results) == 2

    def test_batch_result_format(self, classifier, mock_pipeline) -> None:
        """Each batch result should have the same format as single classify."""
        mock_pipeline.return_value = [MOCK_PIPELINE_RESULT]
        results = classifier.classify_batch(["test text"])
        result = results[0]
        assert "primary_intent" in result
        assert "primary_confidence" in result
        assert "needs_human_review" in result


class TestFormatResult:
    """Tests for result formatting."""

    def test_format_result(self, classifier) -> None:
        """_format_result should produce standardized output."""
        raw = {
            "labels": ["billing", "complaint"],
            "scores": [0.9, 0.1],
        }
        result = classifier._format_result(raw)
        assert result["primary_intent"] == "billing"
        assert result["primary_confidence"] == 0.9
        assert result["secondary_intent"] == "complaint"

    def test_format_result_single_label(self, classifier) -> None:
        """Should handle single-label results."""
        raw = {"labels": ["billing"], "scores": [0.95]}
        result = classifier._format_result(raw)
        assert result["primary_intent"] == "billing"
        assert result["secondary_intent"] is None
        assert result["secondary_confidence"] is None
