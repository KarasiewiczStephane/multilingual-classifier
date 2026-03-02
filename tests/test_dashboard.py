"""Tests for the multilingual classifier dashboard data generators."""

import pandas as pd

from src.dashboard.app import (
    INTENT_CATEGORIES,
    LANGUAGE_NAMES,
    SAMPLE_TEXTS,
    SUPPORTED_LANGUAGES,
    generate_classification_result,
    generate_language_accuracy_data,
)


class TestClassificationResult:
    """Tests for the classification result generator."""

    def test_returns_dict(self) -> None:
        result = generate_classification_result("test input text")
        assert isinstance(result, dict)

    def test_has_required_keys(self) -> None:
        result = generate_classification_result("test input")
        required = {
            "primary_intent",
            "primary_confidence",
            "secondary_intent",
            "secondary_confidence",
            "all_scores",
            "detected_language",
            "language_confidence",
            "language_name",
            "urgency_score",
            "urgency_level",
            "needs_human_review",
            "processing_time_ms",
        }
        assert required.issubset(result.keys())

    def test_confidence_range(self) -> None:
        result = generate_classification_result("billing problem help")
        assert 0.0 <= result["primary_confidence"] <= 1.0
        assert 0.0 <= result["secondary_confidence"] <= 1.0

    def test_urgency_score_range(self) -> None:
        result = generate_classification_result("urgent emergency broken")
        assert 0.0 <= result["urgency_score"] <= 1.0

    def test_urgency_level_valid(self) -> None:
        result = generate_classification_result("please help")
        assert result["urgency_level"] in {"low", "medium", "high", "critical"}

    def test_all_scores_sum_roughly_one(self) -> None:
        result = generate_classification_result("test message")
        total = sum(result["all_scores"].values())
        assert 0.95 <= total <= 1.05

    def test_deterministic_output(self) -> None:
        result1 = generate_classification_result("same text")
        result2 = generate_classification_result("same text")
        assert result1["primary_intent"] == result2["primary_intent"]
        assert result1["primary_confidence"] == result2["primary_confidence"]

    def test_detected_language_valid(self) -> None:
        result = generate_classification_result("some english text")
        assert result["detected_language"] in SUPPORTED_LANGUAGES

    def test_language_name_populated(self) -> None:
        result = generate_classification_result("test")
        assert result["language_name"] in LANGUAGE_NAMES.values()

    def test_processing_time_positive(self) -> None:
        result = generate_classification_result("test")
        assert result["processing_time_ms"] > 0


class TestLanguageAccuracyData:
    """Tests for the language accuracy heatmap data generator."""

    def test_returns_dataframe(self) -> None:
        df = generate_language_accuracy_data()
        assert isinstance(df, pd.DataFrame)

    def test_has_required_columns(self) -> None:
        df = generate_language_accuracy_data()
        assert set(df.columns) == {"language", "intent", "accuracy"}

    def test_covers_all_languages(self) -> None:
        df = generate_language_accuracy_data()
        expected_langs = set(LANGUAGE_NAMES.values())
        assert set(df["language"].unique()) == expected_langs

    def test_covers_all_intents(self) -> None:
        df = generate_language_accuracy_data()
        assert set(df["intent"].unique()) == set(INTENT_CATEGORIES)

    def test_accuracy_range(self) -> None:
        df = generate_language_accuracy_data()
        assert df["accuracy"].min() >= 0.0
        assert df["accuracy"].max() <= 1.0

    def test_correct_row_count(self) -> None:
        df = generate_language_accuracy_data()
        expected = len(SUPPORTED_LANGUAGES) * len(INTENT_CATEGORIES)
        assert len(df) == expected


class TestConstants:
    """Tests for dashboard constants."""

    def test_sample_texts_cover_languages(self) -> None:
        assert set(SAMPLE_TEXTS.keys()) == set(SUPPORTED_LANGUAGES)

    def test_language_names_cover_languages(self) -> None:
        assert set(LANGUAGE_NAMES.keys()) == set(SUPPORTED_LANGUAGES)

    def test_intent_categories_not_empty(self) -> None:
        assert len(INTENT_CATEGORIES) > 0
