"""Tests for the urgency scoring module."""

import pytest

from src.models.urgency_scorer import UrgencyResult, UrgencyScorer
from src.utils.config import reset_config_cache


@pytest.fixture(autouse=True)
def _clear_cache():
    reset_config_cache()
    yield
    reset_config_cache()


@pytest.fixture()
def scorer() -> UrgencyScorer:
    """Create an UrgencyScorer with default config."""
    return UrgencyScorer()


class TestUrgencyResult:
    """Tests for UrgencyResult dataclass."""

    def test_default_values(self) -> None:
        """UrgencyResult should have sensible defaults."""
        result = UrgencyResult(level="low", score=0.2)
        assert result.level == "low"
        assert result.score == 0.2
        assert result.rules_triggered == []
        assert result.should_escalate is False
        assert result.reason == ""


class TestRuleBasedScoring:
    """Tests for keyword-based urgency scoring."""

    def test_critical_keyword_english(self, scorer: UrgencyScorer) -> None:
        """Critical keywords should produce score of 1.0."""
        score, triggered = scorer.score_rules("URGENT: system is down")
        assert score == 1.0
        assert any("critical:" in t for t in triggered)

    def test_high_keyword_english(self, scorer: UrgencyScorer) -> None:
        """High-priority keywords should produce score of 0.7."""
        score, triggered = scorer.score_rules("This is very important, please help")
        assert score == 0.7
        assert any("high:" in t for t in triggered)

    def test_medium_keyword_english(self, scorer: UrgencyScorer) -> None:
        """Medium-priority keywords should produce score of 0.4."""
        score, triggered = scorer.score_rules("I have a question about pricing")
        assert score == 0.4
        assert any("medium:" in t for t in triggered)

    def test_no_keywords(self, scorer: UrgencyScorer) -> None:
        """Text without urgency keywords should get low score."""
        score, triggered = scorer.score_rules("Thank you for the update")
        assert score == 0.2
        assert len(triggered) == 0

    def test_spanish_critical_keyword(self, scorer: UrgencyScorer) -> None:
        """Spanish critical keywords should be detected."""
        score, triggered = scorer.score_rules(
            "Es una emergencia, necesito ayuda", language="es"
        )
        assert score == 1.0
        assert any("es" in t for t in triggered)

    def test_french_critical_keyword(self, scorer: UrgencyScorer) -> None:
        """French critical keywords should be detected."""
        score, triggered = scorer.score_rules(
            "C'est très urgent, mon système est en panne", language="fr"
        )
        assert score == 1.0

    def test_german_critical_keyword(self, scorer: UrgencyScorer) -> None:
        """German critical keywords should be detected."""
        score, triggered = scorer.score_rules(
            "Dringend: mein System ist ausgefallen", language="de"
        )
        assert score == 1.0

    def test_portuguese_critical_keyword(self, scorer: UrgencyScorer) -> None:
        """Portuguese critical keywords should be detected."""
        score, triggered = scorer.score_rules(
            "Urgente: sistema quebrado", language="pt"
        )
        assert score == 1.0

    def test_case_insensitive(self, scorer: UrgencyScorer) -> None:
        """Keyword matching should be case-insensitive."""
        score, _ = scorer.score_rules("URGENT EMERGENCY BROKEN")
        assert score == 1.0


class TestMLScoring:
    """Tests for ML-based urgency scoring."""

    def test_ml_score_default(self, scorer: UrgencyScorer) -> None:
        """ML score should return neutral default."""
        assert scorer.score_ml("any text") == 0.5


class TestDetermineLevel:
    """Tests for score-to-level mapping."""

    def test_critical_level(self, scorer: UrgencyScorer) -> None:
        """Score >= 0.9 should be critical."""
        assert scorer._determine_level(0.95) == "critical"
        assert scorer._determine_level(0.9) == "critical"

    def test_high_level(self, scorer: UrgencyScorer) -> None:
        """Score 0.65-0.89 should be high."""
        assert scorer._determine_level(0.8) == "high"
        assert scorer._determine_level(0.65) == "high"

    def test_medium_level(self, scorer: UrgencyScorer) -> None:
        """Score 0.35-0.64 should be medium."""
        assert scorer._determine_level(0.5) == "medium"
        assert scorer._determine_level(0.35) == "medium"

    def test_low_level(self, scorer: UrgencyScorer) -> None:
        """Score < 0.35 should be low."""
        assert scorer._determine_level(0.2) == "low"
        assert scorer._determine_level(0.0) == "low"


class TestCombinedScoring:
    """Tests for the combined scoring method."""

    def test_critical_combined(self, scorer: UrgencyScorer) -> None:
        """Critical keyword should produce high combined score."""
        result = scorer.score("URGENT: everything is broken and down")
        assert isinstance(result, UrgencyResult)
        assert result.level in ("critical", "high")
        assert result.score >= 0.7

    def test_escalation_flag(self, scorer: UrgencyScorer) -> None:
        """Critical text should trigger escalation."""
        result = scorer.score("URGENT emergency: system down immediately")
        assert result.should_escalate is True

    def test_low_urgency_no_escalation(self, scorer: UrgencyScorer) -> None:
        """Low-urgency text should not trigger escalation."""
        result = scorer.score("Thank you for the update on the feature")
        assert result.should_escalate is False

    def test_reason_populated(self, scorer: UrgencyScorer) -> None:
        """Result reason should be populated for urgent text."""
        result = scorer.score("URGENT: system is broken")
        assert result.reason != ""

    def test_reason_standard_priority(self, scorer: UrgencyScorer) -> None:
        """Non-urgent text should have standard priority reason."""
        result = scorer.score("Thanks for the update")
        assert "standard priority" in result.reason

    def test_classification_confidence_impact(self, scorer: UrgencyScorer) -> None:
        """Low classification confidence should slightly increase urgency."""
        result_high_conf = scorer.score("generic text", classification_confidence=0.95)
        result_low_conf = scorer.score("generic text", classification_confidence=0.1)
        assert result_low_conf.score >= result_high_conf.score

    def test_combined_score_bounded(self, scorer: UrgencyScorer) -> None:
        """Combined score should always be between 0 and 1."""
        result = scorer.score("URGENT EMERGENCY CRITICAL BROKEN")
        assert 0.0 <= result.score <= 1.0

    def test_multilingual_combined(self, scorer: UrgencyScorer) -> None:
        """Multilingual keywords should contribute to combined score."""
        result = scorer.score("Es una emergencia urgente", language="es")
        assert result.score >= 0.7
