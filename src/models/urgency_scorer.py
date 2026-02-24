"""Hybrid urgency scoring combining rule-based and ML-based approaches.

Uses keyword detection across multiple languages for rule-based scoring,
with an optional ML signal for enhanced accuracy.
"""

import logging
from dataclasses import dataclass, field
from typing import Literal

from src.utils.config import load_config

logger = logging.getLogger(__name__)

UrgencyLevel = Literal["low", "medium", "high", "critical"]


@dataclass
class UrgencyResult:
    """Result of urgency scoring for a support ticket.

    Attributes:
        level: Urgency level classification.
        score: Numeric urgency score from 0.0 to 1.0.
        rules_triggered: List of matched rule identifiers.
        should_escalate: Whether the ticket should be escalated.
        reason: Human-readable explanation of the urgency assessment.
    """

    level: UrgencyLevel
    score: float
    rules_triggered: list[str] = field(default_factory=list)
    should_escalate: bool = False
    reason: str = ""


class UrgencyScorer:
    """Scores ticket urgency using a hybrid rule-based and ML approach.

    Combines keyword matching across multiple languages with
    classification confidence signals to produce urgency assessments.

    Args:
        config: Optional urgency config dict override.
    """

    MULTILINGUAL_KEYWORDS: dict[str, dict[str, list[str]]] = {
        "es": {
            "critical": ["urgente", "emergencia", "caído", "roto"],
            "high": ["importante", "ayuda", "problema", "error"],
        },
        "fr": {
            "critical": ["urgent", "urgence", "panne", "cassé"],
            "high": ["important", "aide", "problème", "erreur"],
        },
        "de": {
            "critical": ["dringend", "notfall", "ausgefallen", "kaputt"],
            "high": ["wichtig", "hilfe", "problem", "fehler"],
        },
        "pt": {
            "critical": ["urgente", "emergência", "fora do ar", "quebrado"],
            "high": ["importante", "ajuda", "problema", "erro"],
        },
    }

    def __init__(self, config: dict | None = None) -> None:
        app_config = load_config()
        urgency_config = config or app_config.urgency.model_dump()

        self.keywords: dict[str, list[str]] = {
            "critical": urgency_config.get("keywords", {}).get(
                "critical",
                [
                    "urgent",
                    "emergency",
                    "down",
                    "broken",
                    "asap",
                    "immediately",
                    "critical",
                    "outage",
                    "not working",
                ],
            ),
            "high": urgency_config.get("keywords", {}).get(
                "high",
                [
                    "important",
                    "help",
                    "problem",
                    "issue",
                    "error",
                    "failed",
                    "cannot",
                    "unable",
                ],
            ),
            "medium": urgency_config.get("keywords", {}).get(
                "medium",
                ["question", "how to", "need", "want", "would like"],
            ),
        }
        self.escalation_threshold: float = urgency_config.get(
            "escalation_threshold", 0.8
        )

    def score_rules(self, text: str, language: str = "en") -> tuple[float, list[str]]:
        """Compute rule-based urgency score from keyword matching.

        Scans text for urgency keywords in both English and the
        detected language, returning a score and triggered rules.

        Args:
            text: Input ticket text.
            language: Detected language code.

        Returns:
            Tuple of (score, list_of_triggered_rules).
        """
        text_lower = text.lower()
        triggered: list[str] = []

        for level in ("critical", "high", "medium"):
            for kw in self.keywords.get(level, []):
                if kw in text_lower:
                    triggered.append(f"{level}:{kw}")

        if language in self.MULTILINGUAL_KEYWORDS:
            lang_kw = self.MULTILINGUAL_KEYWORDS[language]
            for level in ("critical", "high"):
                for kw in lang_kw.get(level, []):
                    if kw in text_lower:
                        triggered.append(f"{level}:{kw}:{language}")

        if any(t.startswith("critical:") for t in triggered):
            return 1.0, triggered
        if any(t.startswith("high:") for t in triggered):
            return 0.7, triggered
        if any(t.startswith("medium:") for t in triggered):
            return 0.4, triggered
        return 0.2, triggered

    def score_ml(self, text: str) -> float:
        """Compute ML-based urgency score.

        Placeholder for a fine-tuned urgency model. Currently returns
        a neutral default score. Can be extended with a trained model.

        Args:
            text: Input ticket text.

        Returns:
            ML urgency score between 0.0 and 1.0.
        """
        return 0.5

    def _determine_level(self, score: float) -> UrgencyLevel:
        """Map a numeric score to an urgency level.

        Args:
            score: Numeric urgency score (0.0 to 1.0).

        Returns:
            Corresponding urgency level string.
        """
        if score >= 0.9:
            return "critical"
        if score >= 0.65:
            return "high"
        if score >= 0.35:
            return "medium"
        return "low"

    def score(
        self,
        text: str,
        language: str = "en",
        classification_confidence: float = 0.0,
    ) -> UrgencyResult:
        """Compute combined urgency score from rules and ML signals.

        Blends rule-based keyword matching (70% weight) with ML scoring
        (20% weight) and classification confidence (10% weight) for a
        final urgency assessment.

        Args:
            text: Input ticket text.
            language: Detected language code.
            classification_confidence: Confidence from the intent classifier.

        Returns:
            UrgencyResult with level, score, triggered rules, and escalation flag.
        """
        rule_score, triggered = self.score_rules(text, language)
        ml_score = self.score_ml(text)

        combined = (
            0.7 * rule_score + 0.2 * ml_score + 0.1 * (1 - classification_confidence)
        )
        combined = min(max(combined, 0.0), 1.0)

        level = self._determine_level(combined)
        should_escalate = combined >= self.escalation_threshold

        reasons: list[str] = []
        if triggered:
            reasons.append(f"keywords matched: {len(triggered)}")
        if should_escalate:
            reasons.append("auto-escalated due to high urgency")

        return UrgencyResult(
            level=level,
            score=round(combined, 4),
            rules_triggered=triggered,
            should_escalate=should_escalate,
            reason="; ".join(reasons) if reasons else "standard priority",
        )
