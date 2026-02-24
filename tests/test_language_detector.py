"""Tests for the language detection module."""

from unittest.mock import MagicMock, patch


from src.models.language_detector import LanguageDetector


class TestLanguageDetectorInit:
    """Tests for LanguageDetector initialization."""

    def test_default_fallback(self) -> None:
        """Default fallback language should be English."""
        detector = LanguageDetector()
        assert detector.fallback == "en"

    def test_custom_fallback(self) -> None:
        """Custom fallback language should be set."""
        detector = LanguageDetector(fallback_language="es")
        assert detector.fallback == "es"


class TestLanguageDetection:
    """Tests for language detection functionality."""

    def test_detect_english(self) -> None:
        """Should detect English text correctly."""
        detector = LanguageDetector()
        result = detector.detect_language(
            "This is a test sentence about customer billing support"
        )
        assert result == "en"

    def test_detect_spanish(self) -> None:
        """Should detect Spanish text correctly."""
        detector = LanguageDetector()
        result = detector.detect_language(
            "Necesito ayuda con mi factura de este mes por favor"
        )
        assert result == "es"

    def test_detect_french(self) -> None:
        """Should detect French text correctly."""
        detector = LanguageDetector()
        result = detector.detect_language(
            "Je voudrais annuler mon abonnement immédiatement s'il vous plaît"
        )
        assert result == "fr"

    def test_detect_german(self) -> None:
        """Should detect German text correctly."""
        detector = LanguageDetector()
        result = detector.detect_language(
            "Ich brauche Hilfe mit meiner Rechnung diesen Monat bitte"
        )
        assert result == "de"

    def test_detect_with_confidence(self) -> None:
        """Should return tuple when return_confidence is True."""
        detector = LanguageDetector()
        result = detector.detect_language(
            "This is clearly an English sentence about technology",
            return_confidence=True,
        )
        assert isinstance(result, tuple)
        assert len(result) == 2
        lang, confidence = result
        assert isinstance(lang, str)
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0

    def test_empty_string_returns_fallback(self) -> None:
        """Empty string should return fallback language."""
        detector = LanguageDetector(fallback_language="en")
        assert detector.detect_language("") == "en"

    def test_whitespace_only_returns_fallback(self) -> None:
        """Whitespace-only input should return fallback."""
        detector = LanguageDetector(fallback_language="en")
        assert detector.detect_language("   ") == "en"

    def test_empty_with_confidence(self) -> None:
        """Empty input with confidence should return fallback with 0.0."""
        detector = LanguageDetector()
        result = detector.detect_language("", return_confidence=True)
        assert result == ("en", 0.0)


class TestBatchDetection:
    """Tests for batch language detection."""

    def test_detect_batch(self) -> None:
        """Batch detection should return correct number of results."""
        detector = LanguageDetector()
        texts = [
            "Hello, I need help with my account",
            "Necesito ayuda con mi cuenta por favor",
            "Bonjour, j'ai besoin d'aide avec mon compte",
        ]
        results = detector.detect_batch(texts)
        assert len(results) == 3
        assert all(isinstance(r, str) for r in results)

    def test_detect_batch_with_confidence(self) -> None:
        """Batch with confidence should return tuples."""
        detector = LanguageDetector()
        texts = ["Hello world", "Hola mundo"]
        results = detector.detect_batch_with_confidence(texts)
        assert len(results) == 2
        assert all(isinstance(r, tuple) for r in results)


class TestFallbackBehavior:
    """Tests for fallback when detection fails."""

    @patch("src.models.language_detector.detect_langs")
    def test_fallback_on_langdetect_failure(self, mock_detect: MagicMock) -> None:
        """Should use fallback when langdetect fails."""
        from langdetect import LangDetectException

        mock_detect.side_effect = LangDetectException(0, "error")
        detector = LanguageDetector(fallback_language="en")
        detector._ft_detect = None  # disable fasttext
        result = detector.detect_language("x")
        assert result == "en"
