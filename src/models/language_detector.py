"""Language detection using an ensemble of langdetect and fasttext-langdetect.

Combines multiple detection backends for robust language identification,
with configurable fallback behavior and confidence thresholds.
"""

import logging

from langdetect import LangDetectException, detect_langs

logger = logging.getLogger(__name__)


class LanguageDetector:
    """Detects text language using an ensemble of detection backends.

    Uses langdetect as the primary detector with fasttext-langdetect
    as a secondary signal. Returns the highest-confidence result
    from available backends.

    Args:
        fallback_language: Language code returned when detection fails.
    """

    def __init__(self, fallback_language: str = "en") -> None:
        self.fallback = fallback_language
        self._ft_detect = None
        self._load_fasttext()

    def _load_fasttext(self) -> None:
        """Attempt to load the fasttext-langdetect backend."""
        try:
            from ftlangdetect import detect as ft_detect

            self._ft_detect = ft_detect
            logger.info("fasttext-langdetect backend loaded")
        except ImportError:
            logger.warning("fasttext-langdetect not available, using langdetect only")

    def detect_language(
        self,
        text: str,
        return_confidence: bool = False,
    ) -> tuple[str, float] | str:
        """Detect the language of a text using ensemble voting.

        Runs all available backends and returns the result with
        the highest confidence score.

        Args:
            text: Input text for language detection.
            return_confidence: If True, return (language, confidence) tuple.

        Returns:
            Language code string, or (language, confidence) tuple
            if return_confidence is True.
        """
        if not text or not text.strip():
            return (self.fallback, 0.0) if return_confidence else self.fallback

        results: list[tuple[str, float]] = []

        # Backend 1: langdetect
        try:
            lang_results = detect_langs(text)
            if lang_results:
                top = lang_results[0]
                results.append((top.lang, top.prob))
        except LangDetectException:
            logger.debug("langdetect failed for input text")

        # Backend 2: fasttext-langdetect
        if self._ft_detect is not None:
            try:
                clean_text = text.replace("\n", " ")
                ft_result = self._ft_detect(clean_text)
                results.append((ft_result["lang"], ft_result["score"]))
            except Exception:
                logger.debug("fasttext-langdetect failed for input text")

        if not results:
            logger.warning(
                "All detection backends failed, using fallback=%s", self.fallback
            )
            return (self.fallback, 0.0) if return_confidence else self.fallback

        best = max(results, key=lambda x: x[1])
        return best if return_confidence else best[0]

    def detect_batch(self, texts: list[str]) -> list[str]:
        """Detect languages for a batch of texts.

        Args:
            texts: List of input text strings.

        Returns:
            List of detected language codes.
        """
        return [self.detect_language(t) for t in texts]

    def detect_batch_with_confidence(self, texts: list[str]) -> list[tuple[str, float]]:
        """Detect languages for a batch with confidence scores.

        Args:
            texts: List of input text strings.

        Returns:
            List of (language, confidence) tuples.
        """
        return [self.detect_language(t, return_confidence=True) for t in texts]
