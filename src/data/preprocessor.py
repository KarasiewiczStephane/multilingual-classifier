"""Text preprocessing pipeline for multilingual support tickets.

Handles Unicode normalization, whitespace collapsing, URL/email masking,
and length truncation to prepare raw text for classification models.
"""

import logging
import re
import unicodedata

from src.utils.config import load_config

logger = logging.getLogger(__name__)


class TextPreprocessor:
    """Cleans and normalizes text for downstream NLP tasks.

    Applies a pipeline of transformations: Unicode normalization,
    whitespace collapsing, URL/email masking, and length truncation.

    Args:
        max_length: Maximum character length for output text.
    """

    def __init__(self, max_length: int | None = None) -> None:
        config = load_config()
        self.max_length = max_length or config.model.max_length

    def normalize_unicode(self, text: str) -> str:
        """Normalize Unicode characters to NFKC form.

        Args:
            text: Raw input text.

        Returns:
            Unicode-normalized text.
        """
        return unicodedata.normalize("NFKC", text)

    def remove_excessive_whitespace(self, text: str) -> str:
        """Collapse multiple whitespace characters to a single space.

        Args:
            text: Input text with potential excessive whitespace.

        Returns:
            Text with normalized whitespace.
        """
        return re.sub(r"\s+", " ", text).strip()

    def mask_urls(self, text: str) -> str:
        """Replace URLs with a [URL] placeholder token.

        Args:
            text: Input text potentially containing URLs.

        Returns:
            Text with URLs replaced by [URL].
        """
        return re.sub(r"https?://\S+|www\.\S+", "[URL]", text)

    def mask_emails(self, text: str) -> str:
        """Replace email addresses with an [EMAIL] placeholder token.

        Args:
            text: Input text potentially containing email addresses.

        Returns:
            Text with emails replaced by [EMAIL].
        """
        return re.sub(r"\S+@\S+\.\S+", "[EMAIL]", text)

    def clean_text(self, text: str) -> str:
        """Apply the full preprocessing pipeline to a single text.

        Pipeline order: Unicode normalization -> URL masking ->
        email masking -> whitespace collapsing -> truncation.

        Args:
            text: Raw input text.

        Returns:
            Cleaned and truncated text.
        """
        if not text or not text.strip():
            return ""
        text = self.normalize_unicode(text)
        text = self.mask_urls(text)
        text = self.mask_emails(text)
        text = self.remove_excessive_whitespace(text)
        return text[: self.max_length]

    def preprocess_batch(self, texts: list[str]) -> list[str]:
        """Apply preprocessing to a batch of texts.

        Args:
            texts: List of raw text strings.

        Returns:
            List of cleaned text strings.
        """
        return [self.clean_text(t) for t in texts]
