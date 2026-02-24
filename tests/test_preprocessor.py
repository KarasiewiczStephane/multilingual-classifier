"""Tests for the text preprocessing pipeline."""

import pytest

from src.data.preprocessor import TextPreprocessor
from src.utils.config import reset_config_cache


@pytest.fixture(autouse=True)
def _clear_cache():
    reset_config_cache()
    yield
    reset_config_cache()


@pytest.fixture()
def preprocessor() -> TextPreprocessor:
    """Create a TextPreprocessor with a small max_length for testing."""
    return TextPreprocessor(max_length=200)


class TestUnicodeNormalization:
    """Tests for Unicode normalization."""

    def test_nfkc_normalization(self, preprocessor: TextPreprocessor) -> None:
        """Full-width characters should be normalized."""
        text = "\uff28\uff45\uff4c\uff4c\uff4f"  # Ｈｅｌｌｏ
        result = preprocessor.normalize_unicode(text)
        assert result == "Hello"

    def test_combining_characters(self, preprocessor: TextPreprocessor) -> None:
        """Combining characters should be normalized."""
        text = "caf\u0065\u0301"  # café with combining accent
        result = preprocessor.normalize_unicode(text)
        assert "caf" in result


class TestWhitespaceHandling:
    """Tests for whitespace collapsing."""

    def test_multiple_spaces(self, preprocessor: TextPreprocessor) -> None:
        """Multiple spaces should collapse to one."""
        assert (
            preprocessor.remove_excessive_whitespace("hello   world") == "hello world"
        )

    def test_tabs_and_newlines(self, preprocessor: TextPreprocessor) -> None:
        """Tabs and newlines should collapse to spaces."""
        assert (
            preprocessor.remove_excessive_whitespace("hello\t\n  world")
            == "hello world"
        )

    def test_leading_trailing(self, preprocessor: TextPreprocessor) -> None:
        """Leading and trailing whitespace should be stripped."""
        assert preprocessor.remove_excessive_whitespace("  hello  ") == "hello"


class TestUrlMasking:
    """Tests for URL masking."""

    def test_http_url(self, preprocessor: TextPreprocessor) -> None:
        """HTTP URLs should be replaced with [URL]."""
        text = "Visit http://example.com for details"
        assert "[URL]" in preprocessor.mask_urls(text)
        assert "http" not in preprocessor.mask_urls(text)

    def test_https_url(self, preprocessor: TextPreprocessor) -> None:
        """HTTPS URLs should be replaced with [URL]."""
        text = "Go to https://secure.example.com/path"
        result = preprocessor.mask_urls(text)
        assert "[URL]" in result

    def test_www_url(self, preprocessor: TextPreprocessor) -> None:
        """www URLs should be replaced with [URL]."""
        text = "Check www.example.com"
        result = preprocessor.mask_urls(text)
        assert "[URL]" in result


class TestEmailMasking:
    """Tests for email masking."""

    def test_basic_email(self, preprocessor: TextPreprocessor) -> None:
        """Email addresses should be replaced with [EMAIL]."""
        text = "Contact user@example.com for help"
        result = preprocessor.mask_emails(text)
        assert "[EMAIL]" in result
        assert "user@" not in result


class TestCleanText:
    """Tests for the full preprocessing pipeline."""

    def test_full_pipeline(self, preprocessor: TextPreprocessor) -> None:
        """Full pipeline should apply all transformations."""
        text = "  Visit  https://example.com  or email user@test.com  "
        result = preprocessor.clean_text(text)
        assert "[URL]" in result
        assert "[EMAIL]" in result
        assert "  " not in result

    def test_empty_string(self, preprocessor: TextPreprocessor) -> None:
        """Empty string should return empty."""
        assert preprocessor.clean_text("") == ""

    def test_whitespace_only(self, preprocessor: TextPreprocessor) -> None:
        """Whitespace-only input should return empty."""
        assert preprocessor.clean_text("   ") == ""

    def test_truncation(self) -> None:
        """Text exceeding max_length should be truncated."""
        pp = TextPreprocessor(max_length=10)
        result = pp.clean_text("a" * 100)
        assert len(result) == 10

    def test_multilingual_text(self, preprocessor: TextPreprocessor) -> None:
        """Non-Latin scripts should pass through correctly."""
        text = "お問い合わせありがとうございます"
        result = preprocessor.clean_text(text)
        assert len(result) > 0
        assert result == text


class TestPreprocessBatch:
    """Tests for batch preprocessing."""

    def test_batch_processing(self, preprocessor: TextPreprocessor) -> None:
        """Batch should process all texts."""
        texts = ["hello world", "test  text", ""]
        results = preprocessor.preprocess_batch(texts)
        assert len(results) == 3
        assert results[0] == "hello world"
        assert results[1] == "test text"
        assert results[2] == ""
