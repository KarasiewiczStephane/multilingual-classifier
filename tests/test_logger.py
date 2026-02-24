"""Tests for the structured logging setup."""

import logging

from src.utils.logger import setup_logger


def test_setup_logger_returns_logger() -> None:
    """setup_logger should return a logging.Logger instance."""
    logger = setup_logger("test_basic")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "test_basic"


def test_setup_logger_default_level() -> None:
    """Default logging level should be INFO."""
    logger = setup_logger("test_default_level")
    assert logger.level == logging.INFO


def test_setup_logger_custom_level() -> None:
    """Custom level should be applied correctly."""
    logger = setup_logger("test_custom_level", level="DEBUG")
    assert logger.level == logging.DEBUG


def test_setup_logger_has_handler() -> None:
    """Logger should have at least one stream handler."""
    logger = setup_logger("test_handler")
    assert len(logger.handlers) >= 1
    assert isinstance(logger.handlers[0], logging.StreamHandler)


def test_setup_logger_idempotent() -> None:
    """Calling setup_logger twice should not duplicate handlers."""
    logger1 = setup_logger("test_idempotent")
    handler_count = len(logger1.handlers)
    logger2 = setup_logger("test_idempotent")
    assert logger1 is logger2
    assert len(logger2.handlers) == handler_count


def test_setup_logger_no_propagation() -> None:
    """Logger propagation should be disabled to avoid duplicate logs."""
    logger = setup_logger("test_no_propagate")
    assert logger.propagate is False


def test_setup_logger_custom_format() -> None:
    """Custom format string should be applied to the handler."""
    custom_fmt = "%(levelname)s - %(message)s"
    logger = setup_logger("test_custom_fmt", fmt=custom_fmt)
    handler = logger.handlers[0]
    assert handler.formatter._fmt == custom_fmt
