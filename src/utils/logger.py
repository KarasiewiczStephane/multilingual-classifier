"""Structured logging setup for the multilingual classifier.

Provides a consistent logging configuration across all modules
with configurable format and level via config.yaml or environment.
"""

import logging
import sys


def setup_logger(
    name: str,
    level: str = "INFO",
    fmt: str = "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
) -> logging.Logger:
    """Create and configure a logger instance.

    Args:
        name: Logger name, typically ``__name__`` of the calling module.
        level: Logging level string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        fmt: Log message format string.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, level.upper(), logging.INFO))
    formatter = logging.Formatter(fmt)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.propagate = False
    return logger
