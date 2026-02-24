"""Tests for configuration loading and environment overrides."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from src.utils.config import (
    Config,
    load_config,
    reset_config_cache,
)


@pytest.fixture(autouse=True)
def _clear_config_cache():
    """Ensure config cache is cleared between tests."""
    reset_config_cache()
    yield
    reset_config_cache()


@pytest.fixture()
def config_file(tmp_path: Path) -> Path:
    """Create a temporary config file for testing."""
    config_data = {
        "model": {
            "zero_shot_model": "test-model",
            "device": "cpu",
            "max_length": 256,
        },
        "classification": {
            "intent_categories": ["billing", "technical_support"],
            "confidence_threshold": 0.6,
            "escalation_threshold": 0.9,
        },
        "urgency": {
            "levels": ["low", "high"],
            "keywords": {"critical": ["urgent"]},
            "escalation_threshold": 0.85,
        },
        "languages": {"supported": ["en", "es"]},
        "api": {"batch_size": 50, "host": "127.0.0.1", "port": 9000},
        "database": {"path": "test.db"},
        "logging": {"level": "DEBUG", "format": "%(message)s"},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(config_data))
    return config_path


def test_load_config_from_file(config_file: Path) -> None:
    """Config values should match the YAML file contents."""
    cfg = load_config(str(config_file))
    assert cfg.model.zero_shot_model == "test-model"
    assert cfg.model.device == "cpu"
    assert cfg.model.max_length == 256
    assert cfg.classification.confidence_threshold == 0.6
    assert cfg.api.port == 9000
    assert cfg.database.path == "test.db"
    assert cfg.logging.level == "DEBUG"


def test_load_config_defaults_when_file_missing(tmp_path: Path) -> None:
    """Missing config file should produce valid defaults."""
    cfg = load_config(str(tmp_path / "nonexistent.yaml"))
    assert isinstance(cfg, Config)
    assert cfg.model.zero_shot_model == "facebook/bart-large-mnli"
    assert cfg.model.device == "auto"
    assert cfg.classification.confidence_threshold == 0.7


def test_env_override_model_device(config_file: Path) -> None:
    """MODEL_DEVICE env var should override config file."""
    with patch.dict(os.environ, {"MODEL_DEVICE": "cuda"}):
        cfg = load_config(str(config_file))
        assert cfg.model.device == "cuda"


def test_env_override_log_level(config_file: Path) -> None:
    """LOG_LEVEL env var should override config file."""
    with patch.dict(os.environ, {"LOG_LEVEL": "WARNING"}):
        cfg = load_config(str(config_file))
        assert cfg.logging.level == "WARNING"


def test_env_override_api_port(config_file: Path) -> None:
    """API_PORT env var should override config file."""
    with patch.dict(os.environ, {"API_PORT": "3000"}):
        cfg = load_config(str(config_file))
        assert cfg.api.port == 3000


def test_config_path_from_env(config_file: Path) -> None:
    """CONFIG_PATH env var should be used when no explicit path given."""
    with patch.dict(os.environ, {"CONFIG_PATH": str(config_file)}):
        cfg = load_config()
        assert cfg.model.zero_shot_model == "test-model"


def test_config_caching(config_file: Path) -> None:
    """Repeated loads with the same path should return the cached object."""
    cfg1 = load_config(str(config_file))
    cfg2 = load_config(str(config_file))
    assert cfg1 is cfg2


def test_reset_config_cache(config_file: Path) -> None:
    """reset_config_cache should allow fresh loads."""
    cfg1 = load_config(str(config_file))
    reset_config_cache()
    cfg2 = load_config(str(config_file))
    assert cfg1 is not cfg2
    assert cfg1.model.zero_shot_model == cfg2.model.zero_shot_model


def test_classification_config_categories(config_file: Path) -> None:
    """Intent categories should load correctly from file."""
    cfg = load_config(str(config_file))
    assert "billing" in cfg.classification.intent_categories
    assert len(cfg.classification.intent_categories) == 2


def test_urgency_keywords(config_file: Path) -> None:
    """Urgency keywords should be accessible from config."""
    cfg = load_config(str(config_file))
    assert "critical" in cfg.urgency.keywords
    assert "urgent" in cfg.urgency.keywords["critical"]


def test_languages_supported(config_file: Path) -> None:
    """Supported languages should load from config."""
    cfg = load_config(str(config_file))
    assert "en" in cfg.languages.supported
    assert "es" in cfg.languages.supported
