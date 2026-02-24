"""Configuration management for the multilingual classifier.

Loads YAML-based configuration with environment variable overrides
and provides typed Pydantic models for all settings.
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """Model-related configuration."""

    zero_shot_model: str = "facebook/bart-large-mnli"
    device: str = "auto"
    max_length: int = 512


class ClassificationConfig(BaseModel):
    """Classification behavior configuration."""

    intent_categories: list[str] = Field(
        default_factory=lambda: [
            "billing",
            "technical_support",
            "account",
            "general_inquiry",
            "complaint",
            "feedback",
        ]
    )
    confidence_threshold: float = 0.7
    escalation_threshold: float = 0.8


class UrgencyConfig(BaseModel):
    """Urgency scoring configuration."""

    levels: list[str] = Field(
        default_factory=lambda: ["low", "medium", "high", "critical"]
    )
    keywords: dict[str, list[str]] = Field(default_factory=dict)
    escalation_threshold: float = 0.8


class LanguagesConfig(BaseModel):
    """Language support configuration."""

    supported: list[str] = Field(default_factory=lambda: ["en", "es", "fr", "de", "pt"])


class APIConfig(BaseModel):
    """API server configuration."""

    batch_size: int = 100
    host: str = "0.0.0.0"
    port: int = 8000


class DatabaseConfig(BaseModel):
    """Database configuration."""

    path: str = "data/metrics.db"


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = "INFO"
    format: str = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"


class Config(BaseModel):
    """Root configuration model aggregating all sub-configurations."""

    model: ModelConfig = Field(default_factory=ModelConfig)
    classification: ClassificationConfig = Field(default_factory=ClassificationConfig)
    urgency: UrgencyConfig = Field(default_factory=UrgencyConfig)
    languages: LanguagesConfig = Field(default_factory=LanguagesConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


def _resolve_config_path(config_path: str | None = None) -> Path:
    """Resolve the configuration file path.

    Args:
        config_path: Explicit path or None to use default/env override.

    Returns:
        Resolved Path object pointing to the config file.
    """
    if config_path:
        return Path(config_path)
    env_path = os.environ.get("CONFIG_PATH")
    if env_path:
        return Path(env_path)
    return Path("configs/config.yaml")


def _apply_env_overrides(data: dict[str, Any]) -> dict[str, Any]:
    """Apply environment variable overrides to configuration data.

    Supported overrides:
        MODEL_DEVICE -> model.device
        LOG_LEVEL -> logging.level
        API_HOST -> api.host
        API_PORT -> api.port
        DATABASE_PATH -> database.path

    Args:
        data: Raw configuration dictionary from YAML.

    Returns:
        Configuration dictionary with environment overrides applied.
    """
    env_map = {
        "MODEL_DEVICE": ("model", "device"),
        "LOG_LEVEL": ("logging", "level"),
        "API_HOST": ("api", "host"),
        "API_PORT": ("api", "port"),
        "DATABASE_PATH": ("database", "path"),
    }
    for env_var, (section, key) in env_map.items():
        value = os.environ.get(env_var)
        if value is not None:
            if section not in data:
                data[section] = {}
            data[section][key] = value
    return data


@lru_cache()
def load_config(config_path: str | None = None) -> Config:
    """Load and parse the application configuration.

    Reads the YAML config file, applies environment variable overrides,
    and returns a validated Config instance. Results are cached.

    Args:
        config_path: Optional path to the configuration file.

    Returns:
        Parsed and validated Config instance.
    """
    path = _resolve_config_path(config_path)
    if path.exists():
        with open(path) as f:
            data = yaml.safe_load(f) or {}
    else:
        data = {}
    data = _apply_env_overrides(data)
    return Config(**data)


def reset_config_cache() -> None:
    """Clear the cached configuration. Useful for testing."""
    load_config.cache_clear()
