"""Dataset downloading and preparation for multilingual classification.

Downloads publicly available customer support datasets from HuggingFace
and prepares them in a unified format for training and evaluation.
"""

import logging
from pathlib import Path

import pandas as pd
from datasets import load_dataset

from src.utils.config import load_config

logger = logging.getLogger(__name__)


class DatasetDownloader:
    """Handles downloading and preparing multilingual support ticket datasets.

    Args:
        data_dir: Directory for storing raw downloaded data.
    """

    def __init__(self, data_dir: str | None = None) -> None:
        config = load_config()
        self.data_dir = Path(data_dir or "data/raw")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.supported_languages = config.languages.supported

    def download_multilingual_sentiments(
        self,
        languages: list[str] | None = None,
        samples_per_lang: int = 500,
    ) -> pd.DataFrame:
        """Download multilingual sentiment dataset as a proxy for support tickets.

        Args:
            languages: Language codes to download. Defaults to supported list.
            samples_per_lang: Number of samples per language.

        Returns:
            DataFrame with columns: text, language, source.
        """
        languages = languages or self.supported_languages[:5]
        frames: list[pd.DataFrame] = []

        for lang in languages:
            try:
                ds = load_dataset(
                    "tyqiangz/multilingual-sentiments",
                    lang,
                    split=f"train[:{samples_per_lang}]",
                    trust_remote_code=True,
                )
                df = pd.DataFrame(ds)
                df["language"] = lang
                df["source"] = "multilingual_sentiments"
                frames.append(df[["text", "language", "source"]])
                logger.info("Downloaded %d samples for language=%s", len(df), lang)
            except Exception:
                logger.warning("Failed to download data for language=%s", lang)

        if not frames:
            logger.warning("No data downloaded for any language")
            return pd.DataFrame(columns=["text", "language", "source"])

        combined = pd.concat(frames, ignore_index=True)
        logger.info("Total downloaded samples: %d", len(combined))
        return combined

    def prepare_combined_dataset(
        self,
        target_languages: list[str] | None = None,
        samples_per_lang: int = 500,
    ) -> pd.DataFrame:
        """Download and combine datasets into a unified format.

        The output DataFrame has columns: text, intent, urgency, language.
        Intent and urgency are assigned synthetically for bootstrapping.

        Args:
            target_languages: Languages to include.
            samples_per_lang: Samples per language.

        Returns:
            Combined DataFrame with standardized columns.
        """
        config = load_config()
        languages = target_languages or self.supported_languages[:5]
        intents = config.classification.intent_categories
        urgency_levels = config.urgency.levels

        raw_data = self.download_multilingual_sentiments(languages, samples_per_lang)

        if raw_data.empty:
            return pd.DataFrame(columns=["text", "intent", "urgency", "language"])

        # Assign synthetic labels for bootstrapping (round-robin)
        raw_data["intent"] = [intents[i % len(intents)] for i in range(len(raw_data))]
        raw_data["urgency"] = [
            urgency_levels[i % len(urgency_levels)] for i in range(len(raw_data))
        ]

        output_path = self.data_dir / "combined_dataset.parquet"
        raw_data[["text", "intent", "urgency", "language"]].to_parquet(
            output_path, index=False
        )
        logger.info("Saved combined dataset to %s", output_path)

        return raw_data[["text", "intent", "urgency", "language"]]

    def save_dataset(self, df: pd.DataFrame, filename: str) -> Path:
        """Save a DataFrame to parquet format in the data directory.

        Args:
            df: DataFrame to save.
            filename: Output filename (without extension).

        Returns:
            Path to the saved file.
        """
        output_path = self.data_dir / f"{filename}.parquet"
        df.to_parquet(output_path, index=False)
        logger.info("Saved dataset to %s (%d rows)", output_path, len(df))
        return output_path
