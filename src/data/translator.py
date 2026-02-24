"""Synthetic multilingual data generation via translation models.

Uses Helsinki-NLP/opus-mt models from HuggingFace to translate English
support tickets into target languages for training data augmentation.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


class SyntheticTranslator:
    """Generates synthetic multilingual tickets via translation models.

    Uses Helsinki-NLP/opus-mt MarianMT models for offline translation
    without external API dependencies.

    Args:
        target_languages: List of ISO 639-1 language codes to translate into.
    """

    def __init__(self, target_languages: list[str] | None = None) -> None:
        self.target_languages = target_languages or ["es", "fr", "de", "pt"]
        self._models: dict[str, tuple] = {}

    def _load_translation_model(self, source_lang: str, target_lang: str) -> tuple:
        """Load a MarianMT model for a language pair.

        Args:
            source_lang: Source language code.
            target_lang: Target language code.

        Returns:
            Tuple of (tokenizer, model) for the language pair.

        Raises:
            ImportError: If transformers is not installed.
            OSError: If the model is not available for the language pair.
        """
        key = f"{source_lang}-{target_lang}"
        if key not in self._models:
            from transformers import MarianMTModel, MarianTokenizer

            model_name = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
            logger.info("Loading translation model: %s", model_name)
            tokenizer = MarianTokenizer.from_pretrained(model_name)
            model = MarianMTModel.from_pretrained(model_name)
            self._models[key] = (tokenizer, model)
        return self._models[key]

    def translate_batch(
        self,
        texts: list[str],
        source_lang: str = "en",
        target_lang: str = "es",
        batch_size: int = 16,
    ) -> list[str]:
        """Translate a batch of texts between languages.

        Args:
            texts: List of text strings to translate.
            source_lang: Source language code.
            target_lang: Target language code.
            batch_size: Number of texts to process at once.

        Returns:
            List of translated text strings.
        """
        tokenizer, model = self._load_translation_model(source_lang, target_lang)
        translated: list[str] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True
            )
            outputs = model.generate(**inputs)
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            translated.extend(decoded)

        logger.info(
            "Translated %d texts from %s to %s",
            len(translated),
            source_lang,
            target_lang,
        )
        return translated

    def generate_synthetic_dataset(
        self,
        english_data: pd.DataFrame,
        samples_per_lang: int = 200,
    ) -> pd.DataFrame:
        """Generate synthetic multilingual tickets from English data.

        Takes English-language support tickets and translates them into
        each target language, preserving intent and urgency labels.

        Args:
            english_data: DataFrame with columns: text, intent, urgency.
            samples_per_lang: Maximum samples to generate per language.

        Returns:
            DataFrame with translated tickets across all target languages.
        """
        frames: list[pd.DataFrame] = []
        sample = english_data.head(samples_per_lang)

        for lang in self.target_languages:
            try:
                translated_texts = self.translate_batch(
                    sample["text"].tolist(),
                    source_lang="en",
                    target_lang=lang,
                )
                lang_df = sample[["intent", "urgency"]].copy()
                lang_df["text"] = translated_texts
                lang_df["language"] = lang
                frames.append(lang_df)
                logger.info("Generated %d synthetic samples for %s", len(lang_df), lang)
            except Exception:
                logger.warning("Translation failed for language=%s, skipping", lang)

        if not frames:
            return pd.DataFrame(columns=["text", "intent", "urgency", "language"])

        return pd.concat(frames, ignore_index=True)
