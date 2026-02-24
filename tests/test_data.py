"""Tests for data downloading, translation, and splitting modules."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data.downloader import DatasetDownloader
from src.data.splitter import save_splits, stratified_split
from src.data.translator import SyntheticTranslator
from src.utils.config import reset_config_cache


@pytest.fixture(autouse=True)
def _clear_cache():
    reset_config_cache()
    yield
    reset_config_cache()


# --- Sample data fixtures ---


@pytest.fixture()
def sample_tickets() -> pd.DataFrame:
    """Create sample ticket data for testing."""
    path = Path("data/sample/sample_tickets.json")
    with open(path) as f:
        tickets = json.load(f)
    return pd.DataFrame(tickets)


@pytest.fixture()
def english_tickets(sample_tickets: pd.DataFrame) -> pd.DataFrame:
    """Filter to English-only tickets."""
    return sample_tickets[sample_tickets["language"] == "en"].reset_index(drop=True)


# --- DatasetDownloader tests ---


class TestDatasetDownloader:
    """Tests for DatasetDownloader."""

    def test_init_creates_data_dir(self, tmp_path: Path) -> None:
        """Downloader should create data directory on init."""
        data_dir = tmp_path / "test_data"
        dl = DatasetDownloader(data_dir=str(data_dir))
        assert data_dir.exists()
        assert dl.data_dir == data_dir

    def test_save_dataset(self, tmp_path: Path, sample_tickets: pd.DataFrame) -> None:
        """save_dataset should write a parquet file."""
        dl = DatasetDownloader(data_dir=str(tmp_path))
        path = dl.save_dataset(sample_tickets, "test_output")
        assert path.exists()
        assert path.suffix == ".parquet"
        loaded = pd.read_parquet(path)
        assert len(loaded) == len(sample_tickets)

    @patch("src.data.downloader.load_dataset")
    def test_download_multilingual_sentiments(
        self, mock_load_dataset: MagicMock, tmp_path: Path
    ) -> None:
        """Should call load_dataset for each language and return combined DataFrame."""
        mock_ds = [{"text": "test sentence"}]
        mock_load_dataset.return_value = mock_ds

        dl = DatasetDownloader(data_dir=str(tmp_path))
        result = dl.download_multilingual_sentiments(
            languages=["en"], samples_per_lang=1
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) >= 1
        assert "text" in result.columns
        assert "language" in result.columns

    @patch("src.data.downloader.load_dataset")
    def test_download_handles_failure_gracefully(
        self, mock_load_dataset: MagicMock, tmp_path: Path
    ) -> None:
        """Should return empty DataFrame when download fails."""
        mock_load_dataset.side_effect = Exception("Network error")
        dl = DatasetDownloader(data_dir=str(tmp_path))
        result = dl.download_multilingual_sentiments(
            languages=["en"], samples_per_lang=1
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0


# --- SyntheticTranslator tests ---


class TestSyntheticTranslator:
    """Tests for SyntheticTranslator."""

    def test_init_default_languages(self) -> None:
        """Default target languages should be set."""
        translator = SyntheticTranslator()
        assert len(translator.target_languages) == 4
        assert "es" in translator.target_languages

    def test_init_custom_languages(self) -> None:
        """Custom target languages should override defaults."""
        translator = SyntheticTranslator(target_languages=["fr", "de"])
        assert translator.target_languages == ["fr", "de"]

    @patch("src.data.translator.SyntheticTranslator._load_translation_model")
    def test_translate_batch(self, mock_load: MagicMock) -> None:
        """translate_batch should return translated texts."""
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        mock_model.generate.return_value = MagicMock()
        mock_tokenizer.batch_decode.return_value = ["texto traducido"]
        mock_load.return_value = (mock_tokenizer, mock_model)

        translator = SyntheticTranslator(target_languages=["es"])
        result = translator.translate_batch(["test text"], "en", "es")
        assert result == ["texto traducido"]

    @patch("src.data.translator.SyntheticTranslator.translate_batch")
    def test_generate_synthetic_dataset(
        self, mock_translate: MagicMock, english_tickets: pd.DataFrame
    ) -> None:
        """Should generate translated datasets for each target language."""
        mock_translate.return_value = ["translated"] * len(english_tickets.head(5))
        translator = SyntheticTranslator(target_languages=["es"])
        result = translator.generate_synthetic_dataset(
            english_tickets, samples_per_lang=5
        )
        assert isinstance(result, pd.DataFrame)
        assert "language" in result.columns
        assert "text" in result.columns

    @patch("src.data.translator.SyntheticTranslator.translate_batch")
    def test_generate_synthetic_handles_failure(
        self, mock_translate: MagicMock, english_tickets: pd.DataFrame
    ) -> None:
        """Should return empty DataFrame when all translations fail."""
        mock_translate.side_effect = Exception("Model loading failed")
        translator = SyntheticTranslator(target_languages=["xx"])
        result = translator.generate_synthetic_dataset(
            english_tickets, samples_per_lang=5
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0


# --- Splitter tests ---


class TestStratifiedSplit:
    """Tests for stratified_split utility."""

    def test_basic_split_ratios(self, sample_tickets: pd.DataFrame) -> None:
        """Split sizes should approximate the requested ratios."""
        train, val, test = stratified_split(sample_tickets)
        total = len(sample_tickets)
        assert len(train) + len(val) + len(test) == total
        assert len(train) > len(val)
        assert len(train) > len(test)

    def test_all_rows_preserved(self, sample_tickets: pd.DataFrame) -> None:
        """No rows should be lost during splitting."""
        train, val, test = stratified_split(sample_tickets)
        combined = pd.concat([train, val, test])
        assert len(combined) == len(sample_tickets)

    def test_invalid_ratios_raises(self, sample_tickets: pd.DataFrame) -> None:
        """Ratios not summing to 1.0 should raise ValueError."""
        with pytest.raises(ValueError, match="must sum to 1.0"):
            stratified_split(
                sample_tickets, train_ratio=0.5, val_ratio=0.1, test_ratio=0.1
            )

    def test_custom_ratios(self, sample_tickets: pd.DataFrame) -> None:
        """Custom ratios should be respected."""
        train, val, test = stratified_split(
            sample_tickets, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1
        )
        total = len(sample_tickets)
        assert len(train) + len(val) + len(test) == total

    def test_reproducibility(self, sample_tickets: pd.DataFrame) -> None:
        """Same random_state should produce identical splits."""
        t1, v1, s1 = stratified_split(sample_tickets, random_state=42)
        t2, v2, s2 = stratified_split(sample_tickets, random_state=42)
        pd.testing.assert_frame_equal(
            t1.reset_index(drop=True), t2.reset_index(drop=True)
        )


class TestSaveSplits:
    """Tests for save_splits utility."""

    def test_save_creates_parquet_files(
        self, tmp_path: Path, sample_tickets: pd.DataFrame
    ) -> None:
        """save_splits should create train/val/test parquet files."""
        train, val, test = stratified_split(sample_tickets)
        paths = save_splits(train, val, test, output_dir=str(tmp_path))
        assert "train" in paths
        assert "val" in paths
        assert "test" in paths
        for path in paths.values():
            assert path.exists()
            assert path.suffix == ".parquet"


# --- Sample data integrity ---


class TestSampleData:
    """Tests for the sample ticket data."""

    def test_sample_tickets_load(self, sample_tickets: pd.DataFrame) -> None:
        """Sample tickets should load with expected shape."""
        assert len(sample_tickets) == 100
        assert set(sample_tickets.columns) == {"text", "intent", "urgency", "language"}

    def test_five_languages(self, sample_tickets: pd.DataFrame) -> None:
        """Sample tickets should contain exactly 5 languages."""
        languages = sample_tickets["language"].unique()
        assert len(languages) == 5
        assert set(languages) == {"en", "es", "fr", "de", "pt"}

    def test_twenty_per_language(self, sample_tickets: pd.DataFrame) -> None:
        """Each language should have 20 tickets."""
        counts = sample_tickets["language"].value_counts()
        for lang in ["en", "es", "fr", "de", "pt"]:
            assert counts[lang] == 20

    def test_all_intents_present(self, sample_tickets: pd.DataFrame) -> None:
        """All intent categories should be represented."""
        intents = set(sample_tickets["intent"].unique())
        expected = {
            "billing",
            "technical_support",
            "account",
            "general_inquiry",
            "complaint",
            "feedback",
        }
        assert intents == expected

    def test_urgency_levels_present(self, sample_tickets: pd.DataFrame) -> None:
        """Multiple urgency levels should be present."""
        urgencies = set(sample_tickets["urgency"].unique())
        assert len(urgencies) >= 3
