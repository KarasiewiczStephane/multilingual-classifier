"""Tests for the SQLite metrics database."""

from pathlib import Path

import pytest

from src.utils.config import reset_config_cache
from src.utils.database import MetricsDatabase


@pytest.fixture(autouse=True)
def _clear_cache():
    reset_config_cache()
    yield
    reset_config_cache()


@pytest.fixture()
def db(tmp_path: Path) -> MetricsDatabase:
    """Create a test database in a temp directory."""
    return MetricsDatabase(db_path=str(tmp_path / "test_metrics.db"))


class TestDatabaseInit:
    """Tests for database initialization."""

    def test_creates_db_file(self, tmp_path: Path) -> None:
        """Database file should be created on init."""
        db_path = tmp_path / "test.db"
        MetricsDatabase(db_path=str(db_path))
        assert db_path.exists()

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        """Parent directories should be created if needed."""
        db_path = tmp_path / "subdir" / "test.db"
        MetricsDatabase(db_path=str(db_path))
        assert db_path.exists()

    def test_idempotent_init(self, tmp_path: Path) -> None:
        """Multiple inits on same file should not error."""
        db_path = str(tmp_path / "test.db")
        MetricsDatabase(db_path=db_path)
        MetricsDatabase(db_path=db_path)


class TestLogClassification:
    """Tests for logging classification results."""

    def test_log_returns_id(self, db: MetricsDatabase) -> None:
        """log_classification should return a positive row ID."""
        row_id = db.log_classification(
            text="test text",
            language="en",
            primary_intent="billing",
            confidence=0.85,
            urgency="low",
            was_escalated=False,
            needs_review=False,
            processing_time_ms=42.5,
        )
        assert row_id > 0

    def test_log_multiple(self, db: MetricsDatabase) -> None:
        """Multiple logs should produce incrementing IDs."""
        id1 = db.log_classification(
            text="text1",
            language="en",
            primary_intent="billing",
            confidence=0.9,
            urgency="low",
            was_escalated=False,
            needs_review=False,
            processing_time_ms=10.0,
        )
        id2 = db.log_classification(
            text="text2",
            language="es",
            primary_intent="complaint",
            confidence=0.7,
            urgency="high",
            was_escalated=True,
            needs_review=True,
            processing_time_ms=20.0,
        )
        assert id2 > id1

    def test_text_stored_as_hash(self, db: MetricsDatabase) -> None:
        """Original text should not be stored directly."""
        import sqlite3

        db.log_classification(
            text="sensitive customer data",
            language="en",
            primary_intent="billing",
            confidence=0.9,
            urgency="low",
            was_escalated=False,
            needs_review=False,
            processing_time_ms=10.0,
        )
        with sqlite3.connect(db.db_path) as conn:
            row = conn.execute("SELECT text_hash FROM classifications").fetchone()
        assert row[0] != "sensitive customer data"
        assert len(row[0]) == 16


class TestLogEscalation:
    """Tests for logging escalation events."""

    def test_log_escalation(self, db: MetricsDatabase) -> None:
        """Should log an escalation event."""
        cls_id = db.log_classification(
            text="urgent",
            language="en",
            primary_intent="billing",
            confidence=0.3,
            urgency="critical",
            was_escalated=True,
            needs_review=True,
            processing_time_ms=50.0,
        )
        esc_id = db.log_escalation(cls_id, "low confidence critical ticket")
        assert esc_id > 0


class TestLogAccuracyMetrics:
    """Tests for logging accuracy metrics."""

    def test_log_accuracy(self, db: MetricsDatabase) -> None:
        """Should log accuracy metrics."""
        row_id = db.log_accuracy_metrics(
            language="en", accuracy=0.85, f1_score=0.82, sample_count=100
        )
        assert row_id > 0

    def test_log_accuracy_with_intent(self, db: MetricsDatabase) -> None:
        """Should log accuracy metrics with optional intent."""
        row_id = db.log_accuracy_metrics(
            language="en",
            accuracy=0.9,
            f1_score=0.88,
            sample_count=50,
            intent="billing",
        )
        assert row_id > 0


class TestQueryMethods:
    """Tests for database query methods."""

    def _insert_sample_data(self, db: MetricsDatabase) -> None:
        """Insert sample data for query testing."""
        for i in range(5):
            db.log_classification(
                text=f"english text {i}",
                language="en",
                primary_intent="billing",
                confidence=0.85 + i * 0.01,
                urgency="low",
                was_escalated=False,
                needs_review=False,
                processing_time_ms=10.0 + i,
            )
        for i in range(3):
            db.log_classification(
                text=f"spanish text {i}",
                language="es",
                primary_intent="complaint",
                confidence=0.7 + i * 0.05,
                urgency="high",
                was_escalated=True,
                needs_review=True,
                processing_time_ms=20.0 + i,
            )

    def test_get_accuracy_by_language(self, db: MetricsDatabase) -> None:
        """Should return stats grouped by language."""
        self._insert_sample_data(db)
        stats = db.get_accuracy_by_language()
        assert "en" in stats
        assert "es" in stats
        assert stats["en"]["total"] == 5
        assert stats["es"]["total"] == 3

    def test_get_intent_distribution(self, db: MetricsDatabase) -> None:
        """Should return intent counts and avg confidence."""
        self._insert_sample_data(db)
        dist = db.get_intent_distribution()
        assert "billing" in dist
        assert "complaint" in dist
        assert dist["billing"]["count"] == 5

    def test_get_latency_stats(self, db: MetricsDatabase) -> None:
        """Should return latency statistics."""
        self._insert_sample_data(db)
        stats = db.get_latency_stats()
        assert "mean_ms" in stats
        assert "min_ms" in stats
        assert "max_ms" in stats
        assert stats["min_ms"] > 0

    def test_get_latency_stats_empty(self, db: MetricsDatabase) -> None:
        """Empty database should return zero latency stats."""
        stats = db.get_latency_stats()
        assert stats["mean_ms"] == 0.0

    def test_get_total_classifications(self, db: MetricsDatabase) -> None:
        """Should return correct total count."""
        self._insert_sample_data(db)
        total = db.get_total_classifications()
        assert total == 8

    def test_get_total_classifications_empty(self, db: MetricsDatabase) -> None:
        """Empty database should return 0."""
        assert db.get_total_classifications() == 0

    def test_get_recent_escalations(self, db: MetricsDatabase) -> None:
        """Should return recent escalation events."""
        cls_id = db.log_classification(
            text="urgent",
            language="en",
            primary_intent="billing",
            confidence=0.3,
            urgency="critical",
            was_escalated=True,
            needs_review=True,
            processing_time_ms=50.0,
        )
        db.log_escalation(cls_id, "critical ticket")
        escalations = db.get_recent_escalations(limit=10)
        assert len(escalations) == 1
        assert escalations[0]["reason"] == "critical ticket"

    def test_get_recent_escalations_empty(self, db: MetricsDatabase) -> None:
        """Empty database should return empty list."""
        assert db.get_recent_escalations() == []
