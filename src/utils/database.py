"""SQLite database for storing classification metrics and escalations.

Provides structured storage for classification results, escalation events,
and accuracy metrics with query methods for the metrics API.
"""

import hashlib
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from src.utils.config import load_config

logger = logging.getLogger(__name__)


class MetricsDatabase:
    """SQLite-backed storage for classification metrics.

    Stores classification results, escalation events, and accuracy
    metrics for dashboard and API consumption.

    Args:
        db_path: Path to the SQLite database file.
    """

    def __init__(self, db_path: str | None = None) -> None:
        config = load_config()
        self.db_path = Path(db_path or config.database.path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema with tables and indexes."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS classifications (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    text_hash TEXT NOT NULL,
                    language TEXT NOT NULL,
                    primary_intent TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    urgency TEXT NOT NULL,
                    was_escalated BOOLEAN NOT NULL,
                    needs_review BOOLEAN NOT NULL,
                    processing_time_ms REAL NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS escalations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    classification_id INTEGER,
                    reason TEXT NOT NULL,
                    resolved BOOLEAN DEFAULT FALSE,
                    resolution_notes TEXT,
                    FOREIGN KEY (classification_id) REFERENCES classifications(id)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS accuracy_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    language TEXT NOT NULL,
                    intent TEXT,
                    accuracy REAL NOT NULL,
                    f1_score REAL NOT NULL,
                    sample_count INTEGER NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_class_lang
                ON classifications(language)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_class_timestamp
                ON classifications(timestamp)
            """)
        logger.info("Database initialized at %s", self.db_path)

    def log_classification(
        self,
        text: str,
        language: str,
        primary_intent: str,
        confidence: float,
        urgency: str,
        was_escalated: bool,
        needs_review: bool,
        processing_time_ms: float,
    ) -> int:
        """Log a classification result to the database.

        Args:
            text: Original text (stored as hash for privacy).
            language: Detected language code.
            primary_intent: Classified intent category.
            confidence: Classification confidence score.
            urgency: Urgency level string.
            was_escalated: Whether the ticket was escalated.
            needs_review: Whether human review is needed.
            processing_time_ms: Processing time in milliseconds.

        Returns:
            Row ID of the inserted classification.
        """
        text_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
        now = datetime.now(timezone.utc).isoformat()

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                INSERT INTO classifications
                (timestamp, text_hash, language, primary_intent, confidence,
                 urgency, was_escalated, needs_review, processing_time_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    now,
                    text_hash,
                    language,
                    primary_intent,
                    confidence,
                    urgency,
                    was_escalated,
                    needs_review,
                    processing_time_ms,
                ),
            )
            return cursor.lastrowid

    def log_escalation(
        self,
        classification_id: int,
        reason: str,
    ) -> int:
        """Log an escalation event.

        Args:
            classification_id: ID of the related classification.
            reason: Reason for escalation.

        Returns:
            Row ID of the inserted escalation.
        """
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                INSERT INTO escalations (timestamp, classification_id, reason)
                VALUES (?, ?, ?)
                """,
                (now, classification_id, reason),
            )
            return cursor.lastrowid

    def log_accuracy_metrics(
        self,
        language: str,
        accuracy: float,
        f1_score: float,
        sample_count: int,
        intent: str | None = None,
    ) -> int:
        """Log accuracy metrics for a language/intent combination.

        Args:
            language: Language code.
            accuracy: Accuracy score.
            f1_score: F1 score.
            sample_count: Number of samples evaluated.
            intent: Optional intent category.

        Returns:
            Row ID of the inserted metrics record.
        """
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                INSERT INTO accuracy_metrics
                (timestamp, language, intent, accuracy, f1_score, sample_count)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (now, language, intent, accuracy, f1_score, sample_count),
            )
            return cursor.lastrowid

    def get_accuracy_by_language(self) -> dict[str, dict]:
        """Get aggregated classification stats grouped by language.

        Returns:
            Dictionary mapping language codes to their stats.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT
                    language,
                    COUNT(*) as total,
                    AVG(confidence) as avg_confidence,
                    SUM(CASE WHEN needs_review THEN 1 ELSE 0 END) as review_count,
                    SUM(CASE WHEN was_escalated THEN 1 ELSE 0 END) as escalation_count
                FROM classifications
                GROUP BY language
            """).fetchall()

        return {
            row["language"]: {
                "total": row["total"],
                "avg_confidence": round(row["avg_confidence"], 4),
                "review_count": row["review_count"],
                "escalation_count": row["escalation_count"],
            }
            for row in rows
        }

    def get_intent_distribution(self) -> dict[str, dict]:
        """Get classification counts and average confidence per intent.

        Returns:
            Dictionary mapping intent names to their stats.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT
                    primary_intent,
                    COUNT(*) as count,
                    AVG(confidence) as avg_confidence
                FROM classifications
                GROUP BY primary_intent
            """).fetchall()

        return {
            row["primary_intent"]: {
                "count": row["count"],
                "avg_confidence": round(row["avg_confidence"], 4),
            }
            for row in rows
        }

    def get_latency_stats(self) -> dict[str, float | None]:
        """Get processing time statistics.

        Returns:
            Dictionary with mean, min, max, p50, p95, p99 latency stats.
        """
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("""
                SELECT
                    AVG(processing_time_ms) as mean_ms,
                    MIN(processing_time_ms) as min_ms,
                    MAX(processing_time_ms) as max_ms
                FROM classifications
            """).fetchone()

        if row is None or row[0] is None:
            return {
                "mean_ms": 0.0,
                "min_ms": 0.0,
                "max_ms": 0.0,
                "p50_ms": None,
                "p95_ms": None,
                "p99_ms": None,
            }

        # Calculate percentiles
        with sqlite3.connect(self.db_path) as conn:
            all_times = [
                r[0]
                for r in conn.execute(
                    "SELECT processing_time_ms FROM classifications ORDER BY processing_time_ms"
                ).fetchall()
            ]

        n = len(all_times)
        return {
            "mean_ms": round(row[0], 2),
            "min_ms": round(row[1], 2),
            "max_ms": round(row[2], 2),
            "p50_ms": round(all_times[n // 2], 2) if n > 0 else None,
            "p95_ms": round(all_times[int(n * 0.95)], 2) if n > 1 else None,
            "p99_ms": round(all_times[int(n * 0.99)], 2) if n > 1 else None,
        }

    def get_total_classifications(self) -> int:
        """Get total number of classifications.

        Returns:
            Total classification count.
        """
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("SELECT COUNT(*) FROM classifications").fetchone()
            return row[0] if row else 0

    def get_recent_escalations(self, limit: int = 50) -> list[dict]:
        """Get recent escalation events.

        Args:
            limit: Maximum number of escalations to return.

        Returns:
            List of escalation event dictionaries.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT e.*, c.language, c.primary_intent, c.confidence
                FROM escalations e
                LEFT JOIN classifications c ON e.classification_id = c.id
                ORDER BY e.timestamp DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]
