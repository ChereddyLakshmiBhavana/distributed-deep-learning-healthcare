"""
Prediction Database Module
Stores uploaded image bytes and diagnosis metadata in SQLite.
"""

import os
import sqlite3
from pathlib import Path
from datetime import datetime


class PredictionDatabase:
    """Persist uploaded images and prediction outcomes."""

    def __init__(self, db_path='prediction_logs/predictions.db'):
        base_dir = Path(__file__).resolve().parent
        candidate = Path(db_path)
        self.db_path = (base_dir / candidate).resolve() if not candidate.is_absolute() else candidate
        os.makedirs(self.db_path.parent, exist_ok=True)
        self._initialize_schema()

    def _connect(self):
        return sqlite3.connect(str(self.db_path))

    def _initialize_schema(self):
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    endpoint TEXT NOT NULL,
                    status TEXT NOT NULL,
                    filename TEXT,
                    content_type TEXT,
                    image_bytes BLOB,
                    model TEXT,
                    prediction TEXT,
                    confidence REAL,
                    error TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_predictions_created_at
                ON predictions(created_at)
                """
            )

    def save_prediction(
        self,
        endpoint,
        status,
        image_bytes,
        filename=None,
        content_type=None,
        model=None,
        prediction=None,
        confidence=None,
        error=None
    ):
        created_at = datetime.now().isoformat(timespec='seconds')

        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO predictions (
                    created_at,
                    endpoint,
                    status,
                    filename,
                    content_type,
                    image_bytes,
                    model,
                    prediction,
                    confidence,
                    error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    created_at,
                    endpoint,
                    status,
                    filename,
                    content_type,
                    sqlite3.Binary(image_bytes) if image_bytes is not None else None,
                    model,
                    prediction,
                    float(confidence) if confidence is not None else None,
                    error
                )
            )
            return cursor.lastrowid

    def get_stats(self):
        with self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
            success = conn.execute(
                "SELECT COUNT(*) FROM predictions WHERE status = 'success'"
            ).fetchone()[0]
            failures = total - success

        return {
            'db_path': str(self.db_path),
            'total_records': total,
            'successful_records': success,
            'failed_records': failures
        }
