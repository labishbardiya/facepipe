"""
Feature store — embeddings as first-class records.

Stores each embedding alongside rich metadata: quality, pose, camera,
timestamp, model version, cluster assignment. This is how large-scale
facial recognition systems manage their embedding data.
"""

from __future__ import annotations

import dataclasses
import json
import sqlite3
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from facepipe.config.settings import get_settings
from facepipe.observability.logging import get_logger

logger = get_logger(__name__)


@dataclasses.dataclass
class EmbeddingRecord:
    """A first-class embedding record with full metadata.

    Attributes:
        record_id: Unique record identifier.
        identity_id: Identity this embedding belongs to.
        model_version: Model that produced this embedding.
        quality_score: Quality assessment score at capture time.
        yaw: Head yaw angle at capture time.
        pitch: Head pitch angle at capture time.
        camera_id: Camera that captured this embedding.
        timestamp: Unix timestamp of capture.
        cluster_id: Assigned cluster within the identity.
        is_centroid: Whether this is a cluster centroid.
        source: How this embedding was created (enrollment, active_learning, verification).
    """
    record_id: str
    identity_id: str
    model_version: str
    quality_score: float
    yaw: float
    pitch: float
    camera_id: str
    timestamp: float
    cluster_id: int
    is_centroid: bool
    source: str


class FeatureStore:
    """SQLite-backed feature store for embedding metadata.

    Embeddings are stored encrypted on disk (via EncryptedEmbeddingStore).
    This store manages the metadata records that describe each embedding.

    Args:
        db_path: Path to the SQLite database file.
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        if db_path is None:
            db_path = str(get_settings().data_dir / "feature_store.db")

        self._db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the database schema."""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS embedding_records (
                    record_id TEXT PRIMARY KEY,
                    identity_id TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    quality_score REAL DEFAULT 0.0,
                    yaw REAL DEFAULT 0.0,
                    pitch REAL DEFAULT 0.0,
                    camera_id TEXT DEFAULT 'default',
                    timestamp REAL NOT NULL,
                    cluster_id INTEGER DEFAULT 0,
                    is_centroid INTEGER DEFAULT 0,
                    source TEXT DEFAULT 'enrollment'
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_identity
                ON embedding_records(identity_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_model_version
                ON embedding_records(model_version)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp
                ON embedding_records(timestamp)
            """)
            conn.commit()

    def add_record(self, record: EmbeddingRecord) -> None:
        """Add a single embedding record."""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO embedding_records
                   (record_id, identity_id, model_version, quality_score,
                    yaw, pitch, camera_id, timestamp, cluster_id, is_centroid, source)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    record.record_id, record.identity_id, record.model_version,
                    record.quality_score, record.yaw, record.pitch,
                    record.camera_id, record.timestamp, record.cluster_id,
                    int(record.is_centroid), record.source,
                ),
            )
            conn.commit()

    def get_records_by_identity(self, identity_id: str) -> List[EmbeddingRecord]:
        """Get all embedding records for an identity."""
        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM embedding_records WHERE identity_id = ? ORDER BY timestamp",
                (identity_id,),
            ).fetchall()

        return [self._row_to_record(row) for row in rows]

    def get_records_by_model(self, model_version: str) -> List[EmbeddingRecord]:
        """Get all records for a specific model version (for migration)."""
        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM embedding_records WHERE model_version = ?",
                (model_version,),
            ).fetchall()

        return [self._row_to_record(row) for row in rows]

    def get_record_count(self, identity_id: Optional[str] = None) -> int:
        """Get total record count, optionally filtered by identity."""
        with sqlite3.connect(self._db_path) as conn:
            if identity_id:
                row = conn.execute(
                    "SELECT COUNT(*) FROM embedding_records WHERE identity_id = ?",
                    (identity_id,),
                ).fetchone()
            else:
                row = conn.execute("SELECT COUNT(*) FROM embedding_records").fetchone()
            return row[0] if row else 0

    def delete_by_identity(self, identity_id: str) -> int:
        """Delete all records for an identity. Returns count deleted."""
        with sqlite3.connect(self._db_path) as conn:
            cursor = conn.execute(
                "DELETE FROM embedding_records WHERE identity_id = ?",
                (identity_id,),
            )
            conn.commit()
            return cursor.rowcount

    def get_model_versions(self) -> Dict[str, int]:
        """Get count of records per model version."""
        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute(
                "SELECT model_version, COUNT(*) FROM embedding_records GROUP BY model_version"
            ).fetchall()
        return {row[0]: row[1] for row in rows}

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> EmbeddingRecord:
        return EmbeddingRecord(
            record_id=row["record_id"],
            identity_id=row["identity_id"],
            model_version=row["model_version"],
            quality_score=row["quality_score"],
            yaw=row["yaw"],
            pitch=row["pitch"],
            camera_id=row["camera_id"],
            timestamp=row["timestamp"],
            cluster_id=row["cluster_id"],
            is_centroid=bool(row["is_centroid"]),
            source=row["source"],
        )
