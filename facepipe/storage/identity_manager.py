"""
Identity manager — multi-embedding CRUD operations.

Coordinates between the encrypted embedding store, feature store,
cluster engine, and vector index to provide a unified identity
management interface.
"""

from __future__ import annotations

import dataclasses
import sqlite3
import time
from pathlib import Path

import ulid

from facepipe.config.settings import get_settings
from facepipe.observability.logging import get_logger

logger = get_logger(__name__)


@dataclasses.dataclass
class IdentityRecord:
    """Metadata for an enrolled identity.

    Attributes:
        identity_id: UUID for this identity.
        name: Display name.
        created_at: Unix timestamp of creation.
        last_seen: Unix timestamp of last recognition.
        embedding_count: Total number of stored embeddings.
        cluster_count: Number of appearance clusters.
        model_version: Primary model version used.
        is_active: Whether the identity is active.
    """
    identity_id: str
    name: str
    created_at: float
    last_seen: float
    embedding_count: int
    cluster_count: int
    model_version: str
    is_active: bool


class IdentityManager:
    """Unified identity management with SQLite metadata store.

    Args:
        db_path: Path to the identity metadata database.
    """

    def __init__(self, db_path: str | None = None) -> None:
        if db_path is None:
            db_path = str(get_settings().data_dir / "identities.db")

        self._db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the identity metadata schema."""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS identities (
                    identity_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    last_seen REAL NOT NULL,
                    embedding_count INTEGER DEFAULT 0,
                    cluster_count INTEGER DEFAULT 0,
                    model_version TEXT DEFAULT '',
                    is_active INTEGER DEFAULT 1
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_name ON identities(name)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_active ON identities(is_active)
            """)
            conn.commit()

    def create(
        self,
        name: str,
        embedding_count: int = 0,
        cluster_count: int = 0,
        model_version: str = "",
        identity_id: str | None = None,
    ) -> IdentityRecord:
        """Create a new identity record.

        Args:
            name: Display name.
            embedding_count: Initial embedding count.
            cluster_count: Initial cluster count.
            model_version: Model version used.
            identity_id: Optional UUID. Generated if not provided.

        Returns:
            The created IdentityRecord.
        """
        if identity_id is None:
            identity_id = str(ulid.new())

        now = time.time()
        record = IdentityRecord(
            identity_id=identity_id,
            name=name,
            created_at=now,
            last_seen=now,
            embedding_count=embedding_count,
            cluster_count=cluster_count,
            model_version=model_version,
            is_active=True,
        )

        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """INSERT INTO identities
                   (identity_id, name, created_at, last_seen, embedding_count,
                    cluster_count, model_version, is_active)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    record.identity_id, record.name, record.created_at,
                    record.last_seen, record.embedding_count, record.cluster_count,
                    record.model_version, int(record.is_active),
                ),
            )
            conn.commit()

        logger.info("identity_created", identity_id=identity_id, name=name)
        return record

    def get(self, identity_id: str) -> IdentityRecord | None:
        """Get an identity by ID."""
        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM identities WHERE identity_id = ?",
                (identity_id,),
            ).fetchone()

        return self._row_to_record(row) if row else None

    def get_by_name(self, name: str) -> list[IdentityRecord]:
        """Get identities by display name (may return multiple)."""
        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM identities WHERE name = ? AND is_active = 1",
                (name,),
            ).fetchall()

        return [self._row_to_record(row) for row in rows]

    def list_all(self, active_only: bool = True) -> list[IdentityRecord]:
        """List all identities."""
        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            if active_only:
                rows = conn.execute(
                    "SELECT * FROM identities WHERE is_active = 1 ORDER BY name"
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM identities ORDER BY name"
                ).fetchall()

        return [self._row_to_record(row) for row in rows]

    def update(
        self,
        identity_id: str,
        name: str | None = None,
        embedding_count: int | None = None,
        cluster_count: int | None = None,
        last_seen: float | None = None,
    ) -> bool:
        """Update identity metadata. Returns True if found and updated."""
        updates = []
        params = []

        if name is not None:
            updates.append("name = ?")
            params.append(name)
        if embedding_count is not None:
            updates.append("embedding_count = ?")
            params.append(embedding_count)
        if cluster_count is not None:
            updates.append("cluster_count = ?")
            params.append(cluster_count)
        if last_seen is not None:
            updates.append("last_seen = ?")
            params.append(last_seen)

        if not updates:
            return False

        params.append(identity_id)

        with sqlite3.connect(self._db_path) as conn:
            cursor = conn.execute(
                f"UPDATE identities SET {', '.join(updates)} WHERE identity_id = ?",
                params,
            )
            conn.commit()
            return cursor.rowcount > 0

    def delete(self, identity_id: str, soft: bool = True) -> bool:
        """Delete an identity.

        Args:
            identity_id: Identity to delete.
            soft: If True, mark as inactive. If False, permanently delete.

        Returns:
            True if found and deleted.
        """
        with sqlite3.connect(self._db_path) as conn:
            if soft:
                cursor = conn.execute(
                    "UPDATE identities SET is_active = 0 WHERE identity_id = ?",
                    (identity_id,),
                )
            else:
                cursor = conn.execute(
                    "DELETE FROM identities WHERE identity_id = ?",
                    (identity_id,),
                )
            conn.commit()
            deleted = cursor.rowcount > 0

        if deleted:
            logger.info("identity_deleted", identity_id=identity_id, soft=soft)
        return deleted

    def count(self, active_only: bool = True) -> int:
        """Count total identities."""
        with sqlite3.connect(self._db_path) as conn:
            if active_only:
                row = conn.execute(
                    "SELECT COUNT(*) FROM identities WHERE is_active = 1"
                ).fetchone()
            else:
                row = conn.execute("SELECT COUNT(*) FROM identities").fetchone()
            return row[0] if row else 0

    def search_by_name(self, query: str) -> list[IdentityRecord]:
        """Search identities by name (case-insensitive partial match)."""
        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM identities WHERE name LIKE ? AND is_active = 1 ORDER BY name",
                (f"%{query}%",),
            ).fetchall()
        return [self._row_to_record(row) for row in rows]

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> IdentityRecord:
        return IdentityRecord(
            identity_id=row["identity_id"],
            name=row["name"],
            created_at=row["created_at"],
            last_seen=row["last_seen"],
            embedding_count=row["embedding_count"],
            cluster_count=row["cluster_count"],
            model_version=row["model_version"],
            is_active=bool(row["is_active"]),
        )
