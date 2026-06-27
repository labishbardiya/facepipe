"""
Event sourcing store — append-only, tamper-evident event log.

Every state change becomes an immutable event. The current state can be
reconstructed by replaying events. Enables: debugging by replaying exact
sequences, auditing who/what/when, and rolling back active learning mistakes.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import sqlite3
import time
from pathlib import Path
from typing import Any

import ulid

from facepipe.config.settings import get_settings
from facepipe.observability.logging import get_logger

logger = get_logger(__name__)


class EventType:
    """Event type constants."""
    IDENTITY_ENROLLED = "IDENTITY_ENROLLED"
    IDENTITY_DELETED = "IDENTITY_DELETED"
    IDENTITY_UPDATED = "IDENTITY_UPDATED"
    EMBEDDING_ADDED = "EMBEDDING_ADDED"
    EMBEDDING_REMOVED = "EMBEDDING_REMOVED"
    RECOGNITION_ATTEMPT = "RECOGNITION_ATTEMPT"
    RECOGNITION_VERIFIED = "RECOGNITION_VERIFIED"
    SPOOF_DETECTED = "SPOOF_DETECTED"
    DEEPFAKE_DETECTED = "DEEPFAKE_DETECTED"
    CLUSTER_RECOMPUTED = "CLUSTER_RECOMPUTED"
    ACTIVE_LEARNING_ADD = "ACTIVE_LEARNING_ADD"
    ACTIVE_LEARNING_VERIFY = "ACTIVE_LEARNING_VERIFY"
    SYSTEM_EVENT = "SYSTEM_EVENT"


class EventStore:
    """Append-only event store with HMAC chain for tamper detection.

    Each event includes an HMAC computed over the event data + previous
    event's HMAC, forming a hash chain. Any modification to historical
    events breaks the chain.

    Args:
        db_path: Path to the SQLite database file.
        hmac_key: Secret key for HMAC computation. If None, derived from settings.
    """

    def __init__(
        self,
        db_path: str | None = None,
        hmac_key: bytes | None = None,
    ) -> None:
        if db_path is None:
            db_path = str(get_settings().data_dir / "events.db")

        self._db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        if hmac_key is None:
            # Derive from encryption key or use default
            key_str = get_settings().storage.encryption_key
            if key_str:
                import base64
                self._hmac_key = base64.b64decode(key_str)[:32]
            else:
                self._hmac_key = b"default-dev-hmac-key-not-for-prod"
        else:
            self._hmac_key = hmac_key

        self._init_db()

    def _init_db(self) -> None:
        """Initialize the event store schema."""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY,
                    timestamp REAL NOT NULL,
                    event_type TEXT NOT NULL,
                    identity_id TEXT,
                    payload TEXT NOT NULL,
                    image_hash TEXT,
                    source_ip TEXT,
                    hmac_value TEXT NOT NULL,
                    previous_hmac TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_event_type ON events(event_type)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_identity ON events(identity_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp ON events(timestamp)
            """)
            conn.commit()

    def append(
        self,
        event_type: str,
        identity_id: str | None = None,
        payload: dict[str, Any] | None = None,
        image_hash: str | None = None,
        source_ip: str | None = None,
    ) -> str:
        """Append a new event to the store.

        Args:
            event_type: One of the EventType constants.
            identity_id: Related identity ID (if applicable).
            payload: Event-specific data as a dictionary.
            image_hash: SHA-256 hash of input image (not the image itself).
            source_ip: Source IP address of the request.

        Returns:
            The event ID.
        """
        event_id = str(ulid.new())
        timestamp = time.time()
        payload_json = json.dumps(payload or {}, default=str)

        # Get previous HMAC for chain
        previous_hmac = self._get_last_hmac()

        # Compute HMAC over event data + previous HMAC
        hmac_data = f"{event_id}:{timestamp}:{event_type}:{identity_id}:{payload_json}:{previous_hmac}"
        hmac_value = hmac.new(
            self._hmac_key,
            hmac_data.encode(),
            hashlib.sha256,
        ).hexdigest()

        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """INSERT INTO events
                   (event_id, timestamp, event_type, identity_id, payload,
                    image_hash, source_ip, hmac_value, previous_hmac)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    event_id, timestamp, event_type, identity_id,
                    payload_json, image_hash, source_ip,
                    hmac_value, previous_hmac,
                ),
            )
            conn.commit()

        logger.debug(
            "event_appended",
            event_id=event_id,
            event_type=event_type,
            identity_id=identity_id,
        )
        return event_id

    def query(
        self,
        event_type: str | None = None,
        identity_id: str | None = None,
        since: float | None = None,
        until: float | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Query events with filters.

        Args:
            event_type: Filter by event type.
            identity_id: Filter by identity.
            since: Unix timestamp lower bound.
            until: Unix timestamp upper bound.
            limit: Maximum results.

        Returns:
            List of event dictionaries.
        """
        conditions = []
        params: list[Any] = []

        if event_type:
            conditions.append("event_type = ?")
            params.append(event_type)
        if identity_id:
            conditions.append("identity_id = ?")
            params.append(identity_id)
        if since is not None:
            conditions.append("timestamp >= ?")
            params.append(since)
        if until is not None:
            conditions.append("timestamp <= ?")
            params.append(until)

        where = " AND ".join(conditions) if conditions else "1=1"
        params.append(limit)

        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                f"SELECT * FROM events WHERE {where} ORDER BY timestamp DESC LIMIT ?",
                params,
            ).fetchall()

        return [dict(row) for row in rows]

    def verify_chain(self, last_n: int = 100) -> bool:
        """Verify the HMAC chain for the last N events.

        Returns True if the chain is intact (no tampering detected).
        """
        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM events ORDER BY timestamp ASC LIMIT ?",
                (last_n,),
            ).fetchall()

        if not rows:
            return True

        for i, row in enumerate(rows):
            expected_prev = rows[i - 1]["hmac_value"] if i > 0 else None

            if row["previous_hmac"] != expected_prev:
                logger.error("hmac_chain_broken", event_id=row["event_id"], index=i)
                return False

            # Recompute HMAC
            hmac_data = (
                f"{row['event_id']}:{row['timestamp']}:{row['event_type']}:"
                f"{row['identity_id']}:{row['payload']}:{row['previous_hmac']}"
            )
            expected_hmac = hmac.new(
                self._hmac_key,
                hmac_data.encode(),
                hashlib.sha256,
            ).hexdigest()

            if row["hmac_value"] != expected_hmac:
                logger.error("hmac_mismatch", event_id=row["event_id"])
                return False

        return True

    def count(self, event_type: str | None = None) -> int:
        """Count events, optionally filtered by type."""
        with sqlite3.connect(self._db_path) as conn:
            if event_type:
                row = conn.execute(
                    "SELECT COUNT(*) FROM events WHERE event_type = ?",
                    (event_type,),
                ).fetchone()
            else:
                row = conn.execute("SELECT COUNT(*) FROM events").fetchone()
            return row[0] if row else 0

    def prune(self, older_than_days: int | None = None) -> int:
        """Remove events older than the retention period.

        Args:
            older_than_days: Days to retain. If None, uses config value.

        Returns:
            Number of events pruned.
        """
        if older_than_days is None:
            older_than_days = get_settings().storage.event_retention_days

        cutoff = time.time() - (older_than_days * 86400)

        with sqlite3.connect(self._db_path) as conn:
            cursor = conn.execute(
                "DELETE FROM events WHERE timestamp < ?",
                (cutoff,),
            )
            conn.commit()
            pruned = cursor.rowcount

        if pruned > 0:
            logger.info("events_pruned", count=pruned, older_than_days=older_than_days)

        return pruned

    def _get_last_hmac(self) -> str | None:
        """Get the HMAC of the last event in the chain."""
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute(
                "SELECT hmac_value FROM events ORDER BY rowid DESC LIMIT 1"
            ).fetchone()
        return row[0] if row else None
