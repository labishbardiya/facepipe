"""
Migration script: v1 (pickle + IndexFlatIP) → v2 (encrypted + HNSW + SQLite).

Reads the existing db/ directory files:
  - raw_embeddings.pkl → encrypts and stores per identity
  - labels.pkl → creates identity records
  - faces.index → rebuilt as HNSW

The old db/ directory is preserved as backup.
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np

from facepipe.config.settings import get_settings
from facepipe.observability.logging import get_logger

logger = get_logger(__name__)


def migrate_v1_to_v2() -> dict:
    """Run the v1 → v2 migration.

    Returns:
        Migration summary with counts.
    """
    settings = get_settings()
    legacy_dir = Path(settings.legacy_db_dir)

    raw_path = legacy_dir / "raw_embeddings.pkl"
    labels_path = legacy_dir / "labels.pkl"

    if not raw_path.exists():
        logger.warning("no_v1_data_found", path=str(raw_path))
        return {"status": "skipped", "reason": "No v1 data found"}

    # Load v1 data
    logger.info("loading_v1_data")

    with open(raw_path, "rb") as f:
        raw_embeddings: dict[str, list[np.ndarray]] = pickle.load(f)

    labels: list[str] = []
    if labels_path.exists():
        with open(labels_path, "rb") as f:
            labels = pickle.load(f)

    logger.info("v1_data_loaded", identities=len(raw_embeddings), labels=len(labels))

    # Import v2 modules
    from facepipe.core.clustering.identity_cluster import IdentityClusterEngine
    from facepipe.core.search.faiss_store import FAISSStore
    from facepipe.storage.encrypted_store import EncryptedEmbeddingStore
    from facepipe.storage.event_store import EventStore, EventType
    from facepipe.storage.feature_store import EmbeddingRecord, FeatureStore
    from facepipe.storage.identity_manager import IdentityManager

    # Initialize v2 components
    enc_store = EncryptedEmbeddingStore()
    identity_mgr = IdentityManager()
    event_store = EventStore()
    feature_store = FeatureStore()
    vector_store = FAISSStore(dim=512)
    cluster_engine = IdentityClusterEngine()

    migrated = 0
    total_embeddings = 0

    for name, embeddings in raw_embeddings.items():
        if not embeddings:
            continue

        import ulid
        identity_id = str(ulid.new())

        # Create identity record
        identity_mgr.create(
            name=name,
            embedding_count=len(embeddings),
            model_version="arcface_r100_buffalo_l_v1",
            identity_id=identity_id,
        )

        # Encrypt and store embeddings
        emb_arrays = [e.astype(np.float32) for e in embeddings]
        enc_store.save_identity_embeddings(
            identity_id, emb_arrays,
            model_version="arcface_r100_buffalo_l_v1",
        )

        # Create feature store records
        for i, emb in enumerate(emb_arrays):
            feature_store.add_record(EmbeddingRecord(
                record_id=str(ulid.new()),
                identity_id=identity_id,
                model_version="arcface_r100_buffalo_l_v1",
                quality_score=0.7,  # Unknown quality, assign default
                yaw=0.0,
                pitch=0.0,
                camera_id="v1_migration",
                timestamp=time.time(),
                cluster_id=0,
                is_centroid=False,
                source="migration",
            ))

        # Build clusters
        clusters = cluster_engine.compute_clusters(emb_arrays)

        # Add centroids to vector store
        for cluster in clusters:
            vector_store.add([identity_id], cluster.centroid.reshape(1, -1))

        # Update identity with cluster count
        identity_mgr.update(
            identity_id,
            cluster_count=len(clusters),
        )

        # Emit migration event
        event_store.append(
            EventType.IDENTITY_ENROLLED,
            identity_id=identity_id,
            payload={
                "name": name,
                "embedding_count": len(embeddings),
                "cluster_count": len(clusters),
                "source": "v1_migration",
            },
        )

        migrated += 1
        total_embeddings += len(embeddings)
        logger.info(
            "identity_migrated",
            name=name,
            identity_id=identity_id,
            embeddings=len(embeddings),
            clusters=len(clusters),
        )

    # Save vector store
    index_path = str(settings.data_dir / "index")
    vector_store.save(index_path)

    summary = {
        "status": "complete",
        "identities_migrated": migrated,
        "total_embeddings": total_embeddings,
        "index_size": vector_store.size,
    }

    logger.info("migration_complete", **summary)
    return summary


if __name__ == "__main__":
    from facepipe.observability.logging import setup_logging
    setup_logging(level="INFO", json_output=False)
    result = migrate_v1_to_v2()
    print(f"\nMigration result: {result}")
