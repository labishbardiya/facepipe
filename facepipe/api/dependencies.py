"""
Shared FastAPI dependencies — dependency injection for pipeline, storage, etc.
"""

from __future__ import annotations

import functools
from typing import Generator

from facepipe.core.pipeline import RecognitionPipeline
from facepipe.storage.identity_manager import IdentityManager
from facepipe.storage.event_store import EventStore
from facepipe.storage.feature_store import FeatureStore
from facepipe.storage.encrypted_store import EncryptedEmbeddingStore


@functools.lru_cache(maxsize=1)
def get_pipeline() -> RecognitionPipeline:
    """Singleton recognition pipeline."""
    pipeline = RecognitionPipeline()
    pipeline.initialize()
    return pipeline


@functools.lru_cache(maxsize=1)
def get_identity_manager() -> IdentityManager:
    """Singleton identity manager."""
    return IdentityManager()


@functools.lru_cache(maxsize=1)
def get_event_store() -> EventStore:
    """Singleton event store."""
    return EventStore()


@functools.lru_cache(maxsize=1)
def get_feature_store() -> FeatureStore:
    """Singleton feature store."""
    return FeatureStore()


@functools.lru_cache(maxsize=1)
def get_encrypted_store() -> EncryptedEmbeddingStore:
    """Singleton encrypted embedding store."""
    return EncryptedEmbeddingStore()
