"""
Active learning module.

Confidence-gated continuous improvement of identity representations.
Automatically adds high-confidence embeddings, flags medium-confidence
for verification, and discards low-confidence matches.

Safeguards:
  - Rate limiting per identity per hour
  - Novelty check (only add sufficiently different embeddings)
  - Quality gate (only add above quality threshold)
  - Rollback via event store
"""

from __future__ import annotations

import dataclasses
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from facepipe.config.settings import get_settings, ActiveLearningSettings
from facepipe.observability.logging import get_logger

logger = get_logger(__name__)


@dataclasses.dataclass(frozen=True)
class LearningDecision:
    """Decision from the active learning gate.

    Attributes:
        action: One of "auto_add", "verify", "discard".
        identity_id: The identity to potentially update.
        confidence: The fusion confidence that triggered this decision.
        embedding: The embedding to potentially add.
        reason: Explanation for the decision.
        is_novel: Whether the embedding is sufficiently different from existing.
    """
    action: str
    identity_id: Optional[str]
    confidence: float
    embedding: Optional[np.ndarray]
    reason: str
    is_novel: bool


class ActiveLearningGate:
    """Confidence-gated active learning controller.

    Decides whether to automatically add new embeddings to an identity's
    representation based on recognition confidence, novelty, quality,
    and rate limits.

    Args:
        settings: Active learning settings. If None, loaded from global config.
    """

    def __init__(self, settings: Optional[ActiveLearningSettings] = None) -> None:
        self._settings = settings or get_settings().active_learning
        # Rate limiting: identity_id → list of timestamps of recent auto-adds
        self._rate_tracker: Dict[str, List[float]] = defaultdict(list)

    def evaluate(
        self,
        identity_id: Optional[str],
        confidence: float,
        embedding: Optional[np.ndarray] = None,
        existing_centroids: Optional[List[np.ndarray]] = None,
        quality_score: float = 1.0,
    ) -> LearningDecision:
        """Evaluate whether to learn from this recognition result.

        Args:
            identity_id: The recognized identity (None if unknown).
            confidence: Fusion confidence score.
            embedding: The face embedding to potentially store.
            existing_centroids: Current cluster centroids for novelty check.
            quality_score: Quality score of the face.

        Returns:
            LearningDecision with action and reasoning.
        """
        if identity_id is None or embedding is None:
            return LearningDecision(
                action="discard",
                identity_id=identity_id,
                confidence=confidence,
                embedding=None,
                reason="No identity or embedding available.",
                is_novel=False,
            )

        # Check confidence zones
        if confidence < self._settings.verify_threshold:
            return LearningDecision(
                action="discard",
                identity_id=identity_id,
                confidence=confidence,
                embedding=None,
                reason=f"Confidence {confidence:.3f} below verify threshold {self._settings.verify_threshold:.3f}.",
                is_novel=False,
            )

        if confidence < self._settings.auto_add_threshold:
            return LearningDecision(
                action="verify",
                identity_id=identity_id,
                confidence=confidence,
                embedding=embedding,
                reason=f"Confidence {confidence:.3f} in verify zone [{self._settings.verify_threshold:.3f}, {self._settings.auto_add_threshold:.3f}).",
                is_novel=True,
            )

        # High confidence — check safeguards before auto-add

        # 1. Quality gate
        if quality_score < 0.5:
            return LearningDecision(
                action="discard",
                identity_id=identity_id,
                confidence=confidence,
                embedding=None,
                reason=f"Quality score {quality_score:.3f} too low for auto-add.",
                is_novel=False,
            )

        # 2. Rate limiting
        if not self._check_rate_limit(identity_id):
            return LearningDecision(
                action="discard",
                identity_id=identity_id,
                confidence=confidence,
                embedding=None,
                reason=f"Rate limit exceeded for {identity_id} ({self._settings.max_auto_adds_per_hour}/hr).",
                is_novel=False,
            )

        # 3. Novelty check
        is_novel = self._check_novelty(embedding, existing_centroids)
        if not is_novel:
            return LearningDecision(
                action="discard",
                identity_id=identity_id,
                confidence=confidence,
                embedding=None,
                reason="Embedding too similar to existing centroids (not novel).",
                is_novel=False,
            )

        # All checks passed — auto-add
        self._record_auto_add(identity_id)

        return LearningDecision(
            action="auto_add",
            identity_id=identity_id,
            confidence=confidence,
            embedding=embedding,
            reason=f"Auto-adding: confidence={confidence:.3f}, novel, rate OK.",
            is_novel=True,
        )

    def _check_rate_limit(self, identity_id: str) -> bool:
        """Check if we've exceeded the auto-add rate limit for this identity."""
        now = time.time()
        one_hour_ago = now - 3600.0

        # Clean old entries
        self._rate_tracker[identity_id] = [
            t for t in self._rate_tracker[identity_id] if t > one_hour_ago
        ]

        return len(self._rate_tracker[identity_id]) < self._settings.max_auto_adds_per_hour

    def _record_auto_add(self, identity_id: str) -> None:
        """Record an auto-add event for rate limiting."""
        self._rate_tracker[identity_id].append(time.time())

    def _check_novelty(
        self,
        embedding: np.ndarray,
        existing_centroids: Optional[List[np.ndarray]],
    ) -> bool:
        """Check if the embedding is sufficiently different from existing centroids.

        Returns True if the embedding provides new information (is novel).
        """
        if not existing_centroids:
            return True  # No existing data — always novel

        emb_flat = embedding.flatten()

        for centroid in existing_centroids:
            sim = float(np.dot(emb_flat, centroid.flatten()))
            if sim > (1.0 - self._settings.novelty_threshold):
                return False  # Too similar to an existing centroid

        return True

    def get_rate_stats(self) -> Dict[str, int]:
        """Return current rate limiting stats per identity."""
        now = time.time()
        one_hour_ago = now - 3600.0
        return {
            identity_id: len([t for t in timestamps if t > one_hour_ago])
            for identity_id, timestamps in self._rate_tracker.items()
        }
