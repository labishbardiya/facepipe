"""
Adaptive score normalization for face recognition.

Implements Z-norm, T-norm, and ZT-norm to calibrate similarity scores
per-identity rather than relying on a single global threshold. This is
critical for TAR@FAR at tight operating points (1e-4, 1e-5) where raw
cosine similarity doesn't distribute uniformly across identities.

Used by top NIST FRVT performers but underused in open-source implementations.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from facepipe.config.settings import NormalizationSettings, get_settings
from facepipe.observability.logging import get_logger

logger = get_logger(__name__)


@dataclasses.dataclass
class NormStats:
    """Pre-computed normalization statistics for one identity.

    Attributes:
        identity_id: The identity these stats belong to.
        impostor_mean: Mean impostor similarity score.
        impostor_std: Std dev of impostor similarity scores.
    """
    identity_id: str
    impostor_mean: float
    impostor_std: float


class ScoreNormalizer:
    """Adaptive score normalization using Z-norm / T-norm / ZT-norm.

    Z-norm: Per-identity impostor statistics pre-computed at enrollment.
            At query time: z = (score - μ_impostor) / σ_impostor
    T-norm: Per-query impostor statistics computed at search time.
    ZT-norm: Apply Z-norm then T-norm sequentially.

    Args:
        settings: Normalization settings. If None, loaded from facepipe.config.
    """

    def __init__(self, settings: NormalizationSettings | None = None) -> None:
        self._settings = settings or get_settings().normalization

        # Pre-computed Z-norm stats per identity
        self._z_stats: dict[str, NormStats] = {}

        # Cohort embeddings used for computing impostor distributions
        self._cohort: np.ndarray | None = None

    @property
    def method(self) -> str:
        """Return the active normalization method."""
        return self._settings.method

    @property
    def is_calibrated(self) -> bool:
        """Return True if the normalizer has been calibrated."""
        return self._cohort is not None and len(self._z_stats) > 0

    def set_cohort(self, cohort_embeddings: np.ndarray) -> None:
        """Set the cohort embeddings used for impostor distributions.

        The cohort should be a diverse sample of ~200 embeddings from
        the enrollment database. Larger cohorts (500-1000) give
        diminishing returns.

        Args:
            cohort_embeddings: Array of shape (N, dim), L2-normalized.
        """
        self._cohort = cohort_embeddings.astype(np.float32)
        logger.info("score_normalizer_cohort_set", size=len(cohort_embeddings))

    def build_cohort_from_index(
        self,
        all_embeddings: dict[str, list[np.ndarray]],
    ) -> None:
        """Build a cohort by sampling diverse embeddings from the index.

        Selects up to `cohort_size` embeddings, sampling at most one
        per identity to maximize diversity.

        Args:
            all_embeddings: Dict mapping identity_id → list of embeddings.
        """
        target_size = self._settings.cohort_size
        sampled: list[np.ndarray] = []

        identity_ids = list(all_embeddings.keys())
        np.random.shuffle(identity_ids)

        for identity_id in identity_ids:
            if len(sampled) >= target_size:
                break
            embs = all_embeddings[identity_id]
            if embs:
                # Pick a random embedding from this identity
                idx = np.random.randint(0, len(embs))
                sampled.append(embs[idx])

        if sampled:
            self._cohort = np.stack(sampled).astype(np.float32)
            logger.info("cohort_built", size=len(sampled), identities=len(identity_ids))
        else:
            logger.warning("cohort_empty")

    def calibrate_identity(
        self,
        identity_id: str,
        identity_embeddings: list[np.ndarray],
    ) -> None:
        """Pre-compute Z-norm statistics for a single identity.

        Computes the impostor distribution by comparing the identity's
        embeddings against the cohort.

        Args:
            identity_id: The identity to calibrate.
            identity_embeddings: All embeddings for this identity.
        """
        if self._cohort is None or len(self._cohort) == 0:
            return

        # Compute similarity of each identity embedding against cohort
        impostor_scores: list[float] = []
        for emb in identity_embeddings:
            emb_flat = emb.flatten().astype(np.float32)
            sims = self._cohort @ emb_flat
            impostor_scores.extend(sims.tolist())

        if not impostor_scores:
            return

        self._z_stats[identity_id] = NormStats(
            identity_id=identity_id,
            impostor_mean=float(np.mean(impostor_scores)),
            impostor_std=max(float(np.std(impostor_scores)), 1e-8),
        )

    def calibrate_all(
        self,
        all_embeddings: dict[str, list[np.ndarray]],
    ) -> None:
        """Calibrate Z-norm stats for all enrolled identities.

        Args:
            all_embeddings: Dict mapping identity_id → list of embeddings.
        """
        for identity_id, embs in all_embeddings.items():
            self.calibrate_identity(identity_id, embs)

        logger.info("score_normalizer_calibrated", identities=len(self._z_stats))

    def normalize(
        self,
        raw_score: float,
        identity_id: str | None = None,
        query_embedding: np.ndarray | None = None,
    ) -> float:
        """Normalize a raw similarity score.

        Args:
            raw_score: Raw cosine similarity from vector search.
            identity_id: Target identity ID (required for Z-norm).
            query_embedding: Query embedding (required for T-norm).

        Returns:
            Calibrated score. Higher is still better.
        """
        method = self._settings.method

        if method == "none":
            return raw_score

        if method == "z_norm":
            return self._z_normalize(raw_score, identity_id)

        if method == "t_norm":
            return self._t_normalize(raw_score, query_embedding)

        if method == "zt_norm":
            # Apply Z-norm first, then T-norm
            z_score = self._z_normalize(raw_score, identity_id)
            return self._t_normalize(z_score, query_embedding)

        logger.warning("unknown_normalization_method", method=method)
        return raw_score

    def _z_normalize(self, raw_score: float, identity_id: str | None) -> float:
        """Z-norm: normalize using pre-computed per-identity impostor stats."""
        if identity_id is None or identity_id not in self._z_stats:
            return raw_score

        stats = self._z_stats[identity_id]
        return (raw_score - stats.impostor_mean) / stats.impostor_std

    def _t_normalize(self, raw_score: float, query_embedding: np.ndarray | None) -> float:
        """T-norm: normalize using per-query impostor distribution."""
        if query_embedding is None or self._cohort is None:
            return raw_score

        query_flat = query_embedding.flatten().astype(np.float32)
        cohort_scores = self._cohort @ query_flat

        mean_score = float(np.mean(cohort_scores))
        std_score = max(float(np.std(cohort_scores)), 1e-8)

        return (raw_score - mean_score) / std_score

    def normalize_search_results(
        self,
        results: list,
        query_embedding: np.ndarray | None = None,
    ) -> list:
        """Normalize scores for a list of search results in-place.

        Args:
            results: List of SearchResult objects (must have .score and .identity_id).
            query_embedding: Query embedding for T-norm.

        Returns:
            The same list with normalized scores.
        """
        for result in results:
            result.score = self.normalize(
                result.score,
                identity_id=result.identity_id,
                query_embedding=query_embedding,
            )
        return results
