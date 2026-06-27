"""
Quality-weighted template aggregation module.

Aggregates multiple face embeddings into a single template embedding
using quality-weighted averaging, with adaptive per-template outlier
rejection. This is the single biggest lever for improving TAR@FAR on
multi-frame benchmarks like IJB-C.

Strategies:
  - quality_weighted: weight each embedding by its quality score
  - norm_weighted: weight by raw embedding norm (MagFace-style)
  - top_k: use only the K highest-quality embeddings
"""

from __future__ import annotations

import dataclasses
from typing import List, Optional, Tuple

import numpy as np

from facepipe.config.settings import get_settings, TemplateSettings
from facepipe.observability.logging import get_logger

logger = get_logger(__name__)


@dataclasses.dataclass(frozen=True)
class AggregatedTemplate:
    """Result of template aggregation.

    Attributes:
        embedding: L2-normalized aggregated template embedding.
        num_inputs: Number of embeddings provided.
        num_used: Number of embeddings used (after outlier rejection).
        num_rejected: Number of embeddings rejected as outliers.
        mean_quality: Mean quality score of used embeddings.
        strategy: Aggregation strategy used.
    """
    embedding: np.ndarray
    num_inputs: int
    num_used: int
    num_rejected: int
    mean_quality: float
    strategy: str


class TemplateAggregator:
    """Quality-weighted template aggregation with adaptive outlier rejection.

    Combines multiple embeddings (e.g., from multiple video frames of the
    same face) into a single, higher-quality template embedding. This is
    the #1 accuracy lever on multi-frame benchmarks like IJB-C.

    Args:
        settings: Template aggregation settings. If None, loaded from facepipe.config.
    """

    def __init__(self, settings: Optional[TemplateSettings] = None) -> None:
        self._settings = settings or get_settings().template

    def aggregate(
        self,
        embeddings: List[np.ndarray],
        quality_scores: Optional[List[float]] = None,
        raw_norms: Optional[List[float]] = None,
    ) -> AggregatedTemplate:
        """Aggregate multiple embeddings into a single template.

        Args:
            embeddings: List of L2-normalized 512-d embedding vectors.
            quality_scores: Per-embedding quality scores in [0, 1].
                            Required for quality_weighted strategy.
            raw_norms: Per-embedding raw norms (pre-normalization).
                       Required for norm_weighted strategy.

        Returns:
            AggregatedTemplate with the fused embedding.
        """
        if not embeddings:
            return AggregatedTemplate(
                embedding=np.zeros(512, dtype=np.float32),
                num_inputs=0, num_used=0, num_rejected=0,
                mean_quality=0.0, strategy=self._settings.strategy,
            )

        n = len(embeddings)
        emb_array = np.stack(embeddings).astype(np.float32)

        # Default quality scores
        if quality_scores is None:
            quality_scores = [1.0] * n
        if raw_norms is None:
            raw_norms = [1.0] * n

        # Step 1: Adaptive outlier rejection (per-template)
        emb_array, quality_scores, raw_norms, num_rejected = (
            self._reject_outliers(emb_array, quality_scores, raw_norms)
        )

        num_used = len(quality_scores)

        if num_used == 0:
            # All rejected — fall back to simple mean of originals
            emb_array = np.stack(embeddings).astype(np.float32)
            quality_scores = [1.0] * n
            raw_norms = [1.0] * n
            num_rejected = 0
            num_used = n

        # Step 2: Apply aggregation strategy
        strategy = self._settings.strategy

        if strategy == "quality_weighted":
            template = self._quality_weighted(emb_array, quality_scores)
        elif strategy == "norm_weighted":
            template = self._norm_weighted(emb_array, raw_norms)
        elif strategy == "top_k":
            template = self._top_k(emb_array, quality_scores)
        else:
            logger.warning("unknown_aggregation_strategy", strategy=strategy)
            template = self._quality_weighted(emb_array, quality_scores)

        # L2 normalize the result
        norm = np.linalg.norm(template)
        if norm > 0:
            template = template / norm

        return AggregatedTemplate(
            embedding=template.astype(np.float32),
            num_inputs=n,
            num_used=num_used,
            num_rejected=num_rejected,
            mean_quality=float(np.mean(quality_scores)),
            strategy=strategy,
        )

    def _reject_outliers(
        self,
        emb_array: np.ndarray,
        quality_scores: List[float],
        raw_norms: List[float],
    ) -> Tuple[np.ndarray, List[float], List[float], int]:
        """Adaptive per-template outlier rejection.

        Computes the centroid, then rejects embeddings whose similarity
        to the centroid falls below `mean_similarity - sigma * std`.
        This handles high-variance identities (glasses on/off, etc.)
        better than a fixed global threshold.
        """
        if len(emb_array) <= 2:
            # Not enough samples for meaningful outlier detection
            return emb_array, quality_scores, raw_norms, 0

        # Compute initial centroid (quality-weighted)
        weights = np.array(quality_scores, dtype=np.float32)
        weights = weights / (weights.sum() + 1e-8)
        centroid = np.average(emb_array, axis=0, weights=weights)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-8)

        # Compute similarity of each embedding to the centroid
        similarities = emb_array @ centroid

        # Adaptive threshold: mean - sigma * std
        mean_sim = float(np.mean(similarities))
        std_sim = float(np.std(similarities))
        threshold = mean_sim - self._settings.outlier_sigma * std_sim

        # Keep embeddings above threshold
        mask = similarities >= threshold
        num_rejected = int((~mask).sum())

        if num_rejected > 0:
            logger.debug(
                "template_outliers_rejected",
                rejected=num_rejected,
                threshold=f"{threshold:.4f}",
                mean_sim=f"{mean_sim:.4f}",
                std_sim=f"{std_sim:.4f}",
            )

        kept_embs = emb_array[mask]
        kept_quals = [q for q, m in zip(quality_scores, mask) if m]
        kept_norms = [n for n, m in zip(raw_norms, mask) if m]

        return kept_embs, kept_quals, kept_norms, num_rejected

    @staticmethod
    def _quality_weighted(emb_array: np.ndarray, quality_scores: List[float]) -> np.ndarray:
        """Quality-weighted average: Σ(q_i × e_i) / Σ(q_i)."""
        weights = np.array(quality_scores, dtype=np.float32)
        weights = weights / (weights.sum() + 1e-8)
        return np.average(emb_array, axis=0, weights=weights)

    @staticmethod
    def _norm_weighted(emb_array: np.ndarray, raw_norms: List[float]) -> np.ndarray:
        """MagFace-style: weight by raw embedding norm.

        Low-quality images produce lower-norm embeddings, so they
        naturally get less influence.
        """
        weights = np.array(raw_norms, dtype=np.float32)
        weights = weights / (weights.sum() + 1e-8)
        return np.average(emb_array, axis=0, weights=weights)

    def _top_k(self, emb_array: np.ndarray, quality_scores: List[float]) -> np.ndarray:
        """Use only the top-K highest-quality embeddings."""
        k = min(self._settings.top_k, len(quality_scores))
        indices = np.argsort(quality_scores)[-k:]
        selected = emb_array[indices]
        selected_quals = [quality_scores[i] for i in indices]

        # Quality-weighted average of the top-K
        return self._quality_weighted(selected, selected_quals)
