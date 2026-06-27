"""
Multi-model ensemble recognizer.

Fuses embeddings from multiple recognition models (e.g., ArcFace R100 +
AdaFace IR101) for consistently higher accuracy than any single model.
Top IJB-C entries are always ensembles.

Fusion strategies:
  - concat_pca: Concatenate embeddings + PCA whitening before indexing
  - score_level: Search each model's index independently, average scores
  - quality_gated: Use AdaFace for low-quality inputs, ArcFace for high-quality
"""

from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional, Tuple

import numpy as np

from facepipe.config.settings import get_settings, EnsembleSettings
from facepipe.core.recognition.adaface_recognizer import AdaFaceRecognizer, EmbeddingResult
from facepipe.observability.logging import get_logger

logger = get_logger(__name__)


@dataclasses.dataclass(frozen=True)
class EnsembleEmbedding:
    """Result from ensemble embedding extraction.

    Attributes:
        embeddings: Dict mapping model_name → embedding.
        fused_embedding: The fused/concatenated embedding (if applicable).
        model_used: Which model was used (for quality_gated strategy).
        quality_score: Quality score that influenced model selection.
    """
    embeddings: Dict[str, np.ndarray]
    fused_embedding: np.ndarray
    model_used: str
    quality_score: Optional[float]


class EnsembleRecognizer:
    """Multi-model ensemble for face recognition.

    Wraps multiple recognizer instances and fuses their outputs.

    Args:
        settings: Ensemble settings. If None, loaded from facepipe.config.
        recognizers: Optional dict of name → recognizer instance.
    """

    def __init__(
        self,
        settings: Optional[EnsembleSettings] = None,
        recognizers: Optional[Dict[str, AdaFaceRecognizer]] = None,
    ) -> None:
        self._settings = settings or get_settings().ensemble
        self._recognizers = recognizers or {}
        self._pca_matrix: Optional[np.ndarray] = None
        self._pca_mean: Optional[np.ndarray] = None
        self._pca_fitted = False

    def add_recognizer(self, name: str, recognizer: AdaFaceRecognizer) -> None:
        """Register a recognizer in the ensemble."""
        self._recognizers[name] = recognizer
        logger.info("ensemble_recognizer_added", name=name)

    @property
    def model_count(self) -> int:
        """Number of models in the ensemble."""
        return len(self._recognizers)

    def extract(
        self,
        aligned_face: np.ndarray,
        quality_score: Optional[float] = None,
    ) -> EnsembleEmbedding:
        """Extract embeddings from all models and fuse.

        Args:
            aligned_face: Aligned face crop (112×112, BGR).
            quality_score: Face quality score for quality_gated strategy.

        Returns:
            EnsembleEmbedding with per-model and fused embeddings.
        """
        strategy = self._settings.fusion_strategy

        if strategy == "quality_gated":
            return self._quality_gated_extract(aligned_face, quality_score)

        # Extract from all models
        embeddings: Dict[str, np.ndarray] = {}
        for name, recognizer in self._recognizers.items():
            result = recognizer.extract(aligned_face)
            embeddings[name] = result.embedding

        if strategy == "concat_pca":
            fused = self._concat_pca(embeddings)
        elif strategy == "score_level":
            # For score-level fusion, the "fused" embedding is just the first model's
            # (actual fusion happens at search time by averaging scores)
            first_name = next(iter(embeddings))
            fused = embeddings[first_name]
        else:
            logger.warning("unknown_ensemble_strategy", strategy=strategy)
            first_name = next(iter(embeddings))
            fused = embeddings[first_name]

        return EnsembleEmbedding(
            embeddings=embeddings,
            fused_embedding=fused,
            model_used="ensemble",
            quality_score=quality_score,
        )

    def _quality_gated_extract(
        self,
        aligned_face: np.ndarray,
        quality_score: Optional[float],
    ) -> EnsembleEmbedding:
        """Select model based on input quality.

        AdaFace is specifically designed for low-quality / surveillance
        images. ArcFace performs better on high-quality inputs.
        """
        # Default threshold — ideally learned from a calibration set
        quality_threshold = 0.6

        if quality_score is not None and quality_score < quality_threshold:
            # Low quality → prefer AdaFace
            preferred = "adaface"
        else:
            # High quality → prefer ArcFace
            preferred = "arcface"

        # Fall back to whatever is available
        if preferred in self._recognizers:
            recognizer = self._recognizers[preferred]
            model_used = preferred
        else:
            name, recognizer = next(iter(self._recognizers.items()))
            model_used = name

        result = recognizer.extract(aligned_face)

        return EnsembleEmbedding(
            embeddings={model_used: result.embedding},
            fused_embedding=result.embedding,
            model_used=model_used,
            quality_score=quality_score,
        )

    def _concat_pca(self, embeddings: Dict[str, np.ndarray]) -> np.ndarray:
        """Concatenate embeddings from all models and apply PCA whitening.

        Without PCA whitening, the concatenated dimensions aren't equally
        informative, which degrades FAISS search quality.
        """
        # Concatenate in consistent order
        ordered = [embeddings[name] for name in sorted(embeddings.keys())]
        concat = np.concatenate(ordered).astype(np.float32)

        # Apply PCA whitening if fitted
        if self._pca_fitted and self._pca_matrix is not None and self._pca_mean is not None:
            centered = concat - self._pca_mean
            whitened = centered @ self._pca_matrix.T
            # L2 normalize
            norm = np.linalg.norm(whitened)
            if norm > 0:
                whitened = whitened / norm
            return whitened.astype(np.float32)

        # No PCA fitted yet — return raw concatenation (L2 normalized)
        norm = np.linalg.norm(concat)
        if norm > 0:
            concat = concat / norm
        return concat

    def fit_pca(
        self,
        training_embeddings: Dict[str, List[np.ndarray]],
        output_dim: Optional[int] = None,
    ) -> None:
        """Fit PCA whitening on a set of training embeddings.

        Should be called after enrollment with all enrolled embeddings.

        Args:
            training_embeddings: Dict mapping model_name → list of embeddings.
            output_dim: Output dimensionality. Defaults to 512 (half of 2×512).
        """
        # Build concatenated training matrix
        model_names = sorted(training_embeddings.keys())
        n_samples = min(len(embs) for embs in training_embeddings.values())

        if n_samples < 10:
            logger.warning("pca_too_few_samples", samples=n_samples)
            return

        concat_data = []
        for i in range(n_samples):
            row = np.concatenate([training_embeddings[name][i] for name in model_names])
            concat_data.append(row)

        X = np.stack(concat_data).astype(np.float64)
        total_dim = X.shape[1]

        if output_dim is None:
            output_dim = min(512, total_dim)

        # PCA whitening
        self._pca_mean = X.mean(axis=0).astype(np.float32)
        X_centered = X - self._pca_mean

        # SVD
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

        # Whitening transform: project + scale by 1/sqrt(eigenvalue)
        components = Vt[:output_dim]
        eigenvalues = (S[:output_dim] ** 2) / (n_samples - 1)
        whitening_scale = 1.0 / np.sqrt(eigenvalues + 1e-8)

        self._pca_matrix = (components * whitening_scale[:, np.newaxis]).astype(np.float32)
        self._pca_fitted = True

        logger.info(
            "pca_whitening_fitted",
            input_dim=total_dim,
            output_dim=output_dim,
            samples=n_samples,
        )

    def score_level_fuse(
        self,
        per_model_scores: Dict[str, float],
    ) -> float:
        """Fuse scores from multiple models by averaging.

        Args:
            per_model_scores: Dict mapping model_name → similarity score.

        Returns:
            Fused similarity score.
        """
        if not per_model_scores:
            return 0.0
        return float(np.mean(list(per_model_scores.values())))
