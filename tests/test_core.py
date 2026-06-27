"""
Unit tests for core components that don't require model downloads.

These tests verify the logic of quality assessment, template aggregation,
score normalization, decision fusion, and clustering — using synthetic data
so they run instantly without GPU or model files.
"""

import numpy as np

# ──────────────────────────────────────────────────────────────
# Template Aggregator
# ──────────────────────────────────────────────────────────────


class TestTemplateAggregator:
    """Test quality-weighted template aggregation."""

    def _make_embeddings(self, n: int = 5, dim: int = 512) -> list[np.ndarray]:
        """Create n random L2-normalized embeddings."""
        embs = []
        for _ in range(n):
            e = np.random.randn(dim).astype(np.float32)
            e /= np.linalg.norm(e)
            embs.append(e)
        return embs

    def test_empty_input(self):
        from facepipe.core.recognition.template_aggregator import TemplateAggregator
        agg = TemplateAggregator()
        result = agg.aggregate([], [])
        assert result.num_inputs == 0
        assert result.num_used == 0
        assert np.linalg.norm(result.embedding) == 0.0

    def test_single_embedding(self):
        from facepipe.core.recognition.template_aggregator import TemplateAggregator
        agg = TemplateAggregator()
        embs = self._make_embeddings(1)
        result = agg.aggregate(embs, [0.9])
        assert result.num_inputs == 1
        assert result.num_used == 1
        assert abs(np.linalg.norm(result.embedding) - 1.0) < 1e-5

    def test_quality_weighted_favors_high_quality(self):
        from facepipe.core.recognition.template_aggregator import TemplateAggregator
        agg = TemplateAggregator()

        # Create one high-quality and one low-quality embedding (very different)
        e_high = np.zeros(512, dtype=np.float32)
        e_high[0] = 1.0
        e_low = np.zeros(512, dtype=np.float32)
        e_low[1] = 1.0

        result = agg.aggregate([e_high, e_low], [0.99, 0.01])
        # Result should be much closer to e_high
        assert result.embedding[0] > result.embedding[1]

    def test_outlier_rejection(self):
        from facepipe.core.recognition.template_aggregator import TemplateAggregator
        agg = TemplateAggregator()

        # 4 similar embeddings + 1 outlier
        base = np.random.randn(512).astype(np.float32)
        base /= np.linalg.norm(base)
        embs = [base + np.random.randn(512).astype(np.float32) * 0.01 for _ in range(4)]
        for i in range(len(embs)):
            embs[i] /= np.linalg.norm(embs[i])

        # Add outlier (completely different direction)
        outlier = -base
        outlier /= np.linalg.norm(outlier)
        embs.append(outlier)

        result = agg.aggregate(embs, [0.9] * 5)
        assert result.num_rejected >= 1

    def test_top_k_strategy(self):
        from facepipe.config.settings import TemplateSettings
        from facepipe.core.recognition.template_aggregator import TemplateAggregator
        settings = TemplateSettings(strategy="top_k", top_k=2)
        agg = TemplateAggregator(settings=settings)

        embs = self._make_embeddings(5)
        result = agg.aggregate(embs, [0.1, 0.5, 0.9, 0.3, 0.95])
        assert result.strategy == "top_k"
        assert result.num_inputs == 5


# ──────────────────────────────────────────────────────────────
# Score Normalizer
# ──────────────────────────────────────────────────────────────


class TestScoreNormalizer:
    """Test Z-norm / T-norm score normalization."""

    def test_z_norm_calibration(self):
        from facepipe.core.search.score_normalizer import NormStats, ScoreNormalizer
        norm = ScoreNormalizer()

        # Manually set stats
        norm._z_stats["alice"] = NormStats(
            identity_id="alice", impostor_mean=0.2, impostor_std=0.1,
        )

        # A score of 0.5 should normalize to (0.5 - 0.2) / 0.1 = 3.0
        calibrated = norm.normalize(0.5, identity_id="alice")
        assert abs(calibrated - 3.0) < 1e-5

    def test_no_norm_passthrough(self):
        from facepipe.config.settings import NormalizationSettings
        from facepipe.core.search.score_normalizer import ScoreNormalizer
        settings = NormalizationSettings(method="none")
        norm = ScoreNormalizer(settings=settings)
        assert norm.normalize(0.75) == 0.75

    def test_t_norm(self):
        from facepipe.core.search.score_normalizer import ScoreNormalizer
        norm = ScoreNormalizer()

        # Set a cohort
        cohort = np.random.randn(50, 512).astype(np.float32)
        for i in range(len(cohort)):
            cohort[i] /= np.linalg.norm(cohort[i])
        norm.set_cohort(cohort)

        query = np.random.randn(512).astype(np.float32)
        query /= np.linalg.norm(query)

        calibrated = norm._t_normalize(0.5, query)
        # Should be a finite number
        assert np.isfinite(calibrated)


# ──────────────────────────────────────────────────────────────
# Decision Fusion Engine
# ──────────────────────────────────────────────────────────────


class TestDecisionFusion:
    """Test the 7-signal decision fusion engine."""

    def test_high_confidence_recognized(self):
        from facepipe.core.fusion.decision_engine import DecisionFusionEngine
        engine = DecisionFusionEngine()

        result = engine.decide(
            recognition_score=0.95,
            detection_score=0.99,
            quality_score=0.90,
            liveness_score=0.95,
            tracking_consistency=1.0,
            openset_margin=0.3,
            deepfake_score=0.98,
            identity="alice",
            openset_decision="recognized",
        )
        assert result.is_recognized
        assert result.identity == "alice"
        assert result.confidence > 0.8

    def test_unknown_face_not_recognized(self):
        from facepipe.core.fusion.decision_engine import DecisionFusionEngine
        engine = DecisionFusionEngine()

        result = engine.decide(
            recognition_score=0.1,
            detection_score=0.99,
            quality_score=0.90,
            openset_decision="unknown",
        )
        assert not result.is_recognized
        assert result.identity is None

    def test_ambiguous_overrides_high_score(self):
        from facepipe.core.fusion.decision_engine import DecisionFusionEngine
        engine = DecisionFusionEngine()

        result = engine.decide(
            recognition_score=0.99,
            detection_score=0.99,
            quality_score=0.99,
            liveness_score=0.99,
            identity="alice",
            openset_decision="ambiguous",
        )
        # Ambiguous should override even with high scores
        assert not result.is_recognized


# ──────────────────────────────────────────────────────────────
# Face Restoration
# ──────────────────────────────────────────────────────────────


class TestFaceRestoration:
    """Test quality-gated face restoration."""

    def test_high_quality_skips_restoration(self):
        from facepipe.core.quality.face_restoration import FaceRestorer
        restorer = FaceRestorer()
        dummy = np.zeros((112, 112, 3), dtype=np.uint8)
        result = restorer.restore(dummy, quality_score=0.8)
        assert not result.was_restored
        assert result.method == "none"

    def test_low_quality_triggers_fallback(self):
        from facepipe.core.quality.face_restoration import FaceRestorer
        restorer = FaceRestorer()
        dummy = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
        result = restorer.restore(dummy, quality_score=0.2)
        assert result.was_restored
        assert result.method == "fallback_opencv"

    def test_restored_shape_matches_input(self):
        from facepipe.core.quality.face_restoration import FaceRestorer
        restorer = FaceRestorer()
        dummy = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
        result = restorer.restore(dummy, quality_score=0.2)
        assert result.restored.shape == dummy.shape


# ──────────────────────────────────────────────────────────────
# Identity Clustering
# ──────────────────────────────────────────────────────────────


class TestIdentityClustering:
    """Test per-identity appearance clustering."""

    def test_few_embeddings_returns_individual_clusters(self):
        from facepipe.core.clustering.identity_cluster import IdentityClusterEngine
        engine = IdentityClusterEngine()

        emb = np.random.randn(512).astype(np.float32)
        emb /= np.linalg.norm(emb)

        clusters = engine.compute_clusters([emb])
        assert len(clusters) == 1
        assert clusters[0].member_count == 1

    def test_similar_embeddings_cluster_together(self):
        from facepipe.core.clustering.identity_cluster import IdentityClusterEngine
        engine = IdentityClusterEngine()

        base = np.random.randn(512).astype(np.float32)
        base /= np.linalg.norm(base)

        # 5 very similar embeddings
        embs = []
        for _ in range(5):
            e = base + np.random.randn(512).astype(np.float32) * 0.01
            e /= np.linalg.norm(e)
            embs.append(e)

        clusters = engine.compute_clusters(embs)
        # Should cluster into few groups since they're very similar
        assert len(clusters) <= 3
