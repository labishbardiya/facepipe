"""
Per-identity appearance clustering module.

A person looks different with/without glasses, beard, mask, different
lighting conditions. Storing a single averaged embedding loses these
modes. This module clusters embeddings per identity to capture
appearance variants.

Each identity maintains multiple clusters, each with a centroid embedding.
Search queries are compared against all centroids; the best match across
clusters determines identity.
"""

from __future__ import annotations

import dataclasses

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

from facepipe.config.settings import ClusterSettings, get_settings
from facepipe.observability.logging import get_logger

logger = get_logger(__name__)


@dataclasses.dataclass
class EmbeddingCluster:
    """A cluster of similar face embeddings for one identity.

    Attributes:
        centroid: The cluster centroid (L2-normalized).
        member_count: Number of embeddings in this cluster.
        quality_mean: Mean quality score of member embeddings.
    """
    centroid: np.ndarray
    member_count: int
    quality_mean: float


@dataclasses.dataclass
class IdentityClusters:
    """All clusters for a single identity.

    Attributes:
        identity_id: The identity UUID.
        clusters: List of embedding clusters.
        all_embeddings: All raw embeddings (for re-clustering).
        all_qualities: Quality scores aligned with embeddings.
    """
    identity_id: str
    clusters: list[EmbeddingCluster]
    all_embeddings: list[np.ndarray]
    all_qualities: list[float]


class IdentityClusterEngine:
    """Manages per-identity appearance clusters.

    Args:
        settings: Clustering settings. If None, loaded from global config.
    """

    def __init__(self, settings: ClusterSettings | None = None) -> None:
        self._settings = settings or get_settings().clustering

    def compute_clusters(
        self,
        embeddings: list[np.ndarray],
        qualities: list[float] | None = None,
    ) -> list[EmbeddingCluster]:
        """Compute clusters from a list of embeddings.

        Automatically determines the optimal number of clusters using
        silhouette score analysis.

        Args:
            embeddings: List of L2-normalized embedding vectors.
            qualities: Optional quality scores aligned with embeddings.

        Returns:
            List of EmbeddingCluster objects.
        """
        if not embeddings:
            return []

        if qualities is None:
            qualities = [1.0] * len(embeddings)

        n = len(embeddings)
        emb_array = np.stack(embeddings).astype(np.float32)

        # If too few embeddings, each is its own cluster (or single cluster)
        if n <= self._settings.min_clusters:
            clusters = []
            for i, (emb, q) in enumerate(zip(embeddings, qualities)):
                centroid = emb / (np.linalg.norm(emb) + 1e-8)
                clusters.append(EmbeddingCluster(
                    centroid=centroid,
                    member_count=1,
                    quality_mean=q,
                ))
            return clusters

        # Find optimal k using silhouette score
        best_k = self._settings.min_clusters
        best_score = -1.0

        max_k = min(self._settings.max_clusters, n - 1)
        if max_k < 2:
            max_k = 2

        for k in range(self._settings.min_clusters, max_k + 1):
            try:
                clustering = AgglomerativeClustering(
                    n_clusters=k,
                    metric="cosine",
                    linkage="average",
                )
                labels = clustering.fit_predict(emb_array)
                if len(set(labels)) < 2:
                    continue
                score = silhouette_score(emb_array, labels, metric="cosine")
                if score > best_score:
                    best_score = score
                    best_k = k
            except Exception:
                continue

        # Apply best clustering
        clustering = AgglomerativeClustering(
            n_clusters=best_k,
            metric="cosine",
            linkage="average",
        )
        labels = clustering.fit_predict(emb_array)

        # Build clusters
        clusters: list[EmbeddingCluster] = []
        for cluster_id in range(best_k):
            mask = labels == cluster_id
            if not mask.any():
                continue

            cluster_embs = emb_array[mask]
            cluster_quals = [qualities[i] for i in range(n) if mask[i]]

            centroid = np.mean(cluster_embs, axis=0)
            centroid = centroid / (np.linalg.norm(centroid) + 1e-8)

            clusters.append(EmbeddingCluster(
                centroid=centroid.astype(np.float32),
                member_count=int(mask.sum()),
                quality_mean=float(np.mean(cluster_quals)),
            ))

        return clusters

    def add_embedding(
        self,
        existing: IdentityClusters,
        new_embedding: np.ndarray,
        quality: float = 1.0,
    ) -> IdentityClusters:
        """Add a new embedding to an identity's clusters.

        The embedding is assigned to the nearest existing cluster if close
        enough, otherwise a new cluster is created. If the max cluster count
        is exceeded, the two closest clusters are merged.

        Args:
            existing: Current identity clusters.
            new_embedding: New L2-normalized embedding to add.
            quality: Quality score of the new embedding.

        Returns:
            Updated IdentityClusters.
        """
        existing.all_embeddings.append(new_embedding)
        existing.all_qualities.append(quality)

        # Find nearest cluster
        best_cluster_idx = -1
        best_sim = -1.0

        for i, cluster in enumerate(existing.clusters):
            sim = float(np.dot(new_embedding.flatten(), cluster.centroid.flatten()))
            if sim > best_sim:
                best_sim = sim
                best_cluster_idx = i

        if best_sim >= self._settings.merge_threshold and best_cluster_idx >= 0:
            # Add to existing cluster, recompute centroid
            cluster = existing.clusters[best_cluster_idx]
            new_centroid = (
                cluster.centroid * cluster.member_count + new_embedding
            ) / (cluster.member_count + 1)
            new_centroid = new_centroid / (np.linalg.norm(new_centroid) + 1e-8)

            existing.clusters[best_cluster_idx] = EmbeddingCluster(
                centroid=new_centroid.astype(np.float32),
                member_count=cluster.member_count + 1,
                quality_mean=(cluster.quality_mean * cluster.member_count + quality)
                / (cluster.member_count + 1),
            )
        elif best_sim < self._settings.new_cluster_threshold or best_cluster_idx < 0:
            # Create new cluster
            centroid = new_embedding / (np.linalg.norm(new_embedding) + 1e-8)
            existing.clusters.append(EmbeddingCluster(
                centroid=centroid.astype(np.float32),
                member_count=1,
                quality_mean=quality,
            ))

            # Check if we exceeded max clusters → merge two closest
            if len(existing.clusters) > self._settings.max_clusters:
                self._merge_closest(existing.clusters)
        else:
            # Between thresholds: add to nearest cluster
            cluster = existing.clusters[best_cluster_idx]
            new_centroid = (
                cluster.centroid * cluster.member_count + new_embedding
            ) / (cluster.member_count + 1)
            new_centroid = new_centroid / (np.linalg.norm(new_centroid) + 1e-8)
            existing.clusters[best_cluster_idx] = EmbeddingCluster(
                centroid=new_centroid.astype(np.float32),
                member_count=cluster.member_count + 1,
                quality_mean=(cluster.quality_mean * cluster.member_count + quality)
                / (cluster.member_count + 1),
            )

        return existing

    def _merge_closest(self, clusters: list[EmbeddingCluster]) -> None:
        """Merge the two most similar clusters in-place."""
        if len(clusters) < 2:
            return

        best_i, best_j = 0, 1
        best_sim = -1.0

        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                sim = float(np.dot(
                    clusters[i].centroid.flatten(),
                    clusters[j].centroid.flatten(),
                ))
                if sim > best_sim:
                    best_sim = sim
                    best_i, best_j = i, j

        # Merge j into i
        ci, cj = clusters[best_i], clusters[best_j]
        total = ci.member_count + cj.member_count
        merged_centroid = (ci.centroid * ci.member_count + cj.centroid * cj.member_count) / total
        merged_centroid = merged_centroid / (np.linalg.norm(merged_centroid) + 1e-8)

        clusters[best_i] = EmbeddingCluster(
            centroid=merged_centroid.astype(np.float32),
            member_count=total,
            quality_mean=(ci.quality_mean * ci.member_count + cj.quality_mean * cj.member_count) / total,
        )
        clusters.pop(best_j)

    def get_all_centroids(
        self,
        identity_clusters: dict[str, IdentityClusters],
    ) -> tuple[list[str], np.ndarray]:
        """Extract all centroids from all identities for index building.

        Returns:
            (identity_ids, centroids) where identity_ids[i] is the identity
            that centroid[i] belongs to. Multiple centroids may have the
            same identity_id (one per cluster).
        """
        ids: list[str] = []
        centroids: list[np.ndarray] = []

        for identity_id, ic in identity_clusters.items():
            for cluster in ic.clusters:
                ids.append(identity_id)
                centroids.append(cluster.centroid)

        if not centroids:
            return [], np.zeros((0, 512), dtype=np.float32)

        return ids, np.stack(centroids).astype(np.float32)
