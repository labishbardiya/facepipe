"""
Abstract VectorStore protocol.

Defines the interface for vector storage backends. Start with FAISS HNSW;
swap to Milvus/Qdrant/Weaviate at 100M+ identities without changing
any pipeline code.
"""

from __future__ import annotations

import dataclasses
from typing import Protocol, runtime_checkable

import numpy as np


@dataclasses.dataclass(frozen=True)
class SearchResult:
    """A single search result from the vector store.

    Attributes:
        identity_id: The string identity ID.
        score: Similarity score (higher = more similar). For cosine similarity, in [-1, 1].
        faiss_id: Internal integer ID in the vector index.
    """
    identity_id: str
    score: float
    faiss_id: int


@runtime_checkable
class VectorStore(Protocol):
    """Protocol for vector storage backends.

    Any class implementing this protocol can be used as the search backend
    in the recognition pipeline. This abstraction enables swapping FAISS
    for Milvus, Qdrant, Weaviate, etc. without pipeline changes.
    """

    def add(self, ids: list[str], embeddings: np.ndarray) -> None:
        """Add embeddings to the index.

        Args:
            ids: List of string identity IDs.
            embeddings: Array of shape (N, dim) with L2-normalized embeddings.
        """
        ...

    def search(self, query: np.ndarray, k: int = 5) -> list[SearchResult]:
        """Search for the k nearest neighbors.

        Args:
            query: Query embedding of shape (1, dim) or (dim,), L2-normalized.
            k: Number of results to return.

        Returns:
            List of SearchResult objects, sorted by score descending.
        """
        ...

    def remove(self, ids: list[str]) -> None:
        """Remove embeddings by identity ID.

        Args:
            ids: List of identity IDs to remove.
        """
        ...

    def save(self, path: str) -> None:
        """Persist the index to disk.

        Args:
            path: Directory path to save index files.
        """
        ...

    def load(self, path: str) -> None:
        """Load the index from disk.

        Args:
            path: Directory path containing saved index files.
        """
        ...

    @property
    def size(self) -> int:
        """Return the number of embeddings in the index."""
        ...

    @property
    def dim(self) -> int:
        """Return the embedding dimensionality."""
        ...
