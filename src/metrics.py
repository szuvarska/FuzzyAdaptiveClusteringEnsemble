"""Set-correlation metrics used by ACE's Stage 2, for crisp and fuzzy cluster vectors."""
from abc import ABC, abstractmethod

import numpy as np


class BaseMetrics(ABC):
    @abstractmethod
    def cardinality(self, a: np.ndarray) -> float:
        """Size of a cluster."""

    @abstractmethod
    def overlap_matrix(self, A: np.ndarray, B: np.ndarray, size_a: np.ndarray, size_b: np.ndarray) -> np.ndarray:
        """Similarity between every column of A and every column of B."""

    def cardinalities(self, theta: np.ndarray) -> np.ndarray:
        return np.array([self.cardinality(theta[:, g]) for g in range(theta.shape[1])])

    def set_correlation_matrix(self, A: np.ndarray, B: np.ndarray, n: int,
                               size_a: np.ndarray = None, size_b: np.ndarray = None) -> np.ndarray:
        """
        (n * overlap - sqrt(|a||b|)) / sqrt((n - |a|)(n - |b|)) for every column pair (a in A, b in B);
        0 where a cluster is empty or contains all n objects. Cluster sizes can be passed in if known.
        """
        size_a = self.cardinalities(A) if size_a is None else np.asarray(size_a, dtype=float)
        size_b = self.cardinalities(B) if size_b is None else np.asarray(size_b, dtype=float)
        valid = ((size_a > 0) & (size_a < n))[:, None] & ((size_b > 0) & (size_b < n))[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            numerator = n * self.overlap_matrix(A, B, size_a, size_b) - np.sqrt(np.outer(size_a, size_b))
            denominator = np.sqrt(np.outer(n - size_a, n - size_b))
            scores = numerator / denominator
        return np.where(valid & (denominator > 0), scores, 0.0)

    def set_correlation(self, a: np.ndarray, b: np.ndarray, n: int) -> float:
        return float(self.set_correlation_matrix(a[:, None], b[:, None], n)[0, 0])


class BaselineMetrics(BaseMetrics):
    """Crisp clusters: count of members and cosine overlap of the indicator vectors."""

    def cardinality(self, a):
        return float(np.sum(a > 0))

    def overlap_matrix(self, A, B, size_a, size_b):
        return (A.T @ B) / np.sqrt(np.outer(size_a, size_b))


class FuzzyMetrics(BaseMetrics):
    """Fuzzy clusters: memberships capped at 1 (so merged sizes never exceed n), cosine overlap."""

    def cardinality(self, a):
        return float(np.sum(np.clip(a, 0.0, 1.0)))

    def overlap_matrix(self, A, B, size_a, size_b):
        A, B = np.clip(A, 0.0, 1.0), np.clip(B, 0.0, 1.0)
        return (A.T @ B) / np.outer(np.linalg.norm(A, axis=0), np.linalg.norm(B, axis=0))
