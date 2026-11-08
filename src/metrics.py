import numpy as np
from abc import ABC, abstractmethod


class BaseMetrics(ABC):
    """
    Abstract strategy for mathematical clustering metrics.
    Keeps the core set_correlation equation universal.
    """

    @abstractmethod
    def cardinality(self, a: np.ndarray) -> float:
        """Calculates the size/cardinality of a cluster."""
        pass

    @abstractmethod
    def overlap(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculates the similarity/overlap between two clusters."""
        pass

    def set_correlation(self, a: np.ndarray, b: np.ndarray, n: int) -> float:
        """
        Universal Set Correlation.
        Utilizes the abstract cardinality and overlap methods.
        """
        size_a = self.cardinality(a)
        size_b = self.cardinality(b)

        # Edge case protection: Avoid division by zero if a cluster is empty or encompasses all n points
        if size_a == 0 or size_b == 0 or size_a == n or size_b == n:
            return 0.0

        sim_overlap = self.overlap(a, b)

        numerator = (n * sim_overlap) - np.sqrt(size_a * size_b)
        denominator = np.sqrt((n - size_a) * (n - size_b))

        return numerator / denominator if denominator != 0 else 0.0


class BaselineMetrics(BaseMetrics):
    """
    Baseline metrics strategy using discrete binary representations.
    """

    def cardinality(self, a: np.ndarray) -> float:
        # Discrete count of objects
        return float(np.sum(a))

    def overlap(self, a: np.ndarray, b: np.ndarray) -> float:
        # Cosine Similarity for discrete vectors
        size_a = self.cardinality(a)
        size_b = self.cardinality(b)

        if size_a == 0 or size_b == 0:
            return 0.0

        return float(np.dot(a, b) / np.sqrt(size_a * size_b))
