"""Base ensemble members: each turns a (sub)dataset into an n x k membership matrix."""
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np
import skfuzzy as fuzz
from sklearn.cluster import KMeans

from src.metrics import BaseMetrics, BaselineMetrics, FuzzyMetrics


class BaseMemberClusterer(ABC):
    @abstractmethod
    def get_membership_matrix(self, X_train: np.ndarray, X_full: np.ndarray, k: int,
                              random_state: int) -> np.ndarray:
        """Fits on X_train and returns the (n_full x k) membership matrix of X_full."""

    @abstractmethod
    def get_metrics_strategy(self) -> BaseMetrics:
        """Set metrics matching this member's membership type."""


class KMeansMember(BaseMemberClusterer):
    """Single K-Means run (n_init=1 for diversity) -> binary indicator matrix."""

    def get_membership_matrix(self, X_train, X_full, k, random_state):
        labels = KMeans(n_clusters=k, n_init=1, random_state=random_state).fit(X_train).predict(X_full)
        return np.column_stack([(labels == c).astype(float) for c in np.unique(labels)])

    def get_metrics_strategy(self):
        return BaselineMetrics()


class FCMMember(BaseMemberClusterer):
    """
    Fuzzy C-Means -> continuous membership matrix in [0, 1].
    If ``m_fuzz_range`` is given, the fuzzifier m is drawn uniformly from it for every partition.
    """

    def __init__(self, m_fuzz: float = 2.0, m_fuzz_range: Optional[Tuple[float, float]] = None):
        self.m_fuzz = m_fuzz
        self.m_fuzz_range = m_fuzz_range
        self.last_m_fuzz_ = m_fuzz

    def get_membership_matrix(self, X_train, X_full, k, random_state):
        if self.m_fuzz_range is not None:
            m = float(np.random.default_rng(random_state).uniform(*self.m_fuzz_range))
        else:
            m = self.m_fuzz
        self.last_m_fuzz_ = m

        np.random.seed(random_state)  # skfuzzy draws its initial partition from np.random
        centers, *_ = fuzz.cluster.cmeans(X_train.T, c=k, m=m, error=0.005, maxiter=1000, init=None)
        u_full, *_ = fuzz.cluster.cmeans_predict(X_full.T, centers, m=m, error=0.005, maxiter=1000)
        return u_full.T

    def get_metrics_strategy(self):
        return FuzzyMetrics()
