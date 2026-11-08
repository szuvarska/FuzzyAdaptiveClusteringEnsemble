import numpy as np
from abc import ABC, abstractmethod
from sklearn.cluster import KMeans
from src.metrics import BaseMetrics, BaselineMetrics, FuzzyMetrics
import skfuzzy as fuzz


class BaseMemberClusterer(ABC):
    """
    Abstract base interface for a base clustering model.
    Mandates how a model should output its local membership matrix.
    """

    @abstractmethod
    def get_membership_matrix(self, X: np.ndarray, k: int, random_state: int) -> np.ndarray:
        """Fits the algorithm on X and returns an (n x k) membership matrix."""
        pass

    @abstractmethod
    def get_metrics_strategy(self) -> 'BaseMetrics':
        """Factory method: Returns the correct metrics strategy for this base member."""
        pass


class KMeansMember(BaseMemberClusterer):
    """
    Baseline implementation for a single K-Means run.
    Outputs a discrete binary indicator matrix.
    """

    def get_membership_matrix(self, X: np.ndarray, k: int, random_state: int) -> np.ndarray:
        # n_init=1 because ensemble methods rely on varied weak initializations
        kmeans = KMeans(n_clusters=k, n_init=1, random_state=random_state)
        labels = kmeans.fit_predict(X)

        theta_cols = []
        unique_clusters = np.unique(labels)

        for cluster_id in unique_clusters:
            # Map crisp labels to binary column vectors
            binary_col = (labels == cluster_id).astype(float)
            theta_cols.append(binary_col)

        return np.column_stack(theta_cols)

    def get_metrics_strategy(self) -> 'BaseMetrics':
        return BaselineMetrics()


class FCMMember(BaseMemberClusterer):
    """
    Fuzzy C-Means base model.
    Outputs a continuous degree of membership matrix (u_ij in [0,1]).
    """

    def __init__(self, m_fuzz: float = 2.0):
        self.m_fuzz = m_fuzz

    def get_membership_matrix(self, X: np.ndarray, k: int, random_state: int) -> np.ndarray:
        np.random.seed(random_state)
        # skfuzzy cmeans expects data in shape (n_features, n_samples)
        X_T = X.T
        # Run fuzzy c-means
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            X_T, c=k, m=self.m_fuzz, error=0.005, maxiter=1000, init=None
        )
        # u is returned as (k_clusters, n_samples), transpose it to match (n x k)
        return u.T

    def get_metrics_strategy(self) -> 'BaseMetrics':
        return FuzzyMetrics()
