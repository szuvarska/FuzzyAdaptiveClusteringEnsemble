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
    def get_membership_matrix(self, X_train: np.ndarray, X_full: np.ndarray, k: int, random_state: int) -> np.ndarray:
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

    def get_membership_matrix(self, X_train: np.ndarray, X_full: np.ndarray, k: int, random_state: int) -> np.ndarray:
        # n_init=1 because ensemble methods rely on varied weak initializations
        kmeans = KMeans(n_clusters=k, n_init=1, random_state=random_state)
        kmeans.fit(X_train)
        labels = kmeans.predict(X_full)

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

    def __init__(self, m_fuzz: float = 2.0, m_fuzz_range: tuple = None):
        """
        m_fuzz: fixed fuzzification parameter.
        m_fuzz_range: optional (low, high). If given, a fresh m is drawn uniformly from this interval
                      for every generated partition (plan: [1.5, 2.5]) to inject structural diversity.
        """
        self.m_fuzz = m_fuzz
        self.m_fuzz_range = m_fuzz_range
        self.last_m_fuzz_ = m_fuzz

    def get_membership_matrix(self, X_train: np.ndarray, X_full: np.ndarray, k: int, random_state: int) -> np.ndarray:
        if self.m_fuzz_range is not None:
            low, high = self.m_fuzz_range
            m_fuzz = float(np.random.default_rng(random_state).uniform(low, high))
        else:
            m_fuzz = self.m_fuzz
        self.last_m_fuzz_ = m_fuzz

        np.random.seed(random_state)
        # skfuzzy cmeans expects data in shape (n_features, n_samples)
        X_train_T = X_train.T
        # Run fuzzy c-means on the training subset
        cntr, u_train, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            X_train_T, c=k, m=m_fuzz, error=0.005, maxiter=1000, init=None
        )

        # Predict continuous memberships for the full dataset using the discovered cluster centers
        X_full_T = X_full.T
        u_full, u0_full, d_full, jm_full, p_full, fpc_full = fuzz.cluster.cmeans_predict(
            X_full_T, cntr, m=m_fuzz, error=0.005, maxiter=1000
        )

        # u_full is returned as (k_clusters, n_samples), transpose it to match (n x k)
        return u_full.T

    def get_metrics_strategy(self) -> 'BaseMetrics':
        return FuzzyMetrics()
