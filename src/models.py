import numpy as np
from abc import ABC, abstractmethod
from sklearn.cluster import KMeans


class BaseMemberClusterer(ABC):
    """
    Abstract base interface for a base clustering model.
    Mandates how a model should output its local membership matrix.
    """

    @abstractmethod
    def get_membership_matrix(self, X: np.ndarray, k: int, random_state: int) -> np.ndarray:
        """Fits the algorithm on X and returns an (n x k) membership matrix."""
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
