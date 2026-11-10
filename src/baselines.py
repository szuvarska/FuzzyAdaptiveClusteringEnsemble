"""
Comparison methods benchmarked against Fuzzy-ACE (plan, section 5.1):

* ``KMeansClusterer``  - a single standard K-Means run.
* ``FCMClusterer``     - a single Fuzzy C-Means run, hardened with argmax over memberships.
* ``EACClusterer``     - Evidence Accumulation Clustering (Fred & Jain, 2005): builds the
                         co-association matrix from the *same* K-Means ensemble that ACE uses and
                         extracts the consensus with average-linkage hierarchical clustering
                         (the variant used as "CO" by Alqurashi & Wang, 2019).

Every class exposes ``fit_predict(X) -> labels`` like ``AdaptiveClusteringEnsemble``.
"""
import numpy as np
import skfuzzy as fuzz
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans

from src.ensemble import AdaptiveClusteringEnsemble
from src.models import KMeansMember


class KMeansClusterer:
    """Single K-Means run (one random initialisation by default, i.e. a 'single run')."""

    def __init__(self, k_clusters: int, n_init: int = 1, random_state: int = 42):
        self.k_clusters = k_clusters
        self.n_init = n_init
        self.random_state = random_state

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        model = KMeans(n_clusters=self.k_clusters, n_init=self.n_init, random_state=self.random_state)
        return model.fit_predict(X)


class FCMClusterer:
    """Single Fuzzy C-Means run; crisp labels are the cluster of maximum membership."""

    def __init__(self, k_clusters: int, m_fuzz: float = 2.0, error: float = 0.005,
                 maxiter: int = 1000, random_state: int = 42):
        self.k_clusters = k_clusters
        self.m_fuzz = m_fuzz
        self.error = error
        self.maxiter = maxiter
        self.random_state = random_state

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        np.random.seed(self.random_state)  # skfuzzy draws its initial partition from np.random
        _, u, *_ = fuzz.cluster.cmeans(X.T, c=self.k_clusters, m=self.m_fuzz,
                                       error=self.error, maxiter=self.maxiter, init=None)
        self.membership_ = u.T
        return np.argmax(u, axis=0)


class EACClusterer:
    """
    Evidence Accumulation Clustering.

    The ensemble is generated exactly like ACE's Stage 1 (same K-Means members, same 70% instance /
    70% feature sampling, same fixed or variable k), so the only difference between EAC and ACE is the
    consensus function. The co-association matrix C[i, j] = fraction of members that put i and j in the
    same cluster; 1 - C is used as a distance for hierarchical clustering cut at k clusters.
    Note: O(n^2) memory, which is the bottleneck ACE avoids.
    """

    def __init__(self, k_clusters: int, m_partitions: int = 10, variable_k: bool = False,
                 sampling_ratio: float = 0.7, linkage_method: str = "average", random_state: int = 42):
        self.k_clusters = k_clusters
        self.m_partitions = m_partitions
        self.variable_k = variable_k
        self.sampling_ratio = sampling_ratio
        self.linkage_method = linkage_method
        self.random_state = random_state

    def _generate_ensemble(self, X: np.ndarray) -> np.ndarray:
        generator = AdaptiveClusteringEnsemble(
            base_member=KMeansMember(), m_partitions=self.m_partitions, k_clusters=self.k_clusters,
            variable_k=self.variable_k, sampling_ratio=self.sampling_ratio, random_state=self.random_state)
        return generator.stage1_transformation(X)  # binary n x km matrix

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        theta = self._generate_ensemble(X)
        co_association = (theta @ theta.T) / self.m_partitions
        np.fill_diagonal(co_association, 1.0)
        distance = np.clip(1.0 - co_association, 0.0, None)
        self.co_association_ = co_association

        Z = linkage(squareform(distance, checks=False), method=self.linkage_method)
        labels = fcluster(Z, t=self.k_clusters, criterion="maxclust")
        return labels - 1
