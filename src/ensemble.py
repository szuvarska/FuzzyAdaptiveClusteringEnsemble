"""Adaptive Clustering Ensemble (ACE) with pluggable crisp (K-Means) or fuzzy (FCM) base members."""
import numpy as np

from src.models import BaseMemberClusterer

_EPS = 1e-9


def normalized_similarity(theta: np.ndarray) -> np.ndarray:
    """S_x: every row of theta divided by its maximum (rows of zeros stay zero)."""
    row_max = np.max(theta, axis=1, keepdims=True)
    return theta / np.where(row_max == 0, _EPS, row_max)


class AdaptiveClusteringEnsemble:
    """Three-stage ACE: (1) build the ensemble, (2) merge candidate clusters, (3) extract crisp labels."""

    def __init__(self, base_member: BaseMemberClusterer, m_partitions: int, k_clusters: int,
                 variable_k: bool = False, sampling_ratio: float = 0.7, alpha_1: float = 0.8,
                 alpha_1_min: float = 0.6, alpha_2: float = 0.7, delta_alpha: float = 0.1, random_state: int = 42):
        self.base_member = base_member
        self.metrics = base_member.get_metrics_strategy()
        self.m_partitions = m_partitions
        self.k_clusters = k_clusters
        self.variable_k = variable_k
        self.sampling_ratio = sampling_ratio
        self.alpha_1_initial = alpha_1
        self.alpha_1_min = alpha_1_min
        self.alpha_2 = alpha_2
        self.delta_alpha = delta_alpha
        self.random_state = random_state

    def _cluster_certainty(self, theta: np.ndarray, s_x: np.ndarray) -> np.ndarray:
        """rho_g = sum(S_x[:, g]) / |C_g| for every cluster g (Equation 7)."""
        rho = np.zeros(theta.shape[1])
        for g in range(theta.shape[1]):
            size = self.metrics.cardinality(theta[:, g])
            if size > 0:
                rho[g] = np.sum(s_x[:, g]) / size
        return rho

    # ------------------------------------------------------------------ Stage 1
    def stage1_transformation(self, X: np.ndarray) -> np.ndarray:
        """Fits m base partitions and stacks their membership matrices into Theta_1 (n x sum of k)."""
        rng = np.random.default_rng(self.random_state)
        n_instances, n_features = X.shape
        n_instance_samples = max(1, int(n_instances * self.sampling_ratio))
        n_feature_samples = max(1, int(n_features * self.sampling_ratio))
        half_m = self.m_partitions // 2  # first half samples instances, second half samples features

        matrices, partition_ids = [], []
        for i in range(self.m_partitions):
            seed = int(rng.integers(0, 10000))
            if self.variable_k:
                k = int(rng.integers(max(2, self.k_clusters - 2), self.k_clusters + 3))
            else:
                k = self.k_clusters

            if i < half_m:
                rows = rng.choice(n_instances, n_instance_samples, replace=False)
                X_train, X_full = X[rows, :], X
            else:
                cols = rng.choice(n_features, n_feature_samples, replace=False)
                X_train = X_full = X[:, cols]

            membership = self.base_member.get_membership_matrix(X_train, X_full, k, seed)
            matrices.append(membership)
            partition_ids.extend([i] * membership.shape[1])

        # Partitions may contribute different numbers of columns (variable k, empty K-Means clusters)
        self.partition_ids_ = np.asarray(partition_ids)
        return np.column_stack(matrices)

    # ------------------------------------------------------------------ Stage 2
    def _pair_scores(self, theta: np.ndarray, n: int) -> np.ndarray:
        """Set correlations between all columns of theta (only the upper triangle i < j is used)."""
        return self.metrics.set_correlation_matrix(theta, theta, n)

    def _merge_until_stable(self, theta: np.ndarray, partition_ids, scores: np.ndarray, alpha: float,
                            n: int, min_clusters: int = 1) -> np.ndarray:
        """
        Repeatedly merges the most correlated pair (score >= alpha) of clusters from disjoint partitions;
        the merged cluster replaces the pair at the end of the column order. Ties go to the first pair in
        (i, j) row-major order.

        Columns live in fixed slots (a merge kills two slots and fills the next free one), so slot order
        always equals column order. Each row keeps its best candidate, so a merge only re-scores the new
        column and re-scans the rows whose best partner was removed.
        """
        c0 = theta.shape[1]
        n_slots = 2 * c0 - 1  # at most c0 - 1 merges
        cols = np.zeros((n, n_slots))
        cols[:, :c0] = theta
        sizes = np.zeros(n_slots)
        sizes[:c0] = self.metrics.cardinalities(theta)
        _, partition = np.unique(partition_ids, return_inverse=True)
        origin = np.zeros((n_slots, partition.max() + 1), dtype=bool)  # source partitions of every slot
        origin[np.arange(c0), partition] = True
        active = np.zeros(n_slots, dtype=bool)
        active[:c0] = True

        allowed = (partition[:, None] != partition[None, :]) & (scores >= alpha) & (scores > -1)
        cand = np.full((n_slots, n_slots), -np.inf)
        cand[:c0, :c0] = np.where(np.triu(allowed, k=1), scores, -np.inf)
        best_col = np.argmax(cand, axis=1)
        best_val = cand[np.arange(n_slots), best_col]

        n_active = c0
        for new in range(c0, n_slots):
            if n_active <= min_clusters:
                break
            i = int(np.argmax(best_val))
            if best_val[i] == -np.inf:
                break
            j = int(best_col[i])

            merged, merged_origin = cols[:, i] + cols[:, j], origin[i] | origin[j]
            active[[i, j]] = False
            cand[[i, j], :] = -np.inf
            cand[:, [i, j]] = -np.inf
            best_val[[i, j]] = -np.inf
            for r in np.where(active & np.isin(best_col, [i, j]))[0]:
                best_col[r] = np.argmax(cand[r])
                best_val[r] = cand[r, best_col[r]]

            cols[:, new], sizes[new], origin[new], active[new] = merged, self.metrics.cardinality(merged), \
                merged_origin, True
            rows = np.where(active[:new])[0]
            score = self.metrics.set_correlation_matrix(cols[:, rows], merged[:, None], n,
                                                        sizes[rows], sizes[[new]])[:, 0]
            ok = ~(origin[rows] & merged_origin).any(axis=1) & (score >= alpha) & (score > -1)
            cand[rows, new] = np.where(ok, score, -np.inf)
            better = cand[rows, new] > best_val[rows]
            best_val[rows[better]] = cand[rows[better], new]
            best_col[rows[better]] = new
            n_active -= 1
        return cols[:, active]

    def stage2_consensus_merging(self, theta_1_initial: np.ndarray, n_instances: int, partition_ids=None):
        """Merges candidate clusters until k remain. Returns (Theta_1 kept, Theta_2 eliminated)."""
        if partition_ids is None:
            partition_ids = getattr(self, "partition_ids_", None)
        if partition_ids is None or len(partition_ids) != theta_1_initial.shape[1]:
            raise ValueError("partition_ids must give the source partition of every Theta_1 column; "
                             "run stage1_transformation first or pass them explicitly.")

        scores = self._pair_scores(theta_1_initial, n_instances)  # independent of alpha, computed once
        alpha = self.alpha_1_initial
        while True:
            theta_1 = self._merge_until_stable(theta_1_initial, partition_ids, scores, alpha, n_instances)
            n_active = theta_1.shape[1]
            if n_active >= self.k_clusters:
                break
            alpha += self.delta_alpha
            if alpha > 1.0:
                raise ValueError("Cannot resolve clusters: Merging threshold exceeded 1.0 limit.")

        # If still > k, continue merging down to alpha_1_min (stopping as soon as k remain)
        if n_active > self.k_clusters and self.alpha_1_min < alpha:
            theta_1 = self._merge_until_stable(
                theta_1_initial, partition_ids, scores, self.alpha_1_min, n_instances,
                min_clusters=self.k_clusters,
            )
            n_active = theta_1.shape[1]

        self.final_alpha_1_ = alpha

        if n_active == self.k_clusters:
            return theta_1, np.empty((n_instances, 0))

        # Too many clusters: keep the k most certain ones
        certainty = self._cluster_certainty(theta_1, normalized_similarity(theta_1))
        keep = np.argsort(certainty)[-self.k_clusters:][::-1]
        eliminated = [g for g in range(n_active) if g not in keep]
        return theta_1[:, keep], theta_1[:, eliminated]

    # ------------------------------------------------------------------ Stage 3
    def stage3_hard_clustering(self, theta_1: np.ndarray, theta_2: np.ndarray) -> np.ndarray:
        """
        Certain objects (max S_x > alpha_2) go to their best cluster; each uncertain object then goes to
        the cluster whose variance (Equation 9) changes least. theta_2 is kept for API symmetry.
        """
        n_instances, k = theta_1.shape
        s_x = normalized_similarity(theta_1)
        certain = np.max(s_x, axis=1) > self.alpha_2

        labels = np.full(n_instances, -1, dtype=int)
        labels[certain] = np.argmax(s_x[certain], axis=1)

        rho = self._cluster_certainty(theta_1, s_x)
        sizes = np.zeros(k, dtype=int)
        sum_sq = np.zeros(k)
        variance = np.zeros(k)
        for g in range(k):
            members = np.where(certain & (labels == g))[0]
            sizes[g] = len(members)
            if sizes[g] > 0:
                sum_sq[g] = np.sum((s_x[members, g] - rho[g]) ** 2)
                variance[g] = sum_sq[g] / sizes[g]

        for x in np.where(~certain)[0]:
            new_sum_sq = sum_sq + (s_x[x] - rho) ** 2
            new_variance = new_sum_sq / (sizes + 1)
            g = int(np.argmin(np.abs(new_variance - variance)))
            labels[x] = g
            sizes[g] += 1
            sum_sq[g] = new_sum_sq[g]
            variance[g] = new_variance[g]
        return labels

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        theta = self.stage1_transformation(X)
        theta_1, theta_2 = self.stage2_consensus_merging(theta, X.shape[0])
        return self.stage3_hard_clustering(theta_1, theta_2)
