import numpy as np
from src.models import BaseMemberClusterer


class AdaptiveClusteringEnsemble:
    """
    Master orchestrator for the ACE Framework.
    Handles the transformation, merging, and hard clustering stages.
    """

    def __init__(self, base_member: BaseMemberClusterer, m_partitions: int, k_clusters: int, variable_k: bool = False,
                 sampling_ratio: float = 0.7, alpha_1: float = 0.8, alpha_2: float = 0.7,
                 alpha_1_min: float = 0.6, delta_alpha: float = 0.1, random_state: int = 42):
        self.base_member = base_member
        self.metrics = base_member.get_metrics_strategy()
        self.m_partitions = m_partitions
        self.k_clusters = k_clusters
        self.variable_k = variable_k
        self.sampling_ratio = sampling_ratio
        self.final_alpha_1_ = None
        self.partition_ids_ = None

        # Hyperparameters
        self.alpha_1_initial = alpha_1
        self.alpha_1_min = alpha_1_min
        self.alpha_2 = alpha_2
        self.delta_alpha = delta_alpha

        self.random_state = random_state

    def stage1_transformation(self, X: np.ndarray) -> np.ndarray:
        """Generates m base models and constructs the global Theta_1 matrix."""
        rng = np.random.default_rng(self.random_state)
        all_matrices = []
        partition_ids = []  # which base partition each column of Theta_1 came from

        n_instances, n_features = X.shape
        n_instance_samples = max(1, int(n_instances * self.sampling_ratio))
        n_feature_samples = max(1, int(n_features * self.sampling_ratio))

        # Split partitions: half with sample instances, half with sample features
        half_m = self.m_partitions // 2

        for i in range(self.m_partitions):
            seed = int(rng.integers(0, 10000))

            # Determine target clusters (k) for this specific partition
            if self.variable_k:
                # Interval [k-2, k+2], ensuring the minimum k is at least 2
                min_k = max(2, self.k_clusters - 2)
                max_k = self.k_clusters + 2
                current_k = int(rng.integers(min_k, max_k + 1))
            else:
                current_k = self.k_clusters

            if i < half_m:
                # Subspace Strategy 1: Data Instance Sampling
                sampled_indices = rng.choice(n_instances, n_instance_samples, replace=False)
                X_train = X[sampled_indices, :]
                X_full = X
            else:
                # Subspace Strategy 2: Feature Sampling
                sampled_features = rng.choice(n_features, n_feature_samples, replace=False)
                X_train = X[:, sampled_features]
                X_full = X[:, sampled_features]

            local_matrix = self.base_member.get_membership_matrix(X_train, X_full, current_k, seed)
            all_matrices.append(local_matrix)
            partition_ids.extend([i] * local_matrix.shape[1])

        theta_1 = np.column_stack(all_matrices)
        # Recorded here because partitions can have different numbers of columns
        # (variable k, or K-Means returning an empty cluster).
        self.partition_ids_ = np.asarray(partition_ids)
        return theta_1

    def _find_best_pair(self, theta_1: np.ndarray, origins: list, n_instances: int):
        """Finds the most similar pair of clusters from disjoint base partitions."""
        n_cols = theta_1.shape[1]
        best_pair = None
        best_score = -1.0

        for i in range(n_cols):
            for j in range(i + 1, n_cols):
                if origins[i].isdisjoint(origins[j]):
                    score = self.metrics.set_correlation(theta_1[:, i], theta_1[:, j], n_instances)
                    if score > best_score:
                        best_score = score
                        best_pair = (i, j)

        return best_pair, best_score

    @staticmethod
    def _merge_pair(theta_1: np.ndarray, origins: list, pair: tuple):
        """Merges two cluster columns and their partition origin sets."""
        i, j = pair
        new_col = theta_1[:, i] + theta_1[:, j]
        new_origin = origins[i].union(origins[j])

        theta_1 = np.delete(theta_1, [i, j], axis=1)
        origins.pop(j)
        origins.pop(i)

        theta_1 = np.column_stack((theta_1, new_col))
        origins.append(new_origin)
        return theta_1, origins

    def stage2_consensus_merging(self, theta_1_initial: np.ndarray, n_instances: int, partition_ids=None):
        """
        Iteratively merges candidate clusters from different members.
        Returns the finalized active clusters (Theta_1) and eliminated clusters (Theta_2).
        """
        if partition_ids is None:
            partition_ids = getattr(self, "partition_ids_", None)
        if partition_ids is None or len(partition_ids) != theta_1_initial.shape[1]:
            raise ValueError("partition_ids must give the source partition of every Theta_1 column; "
                             "run stage1_transformation first or pass them explicitly.")

        current_alpha = self.alpha_1_initial

        # Pass 1: Merge at current_alpha; if too few clusters (< k), raise alpha_1 and retry
        while True:
            theta_1 = theta_1_initial.copy()
            origins = [{int(p)} for p in partition_ids]

            while True:
                best_pair, best_score = self._find_best_pair(theta_1, origins, n_instances)
                if best_pair is not None and best_score >= current_alpha:
                    theta_1, origins = self._merge_pair(theta_1, origins, best_pair)
                else:
                    break

            if theta_1.shape[1] >= self.k_clusters:
                break

            current_alpha += self.delta_alpha
            if current_alpha > 1.0:
                raise ValueError("Cannot resolve clusters: Merging threshold exceeded 1.0 limit.")

        # If still > k, adapt alpha_1 down to max S_c (>= alpha_1_min)
        while theta_1.shape[1] > self.k_clusters:
            best_pair, best_score = self._find_best_pair(theta_1, origins, n_instances)
            if best_pair is None or best_score < self.alpha_1_min:
                break
            current_alpha = best_score
            theta_1, origins = self._merge_pair(theta_1, origins, best_pair)

        # If still > k after reaching alpha_1_min, eliminate lowest-certainty clusters
        lam = theta_1.shape[1]
        if lam == self.k_clusters:
            theta_2 = np.empty((n_instances, 0))
        else:
            max_theta_per_row = np.max(theta_1, axis=1, keepdims=True)
            max_theta_per_row[max_theta_per_row == 0] = 1e-9
            S_x = theta_1 / max_theta_per_row

            certainties = []
            for g in range(lam):
                cg_size = self.metrics.cardinality(theta_1[:, g])
                rho = np.sum(S_x[:, g]) / cg_size if cg_size > 0 else 0.0
                certainties.append(rho)

            top_k_indices = np.argsort(certainties)[-self.k_clusters:][::-1]
            eliminated_indices = [idx for idx in range(lam) if idx not in top_k_indices]

            theta_2 = theta_1[:, eliminated_indices]
            theta_1 = theta_1[:, top_k_indices]

        self.final_alpha_1_ = current_alpha
        return theta_1, theta_2

    def stage3_hard_clustering(self, theta_1: np.ndarray, theta_2: np.ndarray) -> np.ndarray:
        """
        Stage 3: Extracts crisp cluster labels via variance-minimization.
        Assigns certain objects directly and uncertain objects by minimizing Delta Var.
        """
        n_instances, k_active = theta_1.shape
        labels = np.full(n_instances, -1, dtype=int)

        # Calculate S_x (Normalized membership similarity)
        max_theta_per_row = np.max(theta_1, axis=1, keepdims=True)
        safe_max = np.where(max_theta_per_row == 0, 1e-9, max_theta_per_row)  # Prevent zero division
        S_x = theta_1 / safe_max

        # 1. Object Classification (Certain vs Uncertain)
        max_Sx = np.max(S_x, axis=1)
        certain_mask = max_Sx > self.alpha_2
        certain_indices = np.where(certain_mask)[0]
        uncertain_indices = np.where(~certain_mask)[0]

        # 2. Certain Object Assignment
        if len(certain_indices) > 0:
            labels[certain_indices] = np.argmax(S_x[certain_indices], axis=1)

        # Calculate global cluster certainty (rho_g) for the variance baseline (Equation 7)
        rho_g = np.zeros(k_active)
        for g in range(k_active):
            cg_size = self.metrics.cardinality(theta_1[:, g])
            if cg_size > 0:
                rho_g[g] = np.sum(S_x[:, g]) / cg_size

        # Initialize cluster sizes and sums of squared differences based on certain objects
        cluster_sizes = np.zeros(k_active, dtype=int)
        sum_sq_diff = np.zeros(k_active, dtype=float)
        current_variance = np.zeros(k_active, dtype=float)

        for g in range(k_active):
            assigned_certain = certain_indices[labels[certain_indices] == g]
            cluster_sizes[g] = len(assigned_certain)
            if cluster_sizes[g] > 0:
                sum_sq_diff[g] = np.sum((S_x[assigned_certain, g] - rho_g[g]) ** 2)
                current_variance[g] = sum_sq_diff[g] / cluster_sizes[g]  # Equation 9

        # 3. Variance-Minimization Assignment for Uncertain Objects
        for xu in uncertain_indices:
            best_g = -1
            min_delta_var = float('inf')

            best_new_var = 0.0
            best_new_sum_sq = 0.0

            for g in range(k_active):
                # Calculate prospective variance if x_u is added to cluster g
                prospective_sum_sq = sum_sq_diff[g] + (S_x[xu, g] - rho_g[g]) ** 2
                prospective_size = cluster_sizes[g] + 1
                prospective_var = prospective_sum_sq / prospective_size

                # Calculate Delta Var
                delta_var = abs(prospective_var - current_variance[g])

                if delta_var < min_delta_var:
                    min_delta_var = delta_var
                    best_g = g
                    best_new_var = prospective_var
                    best_new_sum_sq = prospective_sum_sq

            # Assign uncertain object to the cluster minimizing Delta Var
            labels[xu] = best_g
            cluster_sizes[best_g] += 1
            sum_sq_diff[best_g] = best_new_sum_sq
            current_variance[best_g] = best_new_var

        return labels

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Executes the complete ACE pipeline (Stages 1, 2, and 3).
        Returns the final crisp cluster labels for the input data X.
        """
        n_instances = X.shape[0]

        # Stage 1
        theta_1_initial = self.stage1_transformation(X)

        # Stage 2
        theta_1_final, theta_2_elim = self.stage2_consensus_merging(theta_1_initial, n_instances)

        # Stage 3
        final_labels = self.stage3_hard_clustering(theta_1_final, theta_2_elim)

        return final_labels
