import numpy as np
from src.models import BaseMemberClusterer


class AdaptiveClusteringEnsemble:
    """
    Master orchestrator for the ACE Framework.
    Handles the transformation, merging, and hard clustering stages.
    """

    def __init__(self, base_member: BaseMemberClusterer, m_partitions: int, k_clusters: int,
                 alpha_1: float = 0.8, alpha_2: float = 0.7, delta_alpha: float = 0.1, random_state: int = 42):
        self.base_member = base_member
        self.metrics = base_member.get_metrics_strategy()
        self.m_partitions = m_partitions
        self.k_clusters = k_clusters

        # Hyperparameters
        self.alpha_1_initial = alpha_1
        self.alpha_2 = alpha_2
        self.delta_alpha = delta_alpha

        self.random_state = random_state

    def stage1_transformation(self, X: np.ndarray) -> np.ndarray:
        """Generates m base models and constructs the global Theta_1 matrix."""
        rng = np.random.default_rng(self.random_state)
        all_matrices = []

        for _ in range(self.m_partitions):
            seed = int(rng.integers(0, 10000))
            local_matrix = self.base_member.get_membership_matrix(X, self.k_clusters, seed)
            all_matrices.append(local_matrix)

        theta_1 = np.column_stack(all_matrices)
        return theta_1

    def stage2_consensus_merging(self, theta_1_initial: np.ndarray, n_instances: int):
        """
        Iteratively merges candidate clusters from different members.
        Returns the finalized active clusters (Theta_1) and eliminated clusters (Theta_2).
        """
        current_alpha = self.alpha_1_initial

        while True:
            theta_1 = theta_1_initial.copy()

            # Track origins to ensure we only merge clusters from different base partitions
            origins = [{i // self.k_clusters} for i in range(theta_1.shape[1])]

            while True:
                merged_in_pass = False
                n_cols = theta_1.shape[1]

                best_pair = None
                best_score = -1

                # Find the most similar valid pair
                for i in range(n_cols):
                    for j in range(i + 1, n_cols):
                        if origins[i].isdisjoint(origins[j]):  # Must be from different members
                            score = self.metrics.set_correlation(theta_1[:, i], theta_1[:, j], n_instances)
                            if score >= current_alpha and score > best_score:
                                best_score = score
                                best_pair = (i, j)

                # Execute merge if a valid pair met the threshold
                if best_pair is not None:
                    i, j = best_pair
                    # Aggregate components
                    new_col = theta_1[:, i] + theta_1[:, j]
                    new_origin = origins[i].union(origins[j])

                    # Drop old columns (delete j first to maintain valid index for i)
                    theta_1 = np.delete(theta_1, [i, j], axis=1)
                    origins.pop(j)
                    origins.pop(i)

                    # Append merged cluster
                    theta_1 = np.column_stack((theta_1, new_col))
                    origins.append(new_origin)

                    merged_in_pass = True

                if not merged_in_pass:
                    break  # Iterative merging complete for this alpha threshold

            lam = theta_1.shape[1]  # Active clusters

            if lam == self.k_clusters:
                # Perfect merge count
                theta_2 = np.empty((n_instances, 0))
                break

            elif lam > self.k_clusters:
                # Too many clusters: Filter based on cluster certainty (Equation 7)
                max_theta_per_row = np.max(theta_1, axis=1, keepdims=True)
                max_theta_per_row[max_theta_per_row == 0] = 1e-9  # Prevent division by zero

                S_x = theta_1 / max_theta_per_row  # Normalized membership similarity

                certainties = []
                for g in range(lam):
                    cg_size = self.metrics.cardinality(theta_1[:, g])
                    rho = np.sum(S_x[:, g]) / cg_size if cg_size > 0 else 0.0
                    certainties.append(rho)

                # Retain top k certain clusters in Theta_1, move remainder to Theta_2
                top_k_indices = np.argsort(certainties)[-self.k_clusters:][::-1]
                eliminated_indices = [idx for idx in range(lam) if idx not in top_k_indices]

                theta_2 = theta_1[:, eliminated_indices]
                theta_1 = theta_1[:, top_k_indices]
                break

            else:
                # Too few clusters: Alpha threshold was too lenient (too many merges). Make it stricter.
                current_alpha += self.delta_alpha
                if current_alpha > 1.0:
                    raise ValueError("Cannot resolve clusters: Merging threshold exceeded 1.0 limit.")

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
