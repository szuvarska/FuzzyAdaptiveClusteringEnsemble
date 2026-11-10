"""
Statistical comparison of clustering methods over multiple datasets (Demšar, 2006),
as used by Alqurashi & Wang (2019) and the project plan (section 5.2):

1. Friedman test on the per-dataset ranks of the methods.
2. Iman-Davenport correction of the Friedman statistic (F-distributed) - omnibus test, alpha = 0.1.
3. Nemenyi post-hoc test, only if the omnibus null hypothesis is rejected.
4. Critical Difference (CD) diagram.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

# Critical values q_alpha of the Nemenyi test (studentized range / sqrt(2)), Demšar (2006) Table 5
NEMENYI_Q = {
    0.05: [1.960, 2.343, 2.569, 2.728, 2.850, 2.949, 3.031, 3.102, 3.164],
    0.10: [1.645, 2.052, 2.291, 2.459, 2.589, 2.693, 2.780, 2.855, 2.920],
}


def nemenyi_q(n_methods: int, alpha: float) -> float:
    """q_alpha for k methods. Uses Demšar's table, and the studentized range distribution otherwise."""
    if alpha in NEMENYI_Q and 2 <= n_methods <= 10:
        return NEMENYI_Q[alpha][n_methods - 2]
    return float(stats.studentized_range.ppf(1 - alpha, n_methods, np.inf) / np.sqrt(2))


def rank_methods(scores: pd.DataFrame, higher_is_better: bool = True) -> pd.DataFrame:
    """scores: datasets x methods. Returns ranks (1 = best), ties get the average rank."""
    return scores.rank(axis=1, ascending=not higher_is_better, method="average")


@dataclass
class ComparisonResult:
    metric: str
    setting: str
    n_datasets: int
    n_methods: int
    alpha: float
    average_ranks: pd.Series
    friedman_chi2: float
    friedman_p: float
    iman_davenport_f: float
    iman_davenport_p: float
    df1: int
    df2: int
    reject_h0: bool
    critical_difference: float
    nemenyi_pvalues: Optional[pd.DataFrame]
    significant_pairs: List[tuple]
    datasets_used: List[str]

    def to_dict(self) -> Dict:
        return {
            "metric": self.metric, "setting": self.setting, "alpha": self.alpha,
            "n_datasets": self.n_datasets, "n_methods": self.n_methods,
            "datasets_used": self.datasets_used,
            "average_ranks": self.average_ranks.round(4).to_dict(),
            "friedman": {"chi2": self.friedman_chi2, "p_value": self.friedman_p},
            "iman_davenport": {"F": self.iman_davenport_f, "df1": self.df1, "df2": self.df2,
                               "p_value": self.iman_davenport_p, "reject_h0": self.reject_h0},
            "nemenyi": None if self.nemenyi_pvalues is None else {
                "critical_difference": self.critical_difference,
                "significant_pairs": [{"better": a, "worse": b, "rank_diff": round(d, 4)}
                                      for a, b, d in self.significant_pairs],
            },
        }


def compare_methods(scores: pd.DataFrame, alpha: float = 0.1, metric: str = "",
                    setting: str = "") -> ComparisonResult:
    """
    scores: DataFrame (datasets x methods) of mean scores, higher = better.
    Datasets with any missing value are dropped (every method must have a score on every dataset).
    """
    clean = scores.dropna(axis=0, how="any")
    dropped = sorted(set(scores.index) - set(clean.index))
    if dropped:
        print(f"WARNING [{metric}/{setting}]: dropping datasets with missing scores: {dropped}")
    N, k = clean.shape
    if N < 2 or k < 2:
        raise ValueError(f"Need at least 2 datasets and 2 methods, got N={N}, k={k}")

    ranks = rank_methods(clean)
    R = ranks.mean(axis=0)

    # Friedman statistic (Demšar 2006, eq. 1)
    chi2 = 12 * N / (k * (k + 1)) * (np.sum(R ** 2) - k * (k + 1) ** 2 / 4)
    chi2_p = float(stats.chi2.sf(chi2, k - 1))

    # Iman-Davenport (eq. 2), F-distributed with (k-1) and (k-1)(N-1) degrees of freedom
    df1, df2 = k - 1, (k - 1) * (N - 1)
    denom = N * (k - 1) - chi2
    if denom <= 0:  # every dataset ranks the methods identically
        ff, ff_p = np.inf, 0.0
    else:
        ff = (N - 1) * chi2 / denom
        ff_p = float(stats.f.sf(ff, df1, df2))
    reject = ff_p < alpha

    cd = nemenyi_q(k, alpha) * np.sqrt(k * (k + 1) / (6 * N))
    pvals, pairs = None, []
    if reject:
        se = np.sqrt(k * (k + 1) / (6 * N))
        methods = list(R.index)
        pvals = pd.DataFrame(np.ones((k, k)), index=methods, columns=methods)
        for i, a in enumerate(methods):
            for j, b in enumerate(methods):
                if i < j:
                    z = abs(R[a] - R[b]) / se
                    p = float(stats.studentized_range.sf(z * np.sqrt(2), k, np.inf))
                    pvals.loc[a, b] = pvals.loc[b, a] = p
                    if abs(R[a] - R[b]) > cd:
                        better, worse = (a, b) if R[a] < R[b] else (b, a)
                        pairs.append((better, worse, float(abs(R[a] - R[b]))))

    return ComparisonResult(metric, setting, N, k, alpha, R.sort_values(), float(chi2), chi2_p,
                            float(ff), ff_p, df1, df2, bool(reject), float(cd), pvals, pairs,
                            list(clean.index))


def _cliques(avg_ranks: pd.Series, cd: float) -> List[List[str]]:
    """Maximal groups of methods whose rank range is within CD (connected by a bar in the diagram)."""
    names, r = list(avg_ranks.index), avg_ranks.values
    groups = []
    for i in range(len(names)):
        j = i
        while j + 1 < len(names) and r[j + 1] - r[i] <= cd:
            j += 1
        if j > i:
            groups.append((i, j))
    maximal = [g for g in groups if not any(o != g and o[0] <= g[0] and g[1] <= o[1] for o in groups)]
    return [names[a:b + 1] for a, b in maximal]


def plot_cd_diagram(avg_ranks: pd.Series, cd: float, title: str = "", path: Optional[str] = None,
                    reject_h0: bool = True):
    """Demšar-style critical difference diagram. Lower rank = better (drawn on the left)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    avg_ranks = avg_ranks.sort_values()
    names, ranks = list(avg_ranks.index), avg_ranks.values
    k = len(names)
    lo, hi = 1, max(k, int(np.ceil(ranks.max())))
    cliques = _cliques(avg_ranks, cd) if reject_h0 else [names]

    half = int(np.ceil(k / 2))
    n_rows = max(half, k - half)
    height = 1.6 + 0.32 * n_rows + 0.18 * len(cliques)
    fig, ax = plt.subplots(figsize=(8, height))
    ax.set_xlim(lo - 0.6, hi + 0.6)
    ax.set_ylim(-(n_rows + 1.2 + 0.25 * len(cliques)) * 0.35, 1.3)
    ax.axis("off")

    # Rank axis
    ax.hlines(0, lo, hi, color="black", lw=1)
    for t in range(lo, hi + 1):
        ax.vlines(t, 0, 0.1, color="black", lw=1)
        ax.text(t, 0.18, str(t), ha="center", va="bottom", fontsize=10)
    for t in np.arange(lo + 0.5, hi, 1.0):
        ax.vlines(t, 0, 0.05, color="black", lw=0.7)

    # CD bar
    ax.hlines(0.9, lo, lo + cd, color="black", lw=1.5)
    ax.vlines([lo, lo + cd], 0.83, 0.97, color="black", lw=1.5)
    ax.text(lo + cd / 2, 1.0, f"CD = {cd:.3f}", ha="center", va="bottom", fontsize=9)

    # Method labels: best half on the left, rest on the right
    step = 0.35
    for i, (name, r) in enumerate(zip(names, ranks)):
        if i < half:
            y = -(i + 1) * step - 0.1 * len(cliques)
            ax.plot([r, r, lo - 0.1], [0, y, y], color="black", lw=0.9)
            ax.text(lo - 0.15, y, f"{name} ({r:.2f})", ha="right", va="center", fontsize=9)
        else:
            y = -(k - i) * step - 0.1 * len(cliques)
            ax.plot([r, r, hi + 0.1], [0, y, y], color="black", lw=0.9)
            ax.text(hi + 0.15, y, f"({r:.2f}) {name}", ha="left", va="center", fontsize=9)

    # Cliques of non-significantly different methods
    for c, group in enumerate(cliques):
        y = -0.08 - 0.1 * c
        ax.hlines(y, avg_ranks[group[0]] - 0.03, avg_ranks[group[-1]] + 0.03, color="red", lw=3)

    if title:
        sub = "" if reject_h0 else "  (H0 not rejected: no significant differences)"
        ax.set_title(title + sub, fontsize=10)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    return fig


def run_statistical_tests(experiment_dir: str, alpha: float = 0.1, metrics=("ari", "nmi"),
                          methods: Optional[List[str]] = None, datasets: Optional[List[str]] = None,
                          tag: str = "") -> List[ComparisonResult]:
    """Reads ``summary.csv`` of an experiment, runs all tests and stores results in ``<dir>/statistics``."""
    import json

    summary = pd.read_csv(os.path.join(experiment_dir, "summary.csv"))
    if methods:
        summary = summary[summary["method"].isin(methods)]
    if datasets:
        summary = summary[summary["dataset"].isin(datasets)]

    out_dir = os.path.join(experiment_dir, "statistics" + (f"_{tag}" if tag else ""))
    fig_dir = os.path.join(out_dir, "cd_diagrams")
    os.makedirs(fig_dir, exist_ok=True)

    results, report = [], []
    for setting in sorted(summary["setting"].unique()):
        sub = summary[summary["setting"] == setting]
        for metric in metrics:
            scores = sub.pivot(index="dataset", columns="method", values=f"{metric}_mean")
            res = compare_methods(scores, alpha=alpha, metric=metric, setting=setting)
            results.append(res)
            report.append(res.to_dict())

            rank_table = rank_methods(scores.loc[res.datasets_used])
            rank_table.loc["Average rank"] = rank_table.mean()
            rank_table.to_csv(os.path.join(out_dir, f"ranks_{metric}_{setting}.csv"))
            if res.nemenyi_pvalues is not None:
                res.nemenyi_pvalues.to_csv(os.path.join(out_dir, f"nemenyi_pvalues_{metric}_{setting}.csv"))

            plot_cd_diagram(res.average_ranks, res.critical_difference,
                            title=f"{metric.upper()} - {setting} k (alpha = {alpha}, N = {res.n_datasets})",
                            path=os.path.join(fig_dir, f"cd_{metric}_{setting}.png"),
                            reject_h0=res.reject_h0)

    with open(os.path.join(out_dir, "tests.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    flat = pd.DataFrame([{
        "setting": r.setting, "metric": r.metric, "N": r.n_datasets, "k": r.n_methods,
        "friedman_chi2": r.friedman_chi2, "friedman_p": r.friedman_p,
        "iman_davenport_F": r.iman_davenport_f, "iman_davenport_p": r.iman_davenport_p,
        "reject_h0": r.reject_h0, "CD": r.critical_difference,
        "best_method": r.average_ranks.index[0],
        "significant_pairs": "; ".join(f"{a} > {b}" for a, b, _ in r.significant_pairs),
        **{f"rank_{m}": v for m, v in r.average_ranks.items()},
    } for r in results])
    flat.to_csv(os.path.join(out_dir, "tests_summary.csv"), index=False)
    return results
