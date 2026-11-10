"""
Experiment runner (plan, sections 4-5).

* Two settings: ``fixed`` (every base member uses the true k) and ``variable`` (k drawn from [k-2, k+2]).
* Methods: K-Means, single FCM, EAC, ACE (K-Means members) and Fuzzy-ACE (FCM members).
* 10 independent seeded runs per (setting, dataset, method). Within a run every method gets the same seed.
* Every finished run is appended immediately to ``raw_results.csv`` so an interrupted experiment
  can be resumed: rows already on disk are skipped.

Stored per experiment (``results/<experiment>/``):
    config.json             parameters, package versions, git commit, timestamps
    raw_results.csv         one row per run: ARI, NMI, runtime, status, ...
    predicted_labels.csv    predicted labels of every run (space-separated) for later analysis
    summary.csv             mean / std per (setting, dataset, method)
    tables/<metric>_<setting>.csv   dataset x method "mean ± std" tables, ready for the report
"""
from __future__ import annotations

import csv
import json
import os
import platform
import subprocess
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Callable, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from src.baselines import EACClusterer, FCMClusterer, KMeansClusterer
from src.datasets import ALL_DATASETS, DEFAULT_DATA_DIR, PROJECT_ROOT, load_dataset
from src.ensemble import AdaptiveClusteringEnsemble
from src.models import FCMMember, KMeansMember

DEFAULT_RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
SETTINGS = ("fixed", "variable")
METHODS = ("KMeans", "FCM", "EAC", "ACE", "Fuzzy-ACE")

RAW_COLUMNS = ["experiment", "setting", "dataset", "method", "run", "seed", "k_true", "n_instances",
               "n_features", "ari", "nmi", "n_clusters_found", "runtime_s", "status", "error",
               "final_alpha_1", "timestamp"]
KEY_COLUMNS = ["setting", "dataset", "method", "run"]


@dataclass
class ExperimentConfig:
    """Hyperparameters - defaults replicate the ACE article / project plan."""
    name: str = "main"
    datasets: List[str] = field(default_factory=lambda: list(ALL_DATASETS))
    settings: List[str] = field(default_factory=lambda: list(SETTINGS))
    methods: List[str] = field(default_factory=lambda: list(METHODS))
    n_runs: int = 10
    base_seed: int = 0
    m_partitions: int = 10
    sampling_ratio: float = 0.7
    alpha_1: float = 0.8
    alpha_2: float = 0.7
    delta_alpha: float = 0.1
    fcm_m_fuzz: float = 2.0                      # single FCM baseline
    fuzzy_ace_m_range: tuple = (1.5, 2.5)        # m drawn per base partition in Fuzzy-ACE
    kmeans_n_init: int = 1                       # single K-Means run
    eac_linkage: str = "average"
    data_dir: str = DEFAULT_DATA_DIR
    results_dir: str = DEFAULT_RESULTS_DIR
    n_jobs: int = 1


# ----------------------------------------------------------------------------------------------
# Method factory
# ----------------------------------------------------------------------------------------------
def build_method(method: str, k: int, setting: str, seed: int, cfg: ExperimentConfig):
    variable_k = setting == "variable"
    if method == "KMeans":
        return KMeansClusterer(k, n_init=cfg.kmeans_n_init, random_state=seed)
    if method == "FCM":
        return FCMClusterer(k, m_fuzz=cfg.fcm_m_fuzz, random_state=seed)
    if method == "EAC":
        return EACClusterer(k, m_partitions=cfg.m_partitions, variable_k=variable_k,
                            sampling_ratio=cfg.sampling_ratio, linkage_method=cfg.eac_linkage,
                            random_state=seed)
    if method in ("ACE", "Fuzzy-ACE"):
        member = KMeansMember() if method == "ACE" else FCMMember(m_fuzz_range=tuple(cfg.fuzzy_ace_m_range))
        return AdaptiveClusteringEnsemble(
            base_member=member, m_partitions=cfg.m_partitions, k_clusters=k, variable_k=variable_k,
            sampling_ratio=cfg.sampling_ratio, alpha_1=cfg.alpha_1, alpha_2=cfg.alpha_2,
            delta_alpha=cfg.delta_alpha, random_state=seed)
    raise ValueError(f"Unknown method '{method}'. Available: {METHODS}")


# ----------------------------------------------------------------------------------------------
# Storage helpers
# ----------------------------------------------------------------------------------------------
def _git_commit() -> Optional[str]:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def _package_versions() -> Dict[str, str]:
    import scipy
    import sklearn
    import skfuzzy
    return {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__, "scipy": scipy.__version__,
            "scikit-fuzzy": getattr(skfuzzy, "__version__", "unknown")}


def write_config(out_dir: str, cfg_dict: dict, extra: Optional[dict] = None) -> None:
    path = os.path.join(out_dir, "config.json")
    history = []
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            history = json.load(fh).get("sessions", [])
    history.append({"started": datetime.now().isoformat(timespec="seconds"),
                    "git_commit": _git_commit(), "platform": platform.platform()})
    payload = {"config": cfg_dict, "versions": _package_versions(), "sessions": history}
    if extra:
        payload.update(extra)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)


class CsvAppender:
    """Appends rows to a CSV file, writing the header when the file is new."""

    def __init__(self, path: str, columns: List[str]):
        self.path, self.columns = path, columns
        self._new = not os.path.exists(path) or os.path.getsize(path) == 0

    def append(self, row: dict) -> None:
        with open(self.path, "a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=self.columns, extrasaction="ignore")
            if self._new:
                writer.writeheader()
                self._new = False
            writer.writerow(row)


def _completed_keys(raw_path: str) -> set:
    if not os.path.exists(raw_path) or os.path.getsize(raw_path) == 0:
        return set()
    done = pd.read_csv(raw_path, usecols=KEY_COLUMNS + ["status"])
    done = done[done["status"] == "ok"]
    return {(r.setting, r.dataset, r.method, int(r.run)) for r in done.itertuples()}


# ----------------------------------------------------------------------------------------------
# Single run
# ----------------------------------------------------------------------------------------------
_DATA_CACHE: Dict[str, tuple] = {}


def _get_data(dataset: str, data_dir: str):
    if dataset not in _DATA_CACHE:
        ds = load_dataset(dataset, data_dir)
        _DATA_CACHE[dataset] = (ds, ds.preprocessed())
    return _DATA_CACHE[dataset]


def run_single(cfg: ExperimentConfig, setting: str, dataset: str, method: str, run: int) -> tuple:
    """Runs one (setting, dataset, method, run) and returns (result_row, labels or None)."""
    ds, X = _get_data(dataset, cfg.data_dir)
    seed = cfg.base_seed + run
    row = {"experiment": cfg.name, "setting": setting, "dataset": dataset, "method": method,
           "run": run, "seed": seed, "k_true": ds.k, "n_instances": X.shape[0], "n_features": X.shape[1],
           "ari": np.nan, "nmi": np.nan, "n_clusters_found": np.nan, "runtime_s": np.nan,
           "status": "ok", "error": "", "final_alpha_1": np.nan}
    labels = None
    start = time.perf_counter()
    try:
        model = build_method(method, ds.k, setting, seed, cfg)
        labels = np.asarray(model.fit_predict(X))
        row.update(ari=adjusted_rand_score(ds.y, labels), nmi=normalized_mutual_info_score(ds.y, labels),
                   n_clusters_found=len(np.unique(labels)),
                   final_alpha_1=getattr(model, "final_alpha_1_", np.nan))
    except Exception as exc:  # recorded, never crashes the whole experiment
        row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
    row["runtime_s"] = time.perf_counter() - start
    row["timestamp"] = datetime.now().isoformat(timespec="seconds")
    return row, labels


# ----------------------------------------------------------------------------------------------
# Full experiment
# ----------------------------------------------------------------------------------------------
def run_experiment(cfg: ExperimentConfig, resume: bool = True, verbose: bool = True) -> pd.DataFrame:
    out_dir = os.path.join(cfg.results_dir, cfg.name)
    os.makedirs(out_dir, exist_ok=True)
    write_config(out_dir, asdict(cfg))

    raw_path = os.path.join(out_dir, "raw_results.csv")
    labels_path = os.path.join(out_dir, "predicted_labels.csv")
    if not resume:
        for p in (raw_path, labels_path):
            if os.path.exists(p):
                os.remove(p)
    done = _completed_keys(raw_path) if resume else set()

    # Datasets without local files are skipped (and reported) instead of failing the experiment
    datasets = []
    for d in cfg.datasets:
        try:
            _get_data(d, cfg.data_dir)
            datasets.append(d)
        except FileNotFoundError as exc:
            print(f"SKIP {d}: {exc}")

    tasks = [(s, d, m, r) for s in cfg.settings for d in datasets for m in cfg.methods
             for r in range(cfg.n_runs) if (s, d, m, r) not in done]
    if verbose:
        print(f"[{cfg.name}] {len(tasks)} runs to do ({len(done)} already stored) -> {out_dir}")

    raw_writer = CsvAppender(raw_path, RAW_COLUMNS)
    labels_writer = CsvAppender(labels_path, KEY_COLUMNS + ["seed", "labels"])

    def _store(row, labels, i):
        raw_writer.append(row)
        if labels is not None:
            labels_writer.append({**{k: row[k] for k in KEY_COLUMNS}, "seed": row["seed"],
                                  "labels": " ".join(map(str, labels.tolist()))})
        if verbose:
            print(f"  [{i}/{len(tasks)}] {row['setting']:8s} {row['dataset']:18s} {row['method']:9s} "
                  f"run {row['run']}: ARI={row['ari']:.4f} NMI={row['nmi']:.4f} "
                  f"({row['runtime_s']:.1f}s) {row['status']}")

    if cfg.n_jobs == 1:
        for i, t in enumerate(tasks, 1):
            _store(*run_single(cfg, *t), i)
    else:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=cfg.n_jobs, return_as="generator_unordered")(
            delayed(run_single)(cfg, *t) for t in tasks)
        for i, (row, labels) in enumerate(results, 1):
            _store(row, labels, i)

    raw = load_raw_results(out_dir)
    summarize(raw, out_dir)
    return raw


def load_raw_results(out_dir: str) -> pd.DataFrame:
    raw = pd.read_csv(os.path.join(out_dir, "raw_results.csv"))
    # If a failed run was later re-run successfully, keep the latest row per key
    return raw.drop_duplicates(subset=KEY_COLUMNS, keep="last").reset_index(drop=True)


def summarize(raw: pd.DataFrame, out_dir: str) -> pd.DataFrame:
    ok = raw[raw["status"] == "ok"]
    summary = (ok.groupby(["setting", "dataset", "method"])
                 .agg(ari_mean=("ari", "mean"), ari_std=("ari", "std"),
                      nmi_mean=("nmi", "mean"), nmi_std=("nmi", "std"),
                      runtime_mean_s=("runtime_s", "mean"), n_runs_ok=("run", "count"))
                 .reset_index())
    failed = raw[raw["status"] != "ok"].groupby(["setting", "dataset", "method"]).size()
    summary["n_runs_failed"] = [int(failed.get((r.setting, r.dataset, r.method), 0))
                                for r in summary.itertuples()]
    summary.to_csv(os.path.join(out_dir, "summary.csv"), index=False)

    tables_dir = os.path.join(out_dir, "tables")
    os.makedirs(tables_dir, exist_ok=True)
    method_order = [m for m in METHODS if m in summary["method"].unique()]
    for setting in summary["setting"].unique():
        sub = summary[summary["setting"] == setting]
        for metric in ("ari", "nmi"):
            means = sub.pivot(index="dataset", columns="method", values=f"{metric}_mean")
            stds = sub.pivot(index="dataset", columns="method", values=f"{metric}_std")
            cols = [m for m in method_order if m in means.columns]
            order = [d for d in ALL_DATASETS if d in means.index]
            means, stds = means.loc[order, cols], stds.loc[order, cols]
            pretty = means.round(4).astype(str) + " ± " + stds.round(4).astype(str)
            pretty.loc["Average"] = means.mean().round(4).astype(str)
            pretty.to_csv(os.path.join(tables_dir, f"{metric}_{setting}.csv"))
            means.to_csv(os.path.join(tables_dir, f"{metric}_{setting}_means.csv"))
    return summary


# ----------------------------------------------------------------------------------------------
# alpha_2 grid search (plan, section 4)
# ----------------------------------------------------------------------------------------------
GRID_COLUMNS = ["setting", "dataset", "method", "run", "seed", "alpha_2", "ari", "nmi",
                "n_certain", "status", "error", "timestamp"]


def _grid_single(cfg: ExperimentConfig, setting: str, dataset: str, method: str, run: int,
                 alpha_2_values: Iterable[float]) -> List[dict]:
    """Stages 1-2 do not depend on alpha_2, so they are computed once and Stage 3 is re-run per value."""
    ds, X = _get_data(dataset, cfg.data_dir)
    seed = cfg.base_seed + run
    base = {"setting": setting, "dataset": dataset, "method": method, "run": run, "seed": seed,
            "timestamp": datetime.now().isoformat(timespec="seconds")}
    rows = []
    try:
        ace = build_method(method, ds.k, setting, seed, cfg)
        theta = ace.stage1_transformation(X)
        theta_1, theta_2 = ace.stage2_consensus_merging(theta, X.shape[0])
        max_row = np.max(theta_1, axis=1, keepdims=True)
        s_x_max = np.max(theta_1 / np.where(max_row == 0, 1e-9, max_row), axis=1)
        for a2 in alpha_2_values:
            ace.alpha_2 = float(a2)
            labels = ace.stage3_hard_clustering(theta_1, theta_2)
            rows.append({**base, "alpha_2": float(a2), "ari": adjusted_rand_score(ds.y, labels),
                         "nmi": normalized_mutual_info_score(ds.y, labels),
                         "n_certain": int(np.sum(s_x_max > a2)), "status": "ok", "error": ""})
    except Exception as exc:
        for a2 in alpha_2_values:
            rows.append({**base, "alpha_2": float(a2), "ari": np.nan, "nmi": np.nan, "n_certain": np.nan,
                         "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
    return rows


def run_alpha2_grid_search(cfg: ExperimentConfig, alpha_2_values=(0.5, 0.6, 0.7, 0.8, 0.9),
                           resume: bool = True, verbose: bool = True) -> pd.DataFrame:
    out_dir = os.path.join(cfg.results_dir, cfg.name)
    os.makedirs(out_dir, exist_ok=True)
    write_config(out_dir, asdict(cfg), extra={"alpha_2_values": list(alpha_2_values)})
    raw_path = os.path.join(out_dir, "raw_results.csv")
    if not resume and os.path.exists(raw_path):
        os.remove(raw_path)

    done = set()
    if resume and os.path.exists(raw_path) and os.path.getsize(raw_path) > 0:
        prev = pd.read_csv(raw_path)
        prev = prev[prev["status"] == "ok"]
        done = {(r.setting, r.dataset, r.method, int(r.run)) for r in prev.itertuples()}

    datasets = []
    for d in cfg.datasets:
        try:
            _get_data(d, cfg.data_dir)
            datasets.append(d)
        except FileNotFoundError as exc:
            print(f"SKIP {d}: {exc}")
    methods = [m for m in cfg.methods if m in ("ACE", "Fuzzy-ACE")]
    tasks = [(s, d, m, r) for s in cfg.settings for d in datasets for m in methods
             for r in range(cfg.n_runs) if (s, d, m, r) not in done]
    if verbose:
        print(f"[{cfg.name}] {len(tasks)} ensemble fits to do -> {out_dir}")

    writer = CsvAppender(raw_path, GRID_COLUMNS)
    if cfg.n_jobs == 1:
        results = (_grid_single(cfg, *t, alpha_2_values) for t in tasks)
    else:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=cfg.n_jobs, return_as="generator_unordered")(
            delayed(_grid_single)(cfg, *t, alpha_2_values) for t in tasks)
    for i, rows in enumerate(results, 1):
        for row in rows:
            writer.append(row)
        if verbose:
            r = rows[0]
            best = max(rows, key=lambda x: -1 if np.isnan(x["ari"]) else x["ari"])
            print(f"  [{i}/{len(tasks)}] {r['dataset']:18s} {r['method']:9s} run {r['run']}: "
                  f"best alpha_2={best['alpha_2']} ARI={best['ari']:.4f} {r['status']}")

    raw = pd.read_csv(raw_path).drop_duplicates(subset=KEY_COLUMNS + ["alpha_2"], keep="last")
    summarize_alpha2(raw, out_dir)
    return raw


def summarize_alpha2(raw: pd.DataFrame, out_dir: str) -> pd.DataFrame:
    ok = raw[raw["status"] == "ok"]
    summary = (ok.groupby(["setting", "dataset", "method", "alpha_2"])
                 .agg(ari_mean=("ari", "mean"), ari_std=("ari", "std"), nmi_mean=("nmi", "mean"),
                      nmi_std=("nmi", "std"), certain_mean=("n_certain", "mean"), n_runs=("run", "count"))
                 .reset_index())
    summary.to_csv(os.path.join(out_dir, "summary.csv"), index=False)

    # Stability report: how much does each dataset's score move across the grid?
    stab = []
    for (setting, dataset, method), g in summary.groupby(["setting", "dataset", "method"]):
        for metric in ("ari", "nmi"):
            col = g[f"{metric}_mean"]
            stab.append({"setting": setting, "dataset": dataset, "method": method, "metric": metric,
                         "best_alpha_2": float(g.loc[col.idxmax(), "alpha_2"]),
                         "best": col.max(), "worst": col.min(), "range": col.max() - col.min(),
                         "at_0.7": float(g.loc[np.isclose(g["alpha_2"], 0.7), f"{metric}_mean"].iloc[0])
                         if np.isclose(g["alpha_2"], 0.7).any() else np.nan})
    stability = pd.DataFrame(stab)
    stability.to_csv(os.path.join(out_dir, "stability.csv"), index=False)

    # Average rank of each alpha_2 value across datasets (rank 1 = best)
    ranks = []
    for (setting, method), g in summary.groupby(["setting", "method"]):
        for metric in ("ari", "nmi"):
            piv = g.pivot(index="dataset", columns="alpha_2", values=f"{metric}_mean")
            avg = piv.rank(axis=1, ascending=False).mean()
            ranks.append({"setting": setting, "method": method, "metric": metric,
                          **{f"rank_alpha2_{a}": v for a, v in avg.items()}})
    pd.DataFrame(ranks).to_csv(os.path.join(out_dir, "average_ranks.csv"), index=False)

    _plot_alpha2(summary, out_dir)
    return summary


def _plot_alpha2(summary: pd.DataFrame, out_dir: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    for (setting, method), g in summary.groupby(["setting", "method"]):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
        for ax, metric in zip(axes, ("ari", "nmi")):
            for dataset, gd in g.groupby("dataset"):
                ax.errorbar(gd["alpha_2"], gd[f"{metric}_mean"], yerr=gd[f"{metric}_std"],
                            marker="o", capsize=3, label=dataset)
            ax.axvline(0.7, color="grey", ls="--", lw=1)
            ax.set_xlabel(r"$\alpha_2$")
            ax.set_ylabel(metric.upper())
            ax.grid(alpha=0.3)
        axes[1].legend(fontsize=8, loc="best")
        fig.suptitle(f"{method} - sensitivity to $\\alpha_2$ ({setting} k)")
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, f"alpha2_{method}_{setting}.png"), dpi=150)
        plt.close(fig)
