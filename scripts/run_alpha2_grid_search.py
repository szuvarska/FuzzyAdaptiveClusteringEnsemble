"""Grid search of the certainty threshold alpha_2 over [0.5, 0.9] for ACE and Fuzzy-ACE.

Stages 1-2 are computed once per run; only Stage 3 is re-run for each alpha_2 value.

Examples
    python scripts/run_alpha2_grid_search.py                 # default subset, fixed k, 10 runs
    python scripts/run_alpha2_grid_search.py --datasets iris wine glass ecoli yeast --settings fixed variable
"""
import argparse

import _bootstrap  # noqa: F401

from src.datasets import ALL_DATASETS
from src.experiments import SETTINGS, ExperimentConfig, run_alpha2_grid_search

DEFAULT_SUBSET = ["iris", "wine", "glass", "ecoli", "dermatology"]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", default="alpha2_grid_search")
    p.add_argument("--datasets", nargs="+", default=DEFAULT_SUBSET, choices=ALL_DATASETS)
    p.add_argument("--settings", nargs="+", default=["fixed"], choices=SETTINGS)
    p.add_argument("--methods", nargs="+", default=["ACE", "Fuzzy-ACE"], choices=["ACE", "Fuzzy-ACE"])
    p.add_argument("--alpha-2-values", nargs="+", type=float, default=[0.5, 0.6, 0.7, 0.8, 0.9])
    p.add_argument("--n-runs", type=int, default=10)
    p.add_argument("--base-seed", type=int, default=0)
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--no-resume", action="store_true")
    a = p.parse_args()

    cfg = ExperimentConfig(name=a.name, datasets=a.datasets, settings=a.settings, methods=a.methods,
                           n_runs=a.n_runs, base_seed=a.base_seed, n_jobs=a.n_jobs)
    run_alpha2_grid_search(cfg, alpha_2_values=a.alpha_2_values, resume=not a.no_resume)
    print(f"\nDone. Results in results/{cfg.name}/ (summary.csv, stability.csv, average_ranks.csv, figures/)")


if __name__ == "__main__":
    main()
