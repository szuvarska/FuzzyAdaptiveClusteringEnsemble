"""Main experiment: 5 methods x 12 datasets x {fixed, variable} k x 10 runs.

Examples
    python scripts/run_experiments.py                         # full experiment (resumes if interrupted)
    python scripts/run_experiments.py --n-jobs -1             # use all CPU cores
    python scripts/run_experiments.py --datasets iris wine --n-runs 3 --name quick_test
    python scripts/run_experiments.py --alpha-2 0.8 --name main_alpha2_08
    python scripts/run_experiments.py --no-resume             # wipe stored runs and start over
"""
import argparse

import _bootstrap  # noqa: F401

from src.datasets import ALL_DATASETS
from src.experiments import METHODS, SETTINGS, ExperimentConfig, run_experiment
from src.stats_tests import run_statistical_tests


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", default="main", help="experiment folder name under results/")
    p.add_argument("--datasets", nargs="+", default=ALL_DATASETS, choices=ALL_DATASETS)
    p.add_argument("--settings", nargs="+", default=list(SETTINGS), choices=SETTINGS)
    p.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    p.add_argument("--n-runs", type=int, default=10)
    p.add_argument("--base-seed", type=int, default=0)
    p.add_argument("--alpha-1", type=float, default=0.8)
    p.add_argument("--alpha-2", type=float, default=0.7)
    p.add_argument("--delta-alpha", type=float, default=0.1)
    p.add_argument("--n-jobs", type=int, default=1, help="parallel workers (-1 = all cores)")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--no-stats", action="store_true", help="skip the statistical tests at the end")
    p.add_argument("--stats-alpha", type=float, default=0.1)
    a = p.parse_args()

    cfg = ExperimentConfig(name=a.name, datasets=a.datasets, settings=a.settings, methods=a.methods,
                           n_runs=a.n_runs, base_seed=a.base_seed, alpha_1=a.alpha_1, alpha_2=a.alpha_2,
                           delta_alpha=a.delta_alpha, n_jobs=a.n_jobs)
    run_experiment(cfg, resume=not a.no_resume)

    if not a.no_stats:
        import os
        run_statistical_tests(os.path.join(cfg.results_dir, cfg.name), alpha=a.stats_alpha)
    print(f"\nDone. Results in results/{cfg.name}/")


if __name__ == "__main__":
    main()
