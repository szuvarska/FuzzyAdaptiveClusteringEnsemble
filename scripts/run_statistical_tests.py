"""Iman-Davenport + Nemenyi tests and CD diagrams for a finished experiment.

Examples
    python scripts/run_statistical_tests.py                          # results/main, alpha = 0.1
    python scripts/run_statistical_tests.py --name main --alpha 0.05
    python scripts/run_statistical_tests.py --datasets ecoli yeast vehicle image_segmentation dermatology --tag fuzzy_subset
"""
import argparse
import os

import _bootstrap  # noqa: F401

from src.experiments import DEFAULT_RESULTS_DIR
from src.stats_tests import run_statistical_tests


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", default="main")
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--methods", nargs="+", default=None)
    p.add_argument("--datasets", nargs="+", default=None)
    p.add_argument("--tag", default="", help="suffix for the output folder, e.g. when using a subset")
    a = p.parse_args()

    results = run_statistical_tests(os.path.join(DEFAULT_RESULTS_DIR, a.name), alpha=a.alpha,
                                    methods=a.methods, datasets=a.datasets, tag=a.tag)
    for r in results:
        print(f"\n== {r.metric.upper()} / {r.setting} k  (N={r.n_datasets} datasets, k={r.n_methods} methods)")
        print("Average ranks:", ", ".join(f"{m} {v:.2f}" for m, v in r.average_ranks.items()))
        print(f"Friedman chi2={r.friedman_chi2:.3f} (p={r.friedman_p:.4f}); "
              f"Iman-Davenport F({r.df1},{r.df2})={r.iman_davenport_f:.3f} (p={r.iman_davenport_p:.4f}) "
              f"-> {'REJECT H0' if r.reject_h0 else 'no significant difference'}")
        if r.reject_h0:
            print(f"Nemenyi CD={r.critical_difference:.3f}; significant:",
                  "; ".join(f"{x} > {y}" for x, y, _ in r.significant_pairs) or "none")


if __name__ == "__main__":
    main()
