"""Lists the 12 datasets, whether their raw files are in data/raw, and their size.

    python scripts/check_datasets.py
"""
import _bootstrap  # noqa: F401
import pandas as pd

from src.datasets import DEFAULT_DATA_DIR, dataset_overview

if __name__ == "__main__":
    pd.set_option("display.width", 200)
    print(f"Looking in: {DEFAULT_DATA_DIR}\n")
    print(dataset_overview().to_string(index=False))
