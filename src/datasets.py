"""
Local loader for the 12 UCI benchmark datasets used in the experiments.

Nothing is downloaded here. Each dataset is read from the raw UCI files that you place in
``data/raw/<dataset folder>/`` (see ``data/README.md`` for the exact file names and links).
Iris and Wine fall back to the copies bundled with scikit-learn when the raw files are absent.

Every loader returns a ``Dataset`` with the raw feature matrix (NaN for missing values),
the integer-encoded ground-truth labels (kept strictly for external evaluation) and k.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from src.data_pipeline import DataPreprocessor

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")


@dataclass
class Dataset:
    name: str
    X: np.ndarray            # raw features, NaN = missing
    y: np.ndarray            # integer-encoded ground truth (evaluation only)
    class_names: List[str] = field(default_factory=list)

    @property
    def k(self) -> int:
        return int(len(np.unique(self.y)))

    @property
    def n_instances(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.X.shape[1])

    def preprocessed(self) -> np.ndarray:
        """Median imputation, static-column removal and Z-score scaling."""
        return DataPreprocessor().process_dataset(self.X)


@dataclass
class DatasetSpec:
    """Describes how to parse a raw UCI file."""
    key: str
    display_name: str
    folder: str
    files: List[str]                 # file names or glob patterns, concatenated in order
    sep: str = ","                   # "," or r"\s+"
    label_col: int = -1              # index of the label column in the raw file
    drop_cols: List[int] = field(default_factory=list)  # e.g. ID / sequence-name columns
    na_values: List[str] = field(default_factory=lambda: ["?"])
    expected_shape: Optional[tuple] = None   # (n, d) after dropping label/ids - used as a sanity check
    expected_k: Optional[int] = None
    custom_parser: Optional[Callable[[List[str]], pd.DataFrame]] = None
    sklearn_fallback: Optional[str] = None
    source_url: str = ""


def _parse_segmentation(paths: List[str]) -> pd.DataFrame:
    """Image Segmentation files have a free-text header; keep only data lines (label + 19 numbers)."""
    rows = []
    for path in paths:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                parts = [p.strip() for p in line.strip().split(",")]
                if len(parts) != 20:
                    continue
                try:
                    values = [float(v) for v in parts[1:]]
                except ValueError:
                    continue  # header line with column names
                rows.append([parts[0]] + values)
    return pd.DataFrame(rows)


DATASET_SPECS: Dict[str, DatasetSpec] = {
    spec.key: spec for spec in [
        # --- Datasets 1-7: replicate the original ACE paper ---
        DatasetSpec("iris", "Iris", "iris", ["iris.data"], label_col=-1,
                    expected_shape=(150, 4), expected_k=3, sklearn_fallback="iris",
                    source_url="https://archive.ics.uci.edu/dataset/53/iris"),
        DatasetSpec("wine", "Wine", "wine", ["wine.data"], label_col=0,
                    expected_shape=(178, 13), expected_k=3, sklearn_fallback="wine",
                    source_url="https://archive.ics.uci.edu/dataset/109/wine"),
        DatasetSpec("new_thyroid", "New-Thyroid", "thyroid", ["new-thyroid.data"], label_col=0,
                    expected_shape=(215, 5), expected_k=3,
                    source_url="https://archive.ics.uci.edu/dataset/102/thyroid+disease"),
        DatasetSpec("glass", "Glass", "glass", ["glass.data"], label_col=-1, drop_cols=[0],
                    expected_shape=(214, 9), expected_k=6,
                    source_url="https://archive.ics.uci.edu/dataset/42/glass+identification"),
        DatasetSpec("breast_cancer", "Breast Cancer (Original)", "breast_cancer_wisconsin",
                    ["breast-cancer-wisconsin.data"], label_col=-1, drop_cols=[0],
                    expected_shape=(699, 9), expected_k=2,
                    source_url="https://archive.ics.uci.edu/dataset/15/breast+cancer+wisconsin+original"),
        DatasetSpec("soybean_small", "Soybean (Small)", "soybean", ["soybean-small.data"], label_col=-1,
                    expected_shape=(47, 35), expected_k=4,
                    source_url="https://archive.ics.uci.edu/dataset/91/soybean+small"),
        DatasetSpec("ionosphere", "Ionosphere", "ionosphere", ["ionosphere.data"], label_col=-1,
                    expected_shape=(351, 34), expected_k=2,
                    source_url="https://archive.ics.uci.edu/dataset/52/ionosphere"),
        # --- Datasets 8-12: added to test the fuzzy hypothesis ---
        DatasetSpec("ecoli", "Ecoli", "ecoli", ["ecoli.data"], sep=r"\s+", label_col=-1, drop_cols=[0],
                    expected_shape=(336, 7), expected_k=8,
                    source_url="https://archive.ics.uci.edu/dataset/39/ecoli"),
        DatasetSpec("yeast", "Yeast", "yeast", ["yeast.data"], sep=r"\s+", label_col=-1, drop_cols=[0],
                    expected_shape=(1484, 8), expected_k=10,
                    source_url="https://archive.ics.uci.edu/dataset/110/yeast"),
        DatasetSpec("vehicle", "Statlog (Vehicle)", "vehicle", ["xa?.dat"], sep=r"\s+", label_col=-1,
                    expected_shape=(846, 18), expected_k=4,
                    source_url="https://archive.ics.uci.edu/dataset/149/statlog+vehicle+silhouettes"),
        DatasetSpec("image_segmentation", "Image Segmentation", "image_segmentation",
                    ["segmentation.data", "segmentation.test"], label_col=0,
                    expected_shape=(2310, 19), expected_k=7, custom_parser=_parse_segmentation,
                    source_url="https://archive.ics.uci.edu/dataset/50/image+segmentation"),
        DatasetSpec("dermatology", "Dermatology", "dermatology", ["dermatology.data"], label_col=-1,
                    expected_shape=(366, 34), expected_k=6,
                    source_url="https://archive.ics.uci.edu/dataset/33/dermatology"),
    ]
}

# Groups used by the experiment scripts
ACE_PAPER_DATASETS = ["iris", "wine", "new_thyroid", "glass", "breast_cancer", "soybean_small", "ionosphere"]
FUZZY_HYPOTHESIS_DATASETS = ["ecoli", "yeast", "vehicle", "image_segmentation", "dermatology"]
ALL_DATASETS = ACE_PAPER_DATASETS + FUZZY_HYPOTHESIS_DATASETS


def _resolve_files(spec: DatasetSpec, data_dir: str) -> List[str]:
    folder = os.path.join(data_dir, spec.folder)
    paths: List[str] = []
    for pattern in spec.files:
        matches = sorted(glob.glob(os.path.join(folder, pattern)))
        paths.extend(matches)
    return paths


def _from_dataframe(spec: DatasetSpec, df: pd.DataFrame) -> Dataset:
    df = df.dropna(axis=1, how="all")  # trailing separators create empty columns
    n_cols = df.shape[1]
    label_idx = spec.label_col % n_cols
    drop_idx = {c % n_cols for c in spec.drop_cols}

    labels_raw = df.iloc[:, label_idx].astype(str).str.strip()
    feature_idx = [c for c in range(n_cols) if c != label_idx and c not in drop_idx]
    features = df.iloc[:, feature_idx].apply(pd.to_numeric, errors="coerce")

    class_names, y = np.unique(labels_raw.to_numpy(), return_inverse=True)
    return Dataset(spec.display_name, features.to_numpy(dtype=float), y.astype(int), list(class_names))


def _from_sklearn(spec: DatasetSpec) -> Dataset:
    from sklearn import datasets as skds
    bunch = getattr(skds, f"load_{spec.sklearn_fallback}")()
    return Dataset(spec.display_name, bunch.data.astype(float), bunch.target.astype(int),
                   [str(n) for n in bunch.target_names])


def _validate(spec: DatasetSpec, ds: Dataset, strict: bool) -> None:
    problems = []
    if spec.expected_shape and (ds.n_instances, ds.n_features) != spec.expected_shape:
        problems.append(f"shape {(ds.n_instances, ds.n_features)} != expected {spec.expected_shape}")
    if spec.expected_k and ds.k != spec.expected_k:
        problems.append(f"k={ds.k} != expected {spec.expected_k}")
    if problems:
        msg = f"[{spec.key}] unexpected raw file contents: " + "; ".join(problems)
        if strict:
            raise ValueError(msg)
        print("WARNING:", msg)


def load_dataset(key: str, data_dir: str = DEFAULT_DATA_DIR, strict: bool = False) -> Dataset:
    """Loads one dataset from local raw UCI files (never downloads)."""
    if key not in DATASET_SPECS:
        raise KeyError(f"Unknown dataset '{key}'. Available: {list(DATASET_SPECS)}")
    spec = DATASET_SPECS[key]
    paths = _resolve_files(spec, data_dir)

    if paths:
        if spec.custom_parser is not None:
            df = spec.custom_parser(paths)
        else:
            frames = [pd.read_csv(p, sep=spec.sep, header=None, na_values=spec.na_values,
                                  skipinitialspace=True, engine="python") for p in paths]
            df = pd.concat(frames, ignore_index=True)
        ds = _from_dataframe(spec, df)
    elif spec.sklearn_fallback:
        ds = _from_sklearn(spec)
    else:
        expected = os.path.join(data_dir, spec.folder, " | ".join(spec.files))
        raise FileNotFoundError(
            f"Raw files for '{key}' not found. Place them at: {expected}\n"
            f"Source: {spec.source_url}"
        )

    _validate(spec, ds, strict)
    return ds


def available_datasets(data_dir: str = DEFAULT_DATA_DIR) -> List[str]:
    """Keys of datasets that can be loaded right now (raw files present or sklearn fallback)."""
    return [k for k, s in DATASET_SPECS.items() if _resolve_files(s, data_dir) or s.sklearn_fallback]


def dataset_overview(data_dir: str = DEFAULT_DATA_DIR) -> pd.DataFrame:
    """Table of every dataset, whether it is available, and its size."""
    rows = []
    for key, spec in DATASET_SPECS.items():
        row = {"key": key, "name": spec.display_name, "available": False}
        try:
            ds = load_dataset(key, data_dir)
            row.update(available=True, n=ds.n_instances, d=ds.n_features, k=ds.k,
                       missing_values=int(np.isnan(ds.X).sum()))
        except FileNotFoundError:
            row.update(n=spec.expected_shape[0], d=spec.expected_shape[1], k=spec.expected_k)
        rows.append(row)
    return pd.DataFrame(rows)
