# Fuzzy Adaptive Clustering Ensemble (Fuzzy-ACE)

This repository contains the implementation of the Adaptive Clustering Ensemble (ACE) framework, originally proposed by Alqurashi and Wang (2019), along with a novel extension: Fuzzy-ACE.

By substituting rigid K-Means base models with Fuzzy C-Means (FCM) and adapting the consensus merging mathematics (continuous sums and soft fuzzy products), Fuzzy-ACE is designed to better handle datasets with overlapping features and ambiguous spatial boundaries.

## Project Overview

Ensemble clustering combines multiple base clusterings into a single, highly robust consensus partition. The ACE framework achieves this without relying on computationally expensive $O(N^2)$ co-association matrices. Instead, it utilizes a bipartite-like global membership matrix ($\Theta_1$) and iteratively merges candidate clusters using set correlation and cosine similarity.

## Key Features
- **Baseline ACE**: Full implementation of the original framework using K-Means and discrete binary math.
- **Fuzzy-ACE (Proposed)**: Novel extension utilizing Fuzzy C-Means and continuous bounded math.
- **Subspace Sampling**: Built-in structural diversity using 70% data instance sampling and 70% feature sampling.
- **Variable $k$ Support**: Base partitions can dynamically generate initial clusters within a $[k-2, k+2]$ interval.

## Installation
Ensure you have Python 3.8+ installed. Install the necessary dependencies via pip:

```bash
pip install -r requirements.txt
```
Dependencies: `numpy`, `scikit-learn`, `scikit-fuzzy`, `scipy`, `pandas`, `matplotlib`, `joblib`.

## Repository Structure
- `src/`: Main source code directory.
  - `ensemble.py`: The ACE / Fuzzy-ACE ensemble (Stages 1–3).
  - `models.py`: Base members (K-Means, Fuzzy C-Means).
  - `metrics.py`: Cardinality / overlap / set-correlation strategies (binary and fuzzy).
  - `data_pipeline.py`: Preprocessing (median imputation, constant-column removal, z-score).
  - `datasets.py`: Loader for the 12 UCI datasets from local raw files (no downloading).
  - `baselines.py`: Comparison methods – single K-Means, single FCM, EAC (co-association + average linkage).
  - `experiments.py`: Experiment runner, result storage and the α₂ grid search.
  - `stats_tests.py`: Friedman / Iman–Davenport / Nemenyi tests and critical-difference diagrams.
- `scripts/`: Command-line entry points (see below).
- `data/`: Put raw UCI files in `data/raw/` – see [`data/README.md`](data/README.md).
- `results/`: Stored experiment results.
- `notebooks/`: Jupyter notebooks – `02_framework_test.ipynb` runs the whole framework on Iris, Wine and Breast Cancer.

## Running the Experiments
1. Download the raw UCI files into `data/raw/` (`bash data/download_uci.sh`, details in [`data/README.md`](data/README.md)), then check them:
   ```bash
   python scripts/check_datasets.py
   ```
2. Main experiment – 5 methods × 12 datasets × {fixed, variable} k × 10 seeded runs, followed by the statistical tests:
   ```bash
   python scripts/run_experiments.py --n-jobs -1
   ```
   Every finished run is written to disk immediately; re-running the same command resumes where it stopped.
   Datasets whose files are missing are skipped with a message.
3. α₂ sensitivity (grid search over 0.5–0.9 for ACE and Fuzzy-ACE on a subset of datasets):
   ```bash
   python scripts/run_alpha2_grid_search.py --n-jobs -1
   ```
4. Re-run the statistical tests on stored results (e.g. another significance level or a dataset subset):
   ```bash
   python scripts/run_statistical_tests.py --name main --alpha 0.1
   python scripts/run_statistical_tests.py --name main --datasets ecoli yeast vehicle image_segmentation dermatology --tag fuzzy_subset
   ```

Default hyperparameters follow the plan: m = 10 members (5 × 70 % instances, 5 × 70 % features), α₁ = 0.8, Δα = 0.1,
α₂ = 0.7, Fuzzy-ACE fuzzifier drawn from [1.5, 2.5] per member, single FCM baseline with m = 2.

### Stored results
```
results/<experiment>/
├── config.json              # hyperparameters, package versions, git commit, run sessions
├── raw_results.csv          # one row per run: ARI, NMI, #clusters, runtime, status/error, seed
├── predicted_labels.csv     # predicted labels of every run
├── summary.csv              # mean/std per (setting, dataset, method)
├── tables/                  # dataset × method "mean ± std" tables per metric and setting
└── statistics/
    ├── tests_summary.csv    # Friedman, Iman–Davenport, CD, significant pairs
    ├── tests.json
    ├── ranks_<metric>_<setting>.csv
    ├── nemenyi_pvalues_<metric>_<setting>.csv   # only when H0 is rejected
    └── cd_diagrams/cd_<metric>_<setting>.png
results/alpha2_grid_search/
├── raw_results.csv, summary.csv
├── stability.csv            # best α₂, score range across the grid, score at 0.7
├── average_ranks.csv        # average rank of each α₂ value across datasets
└── figures/alpha2_<method>_<setting>.png
```

## Quick Start
The framework is designed to mimic the standard scikit-learn estimator API. You can easily run it on any preprocessed tabular dataset:

```python
from src.ensemble import AdaptiveClusteringEnsemble
from src.models import FCMMember
from sklearn.datasets import load_iris
from src.data_pipeline import DataPreprocessor

# 1. Load and Scale Data
X_raw = load_iris().data
X_scaled = DataPreprocessor().process_dataset(X_raw)

# 2. Initialize the Fuzzy-ACE framework
ace_fcm = AdaptiveClusteringEnsemble(
    base_member=FCMMember(m_fuzz=2.0), # Swap with KMeansMember() for baseline
    m_partitions=10, 
    k_clusters=3, 
    variable_k=True,  # Enables [k-2, k+2] dynamic cluster intervals
    random_state=42
)

# 3. Fit and Predict
labels = ace_fcm.fit_predict(X_scaled)
print(labels)
```

## Team
- Marta Szuwarska ([@szuvarska](https://github.com/szuvarska))
- Łukasz Grabarski ([@LukaszGrabarski](https://github.com/LukaszGrabarski))
