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
Dependencies: `numpy`, `scikit-learn`, `scikit-fuzzy`.

## Repository Structure
- `src/`: Main source code directory.
  - `ensemble.py`: Contains the main ensemble classes.
  - `models.py`: Contains the base model classes.
  - `data_pipeline.py`: Contains data preprocessing classes.
- `notebooks/`: Jupyter notebooks for experimentation and demonstration.

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
