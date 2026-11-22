import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


class DataPreprocessor:
    """Median imputation -> removal of constant columns -> Z-score. Labels never pass through here."""

    def __init__(self):
        self.imputer = SimpleImputer(strategy="median")
        self.static_filter = VarianceThreshold(threshold=0.0)
        self.scaler = StandardScaler()

    def process_dataset(self, X_raw: np.ndarray) -> np.ndarray:
        X = self.imputer.fit_transform(np.asarray(X_raw, dtype=float))
        X = self.static_filter.fit_transform(X)
        return self.scaler.fit_transform(X)
