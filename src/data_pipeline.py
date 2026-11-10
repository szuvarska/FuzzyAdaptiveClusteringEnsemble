import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold


class DataPreprocessor:
    """
    Utility class to handle the experimental data pipeline.
    Imputes missing values via medians, removes static (constant) columns and applies Z-Score.
    Ground-truth labels are never passed in here - they are kept aside for external evaluation only.
    """

    def __init__(self):
        # Set up the imputation, static-column removal and scaling transformers
        self.imputer = SimpleImputer(strategy='median')
        self.static_filter = VarianceThreshold(threshold=0.0)
        self.scaler = StandardScaler()

    def process_dataset(self, X_raw: np.ndarray) -> np.ndarray:
        """
        Executes the preprocessing pipeline on the raw feature matrix.
        """
        X_raw = np.asarray(X_raw, dtype=float)

        # Impute missing values (represented as NaN) with feature medians
        X_imputed = self.imputer.fit_transform(X_raw)

        # Remove static columns (zero variance)
        X_varying = self.static_filter.fit_transform(X_imputed)

        # Apply Z-score standardization
        X_scaled = self.scaler.fit_transform(X_varying)

        return X_scaled
