from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


class DataPreprocessor:
    """
    Utility class to handle the experimental data pipeline.
    Strips ground truth, imputes missing values via medians, and applies Z-Score.
    """

    def __init__(self):
        # Set up the imputation and scaling transformers
        self.imputer = SimpleImputer(strategy='median')
        self.scaler = StandardScaler()

    def process_dataset(self, X_raw: np.ndarray) -> np.ndarray:
        """
        Executes the preprocessing pipeline on the raw feature matrix.
        """
        # Impute missing values (represented as NaN) with feature medians
        X_imputed = self.imputer.fit_transform(X_raw)

        # Apply Z-score standardization
        X_scaled = self.scaler.fit_transform(X_imputed)

        return X_scaled
