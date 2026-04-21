"""Standalone data preparation pipeline: load, clean, normalize, and split tabular datasets.

Usable independently of Streamlit. No st.* calls here.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class DatasetProcessor:
    """Handles the full data preparation pipeline for any tabular numeric CSV.

    Includes StandardScaler normalization — required before embedding because
    PCA and all distance metrics are sensitive to feature scale.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names_: list[str] = []
        self.label_col_: str = None
        self._fitted = False

    def load(self, source) -> pd.DataFrame:
        """Load a dataset from a file path or return a DataFrame as-is.

        Args:
            source: File path (str or Path) to a CSV, or an existing pd.DataFrame.

        Returns:
            pd.DataFrame with raw data.
        """
        if isinstance(source, pd.DataFrame):
            return source.copy()
        return pd.read_csv(source)

    def get_label_columns(self, df: pd.DataFrame) -> list[str]:
        """Return all column names that could serve as a classification label.

        Filters to columns with fewer than 50 unique values to exclude
        high-cardinality columns (IDs, free text) that are unlikely labels.
        """
        return [
            col for col in df.columns
            if df[col].nunique() < 50
        ]

    def preprocess(
        self,
        df: pd.DataFrame,
        label_col: str,
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Clean, encode, and normalize a DataFrame for embedding.

        Steps:
          1. Separate label column.
          2. Keep only numeric feature columns (drops strings/categoricals).
          3. Fill NaNs with per-column means.
          4. Fit and apply StandardScaler (fit is stored; call transform_new()
             to normalize held-out or drifted batches with the same scale).

        Args:
            df: Raw DataFrame.
            label_col: Name of the target/label column.

        Returns:
            X (n_samples, n_features) — normalized float32 features,
            y (n_samples,)            — label array,
            feature_names             — list of retained feature column names.
        """
        self.label_col_ = label_col

        y_raw = df[label_col]
        # Encode string labels to integers if necessary
        if y_raw.dtype == object or str(y_raw.dtype) == "category":
            y = pd.Categorical(y_raw).codes.astype(np.int64)
        else:
            y = y_raw.values.astype(np.int64)

        feature_df = df.drop(columns=[label_col]).select_dtypes(include=[np.number])
        self.feature_names_ = list(feature_df.columns)

        X = feature_df.values.astype(np.float64)

        # Fill NaNs with column means
        col_means = np.nanmean(X, axis=0)
        nan_mask = np.isnan(X)
        X[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

        # Fit scaler and normalize — must match scale used for all future batches
        X_scaled = self.scaler.fit_transform(X).astype(np.float32)
        self._fitted = True

        return X_scaled, y, self.feature_names_

    def transform_new(self, df: pd.DataFrame) -> np.ndarray:
        """Apply the already-fitted scaler to a new batch (e.g., a drifted batch).

        Drift batches must be normalized with the baseline scaler — refitting
        would absorb the drift signal into the scale parameters.

        Args:
            df: DataFrame with the same feature columns used during preprocess().

        Returns:
            X_scaled (n_samples, n_features) — float32, baseline-normalized.
        """
        if not self._fitted:
            raise RuntimeError("Call preprocess() on baseline data before transform_new().")

        feature_df = df[self.feature_names_] if isinstance(df, pd.DataFrame) else df
        X = np.array(feature_df, dtype=np.float64)

        col_means = np.nanmean(X, axis=0)
        nan_mask = np.isnan(X)
        X[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

        return self.scaler.transform(X).astype(np.float32)

    def split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_ratio: float = 0.2,
        seed: int = 42,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Randomly split X and y into train and test sets.

        Args:
            X: Feature matrix (n_samples, n_features).
            y: Label array (n_samples,).
            test_ratio: Fraction of samples to hold out as test set.
            seed: Random seed for reproducibility.

        Returns:
            X_train, X_test, y_train, y_test
        """
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(X))
        split = int(len(X) * (1 - test_ratio))
        train_idx, test_idx = idx[:split], idx[split:]
        return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
