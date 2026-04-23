"""Tests for DatasetProcessor: loading, preprocessing, splitting."""

import numpy as np
import pandas as pd
import pytest

from data.dataset_processor import DatasetProcessor


def make_df(n=100, n_features=5, n_classes=2, seed=42):
    rng = np.random.default_rng(seed)
    data = {f"feat_{i}": rng.standard_normal(n) for i in range(n_features)}
    data["label"] = rng.integers(0, n_classes, size=n)
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# preprocess
# ---------------------------------------------------------------------------

class TestPreprocess:
    def test_returns_correct_shapes(self):
        df = make_df(n=100, n_features=5)
        proc = DatasetProcessor()
        X, y, features = proc.preprocess(df, label_col="label")
        assert X.shape == (100, 5)
        assert y.shape == (100,)
        assert len(features) == 5

    def test_label_col_excluded_from_features(self):
        df = make_df()
        proc = DatasetProcessor()
        X, y, features = proc.preprocess(df, label_col="label")
        assert "label" not in features

    def test_output_is_float32(self):
        df = make_df()
        proc = DatasetProcessor()
        X, _, _ = proc.preprocess(df, label_col="label")
        assert X.dtype == np.float32

    def test_non_numeric_columns_dropped(self):
        df = make_df()
        df["text_col"] = "hello"
        proc = DatasetProcessor()
        X, _, features = proc.preprocess(df, label_col="label")
        assert "text_col" not in features

    def test_nans_filled(self):
        df = make_df()
        df.loc[0, "feat_0"] = np.nan
        proc = DatasetProcessor()
        X, _, _ = proc.preprocess(df, label_col="label")
        assert not np.isnan(X).any()

    def test_features_standardized(self):
        df = make_df(n=500)
        proc = DatasetProcessor()
        X, _, _ = proc.preprocess(df, label_col="label")
        # StandardScaler: each column should have mean ~0 and std ~1
        np.testing.assert_allclose(X.mean(axis=0), 0.0, atol=1e-5)
        np.testing.assert_allclose(X.std(axis=0), 1.0, atol=1e-5)

    def test_string_labels_encoded(self):
        df = make_df()
        df["label"] = df["label"].map({0: "cat", 1: "dog"})
        proc = DatasetProcessor()
        _, y, _ = proc.preprocess(df, label_col="label")
        assert y.dtype == np.int64
        assert set(np.unique(y)) == {0, 1}


# ---------------------------------------------------------------------------
# get_label_columns
# ---------------------------------------------------------------------------

class TestGetLabelColumns:
    def test_returns_low_cardinality_columns(self):
        df = make_df()
        proc = DatasetProcessor()
        candidates = proc.get_label_columns(df)
        assert "label" in candidates

    def test_excludes_high_cardinality(self):
        df = make_df(n=200)
        df["id"] = range(200)  # unique per row
        proc = DatasetProcessor()
        candidates = proc.get_label_columns(df)
        assert "id" not in candidates


# ---------------------------------------------------------------------------
# transform_new
# ---------------------------------------------------------------------------

class TestTransformNew:
    def test_raises_before_fit(self):
        df = make_df()
        proc = DatasetProcessor()
        with pytest.raises(RuntimeError):
            proc.transform_new(df)

    def test_same_scale_as_baseline(self):
        df = make_df(n=200)
        proc = DatasetProcessor()
        X, _, _ = proc.preprocess(df, label_col="label")
        # Transform the same data again — should give the same result
        X2 = proc.transform_new(df.drop(columns=["label"]))
        np.testing.assert_allclose(X, X2, atol=1e-5)

    def test_output_shape(self):
        df = make_df(n=200, n_features=5)
        proc = DatasetProcessor()
        proc.preprocess(df, label_col="label")
        new_df = make_df(n=50, n_features=5)
        X_new = proc.transform_new(new_df.drop(columns=["label"]))
        assert X_new.shape == (50, 5)


# ---------------------------------------------------------------------------
# split
# ---------------------------------------------------------------------------

class TestSplit:
    def test_split_sizes(self):
        df = make_df(n=100)
        proc = DatasetProcessor()
        X, y, _ = proc.preprocess(df, label_col="label")
        X_train, X_test, y_train, y_test = proc.split(X, y, test_ratio=0.2)
        assert len(X_train) == 80
        assert len(X_test) == 20

    def test_no_overlap(self):
        df = make_df(n=100)
        proc = DatasetProcessor()
        X, y, _ = proc.preprocess(df, label_col="label")
        X_train, X_test, _, _ = proc.split(X, y)
        # Concatenation should equal original length
        assert len(X_train) + len(X_test) == len(X)

    def test_reproducible_with_seed(self):
        df = make_df(n=100)
        proc = DatasetProcessor()
        X, y, _ = proc.preprocess(df, label_col="label")
        X_train_a, _, _, _ = proc.split(X, y, seed=0)
        X_train_b, _, _, _ = proc.split(X, y, seed=0)
        np.testing.assert_array_equal(X_train_a, X_train_b)
