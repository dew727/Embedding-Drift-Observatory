"""Tests for drift generator functions and batch simulator."""

import numpy as np
import pytest

from drift.covariate_shift import apply_covariate_shift
from drift.concept_drift import apply_concept_drift
from drift.noise_injection import inject_gaussian_noise, inject_missingness
from drift.prior_shift import apply_prior_shift
from drift.batch_simulator import simulate_batches, DriftConfig


RNG = np.random.default_rng(42)


def make_data(n=200, d=10, n_classes=2, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    y = rng.integers(0, n_classes, size=n)
    return X, y


# ---------------------------------------------------------------------------
# covariate_shift
# ---------------------------------------------------------------------------

class TestCovariateShift:
    def test_shape_preserved(self):
        X, _ = make_data()
        X_shifted = apply_covariate_shift(X, shift_strength=1.0, rng=RNG)
        assert X_shifted.shape == X.shape

    def test_values_change(self):
        X, _ = make_data()
        X_shifted = apply_covariate_shift(X, shift_strength=1.0, rng=RNG)
        assert not np.allclose(X, X_shifted)

    def test_zero_strength_no_change(self):
        X, _ = make_data()
        X_shifted = apply_covariate_shift(X, shift_strength=0.0, rng=RNG)
        np.testing.assert_array_equal(X, X_shifted)

    def test_feature_indices_respected(self):
        X, _ = make_data()
        X_shifted = apply_covariate_shift(X, shift_strength=2.0, feature_indices=[0], rng=RNG)
        # Only column 0 should change
        assert not np.allclose(X[:, 0], X_shifted[:, 0])
        np.testing.assert_array_equal(X[:, 1:], X_shifted[:, 1:])

    def test_original_not_mutated(self):
        X, _ = make_data()
        X_copy = X.copy()
        apply_covariate_shift(X, shift_strength=1.0, rng=RNG)
        np.testing.assert_array_equal(X, X_copy)


# ---------------------------------------------------------------------------
# concept_drift
# ---------------------------------------------------------------------------

class TestConceptDrift:
    def test_shape_preserved(self):
        X, y = make_data()
        X_out, y_out = apply_concept_drift(X, y, flip_rate=0.2, rng=RNG)
        assert X_out.shape == X.shape
        assert y_out.shape == y.shape

    def test_x_unchanged(self):
        X, y = make_data()
        X_out, _ = apply_concept_drift(X, y, flip_rate=0.2, rng=RNG)
        np.testing.assert_array_equal(X, X_out)

    def test_flip_rate_zero_no_change(self):
        X, y = make_data()
        _, y_out = apply_concept_drift(X, y, flip_rate=0.0, rng=RNG)
        np.testing.assert_array_equal(y, y_out)

    def test_approx_flip_rate(self):
        X, y = make_data(n=1000)
        _, y_out = apply_concept_drift(X, y, flip_rate=0.3, rng=RNG)
        changed = (y != y_out).mean()
        # Allow some tolerance since flipped labels may land on same class by chance
        assert 0.20 <= changed <= 0.35

    def test_labels_stay_valid(self):
        X, y = make_data()
        classes = np.unique(y)
        _, y_out = apply_concept_drift(X, y, flip_rate=0.5, rng=RNG)
        assert set(np.unique(y_out)).issubset(set(classes))


# ---------------------------------------------------------------------------
# noise_injection
# ---------------------------------------------------------------------------

class TestNoiseInjection:
    def test_gaussian_shape_preserved(self):
        X, _ = make_data()
        X_noisy = inject_gaussian_noise(X, noise_std=0.1, rng=RNG)
        assert X_noisy.shape == X.shape

    def test_gaussian_zero_std_no_change(self):
        X, _ = make_data()
        X_noisy = inject_gaussian_noise(X, noise_std=0.0, rng=RNG)
        np.testing.assert_array_almost_equal(X, X_noisy)

    def test_gaussian_changes_values(self):
        X, _ = make_data()
        X_noisy = inject_gaussian_noise(X, noise_std=1.0, rng=RNG)
        assert not np.allclose(X, X_noisy)

    def test_gaussian_original_not_mutated(self):
        X, _ = make_data()
        X_copy = X.copy()
        inject_gaussian_noise(X, noise_std=0.5, rng=RNG)
        np.testing.assert_array_equal(X, X_copy)

    def test_missingness_shape_preserved(self):
        X, _ = make_data()
        X_miss = inject_missingness(X, missing_rate=0.1, rng=RNG)
        assert X_miss.shape == X.shape

    def test_missingness_approx_rate(self):
        X, _ = make_data(n=1000, d=20)
        X_miss = inject_missingness(X, missing_rate=0.2, fill_value=0.0, rng=RNG)
        zero_rate = (X_miss == 0.0).mean()
        assert 0.15 <= zero_rate <= 0.25

    def test_missingness_zero_rate_no_change(self):
        X, _ = make_data()
        X_miss = inject_missingness(X, missing_rate=0.0, rng=RNG)
        np.testing.assert_array_equal(X, X_miss)


# ---------------------------------------------------------------------------
# prior_shift
# ---------------------------------------------------------------------------

class TestPriorShift:
    def test_shape_preserved(self):
        X, y = make_data()
        X_out, y_out = apply_prior_shift(X, y, target_ratio=0.8, majority_class=1, rng=RNG)
        assert X_out.shape == X.shape
        assert y_out.shape == y.shape

    def test_majority_class_ratio(self):
        X, y = make_data(n=1000)
        target = 0.8
        _, y_out = apply_prior_shift(X, y, target_ratio=target, majority_class=1, rng=RNG)
        actual = (y_out == 1).mean()
        assert abs(actual - target) < 0.05

    def test_labels_stay_valid(self):
        X, y = make_data()
        classes = np.unique(y)
        _, y_out = apply_prior_shift(X, y, target_ratio=0.9, majority_class=1, rng=RNG)
        assert set(np.unique(y_out)).issubset(set(classes))


# ---------------------------------------------------------------------------
# batch_simulator
# ---------------------------------------------------------------------------

class TestBatchSimulator:
    def test_returns_correct_n_batches(self):
        X, y = make_data()
        configs = [DriftConfig() for _ in range(4)]
        batches = simulate_batches(X, y, n_batches=4, drift_configs=configs)
        assert len(batches) == 4

    def test_batch_indices_sequential(self):
        X, y = make_data()
        configs = [DriftConfig() for _ in range(3)]
        batches = simulate_batches(X, y, n_batches=3, drift_configs=configs)
        assert [b.index for b in batches] == [0, 1, 2]

    def test_mismatched_configs_raises(self):
        X, y = make_data()
        with pytest.raises(ValueError):
            simulate_batches(X, y, n_batches=3, drift_configs=[DriftConfig()])

    def test_no_drift_config_preserves_shape(self):
        X, y = make_data()
        configs = [DriftConfig() for _ in range(2)]
        batches = simulate_batches(X, y, n_batches=2, drift_configs=configs)
        for b in batches:
            assert b.X.shape[1] == X.shape[1]

    def test_covariate_drift_applied(self):
        X, y = make_data(n=400)
        cfg = DriftConfig(covariate_strength=3.0)
        batches = simulate_batches(X, y, n_batches=2, drift_configs=[cfg, cfg], rng=RNG)
        batch_size = len(X) // 2
        orig = X[:batch_size]
        assert not np.allclose(orig, batches[0].X)
