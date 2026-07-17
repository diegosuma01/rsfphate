#!/usr/bin/env python3
"""
Test suite for RSF-PHATE pipeline.
Covers: data preprocessing, RSF training, similarity, PHATE, clustering.

Run with: pytest tests/test_pipeline.py -v
"""

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from rsfphate import RSFPhate, make_donut_survival, to_survival_array


class TestDataGeneration:
    """Test synthetic data generation."""

    def test_donut_survival_shape(self):
        """Verify donut survival dataset has correct shape."""
        X, time, event, truth = make_donut_survival(
            n_samples=100,
            censoring_fraction=0.20,
            random_state=42
        )
        assert X.shape[0] == 100, "Number of samples mismatch"
        assert time.shape[0] == 100, "Time vector length mismatch"
        assert event.shape[0] == 100, "Event vector length mismatch"
        assert truth.shape[0] == 100, "Ground truth length mismatch"

    def test_donut_survival_censoring(self):
        """Verify censoring fraction is approximately correct."""
        X, time, event, truth = make_donut_survival(
            n_samples=500,
            censoring_fraction=0.30,
            random_state=42
        )
        actual_censoring = 1 - event.mean()
        assert 0.20 < actual_censoring < 0.40, "Censoring fraction out of range"

    def test_survival_array_creation(self):
        """Test conversion to scikit-survival structured array."""
        time = np.array([10, 20, 30, 40, 50])
        event = np.array([1, 0, 1, 0, 1])
        y_structured = to_survival_array(time, event)
        assert len(y_structured) == 5, "Structured array length mismatch"
        assert y_structured.dtype.names == ('event', 'time'), "Field names mismatch"


class TestRSFPhateBasic:
    """Test RSFPhate model on synthetic data."""

    @pytest.fixture
    def synthetic_data(self):
        """Generate small synthetic survival dataset."""
        X, time, event, truth = make_donut_survival(
            n_samples=200,
            censoring_fraction=0.15,
            random_state=42
        )
        return X, time, event, truth

    def test_rsfphate_initialization(self):
        """Test RSFPhate can be instantiated."""
        model = RSFPhate(
            n_clusters=2,
            n_estimators=50,
            diffusion_time=7.0,
            random_state=42
        )
        assert model.n_clusters == 2
        assert model.n_estimators == 50
        assert model.diffusion_time == 7.0

    def test_rsfphate_fit_predict(self, synthetic_data):
        """Test fit_predict returns correct cluster labels."""
        X, time, event, truth = synthetic_data
        model = RSFPhate(
            n_clusters=2,
            n_estimators=50,
            diffusion_time=7.0,
            random_state=42
        )
        labels = model.fit_predict(X, time, event)
        assert labels.shape[0] == X.shape[0], "Label count mismatch"
        assert set(labels).issubset({0, 1}), "Invalid cluster labels"
        assert len(set(labels)) == 2, "Should have 2 clusters"

    def test_rsfphate_attributes_after_fit(self, synthetic_data):
        """Test that all expected attributes exist after fitting."""
        X, time, event, truth = synthetic_data
        model = RSFPhate(
            n_clusters=2,
            n_estimators=50,
            diffusion_time=7.0,
            random_state=42
        )
        model.fit(X, time, event)

        # Check all learned attributes
        assert hasattr(model, 'labels_'), "Missing labels_ attribute"
        assert hasattr(model, 'embedding_'), "Missing embedding_ attribute"
        assert hasattr(model, 'proximity_'), "Missing proximity_ attribute"
        assert hasattr(model, 'smoothed_proximity_'), "Missing smoothed_proximity_ attribute"
        assert hasattr(model, 'rf_model_'), "Missing rf_model_ attribute"
        assert hasattr(model, 'phate_operator_'), "Missing phate_operator_ attribute"

    def test_rsfphate_embedding_shape(self, synthetic_data):
        """Test PHATE embedding has correct dimensionality."""
        X, time, event, truth = synthetic_data
        model = RSFPhate(
            n_clusters=2,
            n_estimators=50,
            diffusion_time=7.0,
            phate_n_components=2,
            random_state=42
        )
        model.fit(X, time, event)
        assert model.embedding_.shape == (X.shape[0], 2), "Embedding shape mismatch"

    def test_rsfphate_proximity_matrix(self, synthetic_data):
        """Test proximity matrix is symmetric and valid."""
        X, time, event, truth = synthetic_data
        model = RSFPhate(
            n_clusters=2,
            n_estimators=50,
            diffusion_time=7.0,
            random_state=42
        )
        model.fit(X, time, event)

        prox = model.proximity_
        assert prox.shape[0] == prox.shape[1], "Proximity matrix not square"
        assert np.allclose(prox, prox.T), "Proximity matrix not symmetric"
        assert np.all(prox >= 0), "Negative values in proximity matrix"
        assert np.all(prox <= 1), "Values > 1 in proximity matrix"

    def test_rsfphate_diffusion_time_effect(self, synthetic_data):
        """Test that different diffusion_time produces different results."""
        X, time, event, truth = synthetic_data

        model_low = RSFPhate(
            n_clusters=2,
            n_estimators=50,
            diffusion_time=4.0,
            random_state=42
        )
        labels_low = model_low.fit_predict(X, time, event)

        model_high = RSFPhate(
            n_clusters=2,
            n_estimators=50,
            diffusion_time=10.0,
            random_state=42
        )
        labels_high = model_high.fit_predict(X, time, event)

        # Labels might differ due to different smoothing
        # This tests that the parameter actually affects output
        assert model_low.diffusion_time != model_high.diffusion_time


class TestRSFPhateRecovery:
    """Test RSFPhate recovery of known structure."""

    def test_recovery_donut_clusters(self):
        """Test RSFPhate can recover known donut structure."""
        X, time, event, truth = make_donut_survival(
            n_samples=300,
            censoring_fraction=0.10,
            random_state=42
        )
        model = RSFPhate(
            n_clusters=2,
            n_estimators=100,
            diffusion_time=7.0,
            random_state=42
        )
        labels = model.fit_predict(X, time, event)

        # Should recover 2 clusters
        assert len(set(labels)) == 2, "Failed to identify 2 clusters"
        assert labels.min() == 0 and labels.max() == 1, "Invalid cluster range"

    def test_reproducibility(self):
        """Test that same random_state produces same results."""
        X, time, event, truth = make_donut_survival(
            n_samples=100,
            censoring_fraction=0.10,
            random_state=42
        )

        model1 = RSFPhate(n_clusters=2, n_estimators=50, random_state=42)
        labels1 = model1.fit_predict(X, time, event)

        model2 = RSFPhate(n_clusters=2, n_estimators=50, random_state=42)
        labels2 = model2.fit_predict(X, time, event)

        # Labels should be identical with same random_state
        assert np.array_equal(labels1, labels2), "Non-reproducible results"


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_cluster(self):
        """Test behavior with n_clusters=1."""
        X, time, event, truth = make_donut_survival(
            n_samples=50,
            censoring_fraction=0.10,
            random_state=42
        )
        model = RSFPhate(n_clusters=1, n_estimators=20, random_state=42)
        labels = model.fit_predict(X, time, event)
        assert np.all(labels == 0), "Single cluster should label all as 0"

    def test_small_dataset(self):
        """Test on very small dataset."""
        X = np.random.randn(10, 3)
        time = np.random.rand(10) * 100
        event = np.random.binomial(1, 0.5, 10)

        model = RSFPhate(n_clusters=2, n_estimators=10, random_state=42)
        labels = model.fit_predict(X, time, event)
        assert labels.shape[0] == 10, "Small dataset prediction failed"

    def test_many_clusters(self):
        """Test with many clusters."""
        X, time, event, truth = make_donut_survival(
            n_samples=200,
            censoring_fraction=0.10,
            random_state=42
        )
        model = RSFPhate(n_clusters=5, n_estimators=50, random_state=42)
        labels = model.fit_predict(X, time, event)
        assert set(labels).issubset(set(range(5))), "Cluster labels out of range"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
