"""Tests for src/data/simulator.py"""
import numpy as np
import pandas as pd
import pytest
from src.data.simulator import (
    simulate_normal, simulate_anomalies, build_dataset,
    OTU_NAMES, N_OTUS, ANOMALY_TYPES,
    CONTAMINANT_IDX, COMMENSAL_IDX, PATHOBIONT_IDX, _clr,
)


class TestSimulateNormal:
    def test_shape(self):
        X = simulate_normal(n=50, seed=0)
        assert X.shape == (50, N_OTUS)

    def test_no_nan(self):
        X = simulate_normal(n=30, seed=1)
        assert not np.isnan(X).any()

    def test_reproducible(self):
        X1 = simulate_normal(n=20, seed=7)
        X2 = simulate_normal(n=20, seed=7)
        np.testing.assert_array_equal(X1, X2)


class TestSimulateAnomalies:
    def test_shape(self):
        X, y = simulate_anomalies(n_per_type=10, seed=0)
        assert X.shape == (40, N_OTUS)
        assert len(y) == 40

    def test_label_range(self):
        _, y = simulate_anomalies(n_per_type=5, seed=0)
        assert set(y).issubset({0, 1, 2, 3})

    def test_all_types_present(self):
        _, y = simulate_anomalies(n_per_type=5, seed=0)
        assert set(y) == {0, 1, 2, 3}

    def test_contamination_has_high_contaminant(self):
        """Contaminated samples should have elevated contaminant OTU scores."""
        X, y = simulate_anomalies(n_per_type=20, seed=42)
        cont_mask = y == 0
        norm_X = simulate_normal(n=20, seed=42)
        cont_mean = X[cont_mask][:, CONTAMINANT_IDX].mean()
        norm_mean = norm_X[:, CONTAMINANT_IDX].mean()
        assert cont_mean > norm_mean


class TestBuildDataset:
    def test_returns_dataframe_and_series(self):
        X, y = build_dataset(n_normal=50, n_anomaly_per_type=5, seed=0)
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)

    def test_correct_total_size(self):
        X, y = build_dataset(n_normal=40, n_anomaly_per_type=5, seed=0)
        assert len(X) == 40 + 5*4
        assert len(y) == len(X)

    def test_label_values(self):
        _, y = build_dataset(n_normal=20, n_anomaly_per_type=5, seed=0)
        valid = {"normal"} | set(ANOMALY_TYPES)
        assert set(y.unique()).issubset(valid)

    def test_normal_count(self):
        X, y = build_dataset(n_normal=30, n_anomaly_per_type=5, seed=0)
        assert (y == "normal").sum() == 30

    def test_otu_columns(self):
        X, _ = build_dataset(n_normal=10, n_anomaly_per_type=3, seed=0)
        assert list(X.columns) == OTU_NAMES

    def test_clr_row_mean_zero_normal(self):
        """CLR rows should sum to ~0."""
        X, y = build_dataset(n_normal=20, n_anomaly_per_type=3, seed=0)
        row_means = X.values.mean(axis=1)
        mask = y != "batch_outlier"
        row_means = X.values[mask.values].mean(axis=1)
        assert np.allclose(row_means, 0, atol=1e-6)
