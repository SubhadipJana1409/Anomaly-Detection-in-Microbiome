"""Tests for AnomalyDetector and visualization."""
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from src.data.simulator import build_dataset, ANOMALY_TYPES
from src.models.detector import AnomalyDetector, DETECTORS


@pytest.fixture(scope="module")
def fitted_detector():
    X_df, y_s = build_dataset(n_normal=150, n_anomaly_per_type=10, seed=0)
    y_str = y_s.values.astype(str)
    y_bin = (y_str != "normal").astype(int)
    X_normal = X_df.values[y_bin == 0]

    det = AnomalyDetector(contamination=0.15, seed=0)
    det.fit(X_normal)
    det.score_samples(X_df.values)
    det.evaluate(y_bin)
    return det, X_df, y_str, y_bin


class TestAnomalyDetector:
    def test_all_detectors_fitted(self, fitted_detector):
        det = fitted_detector[0]
        for name in DETECTORS:
            assert name in det._models or name == "pca_recon"

    def test_scores_shape(self, fitted_detector):
        det, X_df, _, _ = fitted_detector
        for name, scores in det.scores_.items():
            assert len(scores) == len(X_df), f"{name} score length mismatch"

    def test_scores_normalised(self, fitted_detector):
        det = fitted_detector[0]
        for name, scores in det.scores_.items():
            assert scores.min() >= -1e-6, f"{name} score below 0"
            assert scores.max() <= 1 + 1e-6, f"{name} score above 1"

    def test_scores_no_nan(self, fitted_detector):
        det = fitted_detector[0]
        for name, scores in det.scores_.items():
            assert not np.isnan(scores).any(), f"{name} has NaN scores"

    def test_metrics_keys(self, fitted_detector):
        det = fitted_detector[0]
        for name in det.metrics_:
            for key in ["auc_roc", "auc_pr", "precision", "recall", "f1"]:
                assert key in det.metrics_[name]

    def test_auc_range(self, fitted_detector):
        det = fitted_detector[0]
        for name, m in det.metrics_.items():
            assert 0.0 <= m["auc_roc"] <= 1.0, f"{name} AUC out of range"
            assert 0.0 <= m["auc_pr"]  <= 1.0

    def test_anomaly_type_recall_keys(self, fitted_detector):
        det, _, y_str, _ = fitted_detector
        recall = det.anomaly_type_recall(y_str, detector="isolation_forest")
        for atype in ANOMALY_TYPES:
            assert atype in recall

    def test_anomaly_type_recall_range(self, fitted_detector):
        det, _, y_str, _ = fitted_detector
        recall = det.anomaly_type_recall(y_str, detector="isolation_forest")
        for atype, r in recall.items():
            assert 0.0 <= r <= 1.0

    def test_save_load(self, fitted_detector, tmp_path):
        det, X_df, _, _ = fitted_detector
        p = tmp_path / "det.joblib"
        det.save(p)
        det2 = AnomalyDetector.load(p)
        scores2 = det2.score_samples(X_df.values)
        np.testing.assert_array_almost_equal(
            det.scores_["isolation_forest"],
            scores2["isolation_forest"],
        )

    def test_score_before_fit_empty(self):
        det = AnomalyDetector()
        # fit required before score
        rng = np.random.default_rng(0)
        X_n = rng.normal(size=(50, 20))
        det.fit(X_n)
        scores = det.score_samples(rng.normal(size=(10, 20)))
        assert len(scores) > 0

    def test_evaluate_before_score_raises(self):
        det = AnomalyDetector()
        with pytest.raises(RuntimeError):
            det.evaluate(np.array([0, 1]))


class TestPlots:
    def test_fig1_creates_file(self, fitted_detector, tmp_path):
        from src.visualization.plots import fig1_pca_overview
        _, X_df, y_str, _ = fitted_detector
        fig1_pca_overview(X_df.values, y_str, Path(tmp_path))
        assert (tmp_path / "fig1_pca_overview.png").exists()

    def test_fig4_creates_file(self, fitted_detector, tmp_path):
        from src.visualization.plots import fig4_roc_curves
        det = fitted_detector[0]
        fig4_roc_curves(det.metrics_, Path(tmp_path))
        assert (tmp_path / "fig4_roc_curves.png").exists()

    def test_fig6_creates_file(self, fitted_detector, tmp_path):
        from src.visualization.plots import fig6_performance_bar
        det = fitted_detector[0]
        fig6_performance_bar(det.metrics_, Path(tmp_path))
        assert (tmp_path / "fig6_performance_bar.png").exists()

    def test_fig7_creates_file(self, fitted_detector, tmp_path):
        from src.visualization.plots import fig7_per_type_recall
        det, _, y_str, _ = fitted_detector
        recalls = {d: det.anomaly_type_recall(y_str, d) for d in det.scores_}
        fig7_per_type_recall(recalls, Path(tmp_path))
        assert (tmp_path / "fig7_per_type_recall.png").exists()

    def test_fig9_creates_file(self, fitted_detector, tmp_path):
        from src.visualization.plots import fig9_summary
        det, X_df, y_str, _ = fitted_detector
        fig9_summary(det.metrics_, det.scores_, y_str,
                     X_df.index.tolist(), Path(tmp_path))
        assert (tmp_path / "fig9_summary.png").exists()
