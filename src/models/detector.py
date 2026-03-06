"""
Anomaly detection on gut microbiome samples.

Five detectors compared:
  1. IsolationForest   – tree-based anomaly scoring (main method per roadmap)
  2. LocalOutlierFactor – density-based local outlier detection
  3. OneClassSVM        – kernel-based one-class classification
  4. EllipticEnvelope   – Gaussian assumption, Mahalanobis distance
  5. ZScore_PCA         – simple PCA reconstruction error baseline

All detectors are trained on normal samples only, then scored on full dataset.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score,
    roc_curve, precision_recall_curve,
)

logger = logging.getLogger(__name__)

DETECTORS = ["isolation_forest", "lof", "ocsvm", "elliptic", "pca_recon"]


class AnomalyDetector:
    """
    Fits multiple anomaly detectors on normal training data and
    scores all samples.

    Parameters
    ----------
    contamination : expected fraction of anomalies (for threshold tuning)
    seed          : random seed
    """

    def __init__(self, contamination: float = 0.10, seed: int = 42):
        self.contamination = contamination
        self.seed = seed
        self._scalers: dict   = {}
        self._models:  dict   = {}
        self._pca:     Optional[PCA] = None
        self.scores_:  dict[str, np.ndarray] = {}
        self.metrics_: dict[str, dict] = {}

    # ── Fit ───────────────────────────────────────────────────────────────────
    def fit(self, X_normal: np.ndarray) -> "AnomalyDetector":
        """Fit all detectors on normal (inlier) data only."""
        logger.info("Fitting anomaly detectors on %d normal samples …", len(X_normal))

        sc = StandardScaler().fit(X_normal)
        self._scalers["standard"] = sc
        Xs = sc.transform(X_normal)

        # 1. Isolation Forest
        self._models["isolation_forest"] = IsolationForest(
            n_estimators=200,
            contamination=self.contamination,
            random_state=self.seed,
            n_jobs=1,
        ).fit(Xs)
        logger.info("  ✅ Isolation Forest fitted")

        # 2. Local Outlier Factor (novelty=True → fit on normal only)
        self._models["lof"] = LocalOutlierFactor(
            n_neighbors=20,
            contamination=self.contamination,
            novelty=True,
            n_jobs=1,
        ).fit(Xs)
        logger.info("  ✅ LOF fitted")

        # 3. One-Class SVM
        self._models["ocsvm"] = OneClassSVM(
            kernel="rbf", nu=self.contamination, gamma="scale"
        ).fit(Xs)
        logger.info("  ✅ One-Class SVM fitted")

        # 4. Elliptic Envelope
        try:
            self._models["elliptic"] = EllipticEnvelope(
                contamination=self.contamination,
                random_state=self.seed,
                support_fraction=0.85,
            ).fit(Xs)
            logger.info("  ✅ Elliptic Envelope fitted")
        except Exception as e:
            logger.warning("Elliptic Envelope failed: %s — skipping", e)
            self._models["elliptic"] = None

        # 5. PCA reconstruction error
        n_comp = min(20, Xs.shape[1], Xs.shape[0] - 1)
        self._pca = PCA(n_components=n_comp, random_state=self.seed).fit(Xs)
        logger.info("  ✅ PCA (n_components=%d) fitted", n_comp)

        return self

    # ── Score ─────────────────────────────────────────────────────────────────
    def score_samples(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """
        Return anomaly scores for each detector.
        Higher score = more anomalous (we negate sklearn's convention
        where lower = more anomalous).
        """
        Xs = self._scalers["standard"].transform(X)
        raw: dict[str, np.ndarray] = {}

        # sklearn detectors return decision_function where lower = more anomalous
        # We negate so higher = more anomalous (easier to interpret)
        raw["isolation_forest"] = -self._models["isolation_forest"].decision_function(Xs)
        raw["lof"]              = -self._models["lof"].decision_function(Xs)
        raw["ocsvm"]            = -self._models["ocsvm"].decision_function(Xs)

        if self._models["elliptic"] is not None:
            raw["elliptic"] = -self._models["elliptic"].decision_function(Xs)
        else:
            raw["elliptic"] = np.zeros(len(Xs))

        # PCA reconstruction error
        Z     = self._pca.transform(Xs)
        X_hat = self._pca.inverse_transform(Z)
        raw["pca_recon"] = np.mean((Xs - X_hat) ** 2, axis=1)

        # Min-max normalise each score to [0, 1]
        self.scores_ = {}
        for k, v in raw.items():
            lo, hi = v.min(), v.max()
            self.scores_[k] = (v - lo) / (hi - lo + 1e-10)

        return self.scores_

    # ── Evaluate ──────────────────────────────────────────────────────────────
    def evaluate(
        self,
        y_true_binary: np.ndarray,   # 1 = anomaly, 0 = normal
    ) -> dict[str, dict]:
        """
        Compute AUC-ROC, AUC-PR, precision, recall, F1 for all detectors.
        Threshold chosen at contamination fraction.
        """
        if not self.scores_:
            raise RuntimeError("Call score_samples() first.")

        results = {}
        for name, scores in self.scores_.items():
            thr  = np.percentile(scores, 100 * (1 - self.contamination))
            pred = (scores >= thr).astype(int)

            fpr, tpr, _ = roc_curve(y_true_binary, scores)
            prec_c, rec_c, _ = precision_recall_curve(y_true_binary, scores)

            results[name] = {
                "auc_roc":   round(roc_auc_score(y_true_binary, scores), 4),
                "auc_pr":    round(average_precision_score(y_true_binary, scores), 4),
                "precision": round(precision_score(y_true_binary, pred, zero_division=0), 4),
                "recall":    round(recall_score(y_true_binary, pred, zero_division=0), 4),
                "f1":        round(f1_score(y_true_binary, pred, zero_division=0), 4),
                "fpr":       fpr,
                "tpr":       tpr,
                "prec_curve": prec_c,
                "rec_curve":  rec_c,
                "threshold":  thr,
                "pred":       pred,
            }
            logger.info(
                "  %-18s  AUC-ROC=%.3f  AUC-PR=%.3f  F1=%.3f",
                name, results[name]["auc_roc"],
                results[name]["auc_pr"], results[name]["f1"],
            )

        self.metrics_ = results
        return results

    # ── Per-type breakdown ─────────────────────────────────────────────────────
    def anomaly_type_recall(
        self,
        y_str: np.ndarray,
        detector: str = "isolation_forest",
    ) -> dict[str, float]:
        """Recall per anomaly type for a specific detector."""
        if detector not in self.scores_:
            raise KeyError(f"{detector} not scored yet")
        scores = self.scores_[detector]
        thr    = self.metrics_[detector]["threshold"]
        pred   = scores >= thr

        types = [t for t in np.unique(y_str) if t != "normal"]
        recall = {}
        for t in types:
            mask = y_str == t
            if mask.sum() == 0:
                recall[t] = 0.0
            else:
                recall[t] = float(pred[mask].sum() / mask.sum())
        return recall

    # ── Save / load ────────────────────────────────────────────────────────────
    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("Saved detector → %s", path)

    @staticmethod
    def load(path: str | Path) -> "AnomalyDetector":
        return joblib.load(path)
