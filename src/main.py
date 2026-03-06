"""
Day 22 · Anomaly Detection in Microbiome Samples
=================================================
Detect contamination, dysbiosis, low-biomass & batch outliers
using Isolation Forest (+ 4 other detectors for comparison).

Usage
-----
    python -m src.main
    python -m src.main --config configs/config.yaml
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.simulator     import build_dataset, ANOMALY_TYPES
from src.models.detector    import AnomalyDetector
from src.visualization.plots import generate_all
from src.utils.logger       import setup_logging
from src.utils.config       import load_config

logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--outdir", default="outputs")
    p.add_argument("--quiet",  action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = load_config(args.config)
    setup_logging(level=logging.WARNING if args.quiet else logging.INFO)
    out  = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    logger.info("=" * 60)
    logger.info("Day 22 · Anomaly Detection in Microbiome Samples")
    logger.info("=" * 60)

    data_cfg = cfg.get("data", {})
    det_cfg  = cfg.get("detector", {})

    # ── Step 1: Simulate dataset ──────────────────────────────────────────────
    logger.info("[1/5] Building dataset …")
    X_df, y_s = build_dataset(
        n_normal=data_cfg.get("n_normal", 300),
        n_anomaly_per_type=data_cfg.get("n_anomaly_per_type", 15),
        seed=data_cfg.get("seed", 42),
    )
    y_str = y_s.values.astype(str)
    y_bin = (y_str != "normal").astype(int)

    logger.info("Dataset: %d samples (%d normal, %d anomalies)",
                len(X_df), (y_bin == 0).sum(), (y_bin == 1).sum())

    # Split: normal-only train set for fitting detectors
    X_normal = X_df.values[y_bin == 0]

    # ── Step 2: Fit detectors ─────────────────────────────────────────────────
    logger.info("[2/5] Fitting detectors on normal samples …")
    det = AnomalyDetector(
        contamination=det_cfg.get("contamination", 0.15),
        seed=det_cfg.get("seed", 42),
    )
    det.fit(X_normal)

    # ── Step 3: Score all samples ─────────────────────────────────────────────
    logger.info("[3/5] Scoring all samples …")
    scores  = det.score_samples(X_df.values)
    metrics = det.evaluate(y_bin)

    # ── Step 4: Save outputs ──────────────────────────────────────────────────
    logger.info("[4/5] Saving outputs …")
    det.save(out / "models" / "anomaly_detector.joblib")

    # Results table
    score_df = pd.DataFrame(scores, index=X_df.index)
    score_df["true_label"] = y_str
    score_df["is_anomaly"] = y_bin
    score_df.to_csv(out / "anomaly_scores.csv")

    # Metrics table
    rows = [{
        "detector":  d,
        "auc_roc":   m["auc_roc"],
        "auc_pr":    m["auc_pr"],
        "precision": m["precision"],
        "recall":    m["recall"],
        "f1":        m["f1"],
    } for d, m in metrics.items()]
    pd.DataFrame(rows).to_csv(out / "detector_metrics.csv", index=False)

    # Per-type recall
    type_recalls = {
        d: det.anomaly_type_recall(y_str, detector=d)
        for d in scores
    }
    pd.DataFrame(type_recalls).to_csv(out / "per_type_recall.csv")

    # ── Step 5: Figures ────────────────────────────────────────────────────────
    logger.info("[5/5] Generating figures …")
    generate_all(X_df, y_str, det, out)

    # ── Print summary ──────────────────────────────────────────────────────────
    elapsed  = time.time() - t0
    best_det = max(metrics, key=lambda d: metrics[d]["auc_roc"])

    print("\n" + "="*54)
    print("  Day 22 · Anomaly Detection Summary")
    print("="*54)
    print(f"  Samples    : {len(X_df)}  ({(y_bin==0).sum()} normal, {(y_bin==1).sum()} anomalies)")
    print(f"  OTU features: {X_df.shape[1]}")
    print(f"  Anomaly types: {', '.join(ANOMALY_TYPES)}")
    print()
    for _, row in pd.DataFrame(rows).sort_values("auc_roc", ascending=False).iterrows():
        flag = " ← best" if row["detector"] == best_det else ""
        print(f"  {row['detector']:<20}  AUC-ROC={row['auc_roc']:.3f}"
              f"  F1={row['f1']:.3f}{flag}")
    print(f"\n  Figures    : 9 saved to {out}/")
    print(f"  Elapsed    : {elapsed:.1f}s")
    print("="*54 + "\n")


if __name__ == "__main__":
    main()
