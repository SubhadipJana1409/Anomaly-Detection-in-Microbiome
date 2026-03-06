"""
9 publication-quality figures for microbiome anomaly detection.

fig1 : Sample overview PCA – normal vs anomaly types
fig2 : OTU abundance heatmap – top variable OTUs, sorted by anomaly type
fig3 : Anomaly score distributions – violin/box per label, per detector
fig4 : ROC curves – all 5 detectors
fig5 : Precision-Recall curves – all 5 detectors
fig6 : Performance bar chart – AUC-ROC, AUC-PR, F1
fig7 : Per-anomaly-type recall heatmap (detector × anomaly type)
fig8 : Score heatmap – sample × detector (sorted by Isolation Forest score)
fig9 : Summary – contamination sensitivity + top anomalous samples
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)

TYPE_PALETTE = {
    "normal":       "#3498DB",
    "contamination":"#E74C3C",
    "dysbiosis":    "#F39C12",
    "low_biomass":  "#9B59B6",
    "batch_outlier":"#1ABC9C",
}
DET_PALETTE = {
    "isolation_forest": "#E74C3C",
    "lof":              "#3498DB",
    "ocsvm":            "#2ECC71",
    "elliptic":         "#F39C12",
    "pca_recon":        "#9B59B6",
}
DET_LABELS = {
    "isolation_forest": "Isolation Forest",
    "lof":              "Local Outlier Factor",
    "ocsvm":            "One-Class SVM",
    "elliptic":         "Elliptic Envelope",
    "pca_recon":        "PCA Reconstruction",
}
DPI = 150


def _save(fig, out_dir: Path, name: str) -> None:
    p = out_dir / name
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", p)


# ── Fig 1: PCA overview ───────────────────────────────────────────────────────
def fig1_pca_overview(X: np.ndarray, y_str: np.ndarray, out_dir: Path) -> None:
    pca    = PCA(n_components=3)
    coords = pca.fit_transform(X)
    var    = pca.explained_variance_ratio_ * 100

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Sample Overview: PCA of CLR-Transformed OTU Profiles",
                 fontsize=13, fontweight="bold")

    for ax, (xi, yi) in zip(axes, [(0,1),(0,2)]):
        for lbl, col in TYPE_PALETTE.items():
            idx = y_str == lbl
            if not idx.any(): continue
            ax.scatter(coords[idx, xi], coords[idx, yi], c=col, s=20 if lbl=="normal" else 60,
                       alpha=0.5 if lbl=="normal" else 0.9,
                       edgecolors="white" if lbl!="normal" else "none", lw=0.5,
                       label=lbl.replace("_"," ").title(), zorder=3 if lbl!="normal" else 1)
        ax.set_xlabel(f"PC{xi+1} ({var[xi]:.1f}%)")
        ax.set_ylabel(f"PC{yi+1} ({var[yi]:.1f}%)")
        ax.spines[["top","right"]].set_visible(False)
    axes[0].set_title("PC1 vs PC2"); axes[1].set_title("PC1 vs PC3")
    axes[0].legend(fontsize=9, frameon=False, markerscale=1.5)
    plt.tight_layout()
    _save(fig, out_dir, "fig1_pca_overview.png")


# ── Fig 2: Heatmap top variable OTUs ──────────────────────────────────────────
def fig2_otu_heatmap(
    X_df: pd.DataFrame, y_str: np.ndarray, out_dir: Path, n_show: int = 30
) -> None:
    order = np.argsort(X_df.var(axis=0))[-n_show:]
    mat   = X_df.iloc[:, order].copy()
    # Sort samples by label
    label_order = ["normal","contamination","dysbiosis","low_biomass","batch_outlier"]
    sort_idx = np.concatenate([np.where(y_str==l)[0] for l in label_order if (y_str==l).any()])
    mat  = mat.iloc[sort_idx]
    y_s  = y_str[sort_idx]

    row_colors = pd.Series([TYPE_PALETTE.get(l,"gray") for l in y_s],
                            index=mat.index, name="Sample Type")
    g = sns.clustermap(mat.T, cmap="RdBu_r", center=0, vmin=-4, vmax=4,
                       col_colors=row_colors, col_cluster=False,
                       row_cluster=True, figsize=(13, 9),
                       xticklabels=False, yticklabels=True,
                       linewidths=0, cbar_kws={"label":"CLR"})
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), fontsize=7)
    g.figure.suptitle(f"Top {n_show} Variable OTUs by Sample Type",
                      fontsize=13, fontweight="bold", y=1.01)
    patches = [mpatches.Patch(color=v, label=k.replace("_"," ").title())
               for k,v in TYPE_PALETTE.items()]
    g.ax_col_dendrogram.legend(handles=patches, loc="center left",
                                bbox_to_anchor=(1.02,0.5), frameon=False, fontsize=9)
    g.figure.savefig(out_dir/"fig2_otu_heatmap.png", dpi=DPI, bbox_inches="tight")
    plt.close(g.figure)
    logger.info("Saved fig2_otu_heatmap.png")


# ── Fig 3: Score distributions ────────────────────────────────────────────────
def fig3_score_distributions(
    scores: dict[str, np.ndarray], y_str: np.ndarray, out_dir: Path
) -> None:
    det_names = list(scores.keys())
    n = len(det_names)
    fig, axes = plt.subplots(1, n, figsize=(4*n, 5))
    if n == 1: axes = [axes]
    fig.suptitle("Anomaly Score Distributions by Sample Type",
                 fontsize=13, fontweight="bold")

    for ax, name in zip(axes, det_names):
        df_plot = pd.DataFrame({"score": scores[name], "type": y_str})
        order = [l for l in TYPE_PALETTE if l in df_plot["type"].unique()]
        sns.boxplot(data=df_plot, x="type", y="score",
                    palette=TYPE_PALETTE, order=order, ax=ax,
                    linewidth=1.2, fliersize=3)
        ax.set_title(DET_LABELS.get(name, name), fontsize=10, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Anomaly Score (0–1)" if ax == axes[0] else "")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
        ax.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    _save(fig, out_dir, "fig3_score_distributions.png")


# ── Fig 4: ROC curves ─────────────────────────────────────────────────────────
def fig4_roc_curves(metrics: dict, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    for name, res in metrics.items():
        label = f"{DET_LABELS.get(name,name)} (AUC={res['auc_roc']:.3f})"
        ax.plot(res["fpr"], res["tpr"], color=DET_PALETTE.get(name,"gray"),
                lw=2.0, label=label)
    ax.plot([0,1],[0,1],"k--",lw=1,alpha=0.4)
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC Curves: Anomaly vs Normal", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, frameon=False, loc="lower right")
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    _save(fig, out_dir, "fig4_roc_curves.png")


# ── Fig 5: Precision-Recall curves ────────────────────────────────────────────
def fig5_pr_curves(metrics: dict, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    for name, res in metrics.items():
        label = f"{DET_LABELS.get(name,name)} (AP={res['auc_pr']:.3f})"
        ax.plot(res["rec_curve"], res["prec_curve"],
                color=DET_PALETTE.get(name,"gray"), lw=2.0, label=label)
    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title("Precision-Recall Curves", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, frameon=False)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    _save(fig, out_dir, "fig5_pr_curves.png")


# ── Fig 6: Performance bar chart ──────────────────────────────────────────────
def fig6_performance_bar(metrics: dict, out_dir: Path) -> None:
    met_keys = ["auc_roc", "auc_pr", "f1"]
    met_lbls = ["AUC-ROC", "AUC-PR", "F1"]
    names    = list(metrics.keys())
    x = np.arange(len(names)); w = 0.25

    fig, ax = plt.subplots(figsize=(12, 5))
    for j, (k, lbl) in enumerate(zip(met_keys, met_lbls)):
        vals = [metrics[n][k] for n in names]
        bars = ax.bar(x + j*w, vals, w, label=lbl, alpha=0.85, edgecolor="white")
        for b, v in zip(bars, vals):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.005,
                    f"{v:.3f}", ha="center", fontsize=8, fontweight="bold")
    ax.set_xticks(x + w)
    ax.set_xticklabels([DET_LABELS.get(n,n) for n in names], rotation=15, ha="right")
    ax.set_ylim(0, 1.15); ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Detector Performance Comparison", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, frameon=False)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    _save(fig, out_dir, "fig6_performance_bar.png")


# ── Fig 7: Per-type recall heatmap ────────────────────────────────────────────
def fig7_per_type_recall(type_recalls: dict[str, dict], out_dir: Path) -> None:
    dets  = list(type_recalls.keys())
    types = sorted({t for d in type_recalls.values() for t in d})
    mat   = pd.DataFrame(
        {d: [type_recalls[d].get(t, 0) for t in types] for d in dets},
        index=types,
    )
    fig, ax = plt.subplots(figsize=(len(dets)*2+2, len(types)*1.2+1.5))
    sns.heatmap(mat, annot=True, fmt=".2f", cmap="YlOrRd",
                vmin=0, vmax=1, ax=ax, linewidths=0.5,
                xticklabels=[DET_LABELS.get(d,d) for d in dets],
                yticklabels=[t.replace("_"," ").title() for t in types])
    ax.set_title("Recall per Anomaly Type × Detector",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right", fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=10)
    plt.tight_layout()
    _save(fig, out_dir, "fig7_per_type_recall.png")


# ── Fig 8: Score heatmap (sample × detector) ─────────────────────────────────
def fig8_score_heatmap(
    scores: dict[str, np.ndarray],
    y_str: np.ndarray,
    sample_ids: list[str],
    out_dir: Path,
    n_show: int = 80,
) -> None:
    mat = pd.DataFrame(scores, index=sample_ids)
    # Show top anomalous + random normal
    if_scores = mat["isolation_forest"].values
    anom_idx  = np.where(y_str != "normal")[0]
    norm_idx  = np.random.default_rng(0).choice(
        np.where(y_str == "normal")[0],
        size=min(n_show - len(anom_idx), 40), replace=False)
    show_idx  = np.concatenate([anom_idx, norm_idx])
    show_idx  = show_idx[np.argsort(if_scores[show_idx])[::-1]]

    sub = mat.iloc[show_idx]
    row_colors = pd.Series(
        [TYPE_PALETTE.get(y_str[i], "gray") for i in show_idx],
        index=sub.index, name="Type")

    g = sns.clustermap(sub, cmap="YlOrRd", vmin=0, vmax=1,
                       row_colors=row_colors, row_cluster=False,
                       col_cluster=False, figsize=(10, 9),
                       xticklabels=[DET_LABELS.get(c,c) for c in sub.columns],
                       yticklabels=False, linewidths=0,
                       cbar_kws={"label":"Normalised Score"})
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=20, ha="right")
    g.figure.suptitle("Anomaly Scores: Top Flagged Samples × Detectors",
                      fontsize=12, fontweight="bold", y=1.01)
    patches = [mpatches.Patch(color=v, label=k.replace("_"," ").title())
               for k,v in TYPE_PALETTE.items()]
    g.ax_col_dendrogram.legend(handles=patches, loc="center left",
                                bbox_to_anchor=(1.02,0.5), frameon=False, fontsize=9)
    g.figure.savefig(out_dir/"fig8_score_heatmap.png", dpi=DPI, bbox_inches="tight")
    plt.close(g.figure)
    logger.info("Saved fig8_score_heatmap.png")


# ── Fig 9: Summary ────────────────────────────────────────────────────────────
def fig9_summary(
    metrics: dict,
    scores: dict[str, np.ndarray],
    y_str: np.ndarray,
    sample_ids: list[str],
    out_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Anomaly Detection Summary: Gut Microbiome QC",
                 fontsize=13, fontweight="bold")

    # Panel A: Best detector AUC-ROC ranked
    ax = axes[0]
    sorted_dets = sorted(metrics.keys(), key=lambda d: metrics[d]["auc_roc"], reverse=True)
    aucs  = [metrics[d]["auc_roc"] for d in sorted_dets]
    cols  = [DET_PALETTE.get(d,"gray") for d in sorted_dets]
    bars  = ax.barh([DET_LABELS.get(d,d) for d in sorted_dets], aucs,
                    color=cols, edgecolor="white")
    for b, v in zip(bars, aucs):
        ax.text(v+0.005, b.get_y()+b.get_height()/2,
                f"{v:.3f}", va="center", fontsize=10, fontweight="bold")
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("AUC-ROC", fontsize=10)
    ax.set_title("Detector Ranking", fontsize=11)
    ax.spines[["top","right"]].set_visible(False)

    # Panel B: Anomaly type sample counts
    ax = axes[1]
    types, counts = np.unique(y_str, return_counts=True)
    colors = [TYPE_PALETTE.get(t,"gray") for t in types]
    bars = ax.bar([t.replace("_","\n") for t in types], counts,
                  color=colors, edgecolor="white")
    for b, c in zip(bars, counts):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.5,
                str(c), ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("Number of Samples")
    ax.set_title("Dataset Composition", fontsize=11)
    ax.spines[["top","right"]].set_visible(False)

    # Panel C: Score distribution — Isolation Forest
    ax = axes[2]
    if_scores = scores.get("isolation_forest", list(scores.values())[0])
    for lbl, col in TYPE_PALETTE.items():
        mask = y_str == lbl
        if mask.sum() == 0: continue
        ax.hist(if_scores[mask], bins=20, alpha=0.65, color=col,
                label=lbl.replace("_"," ").title(), density=True)
    ax.set_xlabel("Isolation Forest Score", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title("Score Distribution by Type", fontsize=11)
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    _save(fig, out_dir, "fig9_summary.png")


# ── Driver ─────────────────────────────────────────────────────────────────────
def generate_all(
    X_df: "pd.DataFrame",
    y_str: np.ndarray,
    detector: "AnomalyDetector",
    out_dir: Path,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Generating figures → %s", out_dir)

    scores  = detector.scores_
    metrics = detector.metrics_

    # Per-type recall for all detectors
    type_recalls = {
        d: detector.anomaly_type_recall(y_str, detector=d)
        for d in scores
    }

    fig1_pca_overview(X_df.values, y_str, out_dir)
    fig2_otu_heatmap(X_df, y_str, out_dir)
    fig3_score_distributions(scores, y_str, out_dir)
    fig4_roc_curves(metrics, out_dir)
    fig5_pr_curves(metrics, out_dir)
    fig6_performance_bar(metrics, out_dir)
    fig7_per_type_recall(type_recalls, out_dir)
    fig8_score_heatmap(scores, y_str, X_df.index.tolist(), out_dir)
    fig9_summary(metrics, scores, y_str, X_df.index.tolist(), out_dir)

    logger.info("All 9 figures saved.")
