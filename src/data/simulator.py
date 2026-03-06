"""
Simulate gut microbiome OTU data with realistic anomaly types.

Normal samples : Dirichlet-based healthy gut composition (HMP-style)
Anomaly types  :
  1. contamination   – spike of environmental/skin taxa (Staphylococcus, Bacillus)
  2. dysbiosis       – severe depletion of commensals + Proteobacteria bloom
  3. low_biomass     – very sparse profile (< 5 taxa detected), sequencing artefact
  4. batch_outlier   – shifted library composition due to extraction failure

Biological basis: HMP gut profile; dysbiosis markers from Halfvarson et al. 2017.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

# ── Gut taxa pool ─────────────────────────────────────────────────────────────
COMMENSALS = [
    "Faecalibacterium_prausnitzii", "Roseburia_intestinalis", "Blautia_obeum",
    "Eubacterium_rectale", "Eubacterium_hallii", "Ruminococcus_bromii",
    "Subdoligranulum_variabile", "Coprococcus_eutactus", "Lachnospira_multipara",
    "Bacteroides_thetaiotaomicron", "Bacteroides_vulgatus", "Bacteroides_uniformis",
    "Prevotella_copri", "Alistipes_shahii", "Parabacteroides_distasonis",
    "Bifidobacterium_longum", "Akkermansia_muciniphila", "Collinsella_aerofaciens",
]
PATHOBIONTS = [
    "Escherichia_coli", "Klebsiella_pneumoniae", "Ruminococcus_gnavus",
    "Fusobacterium_nucleatum", "Clostridium_difficile",
]
CONTAMINANTS = [
    "Staphylococcus_epidermidis", "Staphylococcus_aureus", "Bacillus_subtilis",
    "Pseudomonas_fluorescens", "Burkholderia_cepacia", "Ralstonia_pickettii",
    "Cutibacterium_acnes", "Corynebacterium_striatum",
]

def _build_otu_list(n: int = 150) -> list[str]:
    named = COMMENSALS + PATHOBIONTS + CONTAMINANTS
    extra = [f"OTU_{i:04d}" for i in range(n - len(named))]
    return (named + extra)[:n]

OTU_NAMES = _build_otu_list(150)
N_OTUS = len(OTU_NAMES)

COMMENSAL_IDX   = [OTU_NAMES.index(t) for t in COMMENSALS  if t in OTU_NAMES]
PATHOBIONT_IDX  = [OTU_NAMES.index(t) for t in PATHOBIONTS if t in OTU_NAMES]
CONTAMINANT_IDX = [OTU_NAMES.index(t) for t in CONTAMINANTS if t in OTU_NAMES]

ANOMALY_TYPES = ["contamination", "dysbiosis", "low_biomass", "batch_outlier"]


def _clr(X: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    X = X + eps
    lx = np.log(X)
    return lx - lx.mean(axis=1, keepdims=True)


def _healthy_alpha() -> np.ndarray:
    alpha = np.ones(N_OTUS) * 0.2
    for i in COMMENSAL_IDX:   alpha[i] = 2.5
    for i in PATHOBIONT_IDX:  alpha[i] = 0.3
    for i in CONTAMINANT_IDX: alpha[i] = 0.05
    return alpha


def simulate_normal(n: int = 300, seed: int = 0) -> np.ndarray:
    """Healthy gut microbiome profiles (CLR-transformed)."""
    rng   = np.random.default_rng(seed)
    alpha = _healthy_alpha()
    raw   = rng.dirichlet(alpha, size=n)
    mask  = rng.random(raw.shape) < 0.4
    raw[mask] = 0
    raw  /= raw.sum(axis=1, keepdims=True) + 1e-10
    return _clr(raw)


def simulate_anomalies(
    n_per_type: int = 15,
    seed: int = 99,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate all four anomaly types.
    Returns X (n_total × N_OTUS CLR) and labels array (int, 0-3).
    """
    rng = np.random.default_rng(seed)
    Xs, ys = [], []

    # ── Type 0: contamination ────────────────────────────────────────────────
    alpha = _healthy_alpha()
    raw = rng.dirichlet(alpha, size=n_per_type)
    # spike contaminants to 30-60 % of the profile
    for i in range(n_per_type):
        frac = rng.uniform(0.3, 0.6)
        cont = np.zeros(N_OTUS)
        sel  = rng.choice(CONTAMINANT_IDX, size=rng.integers(3, 7), replace=False)
        cont[sel] = rng.dirichlet(np.ones(len(sel)))
        raw[i] = (1 - frac) * raw[i] + frac * cont
    raw /= raw.sum(axis=1, keepdims=True) + 1e-10
    Xs.append(_clr(raw)); ys.extend([0] * n_per_type)

    # ── Type 1: dysbiosis ────────────────────────────────────────────────────
    alpha_d = np.ones(N_OTUS) * 0.1
    for i in PATHOBIONT_IDX: alpha_d[i] = 4.0
    for i in COMMENSAL_IDX:  alpha_d[i] = 0.1
    raw = rng.dirichlet(alpha_d, size=n_per_type)
    raw /= raw.sum(axis=1, keepdims=True) + 1e-10
    Xs.append(_clr(raw)); ys.extend([1] * n_per_type)

    # ── Type 2: low biomass ──────────────────────────────────────────────────
    raw = np.zeros((n_per_type, N_OTUS))
    for i in range(n_per_type):
        n_taxa = rng.integers(2, 6)
        sel = rng.choice(N_OTUS, size=n_taxa, replace=False)
        raw[i, sel] = rng.dirichlet(np.ones(n_taxa))
    raw /= raw.sum(axis=1, keepdims=True) + 1e-10
    Xs.append(_clr(raw)); ys.extend([2] * n_per_type)

    # ── Type 3: batch outlier ────────────────────────────────────────────────
    alpha_b = np.ones(N_OTUS) * 1.5   # very flat / uniform
    raw = rng.dirichlet(alpha_b, size=n_per_type)
    # also shift by a large random offset in CLR space
    clr_b = _clr(raw)
    clr_b += rng.normal(0, 3.0, size=(n_per_type, N_OTUS))
    Xs.append(clr_b); ys.extend([3] * n_per_type)

    return np.vstack(Xs), np.array(ys)


def build_dataset(
    n_normal: int = 300,
    n_anomaly_per_type: int = 15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Full dataset: normal + all anomaly types.
    Returns X (DataFrame) and y_true (Series: 'normal' or anomaly type name).
    """
    X_norm = simulate_normal(n_normal, seed=seed)
    X_anom, y_anom_int = simulate_anomalies(n_anomaly_per_type, seed=seed + 1)

    n_tot = n_normal + len(X_anom)
    ids_norm = [f"normal_{i+1:04d}" for i in range(n_normal)]
    ids_anom = [f"{ANOMALY_TYPES[t]}_{i+1:03d}"
                for i, t in enumerate(y_anom_int)]

    X_all = np.vstack([X_norm, X_anom])
    y_all = (["normal"] * n_normal +
             [ANOMALY_TYPES[t] for t in y_anom_int])

    X_df = pd.DataFrame(X_all, columns=OTU_NAMES,
                        index=ids_norm + ids_anom)
    y_s  = pd.Series(y_all, index=X_df.index, name="label")
    return X_df, y_s
