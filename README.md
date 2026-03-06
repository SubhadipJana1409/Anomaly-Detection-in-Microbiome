# Day 22 · Anomaly Detection in Microbiome Samples

**Isolation Forest + 4 detectors · 4 anomaly types · 9 publication-quality figures**

Part of the [#30DaysOfBioinformatics](https://github.com/SubhadipJana1409) challenge.
Previous: [Day 21 – Transfer Learning on Microbiome Data](https://github.com/SubhadipJana1409/day21-transfer-learning-microbiome)

---

## Overview

Microbiome studies are vulnerable to silent quality failures — contamination from skin/reagents, extreme dysbiosis misclassified as biology, low-biomass samples with unreliable profiles, and batch-level extraction failures. This pipeline uses **unsupervised anomaly detection** to automatically flag these outliers before downstream analysis.

**Main method:** Isolation Forest (sklearn), which isolates anomalies by building random trees — anomalous samples are isolated faster (fewer splits needed) because they occupy sparse regions of feature space.

**4 anomaly types detected:**

| Type | Biological Meaning |
|---|---|
| `contamination` | Environmental taxa spike (Staphylococcus, Bacillus) from reagent/skin |
| `dysbiosis` | Pathobiont bloom + commensal collapse (Proteobacteria-dominated gut) |
| `low_biomass` | < 5 taxa detected — unreliable sequencing depth |
| `batch_outlier` | Flat/shifted profile from extraction failure |

---

## Pipeline

```
CLR-transformed OTU matrix (150 features × 360 samples)
                │
                ▼
┌───────────────────────────────┐
│  Fit on normal samples only   │
│  (300 healthy gut profiles)   │
└──────────────┬────────────────┘
               │
       ┌───────┼───────────┐
       ▼       ▼           ▼
  Isolation  LOF       One-Class   Elliptic   PCA
   Forest             SVM         Envelope  Recon
       └───────┴───────────┘
               │
               ▼
    Anomaly score per sample [0,1]
    → Flag top contamination% as outliers
```

---

## 5 Detectors Compared

| Detector | Approach |
|---|---|
| **Isolation Forest** | Random tree isolation — fast, non-parametric |
| Local Outlier Factor | Density ratio to k-nearest neighbours |
| One-Class SVM | Kernel hypersphere around normal data |
| Elliptic Envelope | Mahalanobis distance (Gaussian assumption) |
| PCA Reconstruction | MSE of reconstruction from top 20 PCs |

---

## Figures

| Figure | Description |
|--------|-------------|
| `fig1_pca_overview.png`      | PCA of all samples, coloured by anomaly type |
| `fig2_otu_heatmap.png`       | Top 30 variable OTUs, sorted by sample type |
| `fig3_score_distributions.png` | Anomaly score boxplots by sample type |
| `fig4_roc_curves.png`        | ROC curves for all 5 detectors |
| `fig5_pr_curves.png`         | Precision-Recall curves |
| `fig6_performance_bar.png`   | AUC-ROC, AUC-PR, F1 grouped bar chart |
| `fig7_per_type_recall.png`   | Recall heatmap: detector × anomaly type |
| `fig8_score_heatmap.png`     | Score heatmap: top flagged samples × detectors |
| `fig9_summary.png`           | Ranking + dataset composition + IF score distribution |

---

## Quick Start

```bash
git clone https://github.com/SubhadipJana1409/day22-anomaly-detection-microbiome
cd day22-anomaly-detection-microbiome
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.main
```

### Use on Real Data

```python
import pandas as pd
from src.models.detector import AnomalyDetector

counts = pd.read_csv("my_clr_otus.csv", index_col=0)   # samples × OTUs
normal_idx = my_known_normal_samples                    # index of clean samples

det = AnomalyDetector(contamination=0.10)
det.fit(counts.loc[normal_idx].values)
scores = det.score_samples(counts.values)

# Top 10 most anomalous samples
import numpy as np
top10 = counts.index[np.argsort(scores["isolation_forest"])[::-1][:10]]
print(top10.tolist())
```

---

## Project Structure

```
day22-anomaly-detection-microbiome/
├── src/
│   ├── data/
│   │   └── simulator.py        # 4-type anomaly + normal data generator
│   ├── models/
│   │   └── detector.py         # AnomalyDetector (5 sklearn detectors)
│   ├── visualization/
│   │   └── plots.py            # All 9 figures
│   └── main.py
├── tests/
│   ├── test_simulator.py       # 13 tests
│   └── test_detector_plots.py  # 16 tests
├── configs/config.yaml
└── requirements.txt
```

---

## Tests

```bash
pytest tests/ -v
# 29 passed
```

---

## Methods

**Data**: CLR-transformed (centered log-ratio) OTU relative abundances with zero-inflation. CLR normalises for compositionality. Four anomaly types injected with realistic biological profiles (Halfvarson et al. 2017; Salter et al. 2014 for contamination models).

**Isolation Forest**: Ensemble of 200 isolation trees. Anomaly score = mean path length to isolation, normalised. Contamination fraction set to 0.15.

**Evaluation**: Since all anomaly labels are known in simulation, AUC-ROC and Average Precision (AUC-PR) are computed. Threshold set at the `1 - contamination` score percentile.

---

## References

1. Liu FT, Ting KM, Zhou Z-H (2008). Isolation Forest. *ICDM 2008*.
2. Salter SJ et al. (2014). Reagent and laboratory contamination can critically impact sequence-based microbiome analyses. *BMC Biology*.
3. Halfvarson J et al. (2017). Dynamics of the human gut microbiota in inflammatory bowel disease. *Nature Microbiology*.

---

## License

MIT
