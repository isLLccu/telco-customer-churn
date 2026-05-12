# Telco Customer Churn Prediction & Value Segmentation

A machine learning pipeline that predicts customer churn and segments customers by value for a telecommunications dataset. The system covers the full workflow from raw data to actionable insights: data cleaning, exploratory analysis, classification modeling, threshold optimization, feature importance, and K-Means clustering.

## Overview

Customer churn is a critical business problem in telecommunications. This project builds a dual-module system:

- **Churn Prediction** — classifies whether a customer will churn, with threshold tuning to maximize business recall
- **Customer Segmentation** — groups customers into four behavioral segments to guide targeted retention strategies

**Dataset**: [IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) — 7,043 customers, 21 features, 26.5% churn rate.

## Results

| Model | ROC-AUC | F1 (threshold=0.28) | Recall |
|---|---|---|---|
| **Gradient Boosting** | **0.847** | **0.637** | **0.794** |
| Random Forest | 0.846 | 0.634 | 0.765 |
| Logistic Regression + SMOTE | 0.844 | 0.622 | 0.789 |

Cross-validation ROC-AUC (3-fold stratified): **0.848 ± 0.008**

**Top predictive features**: `ChargePerTenure`, `Contract_Month-to-month`, `InternetService_Fiber optic`

**Customer segments** (K-Means, k=4):

| Segment | Customers | Churn Rate | Avg Tenure | Avg Monthly Charge |
|---|---|---|---|---|
| High-risk new | 755 | 66.9% | 1.7 mo | $69.73 |
| At-risk growing | 1,987 | 39.3% | 19.4 mo | $78.71 |
| Stable high-value | 2,061 | 14.5% | 59.9 mo | $89.96 |
| Low-spend loyal | 2,240 | 12.8% | 28.9 mo | $27.53 |

## Project Structure

```
telco_project/
├── data/
│   └── raw/                        # Source CSV (included)
├── src/
│   └── run_analysis.py             # Reproducible end-to-end pipeline
├── outputs/
│   ├── figures/                    # EDA, ROC/PR curves, confusion matrix, segments
│   └── tables/                     # Model metrics, feature importance, cluster summary
├── Telco_Churn_Mining.ipynb        # Notebook with full narrative
└── requirements.txt
```

## Quickstart

**Requirements**: Python 3.10+

```bash
pip install -r requirements.txt
python src/run_analysis.py
```

All figures and tables are written to `outputs/`. The notebook `Telco_Churn_Mining.ipynb` provides the same pipeline with inline explanations.

## Pipeline Details

### Feature Engineering
Three derived features are created on top of the original 21:

- `ChargePerTenure` — monthly charge divided by `(tenure + 1)`, capturing cost-efficiency pressure on newer customers
- `AvgMonthlyCharge` — `TotalCharges / tenure`, falling back to `MonthlyCharges` for zero-tenure accounts
- `TenureSegment` — tenure bucketed into five lifecycle bands (0–6, 7–12, 13–24, 25–48, 49–72 months)

### Class Imbalance
The dataset is imbalanced (26.5% churn). Two strategies are applied and compared:
- SMOTE oversampling (Logistic Regression pipeline)
- `class_weight="balanced"` (Random Forest, Gradient Boosting)

### Threshold Optimization
Default threshold (0.5) maximizes accuracy but undershoots recall. The pipeline sweeps thresholds via the precision-recall curve and selects the one that maximizes F1, yielding a threshold of **0.28** and lifting recall from 51% to **79%**.

### Segmentation
K-Means (k=4) is applied to `[tenure, MonthlyCharges, TotalCharges, ChargePerTenure]` after standard scaling. Segments are ranked by churn rate to produce an actionable retention priority list.

## Reproducibility

- Random seed fixed at `42` throughout
- No hardcoded absolute paths — the script resolves paths relative to the project root
- All outputs are regenerated from scratch on each run
