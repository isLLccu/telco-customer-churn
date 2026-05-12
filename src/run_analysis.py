"""Run the full Telco customer churn mining pipeline.

This script covers data cleaning, feature engineering, classification models,
customer segmentation, and visualization. It is intentionally self-contained so
that the course project can be reproduced with a single command.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
BASE = Path(__file__).resolve().parents[1]
DATA_PATH = BASE / "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
FIG_DIR = BASE / "outputs/figures"
TABLE_DIR = BASE / "outputs/tables"
PROCESSED_DIR = BASE / "data/processed"
for directory in [FIG_DIR, TABLE_DIR, PROCESSED_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


def savefig(name: str) -> None:
    """Save the active matplotlib figure into the project figure folder."""
    plt.tight_layout()
    plt.savefig(FIG_DIR / name, dpi=180)
    plt.close()


def load_and_clean_data() -> pd.DataFrame:
    """Load the raw CSV, clean missing values, and create derived features."""
    df = pd.read_csv(DATA_PATH)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].replace(" ", np.nan), errors="coerce")
    # The 11 missing TotalCharges records are zero-tenure accounts; filling with 0 is business-consistent.
    df["TotalCharges"] = df["TotalCharges"].fillna(0)
    df["ChurnLabel"] = (df["Churn"] == "Yes").astype(int)
    df["AvgMonthlyCharge"] = (df["TotalCharges"] / df["tenure"].replace(0, np.nan)).fillna(
        df["MonthlyCharges"]
    )
    df["TenureSegment"] = pd.cut(
        df["tenure"],
        [-1, 6, 12, 24, 48, 72],
        labels=["0-6", "7-12", "13-24", "25-48", "49-72"],
    )
    df["ChargePerTenure"] = df["MonthlyCharges"] / (df["tenure"] + 1)
    df.to_csv(PROCESSED_DIR / "telco_cleaned.csv", index=False)
    return df


def make_eda_figures(df: pd.DataFrame) -> None:
    """Generate exploratory data analysis figures."""
    summary = {
        "n_samples": int(len(df)),
        "n_columns_original": 21,
        "churn_count": int(df["ChurnLabel"].sum()),
        "non_churn_count": int((1 - df["ChurnLabel"]).sum()),
        "churn_rate": float(df["ChurnLabel"].mean()),
        "missing_totalcharges_filled": 11,
    }
    (TABLE_DIR / "data_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    plt.figure(figsize=(5, 4))
    counts = df["Churn"].value_counts()
    plt.bar(counts.index, counts.values)
    plt.title("Churn class distribution")
    plt.ylabel("Count")
    for i, v in enumerate(counts.values):
        plt.text(i, v + 50, str(v), ha="center")
    savefig("class_distribution.png")

    plt.figure(figsize=(7, 4))
    sns.histplot(data=df, x="tenure", hue="Churn", multiple="stack", bins=24)
    plt.title("Tenure distribution by churn")
    savefig("tenure_churn.png")

    plt.figure(figsize=(7, 4))
    sns.boxplot(data=df, x="Churn", y="MonthlyCharges")
    plt.title("Monthly charges by churn")
    savefig("monthly_charges_box.png")

    plt.figure(figsize=(8, 4.5))
    rate = df.groupby("Contract")["ChurnLabel"].mean().sort_values(ascending=False)
    plt.bar(rate.index, rate.values)
    plt.ylabel("Churn rate")
    plt.title("Churn rate by contract type")
    for i, v in enumerate(rate.values):
        plt.text(i, v + 0.015, f"{v:.1%}", ha="center")
    savefig("contract_churn_rate.png")


def build_preprocessor(X: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    """Create preprocessing pipeline for numerical and categorical features."""
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(exclude=["object", "category"]).columns.tolist()
    preprocessor = ColumnTransformer(
        [
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
                num_cols,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_cols,
            ),
        ]
    )
    return preprocessor, num_cols, cat_cols


def train_and_evaluate_models(df: pd.DataFrame) -> None:
    """Train baseline and ensemble models, then export evaluation tables and curves."""
    X = df.drop(columns=["customerID", "Churn", "ChurnLabel"])
    y = df["ChurnLabel"]
    preprocessor, num_cols, cat_cols = build_preprocessor(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    models = {
        "LogisticRegression_SMOTE": ImbPipeline(
            [
                ("preprocess", preprocessor),
                ("smote", SMOTE(random_state=RANDOM_STATE)),
                (
                    "model",
                    LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
                ),
            ]
        ),
        "RandomForest": Pipeline(
            [
                ("preprocess", preprocessor),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=120,
                        max_depth=10,
                        min_samples_leaf=8,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "GradientBoosting": Pipeline(
            [
                ("preprocess", preprocessor),
                (
                    "model",
                    GradientBoostingClassifier(
                        n_estimators=120, learning_rate=0.05, max_depth=3, random_state=RANDOM_STATE
                    ),
                ),
            ]
        ),
    }

    metrics, roc_data, pr_data = [], {}, {}
    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        proba = pipeline.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        metrics.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(y_test, pred),
                "Precision": precision_score(y_test, pred),
                "Recall": recall_score(y_test, pred),
                "F1": f1_score(y_test, pred),
                "ROC_AUC": roc_auc_score(y_test, proba),
                "PR_AUC": average_precision_score(y_test, proba),
            }
        )
        fpr, tpr, _ = roc_curve(y_test, proba)
        precision, recall, _ = precision_recall_curve(y_test, proba)
        roc_data[name] = (fpr, tpr, roc_auc_score(y_test, proba))
        pr_data[name] = (precision, recall, average_precision_score(y_test, proba))

    metrics_df = pd.DataFrame(metrics).sort_values("ROC_AUC", ascending=False)
    metrics_df.to_csv(TABLE_DIR / "model_metrics.csv", index=False)

    cv_rows = []
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    for name in ["LogisticRegression_SMOTE", "GradientBoosting"]:
        scores = cross_val_score(models[name], X, y, scoring="roc_auc", cv=cv, n_jobs=1)
        cv_rows.append({"Model": name, "CV_ROC_AUC_Mean": scores.mean(), "CV_ROC_AUC_Std": scores.std()})
    pd.DataFrame(cv_rows).to_csv(TABLE_DIR / "cv_metrics.csv", index=False)

    plt.figure(figsize=(7, 5))
    for name, (fpr, tpr, roc_auc) in roc_data.items():
        plt.plot(fpr, tpr, label=f"{name} AUC={roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.title("ROC curves")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(fontsize=8)
    savefig("roc_curves.png")

    plt.figure(figsize=(7, 5))
    for name, (precision, recall, pr_auc) in pr_data.items():
        plt.plot(recall, precision, label=f"{name} AP={pr_auc:.3f}")
    plt.title("Precision-Recall curves")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(fontsize=8)
    savefig("pr_curves.png")

    best_name = str(metrics_df.iloc[0]["Model"])
    best_pipeline = models[best_name]
    proba = best_pipeline.predict_proba(X_test)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, proba)
    f1_values = 2 * precision * recall / (precision + recall + 1e-9)
    best_idx = int(np.nanargmax(f1_values[:-1]))
    best_threshold = float(thresholds[best_idx])
    best_pred = (proba >= best_threshold).astype(int)
    cm = confusion_matrix(y_test, best_pred)

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"Confusion matrix: {best_name}\nthreshold={best_threshold:.3f}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    savefig("confusion_matrix_best.png")

    onehot = best_pipeline.named_steps["preprocess"].named_transformers_["cat"].named_steps["onehot"]
    feature_names = num_cols + list(onehot.get_feature_names_out(cat_cols))
    model = best_pipeline.named_steps["model"]
    importance = getattr(model, "feature_importances_", np.zeros(len(feature_names)))
    importance_df = (
        pd.DataFrame({"Feature": feature_names, "Importance": importance})
        .sort_values("Importance", ascending=False)
        .head(20)
    )
    importance_df.to_csv(TABLE_DIR / "feature_importance_top20.csv", index=False)
    plt.figure(figsize=(8, 6))
    plt.barh(importance_df["Feature"][::-1], importance_df["Importance"][::-1])
    plt.title(f"Top 20 feature importance ({best_name})")
    plt.xlabel("Importance")
    savefig("feature_importance_top20.png")

    result = {
        "best_model": best_name,
        "best_threshold": best_threshold,
        "test_metrics_best_default_threshold": metrics_df.iloc[0].to_dict(),
        "test_metrics_best_f1_threshold": {
            "Accuracy": accuracy_score(y_test, best_pred),
            "Precision": precision_score(y_test, best_pred),
            "Recall": recall_score(y_test, best_pred),
            "F1": f1_score(y_test, best_pred),
            "ROC_AUC": roc_auc_score(y_test, proba),
            "PR_AUC": average_precision_score(y_test, proba),
            "ConfusionMatrix": cm.tolist(),
        },
    }
    (TABLE_DIR / "final_results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


def segment_customers(df: pd.DataFrame) -> None:
    """Perform K-Means segmentation for customer value and churn-risk analysis."""
    cluster_features = ["tenure", "MonthlyCharges", "TotalCharges", "ChargePerTenure"]
    features_scaled = StandardScaler().fit_transform(df[cluster_features])
    df["CustomerSegment"] = KMeans(n_clusters=4, random_state=RANDOM_STATE, n_init=20).fit_predict(
        features_scaled
    )
    cluster_summary = (
        df.groupby("CustomerSegment")
        .agg(
            Customers=("customerID", "count"),
            ChurnRate=("ChurnLabel", "mean"),
            AvgTenure=("tenure", "mean"),
            AvgMonthlyCharges=("MonthlyCharges", "mean"),
            AvgTotalCharges=("TotalCharges", "mean"),
        )
        .reset_index()
        .sort_values("ChurnRate", ascending=False)
    )
    cluster_summary.to_csv(TABLE_DIR / "cluster_summary.csv", index=False)
    plt.figure(figsize=(7, 5))
    for segment in sorted(df["CustomerSegment"].unique()):
        subset = df[df["CustomerSegment"] == segment]
        plt.scatter(subset["tenure"], subset["MonthlyCharges"], s=10, alpha=0.45, label=f"Segment {segment}")
    plt.title("K-Means customer segmentation")
    plt.xlabel("tenure")
    plt.ylabel("MonthlyCharges")
    plt.legend(fontsize=8)
    savefig("customer_segments.png")


def main() -> None:
    """Execute the end-to-end project pipeline."""
    df = load_and_clean_data()
    make_eda_figures(df)
    train_and_evaluate_models(df)
    segment_customers(df)


if __name__ == "__main__":
    main()
