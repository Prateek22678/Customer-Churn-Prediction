"""
evaluate.py
-----------
Evaluation module for the Churn Prediction project.

Computes and visualizes:
  - Accuracy, Precision, Recall, F1, ROC-AUC for all models
  - Confusion matrix
  - ROC curve
  - Precision-Recall curve
  - Threshold tuning (finding optimal decision threshold)
  - Error analysis

Business context:
  Missing a churner (False Negative) costs more than a false alarm
  (False Positive).  We tune threshold to maximize Recall for churners
  while keeping Precision reasonable.

Author: Churn Prediction Project
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (works without display)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    classification_report,
    average_precision_score,
)
from sklearn.pipeline import Pipeline


# ── Paths ──────────────────────────────────────────────────────────────────
FIGURES_DIR = Path(__file__).parent.parent / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Plot style ─────────────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-whitegrid")
COLORS = ["#2196F3", "#4CAF50", "#FF9800"]  # Blue, Green, Orange for 3 models
CHURN_COLOR = "#E91E63"                      # Pink for churn class


def evaluate_model(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute key classification metrics for a single model.

    Parameters
    ----------
    model : fitted sklearn Pipeline
    X_test, y_test : test data
    threshold : float
        Decision threshold (default 0.5). Values above this → predicted churn.

    Returns
    -------
    dict with keys: accuracy, precision, recall, f1, roc_auc
    """
    y_prob = model.predict_proba(X_test)[:, 1]  # Probability of churn
    y_pred = (y_prob >= threshold).astype(int)

    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_prob),
        "Avg Precision": average_precision_score(y_test, y_prob),
    }


def build_comparison_table(
    models: Dict[str, Pipeline],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Build a comparison table of all models' metrics.

    Parameters
    ----------
    models : dict mapping name → fitted pipeline
    X_test, y_test : test data
    threshold : float

    Returns
    -------
    pd.DataFrame with one row per model.
    """
    rows = []
    for name, model in models.items():
        metrics = evaluate_model(model, X_test, y_test, threshold)
        metrics["Model"] = name
        rows.append(metrics)

    df = pd.DataFrame(rows).set_index("Model")
    df = df[["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "Avg Precision"]]
    return df.round(4)


def plot_confusion_matrix(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    threshold: float = 0.5,
) -> None:
    """
    Plot and save a confusion matrix heatmap.

    Reading the confusion matrix:
      - True Negatives (top-left):  Correctly predicted "will NOT churn"
      - False Positives (top-right): Wrongly predicted "will churn" (false alarm)
      - False Negatives (bottom-left): Missed churners ← most costly!
      - True Positives (bottom-right): Correctly predicted "will churn"

    Parameters
    ----------
    model : fitted pipeline
    X_test, y_test : test data
    model_name : str
    threshold : float
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Predicted: Stay", "Predicted: Churn"],
        yticklabels=["Actual: Stay", "Actual: Churn"],
        ax=ax,
        linewidths=0.5,
        annot_kws={"size": 14, "weight": "bold"},
    )
    ax.set_title(f"Confusion Matrix — {model_name}\n(threshold={threshold:.2f})", fontsize=13)
    ax.set_ylabel("Actual Label", fontsize=11)
    ax.set_xlabel("Predicted Label", fontsize=11)

    # Add interpretation text
    tn, fp, fn, tp = cm.ravel()
    fig.text(
        0.5, -0.02,
        f"TN={tn} | FP={fp} | FN={fn} (missed churners!) | TP={tp}",
        ha="center", fontsize=10, color="gray",
    )

    plt.tight_layout()
    safe_name = model_name.lower().replace(" ", "_")
    path = FIGURES_DIR / f"confusion_matrix_{safe_name}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 Saved confusion matrix: {path}")


def plot_roc_curves(
    models: Dict[str, Pipeline],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    """
    Plot ROC curves for all models on the same axes.

    ROC curve shows: as we lower the threshold (predict more churners),
    how does our True Positive Rate improve vs False Positive Rate?

    AUC (Area Under Curve) = 1.0 is perfect, 0.5 is random guessing.

    Parameters
    ----------
    models : dict of fitted pipelines
    X_test, y_test : test data
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for i, (name, model) in enumerate(models.items()):
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})", color=COLORS[i], linewidth=2.5)

    # Random baseline
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random (AUC = 0.500)")

    ax.set_xlabel("False Positive Rate (false alarms)", fontsize=12)
    ax.set_ylabel("True Positive Rate (churners caught)", fontsize=12)
    ax.set_title("ROC Curves — Model Comparison", fontsize=14)
    ax.legend(fontsize=10)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])

    plt.tight_layout()
    path = FIGURES_DIR / "roc_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 Saved ROC curves: {path}")


def plot_pr_curves(
    models: Dict[str, Pipeline],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    """
    Plot Precision-Recall curves for all models.

    PR curve is more informative than ROC when classes are imbalanced.
    It shows: for each threshold, what's the trade-off between precision
    (when we predict churn, how often are we right?) and recall
    (of all actual churners, how many do we catch?).

    Parameters
    ----------
    models : dict of fitted pipelines
    X_test, y_test : test data
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for i, (name, model) in enumerate(models.items()):
        y_prob = model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        avg_prec = average_precision_score(y_test, y_prob)
        ax.plot(recall, precision, label=f"{name} (AP = {avg_prec:.3f})",
                color=COLORS[i], linewidth=2.5)

    # Baseline = random classifier at churn rate
    baseline = y_test.mean()
    ax.axhline(y=baseline, color="k", linestyle="--", linewidth=1,
               label=f"Random (AP = {baseline:.3f})")

    ax.set_xlabel("Recall (churners caught)", fontsize=12)
    ax.set_ylabel("Precision (accuracy of churn predictions)", fontsize=12)
    ax.set_title("Precision-Recall Curves — Model Comparison", fontsize=14)
    ax.legend(fontsize=10)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])

    plt.tight_layout()
    path = FIGURES_DIR / "pr_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 Saved PR curves: {path}")


def find_optimal_threshold(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    metric: str = "f1",
) -> float:
    """
    Find the decision threshold that maximizes a given metric.

    Business rationale:
    The default 0.5 threshold assumes catching a churner and falsely flagging
    a loyal customer are equally bad. For a telecom company, missing a churner
    is much more expensive (lost revenue ~$1,000+ per customer over their
    lifetime). So we may prefer a lower threshold (catch more churners at the
    cost of some false alarms).

    Parameters
    ----------
    model : fitted pipeline
    X_test, y_test : test data
    metric : 'f1' | 'recall' | 'precision'
        Metric to optimize.

    Returns
    -------
    float
        The optimal threshold value.
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    thresholds = np.arange(0.1, 0.9, 0.01)

    best_threshold = 0.5
    best_score = 0.0

    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        if metric == "f1":
            score = f1_score(y_test, y_pred, zero_division=0)
        elif metric == "recall":
            score = recall_score(y_test, y_pred, zero_division=0)
        elif metric == "precision":
            score = precision_score(y_test, y_pred, zero_division=0)
        else:
            score = f1_score(y_test, y_pred, zero_division=0)

        if score > best_score:
            best_score = score
            best_threshold = thresh

    print(f"  🎯 Optimal threshold (max {metric}): {best_threshold:.2f}  →  {metric}={best_score:.4f}")
    return best_threshold


def plot_threshold_analysis(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
) -> None:
    """
    Plot how Precision, Recall, and F1 change across thresholds.

    This helps you pick the right threshold for your business goal.

    Parameters
    ----------
    model : fitted pipeline
    X_test, y_test : test data
    model_name : str
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    thresholds = np.arange(0.1, 0.9, 0.01)

    precisions, recalls, f1s = [], [], []
    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        precisions.append(precision_score(y_test, y_pred, zero_division=0))
        recalls.append(recall_score(y_test, y_pred, zero_division=0))
        f1s.append(f1_score(y_test, y_pred, zero_division=0))

    best_f1_idx = np.argmax(f1s)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(thresholds, precisions, label="Precision", color="#2196F3", linewidth=2)
    ax.plot(thresholds, recalls, label="Recall", color="#4CAF50", linewidth=2)
    ax.plot(thresholds, f1s, label="F1 Score", color="#FF9800", linewidth=2.5)
    ax.axvline(x=thresholds[best_f1_idx], color=CHURN_COLOR, linestyle="--",
               linewidth=2, label=f"Best F1 threshold = {thresholds[best_f1_idx]:.2f}")

    ax.set_xlabel("Decision Threshold", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(f"Precision / Recall / F1 vs Threshold — {model_name}", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_ylim([0, 1.05])

    plt.tight_layout()
    safe_name = model_name.lower().replace(" ", "_")
    path = FIGURES_DIR / f"threshold_analysis_{safe_name}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 Saved threshold analysis: {path}")


def error_analysis(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Analyze where the model makes mistakes.

    Returns a DataFrame of misclassified examples with their predicted
    probability, so you can study patterns in errors.

    Parameters
    ----------
    model : fitted pipeline
    X_test, y_test : test data
    threshold : float

    Returns
    -------
    pd.DataFrame  with columns: true_label, predicted_label, churn_prob
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    errors = X_test.copy()
    errors["true_label"] = y_test.values
    errors["predicted_label"] = y_pred
    errors["churn_prob"] = y_prob
    errors["error_type"] = "correct"
    errors.loc[(errors["true_label"] == 1) & (errors["predicted_label"] == 0), "error_type"] = "false_negative"
    errors.loc[(errors["true_label"] == 0) & (errors["predicted_label"] == 1), "error_type"] = "false_positive"

    error_df = errors[errors["error_type"] != "correct"].copy()
    print(f"\n  🔍 Error analysis:")
    print(f"    False Negatives (missed churners): {(error_df['error_type']=='false_negative').sum()}")
    print(f"    False Positives (false alarms):    {(error_df['error_type']=='false_positive').sum()}")

    return error_df


def run_evaluation(
    models: Dict[str, Pipeline],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    best_model_name: str = "XGBoost",
) -> Tuple[pd.DataFrame, float]:
    """
    Master evaluation function.

    Parameters
    ----------
    models : dict of fitted pipelines
    X_test, y_test : test data
    best_model_name : str
        Which model to do detailed threshold/error analysis on.

    Returns
    -------
    comparison_table : pd.DataFrame
    optimal_threshold : float
    """
    print("\n" + "=" * 60)
    print("📊 MODEL EVALUATION")
    print("=" * 60)

    # Default comparison at threshold = 0.5
    print("\n📋 Metrics at threshold=0.50:")
    table = build_comparison_table(models, X_test, y_test, threshold=0.5)
    print(table.to_string())

    # Plot curves
    print("\n🎨 Generating plots...")
    plot_roc_curves(models, X_test, y_test)
    plot_pr_curves(models, X_test, y_test)

    # Confusion matrices for all models
    for name, model in models.items():
        plot_confusion_matrix(model, X_test, y_test, name, threshold=0.5)

    # Threshold tuning for the best model
    best_model = models[best_model_name]
    print(f"\n🎯 Threshold tuning for {best_model_name}:")
    optimal_threshold = find_optimal_threshold(best_model, X_test, y_test, metric="f1")
    plot_threshold_analysis(best_model, X_test, y_test, best_model_name)

    # Metrics at optimal threshold
    print(f"\n📋 {best_model_name} metrics at optimal threshold={optimal_threshold:.2f}:")
    metrics = evaluate_model(best_model, X_test, y_test, threshold=optimal_threshold)
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # Confusion matrix at optimal threshold
    plot_confusion_matrix(best_model, X_test, y_test,
                          f"{best_model_name} (optimized)", threshold=optimal_threshold)

    # Error analysis
    error_df = error_analysis(best_model, X_test, y_test, threshold=optimal_threshold)

    # Detailed classification report
    y_prob = best_model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= optimal_threshold).astype(int)
    print(f"\n📋 Detailed Classification Report ({best_model_name}):")
    print(classification_report(y_test, y_pred, target_names=["Stay", "Churn"]))

    return table, optimal_threshold
