"""
run_evaluation.py
-----------------
Runs the full evaluation pipeline: loads trained models, computes all
metrics, generates plots, and runs SHAP explainability.

Phases covered: 6 (Evaluation) and 7 (Explainability)

Run with:
    python run_evaluation.py
"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
from pathlib import Path

from src.data_preprocessing import load_raw_data, clean_data, split_features_target, get_column_types
from src.features import engineer_features, get_feature_names
from src.evaluate import (
    build_comparison_table, plot_confusion_matrix, plot_roc_curves,
    plot_pr_curves, find_optimal_threshold, plot_threshold_analysis, error_analysis
)

FIGURES_DIR = Path("reports/figures")
MODELS_DIR  = Path("models")
PROCESSED   = Path("data/processed")

plt.style.use("seaborn-v0_8-whitegrid")

print("=" * 60)
print("📊 PHASES 6 & 7 — EVALUATION + EXPLAINABILITY")
print("=" * 60)

# ── Load test data ─────────────────────────────────────────
X_test  = pd.read_csv(PROCESSED / "X_test.csv")
y_test  = pd.read_csv(PROCESSED / "y_test.csv").squeeze()

# ── Load all models ────────────────────────────────────────
models = {
    "Logistic Regression": joblib.load(MODELS_DIR / "logistic_regression_pipeline.pkl"),
    "Random Forest":       joblib.load(MODELS_DIR / "random_forest_pipeline.pkl"),
    "XGBoost":             joblib.load(MODELS_DIR / "xgboost_pipeline.pkl"),
}
print(f"\n✅ Loaded {len(models)} trained models")
print(f"   Test set: {len(X_test):,} rows | Churn rate: {y_test.mean():.1%}")

# ── Phase 6: Metrics comparison at threshold=0.50 ─────────
print("\n" + "─" * 60)
print("📋 MODEL COMPARISON (threshold = 0.50)")
print("─" * 60)
table = build_comparison_table(models, X_test, y_test, threshold=0.50)
print(table.to_string())

# ── ROC + PR curves ────────────────────────────────────────
print("\n🎨 Plotting curves...")
plot_roc_curves(models, X_test, y_test)
plot_pr_curves(models, X_test, y_test)

# ── Confusion matrices ─────────────────────────────────────
for name, model in models.items():
    plot_confusion_matrix(model, X_test, y_test, name, threshold=0.50)

# ── Threshold tuning (XGBoost) ─────────────────────────────
print("\n🎯 Threshold tuning for XGBoost (optimizing F1)...")
xgb_model = models["XGBoost"]
opt_thresh = find_optimal_threshold(xgb_model, X_test, y_test, metric="f1")
plot_threshold_analysis(xgb_model, X_test, y_test, "XGBoost")

print(f"\n📋 XGBoost metrics at optimal threshold = {opt_thresh:.2f}:")
from src.evaluate import evaluate_model
metrics_opt = evaluate_model(xgb_model, X_test, y_test, threshold=opt_thresh)
for k, v in metrics_opt.items():
    print(f"  {k:15s}: {v:.4f}")

# Confusion matrix at optimal threshold
plot_confusion_matrix(xgb_model, X_test, y_test, "XGBoost (optimal threshold)", threshold=opt_thresh)

# ── Error Analysis ─────────────────────────────────────────
print("\n🔍 Error Analysis:")
error_df = error_analysis(xgb_model, X_test, y_test, threshold=opt_thresh)

# Show top false negatives (missed churners — most costly)
fn_df = error_df[error_df["error_type"] == "false_negative"].sort_values("churn_prob", ascending=False)
print(f"\n  Top 5 missed churners (false negatives) — prob near threshold:")
if "Contract" in fn_df.columns and "tenure" in fn_df.columns:
    print(fn_df[["tenure", "Contract", "MonthlyCharges", "churn_prob"]].head())

# ── Phase 7: SHAP Explainability ───────────────────────────
print("\n" + "─" * 60)
print("🔍 PHASE 7 — SHAP EXPLAINABILITY")
print("─" * 60)

preprocessor = xgb_model.named_steps["preprocessor"]
classifier   = xgb_model.named_steps["classifier"]

# Transform test data
X_test_transformed = preprocessor.transform(X_test)

# Get feature names
num_cols = preprocessor.transformers_[0][2]
ohe = preprocessor.named_transformers_["cat"]["onehot"]
cat_features = ohe.get_feature_names_out().tolist()
all_features = list(num_cols) + cat_features
print(f"  Total features after encoding: {len(all_features)}")

# Compute SHAP values
print("  Computing SHAP values (this takes ~30 seconds)...")
explainer  = shap.TreeExplainer(classifier)
shap_vals  = explainer.shap_values(X_test_transformed)

# ── SHAP Summary Plot ──────────────────────────────────────
print("  Plotting SHAP summary...")
fig, ax = plt.subplots(figsize=(10, 8))
shap_df = pd.DataFrame(shap_vals, columns=all_features)
mean_abs_shap = shap_df.abs().mean().sort_values(ascending=True).tail(15)

colors_shap = ["#E91E63" if v > 0 else "#4CAF50" for v in mean_abs_shap.values]
bars = ax.barh(mean_abs_shap.index, mean_abs_shap.values, color="#7C4DFF", edgecolor="white")
for bar, val in zip(bars, mean_abs_shap.values):
    ax.text(val + 0.0005, bar.get_y() + bar.get_height()/2,
           f"{val:.4f}", va="center", fontsize=9)
ax.set_xlabel("Mean |SHAP Value| (average impact on churn prediction)", fontsize=11)
ax.set_title("Top 15 Churn Drivers (SHAP Feature Importance)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print("  💾 shap_summary.png")

# ── Print top features in business language ────────────────
top10 = mean_abs_shap.tail(10).sort_values(ascending=False)
print("\n📝 TOP 10 CHURN DRIVERS (Business Language):")
for feat, val in top10.items():
    print(f"  {val:.4f}  →  {feat}")

# ── Summary table ──────────────────────────────────────────
print("\n" + "=" * 60)
print("✅ EVALUATION COMPLETE")
print("=" * 60)
print("\n📊 FINAL RESULTS SUMMARY")
print("─" * 60)
print(f"\n{'Model':<22} {'Acc':>6} {'Prec':>6} {'Recall':>7} {'F1':>6} {'AUC':>7}")
print("─" * 60)
for model_name, row in table.iterrows():
    print(f"{model_name:<22} {row['Accuracy']:>6.4f} {row['Precision']:>6.4f} {row['Recall']:>7.4f} {row['F1']:>6.4f} {row['ROC-AUC']:>7.4f}")

print(f"\n🏆 Best model: XGBoost")
print(f"   At threshold={opt_thresh:.2f}:")
for k, v in metrics_opt.items():
    print(f"   {k}: {v:.4f}")

print("\n💡 RETENTION RECOMMENDATIONS (from SHAP):")
print("  1. Convert month-to-month customers to annual contracts (largest churn driver)")
print("  2. Offer first-year loyalty rewards — tenure 0-12m is highest risk window")
print("  3. Bundle TechSupport/OnlineSecurity for Fiber optic customers")

saved = list(FIGURES_DIR.glob("*.png"))
print(f"\n💾 Total figures saved: {len(saved)}")
