"""
run_eda.py
----------
Runs the full EDA pipeline and saves all charts to reports/figures/.
This is the script version of notebooks/01_eda.ipynb.

Run with:
    python run_eda.py
"""
import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from src.data_preprocessing import load_raw_data, clean_data

FIGURES_DIR = Path("reports/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["figure.dpi"] = 120
CHURN_COLORS = {"No": "#4CAF50", "Yes": "#E91E63"}

print("=" * 55)
print("📊 RUNNING EDA")
print("=" * 55)

# ── Load data ──────────────────────────────────────────────
df_raw = load_raw_data()
df_clean = clean_data(df_raw)

# ── Data Quality ───────────────────────────────────────────
print("\n--- Blank TotalCharges ---")
blank = (df_raw["TotalCharges"].str.strip() == "").sum()
print(f"  Found {blank} blank values → fixed to 0.0")

print("\n--- Duplicates ---")
print(f"  Duplicate rows: {df_raw.duplicated().sum()}")
print(f"  Unique customers: {df_raw['customerID'].nunique():,}")

print("\n--- Churn Balance ---")
vc = df_clean["Churn"].value_counts()
print(f"  Not churned: {vc[0]:,} ({vc[0]/len(df_clean):.1%})")
print(f"  Churned:     {vc[1]:,} ({vc[1]/len(df_clean):.1%})")

# ── Figure 1: Churn distribution ───────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
churn_counts = df_raw["Churn"].value_counts()
churn_pct = df_raw["Churn"].value_counts(normalize=True) * 100

bars = axes[0].bar(churn_counts.index, churn_counts.values,
                   color=[CHURN_COLORS["No"], CHURN_COLORS["Yes"]],
                   edgecolor="white", width=0.5)
for bar, val, pct in zip(bars, churn_counts.values, churn_pct.values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
                f"{val:,}\n({pct:.1f}%)", ha="center", fontweight="bold", fontsize=12)
axes[0].set_title("Customer Churn Distribution", fontsize=14, fontweight="bold")
axes[0].set_xlabel("Churn"); axes[0].set_ylabel("Count"); axes[0].set_ylim(0, 6500)

wedges, texts, autotexts = axes[1].pie(
    churn_counts.values, labels=['Stayed', 'Churned'],
    colors=[CHURN_COLORS["No"], CHURN_COLORS["Yes"]],
    autopct="%1.1f%%", startangle=90,
    wedgeprops={"edgecolor": "white", "linewidth": 2},
    textprops={"fontsize": 13})
for at in autotexts: at.set_fontweight("bold")
axes[1].set_title("Churn Proportion", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "churn_distribution.png", dpi=150, bbox_inches="tight")
plt.close(); print("\n  💾 churn_distribution.png")

# ── Figure 2: Churn vs Contract ────────────────────────────
contract_churn = df_raw.groupby(["Contract", "Churn"]).size().unstack(fill_value=0)
contract_churn["rate"] = (contract_churn["Yes"] / contract_churn.sum(axis=1) * 100).round(1)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
contract_churn[["No", "Yes"]].plot(kind="bar", ax=axes[0], stacked=True,
    color=[CHURN_COLORS["No"], CHURN_COLORS["Yes"]], edgecolor="white")
axes[0].set_title("Churn Count by Contract Type", fontsize=13, fontweight="bold")
axes[0].set_xlabel("Contract"); axes[0].set_ylabel("Count")
axes[0].tick_params(axis="x", rotation=0); axes[0].legend(["Stayed", "Churned"])

bars = axes[1].bar(contract_churn.index, contract_churn["rate"],
                   color=["#FF7043", "#FF9800", "#4CAF50"], edgecolor="white")
for bar, val in zip(bars, contract_churn["rate"]):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val}%", ha="center", fontweight="bold", fontsize=12)
axes[1].set_title("Churn Rate (%) by Contract Type", fontsize=13, fontweight="bold")
axes[1].set_xlabel("Contract"); axes[1].set_ylabel("Churn Rate (%)"); axes[1].set_ylim(0, 55)
axes[1].tick_params(axis="x", rotation=0)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "churn_vs_contract.png", dpi=150, bbox_inches="tight")
plt.close(); print("  💾 churn_vs_contract.png")

# ── Figure 3: Churn vs Tenure ──────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for cv, color in CHURN_COLORS.items():
    axes[0].hist(df_raw[df_raw["Churn"] == cv]["tenure"], bins=30,
                alpha=0.6, color=color, label=f"Churn={cv}", edgecolor="white", density=True)
axes[0].set_title("Tenure Distribution by Churn", fontsize=13, fontweight="bold")
axes[0].set_xlabel("Tenure (months)"); axes[0].set_ylabel("Density"); axes[0].legend()

stayed_t = df_raw[df_raw["Churn"] == "No"]["tenure"]
churned_t = df_raw[df_raw["Churn"] == "Yes"]["tenure"]
bp = axes[1].boxplot([stayed_t, churned_t], tick_labels=['Stayed', 'Churned'],  # mpl 3.11+
                     patch_artist=True, widths=0.4,
                     medianprops=dict(color="white", linewidth=2.5))
bp["boxes"][0].set_facecolor(CHURN_COLORS["No"])
bp["boxes"][1].set_facecolor(CHURN_COLORS["Yes"])
axes[1].set_title("Tenure Boxplot: Stayed vs Churned", fontsize=13, fontweight="bold")
axes[1].set_xlabel("Outcome"); axes[1].set_ylabel("Tenure (months)")
print(f"  Median tenure — Stayed: {stayed_t.median():.0f}m | Churned: {churned_t.median():.0f}m")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "churn_vs_tenure.png", dpi=150, bbox_inches="tight")
plt.close(); print("  💾 churn_vs_tenure.png")

# ── Figure 4: Churn vs Monthly Charges ────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for cv, color in CHURN_COLORS.items():
    axes[0].hist(df_raw[df_raw["Churn"] == cv]["MonthlyCharges"], bins=30,
                alpha=0.6, color=color, label=f"Churn={cv}", edgecolor="white", density=True)
axes[0].set_title("Monthly Charges Distribution by Churn", fontsize=13, fontweight="bold")
axes[0].set_xlabel("Monthly Charges ($)"); axes[0].set_ylabel("Density"); axes[0].legend()

stayed_mc = df_raw[df_raw["Churn"] == "No"]["MonthlyCharges"]
churned_mc = df_raw[df_raw["Churn"] == "Yes"]["MonthlyCharges"]
bp = axes[1].boxplot([stayed_mc, churned_mc], tick_labels=['Stayed', 'Churned'],  # mpl 3.11+
                     patch_artist=True, widths=0.4,
                     medianprops=dict(color="white", linewidth=2.5))
bp["boxes"][0].set_facecolor(CHURN_COLORS["No"])
bp["boxes"][1].set_facecolor(CHURN_COLORS["Yes"])
axes[1].set_title("Monthly Charges: Stayed vs Churned", fontsize=13, fontweight="bold")
axes[1].set_xlabel("Outcome"); axes[1].set_ylabel("Monthly Charges ($)")
print(f"  Mean charges — Stayed: ${stayed_mc.mean():.2f} | Churned: ${churned_mc.mean():.2f}")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "churn_vs_monthly_charges.png", dpi=150, bbox_inches="tight")
plt.close(); print("  💾 churn_vs_monthly_charges.png")

# ── Figure 5: Churn vs Payment Method ─────────────────────
payment_churn = df_raw.groupby("PaymentMethod")["Churn"].apply(
    lambda x: (x == "Yes").mean() * 100).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(10, 5))
colors = ["#4CAF50" if v < 25 else "#FF9800" if v < 40 else "#E91E63" for v in payment_churn.values]
bars = ax.barh(payment_churn.index, payment_churn.values, color=colors, edgecolor="white")
for bar, val in zip(bars, payment_churn.values):
    ax.text(val + 0.5, bar.get_y() + bar.get_height()/2, f"{val:.1f}%", va="center", fontweight="bold")
ax.set_title("Churn Rate (%) by Payment Method", fontsize=13, fontweight="bold")
ax.set_xlabel("Churn Rate (%)"); ax.set_xlim(0, 55)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "churn_vs_payment.png", dpi=150, bbox_inches="tight")
plt.close(); print("  💾 churn_vs_payment.png")

# ── Figure 6: Churn vs Internet Service ───────────────────
internet_churn = df_raw.groupby("InternetService")["Churn"].apply(
    lambda x: (x == "Yes").mean() * 100).sort_values(ascending=False)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
bars = axes[0].bar(internet_churn.index, internet_churn.values,
                   color=["#E91E63", "#FF9800", "#4CAF50"], edgecolor="white")
for bar, val in zip(bars, internet_churn.values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", fontweight="bold", fontsize=12)
axes[0].set_title("Churn Rate (%) by Internet Service", fontsize=13, fontweight="bold")
axes[0].set_xlabel("Internet Service"); axes[0].set_ylabel("Churn Rate (%)"); axes[0].set_ylim(0, 55)

internet_counts = df_raw.groupby(["InternetService", "Churn"]).size().unstack(fill_value=0)
internet_counts[["No", "Yes"]].plot(kind="bar", ax=axes[1], stacked=True,
    color=[CHURN_COLORS["No"], CHURN_COLORS["Yes"]], edgecolor="white")
axes[1].set_title("Churn Count by Internet Service", fontsize=13, fontweight="bold")
axes[1].set_xlabel("Internet Service"); axes[1].set_ylabel("Count")
axes[1].tick_params(axis="x", rotation=0); axes[1].legend(["Stayed", "Churned"])
plt.tight_layout()
plt.savefig(FIGURES_DIR / "churn_vs_internet.png", dpi=150, bbox_inches="tight")
plt.close(); print("  💾 churn_vs_internet.png")

# ── Figure 7: Correlation heatmap ─────────────────────────
numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
corr = df_clean[numeric_cols].corr()
fig, ax = plt.subplots(figsize=(8, 6))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
            center=0, vmin=-1, vmax=1, linewidths=0.5, annot_kws={"size": 11}, ax=ax)
ax.set_title("Correlation Matrix (Numeric Features)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close(); print("  💾 correlation_heatmap.png")

# ── Summary ────────────────────────────────────────────────
saved = list(FIGURES_DIR.glob("*.png"))
print(f"\n{'='*55}")
print(f"✅ EDA COMPLETE — {len(saved)} figures saved to {FIGURES_DIR}/")
print(f"{'='*55}")
print("\n📝 KEY BUSINESS INSIGHTS:")
print(f"  1. Churn rate: {vc[1]/len(df_clean):.1%} — 1 in 4 customers leaves")
print(f"  2. Month-to-month churn: {contract_churn.loc['Month-to-month','rate']:.0f}% vs Two-year: {contract_churn.loc['Two year','rate']:.0f}%")
print(f"  3. New customers (0-12m) have highest churn risk")
print(f"  4. Churners pay ${churned_mc.mean():.0f}/mo vs ${stayed_mc.mean():.0f}/mo for those who stay")
print(f"  5. Fiber optic churn: {internet_churn.get('Fiber optic', 0):.0f}% vs DSL: {internet_churn.get('DSL', 0):.0f}%")
print(f"  6. Electronic check churn: {payment_churn.max():.0f}% — highest of all methods")
print(f"  7. TotalCharges: 11 blank strings found and fixed (new customers)")
