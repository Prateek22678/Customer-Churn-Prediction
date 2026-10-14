"""
features.py
-----------
Feature engineering for the Churn Prediction project.

We create three new features that give the model extra useful signals:

  1. tenure_group     — Groups tenure (months) into buckets.
                        Why: Raw tenure is noisy. Customers in 0-12 months
                        behave very differently from 5-year customers.

  2. num_services     — Count of optional services a customer has.
                        Why: Customers with more services are more "sticky"
                        (it's painful to leave) and less likely to churn.

  3. avg_monthly_spend — TotalCharges / (tenure + 1).
                         Why: Captures how much a customer actually pays on
                         average — important because MonthlyCharges can change
                         over time (promotions, upgrades, etc.).

Also builds and returns the sklearn ColumnTransformer preprocessing pipeline.

Author: Churn Prediction Project
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from typing import Tuple, List


# ── Services that count toward "stickiness" ────────────────────────────────
SERVICE_COLS = [
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


def add_tenure_group(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bin tenure (in months) into 5 human-readable categories.

    Bins: 0-12 → 'new', 13-24 → '1-2yr', 25-36 → '2-3yr',
          37-48 → '3-4yr', 49-60 → '4-5yr', 61+ → '5+yr'

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame with new 'tenure_group' column.
    """
    df = df.copy()
    bins = [0, 12, 24, 36, 48, 60, 72]
    labels = ["0-1yr", "1-2yr", "2-3yr", "3-4yr", "4-5yr", "5+yr"]
    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=bins,
        labels=labels,
        include_lowest=True,
    ).astype(str)  # Convert to string so OHE can handle it
    return df


def add_num_services(df: pd.DataFrame) -> pd.DataFrame:
    """
    Count how many optional services a customer subscribes to.

    We look for 'Yes' in each service column.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame with new 'num_services' column (integer 0-8).
    """
    df = df.copy()
    # For each service column: 1 if value contains "Yes", else 0
    service_counts = df[SERVICE_COLS].apply(lambda col: (col == "Yes").astype(int))
    df["num_services"] = service_counts.sum(axis=1)
    return df


def add_avg_monthly_spend(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute average monthly spend = TotalCharges / (tenure + 1).

    Adding 1 to tenure avoids division-by-zero for new customers.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame with new 'avg_monthly_spend' column.
    """
    df = df.copy()
    df["avg_monthly_spend"] = df["TotalCharges"] / (df["tenure"] + 1)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering steps in order.

    Call this after clean_data() and before building the pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe (output of clean_data()).

    Returns
    -------
    pd.DataFrame
        DataFrame with 3 new engineered features added.
    """
    print("🔧 Engineering features...")
    df = add_tenure_group(df)
    df = add_num_services(df)
    df = add_avg_monthly_spend(df)
    print(f"  ✅ Added: tenure_group, num_services, avg_monthly_spend")
    print(f"  Shape after engineering: {df.shape}")
    return df


def build_preprocessor(
    numeric_cols: List[str],
    categorical_cols: List[str],
) -> ColumnTransformer:
    """
    Build a scikit-learn ColumnTransformer that:
      - Scales numeric columns with StandardScaler
        (so all numbers are on the same scale — important for Logistic Regression)
      - One-hot encodes categorical columns
        (converts text categories to numbers the model can understand)

    IMPORTANT: This preprocessor is only FITTED on training data.
    We apply it to test data without refitting — this prevents data leakage.

    Parameters
    ----------
    numeric_cols : list of str
        Column names that are numeric.
    categorical_cols : list of str
        Column names that are categorical (text).

    Returns
    -------
    ColumnTransformer
        Unfitted preprocessor ready to be used in a Pipeline.
    """
    numeric_pipeline = Pipeline(steps=[
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",  # Drop any columns not specified (safety net)
    )

    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer, numeric_cols: List[str]) -> List[str]:
    """
    Extract human-readable feature names after the ColumnTransformer is fitted.

    Useful for SHAP plots and feature importance.

    Parameters
    ----------
    preprocessor : ColumnTransformer
        A FITTED ColumnTransformer.
    numeric_cols : list of str

    Returns
    -------
    list of str
        Feature names in the same order as the transformer output columns.
    """
    ohe = preprocessor.named_transformers_["cat"]["onehot"]
    cat_feature_names = ohe.get_feature_names_out().tolist()
    return numeric_cols + cat_feature_names
