"""
data_preprocessing.py
---------------------
Handles loading and cleaning of the IBM Telco Customer Churn dataset.

Key issues we fix:
  1. TotalCharges has blank strings ("") instead of NaN for new customers
     (tenure=0 → they haven't been charged yet).
  2. customerID is not a feature — we drop it.
  3. Churn target is "Yes"/"No" — we convert to 1/0.

Author: Churn Prediction Project
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple


# ── Paths ──────────────────────────────────────────────────────────────────
RAW_DATA_PATH = Path(__file__).parent.parent / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
PROCESSED_DATA_PATH = Path(__file__).parent.parent / "data" / "processed"


def load_raw_data(filepath: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Load the raw CSV file from disk.

    Parameters
    ----------
    filepath : Path
        Location of the raw CSV file.

    Returns
    -------
    pd.DataFrame
        Raw dataframe with no modifications.

    Raises
    ------
    FileNotFoundError
        If the CSV is not found at the expected location.
    """
    if not filepath.exists():
        raise FileNotFoundError(
            f"\n❌ Dataset not found at: {filepath}\n"
            "Please download it from Kaggle:\n"
            "  https://www.kaggle.com/datasets/blastchar/telco-customer-churn\n"
            "  File name: WA_Fn-UseC_-Telco-Customer-Churn.csv\n"
            f"  Place it in: {filepath.parent}\n"
        )
    df = pd.read_csv(filepath)
    print(f"✅ Loaded raw data: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


def fix_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix the TotalCharges column which contains blank strings for new customers.

    The dataset stores TotalCharges as a STRING column.  Customers with
    tenure=0 (brand new) have a blank string "" instead of 0.  We:
      1. Replace blank strings with NaN.
      2. Convert the column to float.
      3. Fill NaN with 0 (because new customers haven't been charged yet).

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        DataFrame with TotalCharges as float.
    """
    df = df.copy()

    # Count the blanks before fixing (for transparency)
    blank_count = (df["TotalCharges"].str.strip() == "").sum()
    if blank_count > 0:
        print(f"  ⚠️  Found {blank_count} blank TotalCharges values (new customers) → filling with 0")

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)
    return df


def drop_customer_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop the customerID column — it's a unique identifier with no predictive value.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
        print("  🗑️  Dropped customerID column")
    return df


def encode_target(df: pd.DataFrame, target_col: str = "Churn") -> pd.DataFrame:
    """
    Encode the target variable: 'Yes' → 1, 'No' → 0.

    Parameters
    ----------
    df : pd.DataFrame
    target_col : str
        Name of the target column (default: 'Churn').

    Returns
    -------
    pd.DataFrame
        DataFrame with target as integer 0/1.
    """
    df = df.copy()
    df[target_col] = (df[target_col] == "Yes").astype(int)
    churn_rate = df[target_col].mean()
    print(f"  🎯 Churn rate: {churn_rate:.1%}  ({df[target_col].sum():,} churners / {len(df):,} total)")
    return df


def fix_senior_citizen(df: pd.DataFrame) -> pd.DataFrame:
    """
    SeniorCitizen is already 0/1 integer — but let's make it consistent
    by converting to 'Yes'/'No' so it matches other binary columns and
    gets treated uniformly in the pipeline.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Master cleaning function — applies all fixes in the right order.

    Call this after load_raw_data().

    Steps:
      1. Fix TotalCharges blank strings → float
      2. Convert SeniorCitizen 0/1 → 'No'/'Yes' for consistency
      3. Drop customerID
      4. Encode Churn target as 0/1

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe from load_raw_data().

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe ready for feature engineering.
    """
    print("\n🧹 Cleaning data...")
    df = fix_total_charges(df)
    df = fix_senior_citizen(df)
    df = drop_customer_id(df)
    df = encode_target(df)
    print(f"✅ Cleaning complete: {df.shape[0]:,} rows × {df.shape[1]} columns\n")
    return df


def split_features_target(df: pd.DataFrame, target_col: str = "Churn") -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split dataframe into features (X) and target (y).

    Parameters
    ----------
    df : pd.DataFrame
    target_col : str

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        (X, y)
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def get_column_types(X: pd.DataFrame) -> Tuple[list, list]:
    """
    Automatically detect numeric and categorical columns.

    We treat binary columns (e.g., 'Yes'/'No') as categorical so they
    get one-hot encoded consistently.

    Parameters
    ----------
    X : pd.DataFrame

    Returns
    -------
    Tuple[list, list]
        (numeric_cols, categorical_cols)
    """
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    print(f"  📊 Numeric columns ({len(numeric_cols)}): {numeric_cols}")
    print(f"  📝 Categorical columns ({len(categorical_cols)}): {categorical_cols}")
    return numeric_cols, categorical_cols


if __name__ == "__main__":
    # Quick smoke test — run this file directly to check loading works
    df_raw = load_raw_data()
    df_clean = clean_data(df_raw)
    print(df_clean.dtypes)
    print(df_clean.head(3))
