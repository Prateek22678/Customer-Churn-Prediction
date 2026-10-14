"""
test_preprocessing.py
---------------------
Unit tests for data_preprocessing.py

Tests verify that:
  1. TotalCharges blank strings are correctly handled
  2. SeniorCitizen is converted to Yes/No strings
  3. customerID is dropped
  4. Churn is encoded as 0/1
  5. The full clean_data() pipeline works end-to-end

Run with:
    pytest tests/ -v

Author: Churn Prediction Project
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pytest

# Make src importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_preprocessing import (
    fix_total_charges,
    drop_customer_id,
    encode_target,
    fix_senior_citizen,
    clean_data,
    split_features_target,
    get_column_types,
)


# ── Fixtures (reusable test data) ──────────────────────────────────────────

@pytest.fixture
def sample_df() -> pd.DataFrame:
    """
    A small fake DataFrame that mimics the IBM Telco dataset structure.
    This is enough to test all cleaning functions without needing the real file.
    """
    return pd.DataFrame({
        "customerID": ["1111-AAA", "2222-BBB", "3333-CCC"],
        "gender": ["Male", "Female", "Male"],
        "SeniorCitizen": [0, 1, 0],       # Raw: 0/1 integer
        "Partner": ["Yes", "No", "Yes"],
        "Dependents": ["No", "Yes", "No"],
        "tenure": [0, 24, 60],
        "PhoneService": ["Yes", "Yes", "No"],
        "MultipleLines": ["No", "Yes", "No phone service"],
        "InternetService": ["DSL", "Fiber optic", "No"],
        "OnlineSecurity": ["No", "Yes", "No internet service"],
        "OnlineBackup": ["Yes", "No", "No internet service"],
        "DeviceProtection": ["No", "Yes", "No internet service"],
        "TechSupport": ["No", "No", "No internet service"],
        "StreamingTV": ["No", "Yes", "No internet service"],
        "StreamingMovies": ["No", "Yes", "No internet service"],
        "Contract": ["Month-to-month", "One year", "Two year"],
        "PaperlessBilling": ["Yes", "Yes", "No"],
        "PaymentMethod": ["Electronic check", "Mailed check", "Bank transfer (automatic)"],
        "MonthlyCharges": [29.85, 56.95, 53.85],
        "TotalCharges": [" ", "1394.55", "3226.1"],  # ← Blank string for new customer!
        "Churn": ["No", "Yes", "No"],
    })


# ── Tests ──────────────────────────────────────────────────────────────────

class TestFixTotalCharges:
    """Tests for the TotalCharges blank-string fix."""

    def test_blank_string_becomes_zero(self, sample_df):
        """New customer with tenure=0 has blank TotalCharges → should become 0.0."""
        result = fix_total_charges(sample_df)
        assert result.loc[0, "TotalCharges"] == 0.0, "Blank TotalCharges should be 0.0"

    def test_valid_values_preserved(self, sample_df):
        """Valid numeric strings should be converted to float correctly."""
        result = fix_total_charges(sample_df)
        assert abs(result.loc[1, "TotalCharges"] - 1394.55) < 0.01
        assert abs(result.loc[2, "TotalCharges"] - 3226.1) < 0.01

    def test_output_dtype_is_float(self, sample_df):
        """TotalCharges column should be float64 after fixing."""
        result = fix_total_charges(sample_df)
        assert result["TotalCharges"].dtype in [np.float64, float], \
            "TotalCharges must be float after fix"

    def test_no_nulls_remaining(self, sample_df):
        """There should be no NaN values after the fix."""
        result = fix_total_charges(sample_df)
        assert result["TotalCharges"].isna().sum() == 0, "No NaN should remain after fix"


class TestDropCustomerID:
    """Tests for customerID removal."""

    def test_customer_id_removed(self, sample_df):
        """customerID column should not exist after dropping."""
        result = drop_customer_id(sample_df)
        assert "customerID" not in result.columns

    def test_other_columns_intact(self, sample_df):
        """All other columns should be preserved."""
        result = drop_customer_id(sample_df)
        original_cols = set(sample_df.columns) - {"customerID"}
        assert set(result.columns) == original_cols

    def test_no_error_if_already_dropped(self, sample_df):
        """Should not raise an error if customerID already doesn't exist."""
        df_no_id = sample_df.drop(columns=["customerID"])
        result = drop_customer_id(df_no_id)  # Should not raise
        assert "customerID" not in result.columns


class TestEncodeTarget:
    """Tests for Churn Yes/No → 1/0 encoding."""

    def test_yes_becomes_one(self, sample_df):
        """'Yes' churn should be encoded as 1."""
        result = encode_target(sample_df)
        assert result.loc[1, "Churn"] == 1

    def test_no_becomes_zero(self, sample_df):
        """'No' churn should be encoded as 0."""
        result = encode_target(sample_df)
        assert result.loc[0, "Churn"] == 0
        assert result.loc[2, "Churn"] == 0

    def test_output_dtype_is_integer(self, sample_df):
        """Churn column should be integer (0 or 1)."""
        result = encode_target(sample_df)
        assert result["Churn"].dtype in [np.int64, np.int32, int, np.int8]

    def test_only_zero_and_one(self, sample_df):
        """Churn column should only contain 0 and 1."""
        result = encode_target(sample_df)
        assert set(result["Churn"].unique()).issubset({0, 1})


class TestFixSeniorCitizen:
    """Tests for SeniorCitizen 0/1 → No/Yes conversion."""

    def test_zero_becomes_no(self, sample_df):
        """SeniorCitizen=0 should become 'No'."""
        result = fix_senior_citizen(sample_df)
        assert result.loc[0, "SeniorCitizen"] == "No"
        assert result.loc[2, "SeniorCitizen"] == "No"

    def test_one_becomes_yes(self, sample_df):
        """SeniorCitizen=1 should become 'Yes'."""
        result = fix_senior_citizen(sample_df)
        assert result.loc[1, "SeniorCitizen"] == "Yes"

    def test_dtype_is_object(self, sample_df):
        """After conversion, SeniorCitizen should be a string-like dtype.
        
        Accepts both 'object' (pandas < 3.0) and 'StringDtype' (pandas >= 3.0),
        since pandas 3.x defaults string columns to the new StringDtype.
        """
        result = fix_senior_citizen(sample_df)
        dtype = result["SeniorCitizen"].dtype
        is_string_like = dtype == object or isinstance(dtype, pd.StringDtype) or str(dtype) == "string"
        assert is_string_like, f"Expected string dtype, got {dtype}"


class TestCleanData:
    """Integration test for the master clean_data() function."""

    def test_returns_dataframe(self, sample_df):
        """clean_data() should return a DataFrame."""
        result = clean_data(sample_df)
        assert isinstance(result, pd.DataFrame)

    def test_correct_shape(self, sample_df):
        """Output should have one fewer column (customerID dropped)."""
        result = clean_data(sample_df)
        assert result.shape[0] == sample_df.shape[0]       # Same rows
        assert result.shape[1] == sample_df.shape[1] - 1   # One less column

    def test_no_customer_id(self, sample_df):
        """customerID must not be in output."""
        result = clean_data(sample_df)
        assert "customerID" not in result.columns

    def test_churn_is_binary(self, sample_df):
        """Churn column must be 0/1."""
        result = clean_data(sample_df)
        assert set(result["Churn"].unique()).issubset({0, 1})

    def test_total_charges_is_float(self, sample_df):
        """TotalCharges must be float."""
        result = clean_data(sample_df)
        assert pd.api.types.is_float_dtype(result["TotalCharges"])

    def test_no_missing_values(self, sample_df):
        """After cleaning there should be no missing values."""
        result = clean_data(sample_df)
        assert result.isnull().sum().sum() == 0


class TestSplitFeaturesTarget:
    """Tests for the X/y split helper."""

    def test_x_does_not_contain_target(self, sample_df):
        """X should not include the Churn column."""
        df_clean = clean_data(sample_df)
        X, y = split_features_target(df_clean)
        assert "Churn" not in X.columns

    def test_y_is_series(self, sample_df):
        """y should be a pandas Series."""
        df_clean = clean_data(sample_df)
        X, y = split_features_target(df_clean)
        assert isinstance(y, pd.Series)

    def test_lengths_match(self, sample_df):
        """X and y must have the same number of rows."""
        df_clean = clean_data(sample_df)
        X, y = split_features_target(df_clean)
        assert len(X) == len(y)
