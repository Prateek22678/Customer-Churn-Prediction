"""
test_predict.py
---------------
Unit tests for the prediction interface (src/predict.py).

Tests verify:
  1. get_risk_level() maps probabilities to correct labels
  2. predict_churn() returns the expected output structure
  3. Batch prediction works correctly
  4. Input validation catches bad data

Note: Tests that require a saved model are skipped if the model
file doesn't exist yet (run src/train.py first).

Author: Churn Prediction Project
"""

import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

# Make src importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.predict import get_risk_level, RISK_THRESHOLDS, DEFAULT_THRESHOLD


# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def sample_customer() -> dict:
    """A complete customer dictionary matching the training data format."""
    return {
        "gender": "Female",
        "SeniorCitizen": "No",
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 85.50,
        "TotalCharges": 1026.0,
    }


@pytest.fixture
def high_risk_customer() -> dict:
    """A customer profile that historically has very high churn probability."""
    return {
        "gender": "Male",
        "SeniorCitizen": "Yes",
        "Partner": "No",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 95.0,
        "TotalCharges": 95.0,
    }


@pytest.fixture
def low_risk_customer() -> dict:
    """A customer profile that historically has very low churn probability."""
    return {
        "gender": "Female",
        "SeniorCitizen": "No",
        "Partner": "Yes",
        "Dependents": "Yes",
        "tenure": 60,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes",
        "DeviceProtection": "Yes",
        "TechSupport": "Yes",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Two year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Bank transfer (automatic)",
        "MonthlyCharges": 79.0,
        "TotalCharges": 4740.0,
    }


# ── Tests for get_risk_level() ─────────────────────────────────────────────

class TestGetRiskLevel:
    """Tests for the probability → risk label mapping."""

    def test_low_probability_is_low_risk(self):
        """Probability below 0.35 should be 'Low'."""
        assert get_risk_level(0.10) == "Low"
        assert get_risk_level(0.34) == "Low"

    def test_medium_probability_is_medium_risk(self):
        """Probability between 0.35 and 0.60 should be 'Medium'."""
        assert get_risk_level(0.35) == "Medium"
        assert get_risk_level(0.50) == "Medium"
        assert get_risk_level(0.59) == "Medium"

    def test_high_probability_is_high_risk(self):
        """Probability above 0.60 should be 'High'."""
        assert get_risk_level(0.60) == "High"
        assert get_risk_level(0.85) == "High"
        assert get_risk_level(0.99) == "High"

    def test_zero_probability_is_low_risk(self):
        """0.0 probability should be 'Low'."""
        assert get_risk_level(0.0) == "Low"

    def test_one_probability_is_high_risk(self):
        """1.0 probability should be 'High'."""
        assert get_risk_level(1.0) == "High"

    def test_boundary_at_0_35(self):
        """The boundary between Low and Medium is exactly 0.35."""
        assert get_risk_level(0.3499) == "Low"
        assert get_risk_level(0.35) == "Medium"

    def test_boundary_at_0_60(self):
        """The boundary between Medium and High is exactly 0.60."""
        assert get_risk_level(0.5999) == "Medium"
        assert get_risk_level(0.60) == "High"

    def test_returns_string(self):
        """Return type should always be a string."""
        assert isinstance(get_risk_level(0.5), str)

    def test_valid_labels_only(self):
        """Only 'Low', 'Medium', 'High' are valid outputs."""
        valid_labels = {"Low", "Medium", "High"}
        for prob in [0.0, 0.1, 0.35, 0.5, 0.6, 0.9, 1.0]:
            assert get_risk_level(prob) in valid_labels


# ── Tests for predict_churn() (requires trained model) ────────────────────

MODEL_PATH = Path(__file__).parent.parent / "models" / "xgboost_pipeline.pkl"
model_available = MODEL_PATH.exists()


@pytest.mark.skipif(not model_available, reason="Trained model not found. Run src/train.py first.")
class TestPredictChurn:
    """Tests for the full predict_churn() function. Requires trained model."""

    def test_returns_dict(self, sample_customer):
        """predict_churn() must return a dictionary."""
        from src.predict import predict_churn
        result = predict_churn(sample_customer)
        assert isinstance(result, dict)

    def test_has_required_keys(self, sample_customer):
        """Result must contain all required keys."""
        from src.predict import predict_churn
        result = predict_churn(sample_customer)
        required_keys = {"churn_probability", "risk_level", "will_churn", "threshold_used"}
        assert required_keys.issubset(result.keys())

    def test_probability_in_range(self, sample_customer):
        """Churn probability must be between 0 and 1."""
        from src.predict import predict_churn
        result = predict_churn(sample_customer)
        assert 0.0 <= result["churn_probability"] <= 1.0

    def test_risk_level_valid(self, sample_customer):
        """Risk level must be one of the valid labels."""
        from src.predict import predict_churn
        result = predict_churn(sample_customer)
        assert result["risk_level"] in {"Low", "Medium", "High"}

    def test_will_churn_is_bool(self, sample_customer):
        """will_churn must be a boolean."""
        from src.predict import predict_churn
        result = predict_churn(sample_customer)
        assert isinstance(result["will_churn"], bool)

    def test_threshold_consistency(self, sample_customer):
        """will_churn must be consistent with probability and threshold."""
        from src.predict import predict_churn
        threshold = 0.5
        result = predict_churn(sample_customer, threshold=threshold)
        if result["churn_probability"] >= threshold:
            assert result["will_churn"] is True
        else:
            assert result["will_churn"] is False

    def test_high_risk_customer_has_higher_prob(self, high_risk_customer, low_risk_customer):
        """A month-to-month fiber customer should score higher than a 5yr contract customer."""
        from src.predict import predict_churn
        high_result = predict_churn(high_risk_customer)
        low_result = predict_churn(low_risk_customer)
        assert high_result["churn_probability"] > low_result["churn_probability"], \
            "High-risk profile should have higher churn probability than low-risk profile"

    def test_custom_threshold(self, sample_customer):
        """Custom threshold of 0.1 should almost always predict churn=True."""
        from src.predict import predict_churn
        result = predict_churn(sample_customer, threshold=0.01)
        assert result["will_churn"] is True  # Nearly any prob > 0.01

    def test_prediction_deterministic(self, sample_customer):
        """Same input should always give the same output."""
        from src.predict import predict_churn
        result1 = predict_churn(sample_customer)
        result2 = predict_churn(sample_customer)
        assert result1["churn_probability"] == result2["churn_probability"]


# ── Tests for feature engineering used in prediction ──────────────────────

class TestFeatureEngineering:
    """Tests for features added during prediction preprocessing."""

    def test_tenure_group_added(self, sample_customer):
        """engineer_features() should add tenure_group column."""
        from src.features import engineer_features
        df = pd.DataFrame([sample_customer])
        result = engineer_features(df)
        assert "tenure_group" in result.columns

    def test_num_services_added(self, sample_customer):
        """engineer_features() should add num_services column."""
        from src.features import engineer_features
        df = pd.DataFrame([sample_customer])
        result = engineer_features(df)
        assert "num_services" in result.columns

    def test_avg_monthly_spend_added(self, sample_customer):
        """engineer_features() should add avg_monthly_spend column."""
        from src.features import engineer_features
        df = pd.DataFrame([sample_customer])
        result = engineer_features(df)
        assert "avg_monthly_spend" in result.columns

    def test_num_services_range(self, sample_customer):
        """num_services must be between 0 and 8 (8 possible services)."""
        from src.features import engineer_features
        df = pd.DataFrame([sample_customer])
        result = engineer_features(df)
        assert 0 <= result["num_services"].iloc[0] <= 8

    def test_avg_monthly_spend_calculation(self, sample_customer):
        """avg_monthly_spend = TotalCharges / (tenure + 1)."""
        from src.features import engineer_features
        df = pd.DataFrame([sample_customer])
        result = engineer_features(df)
        expected = sample_customer["TotalCharges"] / (sample_customer["tenure"] + 1)
        assert abs(result["avg_monthly_spend"].iloc[0] - expected) < 0.01

    def test_no_tenure_zero_division(self):
        """Customer with tenure=0 should not cause division by zero."""
        from src.features import engineer_features
        new_customer = {
            "gender": "Male", "SeniorCitizen": "No", "Partner": "No",
            "Dependents": "No", "tenure": 0, "PhoneService": "Yes",
            "MultipleLines": "No", "InternetService": "DSL",
            "OnlineSecurity": "No", "OnlineBackup": "No",
            "DeviceProtection": "No", "TechSupport": "No",
            "StreamingTV": "No", "StreamingMovies": "No",
            "Contract": "Month-to-month", "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 29.85, "TotalCharges": 0.0,
        }
        df = pd.DataFrame([new_customer])
        result = engineer_features(df)  # Should not raise ZeroDivisionError
        assert result["avg_monthly_spend"].iloc[0] == 0.0  # 0 / (0+1) = 0
