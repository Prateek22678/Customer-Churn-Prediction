"""
predict.py
----------
Prediction interface for the Churn Prediction project.

This module provides a clean, simple function that:
  1. Accepts a customer's data as a Python dictionary
  2. Loads the saved model pipeline
  3. Returns the churn probability and a risk label

This is the module you'd use in production (API endpoint, web app, etc.)

Usage:
    from src.predict import predict_churn

    customer = {
        "gender": "Female",
        "SeniorCitizen": "No",
        "Partner": "Yes",
        ...
    }
    result = predict_churn(customer)
    print(result)
    # → {'churn_probability': 0.72, 'risk_level': 'High', 'will_churn': True}

Author: Churn Prediction Project
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional


# ── Constants ──────────────────────────────────────────────────────────────
MODELS_DIR = Path(__file__).parent.parent / "models"
DEFAULT_MODEL_PATH = MODELS_DIR / "xgboost_pipeline.pkl"

# Risk thresholds — adjust based on business needs
RISK_THRESHOLDS = {
    "Low":    (0.00, 0.35),   # < 35% chance of churn
    "Medium": (0.35, 0.60),   # 35–60% chance of churn
    "High":   (0.60, 1.00),   # > 60% chance of churn — act now!
}

# Default decision threshold — tuned during evaluation to maximize F1
# (evaluated on held-out test set, not training data)
# At 0.52: Precision=0.534, Recall=0.800, F1=0.640, ROC-AUC=0.847
DEFAULT_THRESHOLD = 0.52


def load_model(model_path: Path = DEFAULT_MODEL_PATH):
    """
    Load a saved model pipeline from disk.

    Parameters
    ----------
    model_path : Path
        Path to the .pkl file saved by joblib.

    Returns
    -------
    sklearn Pipeline (fitted)

    Raises
    ------
    FileNotFoundError
        If the model file doesn't exist (run train.py first).
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"\n❌ Model not found at: {model_path}\n"
            "Please run training first:\n"
            "  python -m src.train\n"
        )
    model = joblib.load(model_path)
    return model


def get_risk_level(probability: float) -> str:
    """
    Convert a probability (0–1) to a human-readable risk label.

    Parameters
    ----------
    probability : float
        Predicted churn probability.

    Returns
    -------
    str
        'Low', 'Medium', or 'High'
    """
    for level, (low, high) in RISK_THRESHOLDS.items():
        if low <= probability < high:
            return level
    return "High"  # fallback for probability == 1.0


def engineer_prediction_features(customer_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the same feature engineering used during training.

    IMPORTANT: We must apply the EXACT SAME transformations as in training,
    otherwise the model will see different feature columns and fail or give
    wrong results.

    Parameters
    ----------
    customer_df : pd.DataFrame
        Raw customer data (before feature engineering).

    Returns
    -------
    pd.DataFrame
        With tenure_group, num_services, avg_monthly_spend added.
    """
    from src.features import engineer_features
    return engineer_features(customer_df)


def predict_churn(
    customer: Dict[str, Any],
    model_path: Optional[Path] = None,
    threshold: float = DEFAULT_THRESHOLD,
) -> Dict[str, Any]:
    """
    Predict whether a single customer will churn.

    This is the main function you'll call from the web app or any other
    integration. It handles everything: feature engineering, model loading,
    prediction, and risk labeling.

    Parameters
    ----------
    customer : dict
        Customer data. Keys must match the training dataset columns
        (excluding customerID and Churn). Example:

        {
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

    model_path : Path, optional
        Override the default model path.
    threshold : float
        Decision threshold (default 0.40 — tuned for recall).

    Returns
    -------
    dict with keys:
        - churn_probability : float  (0.0 to 1.0)
        - risk_level        : str    ('Low', 'Medium', 'High')
        - will_churn        : bool   (True if prob >= threshold)
        - threshold_used    : float
    """
    # 1. Convert customer dict to DataFrame (model expects a DataFrame)
    customer_df = pd.DataFrame([customer])

    # 2. Apply feature engineering (same as training)
    customer_df = engineer_prediction_features(customer_df)

    # 3. Load model
    path = model_path or DEFAULT_MODEL_PATH
    model = load_model(path)

    # 4. Predict
    churn_prob = float(model.predict_proba(customer_df)[:, 1][0])

    # 5. Apply threshold and get risk label
    will_churn = churn_prob >= threshold
    risk_level = get_risk_level(churn_prob)

    return {
        "churn_probability": round(churn_prob, 4),
        "risk_level": risk_level,
        "will_churn": bool(will_churn),
        "threshold_used": threshold,
    }


def predict_batch(
    customers_df: pd.DataFrame,
    model_path: Optional[Path] = None,
    threshold: float = DEFAULT_THRESHOLD,
) -> pd.DataFrame:
    """
    Predict churn for a batch of customers (multiple rows at once).

    More efficient than calling predict_churn() in a loop.

    Parameters
    ----------
    customers_df : pd.DataFrame
        Multiple customer records. Same columns as predict_churn().
    model_path : Path, optional
    threshold : float

    Returns
    -------
    pd.DataFrame
        Original dataframe with added columns:
        'churn_probability', 'risk_level', 'will_churn'
    """
    # Apply feature engineering
    df = engineer_prediction_features(customers_df.copy())

    # Load model and predict
    path = model_path or DEFAULT_MODEL_PATH
    model = load_model(path)
    probs = model.predict_proba(df)[:, 1]

    result = customers_df.copy()
    result["churn_probability"] = probs.round(4)
    result["risk_level"] = [get_risk_level(p) for p in probs]
    result["will_churn"] = (probs >= threshold)

    return result


if __name__ == "__main__":
    # Demo — predict for a single high-risk customer
    sample_customer = {
        "gender": "Female",
        "SeniorCitizen": "No",
        "Partner": "No",
        "Dependents": "No",
        "tenure": 2,
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
        "TotalCharges": 171.0,
    }

    print("🔮 Predicting for sample customer...")
    result = predict_churn(sample_customer)
    print(f"\nResult:")
    for k, v in result.items():
        print(f"  {k}: {v}")
