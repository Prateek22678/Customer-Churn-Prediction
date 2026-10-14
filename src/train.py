"""
train.py
--------
Model training script for the Churn Prediction project.

Trains and compares three models:
  1. Logistic Regression  — simple, fast, interpretable baseline
  2. Random Forest        — ensemble of decision trees, handles non-linearity
  3. XGBoost              — gradient boosting, often wins on tabular data

Strategy for class imbalance (~26% churn / ~74% not churn):
  - We use class_weight='balanced' for LR and RF
  - For XGBoost, we use scale_pos_weight to upweight the minority class
  - This tells the model: "missing a churner is more costly than a false alarm"

We use RandomizedSearchCV with stratified 5-fold cross-validation.

Usage:
    python -m src.train

Author: Churn Prediction Project
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.data_preprocessing import load_raw_data, clean_data, split_features_target, get_column_types
from src.features import engineer_features, build_preprocessor


# ── Constants ──────────────────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_CV_FOLDS = 5
N_ITER_SEARCH = 20          # Number of random hyperparameter combos to try
MODELS_DIR = Path(__file__).parent.parent / "models"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"


def make_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Stratified train/test split (80/20).

    Stratified = maintains the same churn ratio in both train and test sets.
    This is important because churn is imbalanced (~26% positive).

    Parameters
    ----------
    X : pd.DataFrame
    y : pd.Series

    Returns
    -------
    Tuple of (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,          # ← Key: preserve class ratio in both splits
    )
    print(f"  Train: {len(X_train):,} rows  |  Test: {len(X_test):,} rows")
    print(f"  Train churn rate: {y_train.mean():.1%}  |  Test churn rate: {y_test.mean():.1%}")
    return X_train, X_test, y_train, y_test


def build_model_pipelines(
    preprocessor,
    neg_pos_ratio: float,
) -> Dict[str, Pipeline]:
    """
    Build sklearn Pipelines for each model.

    Each pipeline has two steps:
      1. preprocessor — scales and encodes features
      2. classifier   — the actual ML model

    Combining them in a Pipeline ensures the preprocessor is only
    fitted on training data during cross-validation (no data leakage).

    Parameters
    ----------
    preprocessor : ColumnTransformer
        The unfitted feature preprocessor.
    neg_pos_ratio : float
        Ratio of negative to positive class samples.
        Used to set scale_pos_weight for XGBoost.

    Returns
    -------
    dict mapping model name → Pipeline
    """
    pipelines = {
        "Logistic Regression": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(
                class_weight="balanced",   # Automatically upweights churners
                max_iter=1000,             # More iterations for convergence
                random_state=RANDOM_STATE,
                solver="lbfgs",
                # Note: penalty removed — sklearn 1.9+ defaults to l2 via l1_ratio
            )),
        ]),

        "Random Forest": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(
                class_weight="balanced",
                n_estimators=100,
                random_state=RANDOM_STATE,
                n_jobs=-1,                 # Use all CPU cores
            )),
        ]),

        "XGBoost": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", XGBClassifier(
                scale_pos_weight=neg_pos_ratio,  # Handles imbalance
                random_state=RANDOM_STATE,
                eval_metric="logloss",
                verbosity=0,
            )),
        ]),
    }
    return pipelines


def get_param_grids() -> Dict[str, Dict[str, Any]]:
    """
    Hyperparameter search spaces for RandomizedSearchCV.

    We define a range of values for each model's key hyperparameters.
    RandomizedSearchCV will randomly sample N_ITER_SEARCH combinations.

    Returns
    -------
    dict mapping model name → hyperparameter grid
    """
    return {
        "Logistic Regression": {
            "classifier__C": [0.001, 0.01, 0.1, 1, 10, 100],
            "classifier__penalty": ["l2"],
        },
        "Random Forest": {
            "classifier__n_estimators": [100, 200, 300],
            "classifier__max_depth": [None, 5, 10, 20],
            "classifier__min_samples_split": [2, 5, 10],
            "classifier__min_samples_leaf": [1, 2, 4],
        },
        "XGBoost": {
            "classifier__n_estimators": [100, 200, 300],
            "classifier__max_depth": [3, 4, 5, 6],
            "classifier__learning_rate": [0.01, 0.05, 0.1, 0.2],
            "classifier__subsample": [0.7, 0.8, 0.9, 1.0],
            "classifier__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        },
    }


def train_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    pipelines: Dict[str, Pipeline],
    param_grids: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Run RandomizedSearchCV for each model and return the best estimators.

    We optimize for ROC-AUC because it's robust to class imbalance
    and measures overall discriminative ability.

    Parameters
    ----------
    X_train, y_train : training data
    pipelines : dict of model pipelines
    param_grids : dict of hyperparameter grids

    Returns
    -------
    dict mapping model name → best fitted pipeline
    """
    cv = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    best_models = {}

    for name, pipeline in pipelines.items():
        print(f"\n  🔍 Training {name}...")
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_grids[name],
            n_iter=N_ITER_SEARCH,
            scoring="roc_auc",        # Optimize for AUC
            cv=cv,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=0,
            refit=True,               # Refit the best model on all training data
        )
        search.fit(X_train, y_train)
        best_models[name] = search.best_estimator_
        print(f"    ✅ Best CV ROC-AUC: {search.best_score_:.4f}")
        print(f"    Best params: {search.best_params_}")

    return best_models


def save_model(pipeline: Pipeline, model_name: str) -> Path:
    """
    Save a fitted pipeline to disk using joblib.

    joblib is better than pickle for sklearn objects because it handles
    large numpy arrays efficiently.

    Parameters
    ----------
    pipeline : fitted sklearn Pipeline
    model_name : str  (used to name the file)

    Returns
    -------
    Path  (where the model was saved)
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = model_name.lower().replace(" ", "_")
    filepath = MODELS_DIR / f"{safe_name}_pipeline.pkl"
    joblib.dump(pipeline, filepath)
    print(f"  💾 Saved: {filepath}")
    return filepath


def run_training() -> Tuple[Dict, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Master training function — runs the full training pipeline.

    Returns
    -------
    best_models : dict of fitted pipelines
    X_train, X_test, y_train, y_test : train/test splits
    """
    print("=" * 60)
    print("🚀 CHURN PREDICTION — MODEL TRAINING")
    print("=" * 60)

    # 1. Load and clean data
    df_raw = load_raw_data()
    df_clean = clean_data(df_raw)

    # 2. Feature engineering
    df_featured = engineer_features(df_clean)

    # 3. Split X and y
    X, y = split_features_target(df_featured)

    # 4. Get column types (after feature engineering adds new cols)
    numeric_cols, categorical_cols = get_column_types(X)

    # 5. Train/test split
    print("\n📊 Splitting data...")
    X_train, X_test, y_train, y_test = make_train_test_split(X, y)

    # Save processed splits for later use
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(PROCESSED_DIR / "X_train.csv", index=False)
    X_test.to_csv(PROCESSED_DIR / "X_test.csv", index=False)
    y_train.to_csv(PROCESSED_DIR / "y_train.csv", index=False)
    y_test.to_csv(PROCESSED_DIR / "y_test.csv", index=False)
    print(f"  💾 Saved processed splits to {PROCESSED_DIR}")

    # 6. Build preprocessor and pipelines
    neg_pos_ratio = (y_train == 0).sum() / (y_train == 1).sum()
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    pipelines = build_model_pipelines(preprocessor, neg_pos_ratio)
    param_grids = get_param_grids()

    # 7. Train all models
    print("\n🏋️ Training models (this may take a few minutes)...")
    best_models = train_all_models(X_train, y_train, pipelines, param_grids)

    # 8. Save all models
    print("\n💾 Saving models...")
    for name, model in best_models.items():
        save_model(model, name)

    # Save the best model (XGBoost typically) as the "production" model
    # We'll determine the actual best after evaluation
    print("\n✅ Training complete!")
    return best_models, X_train, X_test, y_train, y_test


if __name__ == "__main__":
    run_training()
