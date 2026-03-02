"""
train_model.py
==============
Train Linear Regression and Random Forest Regressor on the Melbourne Housing
dataset, evaluate both, and persist the best model as a .pkl file.

Author : Project 9 — Intelligent Property Price Prediction
"""

import json
import os

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from data_preprocessing import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    load_and_prepare_data,
)


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------
def evaluate_model(name: str, pipeline: Pipeline, X_test, y_test) -> dict:
    """
    Compute and print R², MAE, and RMSE for a fitted pipeline.

    Parameters
    ----------
    name : str
        Human-readable model name (for printing).
    pipeline : sklearn Pipeline
        Already-fitted pipeline (preprocessor + regressor).
    X_test : pd.DataFrame
        Test features.
    y_test : pd.Series
        True target values.

    Returns
    -------
    metrics : dict
        Dictionary with keys ``r2``, ``mae``, ``rmse``.
    """
    y_pred = pipeline.predict(X_test)

    r2   = r2_score(y_test, y_pred)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"\n{'=' * 50}")
    print(f"  Model : {name}")
    print(f"{'=' * 50}")
    print(f"  R²    : {r2:.4f}")
    print(f"  MAE   : ${mae:,.2f}")
    print(f"  RMSE  : ${rmse:,.2f}")
    print(f"{'=' * 50}")

    return {"r2": r2, "mae": mae, "rmse": rmse}


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------
def main():
    # 1. Load data ----------------------------------------------------------
    print("Loading and preparing data …")
    X, y, preprocessor = load_and_prepare_data()
    print(f"  Dataset size : {X.shape[0]} samples, {X.shape[1]} raw features")

    # 2. Train/Test split ---------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"  Train size   : {X_train.shape[0]}")
    print(f"  Test size    : {X_test.shape[0]}")

    # 3. Build pipelines ----------------------------------------------------
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=200, random_state=42, n_jobs=-1
        ),
    }

    results = {}
    fitted_pipelines = {}

    for name, regressor in models.items():
        pipe = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("regressor", regressor),
        ])
        print(f"\nTraining {name} …")
        pipe.fit(X_train, y_train)

        metrics = evaluate_model(name, pipe, X_test, y_test)
        results[name] = metrics
        fitted_pipelines[name] = pipe

    # 4. Select the best model (by R²) --------------------------------------
    best_name = max(results, key=lambda k: results[k]["r2"])
    best_pipeline = fitted_pipelines[best_name]
    print(f"\n★  Best model: {best_name} (R² = {results[best_name]['r2']:.4f})")

    # 5. Save model and metadata --------------------------------------------
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    model_path = os.path.join(models_dir, "best_model.pkl")
    joblib.dump(best_pipeline, model_path)
    print(f"  Saved model  → {model_path}")

    metadata = {
        "best_model": best_name,
        "numerical_features": NUMERICAL_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "metrics": {
            name: {k: round(v, 4) for k, v in m.items()}
            for name, m in results.items()
        },
    }
    meta_path = os.path.join(models_dir, "model_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved metadata → {meta_path}")


if __name__ == "__main__":
    main()
