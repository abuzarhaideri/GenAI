"""
data_preprocessing.py
=====================
Data loading, cleaning, feature engineering, and Scikit-Learn preprocessing
pipeline for the Melbourne Housing dataset.

Author : Project 9 — Intelligent Property Price Prediction
"""

import os
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ---------------------------------------------------------------------------
# Feature definitions
# ---------------------------------------------------------------------------
NUMERICAL_FEATURES = [
    "Rooms", "Distance", "Bedroom2", "Bathroom",
    "Car", "Landsize", "BuildingArea", "HouseAge",
]

CATEGORICAL_FEATURES = ["Type", "Regionname"]

TARGET = "Price"


# ---------------------------------------------------------------------------
# Data loading & feature engineering
# ---------------------------------------------------------------------------
def load_and_prepare_data(data_path: Optional[str] = None):
    """
    Load the Melbourne Housing CSV, engineer features, and return
    X (DataFrame), y (Series), and the fitted ColumnTransformer preprocessor.

    Parameters
    ----------
    data_path : str, optional
        Path to the CSV file.  Defaults to ``data/melbourne_housing.csv``
        relative to the directory that contains this script.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix (raw, before preprocessing).
    y : pd.Series
        Target vector (Price).
    preprocessor : sklearn ColumnTransformer
        Unfitted preprocessing pipeline.
    """
    if data_path is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(base_dir, "data", "melbourne_housing.csv")

    df = pd.read_csv(data_path)

    # --- 1.  Drop rows where the target is missing ---
    df = df.dropna(subset=[TARGET]).copy()

    # --- 2.  Parse sale date and compute HouseAge ---
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    df["SaleYear"] = df["Date"].dt.year
    df["HouseAge"] = df["SaleYear"] - df["YearBuilt"]  # NaN where YearBuilt is NaN

    # --- 3.  Select features ---
    X = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES].copy()
    y = df[TARGET].copy()

    # --- 4.  Build preprocessor ---
    preprocessor = get_preprocessor()

    return X, y, preprocessor


# ---------------------------------------------------------------------------
# Preprocessing pipeline
# ---------------------------------------------------------------------------
def get_preprocessor() -> ColumnTransformer:
    """
    Build and return a ``ColumnTransformer`` that applies:

    * **Numerical features** — median imputation → standard scaling.
    * **Categorical features** — most-frequent imputation → one-hot encoding.

    Returns
    -------
    preprocessor : sklearn ColumnTransformer
    """
    numerical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, NUMERICAL_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    return preprocessor


# ---------------------------------------------------------------------------
# Convenience: quick sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    X, y, preprocessor = load_and_prepare_data()
    print(f"Features shape : {X.shape}")
    print(f"Target shape   : {y.shape}")
    print(f"\nFeature dtypes:\n{X.dtypes}")
    print(f"\nTarget stats:\n{y.describe()}")
    print(f"\nMissing values per column:\n{X.isnull().sum()}")
