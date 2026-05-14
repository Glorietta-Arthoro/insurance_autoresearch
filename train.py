"""
EDITABLE -- This is the only file the agent may modify.
Exp: CatBoost class_weights=[1,4] native cat features threshold=0.5.
[1,3] got 1 TN but 2 FN. [1,4] aims to recover FN while keeping TN.
"""
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier


def build_model():
    return CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        class_weights=[1, 4],
        cat_features=["Diagnosis Code", "Procedure Code", "Insurance Type", "Follow-up Required"],
        random_seed=42,
        verbose=0,
    )


def preprocess(X):
    X = X.copy()
    if "Date of Service" in X.columns:
        dt = pd.to_datetime(X["Date of Service"], errors="coerce")
        X["month"] = dt.dt.month.fillna(0).astype(int)
        X["quarter"] = dt.dt.quarter.fillna(0).astype(int)
        X = X.drop(columns=["Date of Service"])

    if "Billed Amount" in X.columns and "Paid Amount" in X.columns:
        X["billed_paid_ratio"] = X["Billed Amount"] / (X["Paid Amount"] + 1)
        X["billed_paid_diff"] = X["Billed Amount"] - X["Paid Amount"]
    if "Billed Amount" in X.columns and "Allowed Amount" in X.columns:
        X["billed_allowed_ratio"] = X["Billed Amount"] / (X["Allowed Amount"] + 1)
        X["billed_allowed_diff"] = X["Billed Amount"] - X["Allowed Amount"]
    if "Allowed Amount" in X.columns and "Paid Amount" in X.columns:
        X["allowed_paid_diff"] = X["Allowed Amount"] - X["Paid Amount"]

    for col in ["Diagnosis Code", "Procedure Code", "Insurance Type", "Follow-up Required"]:
        if col in X.columns:
            X[col] = X[col].fillna("UNKNOWN").astype(str)

    for col in X.select_dtypes(include=["number"]).columns:
        X[col] = X[col].fillna(0)

    return X
