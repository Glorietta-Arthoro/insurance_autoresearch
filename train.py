"""
EDITABLE -- This is the only file the agent may modify.
Experiment 2: Random Forest + feature engineering (month, ratios) + lower threshold.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier

class ThresholdClassifier(BaseEstimator, ClassifierMixin):
    """Wraps a classifier and applies a custom predict threshold."""
    def __init__(self, base, threshold=0.35):
        self.base = base
        self.threshold = threshold

    def fit(self, X, y):
        self.base.fit(X, y)
        self.classes_ = self.base.classes_
        return self

    def predict(self, X):
        proba = self.base.predict_proba(X)[:, 1]
        return (proba >= self.threshold).astype(int)

    def predict_proba(self, X):
        return self.base.predict_proba(X)

def build_model():
    """
    Returns a trained sklearn-compatible estimator.
    Must expose a .predict(X) method.
    """
    rf = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        max_depth=None,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    return ThresholdClassifier(base=rf, threshold=0.35)

def preprocess(X):
    """
    Feature engineering and encoding.
    Agent may modify this function.
    """
    X = X.copy()

    # Extract month from Date of Service
    if "Date of Service" in X.columns:
        X["month"] = pd.to_datetime(X["Date of Service"], errors="coerce").dt.month.fillna(0)
        X = X.drop(columns=["Date of Service"])

    # Ratio features
    if "Billed Amount" in X.columns and "Paid Amount" in X.columns:
        X["billed_paid_ratio"] = X["Billed Amount"] / (X["Paid Amount"] + 1)
    if "Billed Amount" in X.columns and "Allowed Amount" in X.columns:
        X["billed_allowed_ratio"] = X["Billed Amount"] / (X["Allowed Amount"] + 1)

    # Frequency encoding for categoricals
    for col in ["Procedure Code", "Diagnosis Code", "Insurance Type", "Follow-up Required"]:
        if col in X.columns:
            freq = X[col].value_counts() / len(X)
            X[col] = X[col].map(freq).fillna(0)

    return X
