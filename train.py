"""
EDITABLE -- This is the only file the agent may modify.
Baseline: Logistic Regression on claims features.
"""
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def build_model():
    """
    Returns a trained sklearn pipeline.
    Must expose a .predict(X) method.
    Agent may change anything in this file.
    """
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            random_state=42,
            max_iter=1000
        ))
    ])
    return model

def preprocess(X):
    """
    Feature engineering and encoding.
    Agent may modify this function.
    """
    X = X.copy()

    # Encode categorical columns with frequency encoding
    for col in ["Procedure Code", "Diagnosis Code", "Insurance Type", "Follow-up Required"]:
        if col in X.columns:
            freq = X[col].value_counts() / len(X)
            X[col] = X[col].map(freq).fillna(0)

    # Drop date column for now
    if "Date of Service" in X.columns:
        X = X.drop(columns=["Date of Service"])

    return X
