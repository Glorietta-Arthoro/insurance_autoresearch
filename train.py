"""
EDITABLE -- This is the only file the agent may modify.
Experiment 1: Random Forest with class_weight='balanced'.
"""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def build_model():
    """
    Returns a trained sklearn pipeline.
    Must expose a .predict(X) method.
    Agent may change anything in this file.
    """
    model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        max_depth=None,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
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
