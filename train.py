"""
EDITABLE -- This is the only file the agent may modify.
Experiment 5: RF + ExtraTrees soft-voting ensemble + auto-threshold + richer features.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import fbeta_score


class AutoThresholdClassifier(BaseEstimator, ClassifierMixin):
    """Fits an ensemble, picks the threshold that maximises OOF F2."""

    def __init__(self, base):
        self.base = base
        self.threshold_ = 0.5

    def fit(self, X, y):
        oof_proba = np.zeros(len(y))
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for train_idx, val_idx in cv.split(X, y):
            clf = clone(self.base)
            X_tr = X.iloc[train_idx] if hasattr(X, "iloc") else X[train_idx]
            X_vl = X.iloc[val_idx]  if hasattr(X, "iloc") else X[val_idx]
            y_tr = y.iloc[train_idx] if hasattr(y, "iloc") else y[train_idx]
            clf.fit(X_tr, y_tr)
            oof_proba[val_idx] = clf.predict_proba(X_vl)[:, 1]

        best_t, best_f2 = 0.5, -1.0
        for t in np.linspace(0.05, 0.65, 61):
            preds = (oof_proba >= t).astype(int)
            f2 = fbeta_score(y, preds, beta=2, zero_division=0)
            if f2 > best_f2:
                best_f2, best_t = f2, t
        self.threshold_ = best_t

        self.base.fit(X, y)
        self.classes_ = self.base.classes_
        return self

    def predict(self, X):
        proba = self.base.predict_proba(X)[:, 1]
        return (proba >= self.threshold_).astype(int)

    def predict_proba(self, X):
        return self.base.predict_proba(X)


def build_model():
    """
    Returns a trained sklearn-compatible estimator.
    Must expose a .predict(X) method.
    """
    rf = RandomForestClassifier(
        n_estimators=200, class_weight="balanced",
        min_samples_leaf=2, random_state=42, n_jobs=-1
    )
    et = ExtraTreesClassifier(
        n_estimators=200, class_weight="balanced",
        min_samples_leaf=2, random_state=0, n_jobs=-1
    )
    ensemble = VotingClassifier(
        estimators=[("rf", rf), ("et", et)],
        voting="soft",
        n_jobs=-1
    )
    return AutoThresholdClassifier(base=ensemble)


def preprocess(X):
    """
    Feature engineering and encoding.
    Agent may modify this function.
    """
    X = X.copy()

    # Extract month and quarter from Date of Service
    if "Date of Service" in X.columns:
        dt = pd.to_datetime(X["Date of Service"], errors="coerce")
        X["month"] = dt.dt.month.fillna(0)
        X["quarter"] = dt.dt.quarter.fillna(0)
        X = X.drop(columns=["Date of Service"])

    # Ratio / difference features
    if "Billed Amount" in X.columns and "Paid Amount" in X.columns:
        X["billed_paid_ratio"] = X["Billed Amount"] / (X["Paid Amount"] + 1)
        X["billed_paid_diff"] = X["Billed Amount"] - X["Paid Amount"]
    if "Billed Amount" in X.columns and "Allowed Amount" in X.columns:
        X["billed_allowed_ratio"] = X["Billed Amount"] / (X["Allowed Amount"] + 1)
        X["billed_allowed_diff"] = X["Billed Amount"] - X["Allowed Amount"]
    if "Allowed Amount" in X.columns and "Paid Amount" in X.columns:
        X["allowed_paid_diff"] = X["Allowed Amount"] - X["Paid Amount"]

    # Frequency encoding for categoricals
    for col in ["Procedure Code", "Diagnosis Code", "Insurance Type", "Follow-up Required"]:
        if col in X.columns:
            freq = X[col].value_counts() / len(X)
            X[col] = X[col].map(freq).fillna(0)

    return X
