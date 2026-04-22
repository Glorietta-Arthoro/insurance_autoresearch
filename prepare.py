"""
FROZEN -- Do not modify this file.
Data loading, train/val/test split, F2 evaluation, logging, and plotting.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score, recall_score, precision_score
from matplotlib.lines import Line2D

RANDOM_SEED = 42
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15
RESULTS_FILE = "results.tsv"
DATA_PATH = "data/claim_data.csv"

def load_data():
    df = pd.read_csv(DATA_PATH)

    # Binary target: Denied or Partially Paid = 1, Paid = 0
    df["target"] = df["Outcome"].apply(
        lambda x: 1 if str(x).strip() in ["Denied", "Partially Paid"] else 0
    )

    # Drop identifiers, target, and post-submission leakage columns
    drop_cols = [
        "Claim ID", "Provider ID", "Patient ID",
        "Outcome", "Claim Status", "Reason Code", "AR Status"
    ]
    feature_cols = [c for c in df.columns if c not in drop_cols + ["target"]]

    X = df[feature_cols]
    y = df["target"]

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=TEST_FRACTION, random_state=RANDOM_SEED, stratify=y
    )
    val_adjusted = VAL_FRACTION / (1 - TEST_FRACTION)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_adjusted, random_state=RANDOM_SEED, stratify=y_temp
    )

    print(f"Data loaded: {len(X_train)} train, {len(X_val)} val, {len(X_test)} test")
    print(f"Denial rate — train: {y_train.mean():.1%}, val: {y_val.mean():.1%}, test: {y_test.mean():.1%}")
    print(f"Features used: {list(X.columns)}")

    return X_train, y_train, X_val, y_val, X_test, y_test

def evaluate(model, X_val, y_val):
    y_pred = model.predict(X_val)
    f2 = float(fbeta_score(y_val, y_pred, beta=2, zero_division=0))
    recall = float(recall_score(y_val, y_pred, zero_division=0))
    precision = float(precision_score(y_val, y_pred, zero_division=0))
    return f2, recall, precision

def log_result(experiment_id, val_f2, val_recall, val_precision, status, description, runtime=None):
    file_exists = os.path.exists(RESULTS_FILE)
    with open(RESULTS_FILE, "a", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        if not file_exists:
            writer.writerow(["experiment", "val_f2", "val_recall", "val_precision", "runtime_sec", "status", "description"])
        writer.writerow([
            experiment_id,
            f"{val_f2:.6f}",
            f"{val_recall:.6f}",
            f"{val_precision:.6f}",
            f"{runtime:.2f}" if runtime else "N/A",
            status,
            description
        ])

def plot_results(save_path="performance.png"):
    if not os.path.exists(RESULTS_FILE):
        print("No results.tsv found. Run some experiments first.")
        return

    experiments, f2s, recalls, statuses, descriptions = [], [], [], [], []
    with open(RESULTS_FILE) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            experiments.append(row["experiment"])
            f2s.append(float(row["val_f2"]))
            recalls.append(float(row["val_recall"]))
            statuses.append(row["status"])
            descriptions.append(row["description"])

    color_map = {"keep": "#2ecc71", "discard": "#e74c3c", "baseline": "#3498db"}
    colors = [color_map.get(s, "#95a5a6") for s in statuses]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    ax1.scatter(range(len(f2s)), f2s, c=colors, s=80, zorder=3, edgecolors="white", linewidth=0.5)
    ax1.plot(range(len(f2s)), f2s, "k--", alpha=0.2, zorder=2)
    best_so_far = []
    current_best = 0
    for r in f2s:
        current_best = max(current_best, r)
        best_so_far.append(current_best)
    ax1.plot(range(len(f2s)), best_so_far, color="#2ecc71", linewidth=2.5, label="Best so far")
    ax1.axhline(y=0.70, color="orange", linestyle="--", linewidth=1.5, label="Target F2 = 0.70")
    ax1.set_ylabel("Validation F2 Score (higher is better)", fontsize=12)
    ax1.set_title("AutoResearch: Healthcare Claims Denial Prediction", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    ax2.scatter(range(len(recalls)), recalls, c=colors, s=80, zorder=3, edgecolors="white", linewidth=0.5)
    ax2.plot(range(len(recalls)), recalls, "k--", alpha=0.2, zorder=2)
    best_recall = []
    current_best_recall = 0
    for r in recalls:
        current_best_recall = max(current_best_recall, r)
        best_recall.append(current_best_recall)
    ax2.plot(range(len(recalls)), best_recall, color="#2ecc71", linewidth=2.5, label="Best so far")
    ax2.axhline(y=0.75, color="orange", linestyle="--", linewidth=1.5, label="Target Recall = 0.75")
    ax2.set_xlabel("Experiment #", fontsize=12)
    ax2.set_ylabel("Validation Recall on Denied Class", fontsize=12)
    ax2.grid(True, alpha=0.3)

    short_labels = [d[:22] + ".." if len(d) > 24 else d for d in descriptions]
    ax2.set_xticks(range(len(f2s)))
    ax2.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=8)

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#3498db", markersize=10, label="baseline"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2ecc71", markersize=10, label="keep"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#e74c3c", markersize=10, label="discard"),
        Line2D([0], [0], color="#2ecc71", linewidth=2.5, label="Best so far"),
        Line2D([0], [0], color="orange", linestyle="--", linewidth=1.5, label="Target threshold"),
    ]
    ax1.legend(handles=legend_elements, loc="lower right", fontsize=9)
    ax2.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved {save_path}")

if __name__ == "__main__":
    plot_results()
