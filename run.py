"""
FROZEN -- Do not modify this file.
Run one experiment: build model, train, evaluate, log result.

Usage:
    python3 run.py "description"             # logs as status=keep
    python3 run.py "description" --baseline  # logs as status=baseline
    python3 run.py "description" --discard   # logs as status=discard
"""
import sys
import time
import csv
import os
from prepare import load_data, evaluate, log_result

RESULTS_FILE = "results.tsv"

def get_next_experiment_id():
    if not os.path.exists(RESULTS_FILE):
        return 1
    with open(RESULTS_FILE) as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)
    exp_nums = []
    for row in rows:
        try:
            exp_nums.append(int(row["experiment"]))
        except:
            pass
    return max(exp_nums) + 1 if exp_nums else 1

def main():
    args = sys.argv[1:]
    status = "keep"
    description_parts = []
    for a in args:
        if a == "--baseline":
            status = "baseline"
        elif a == "--discard":
            status = "discard"
        else:
            description_parts.append(a)
    description = " ".join(description_parts) if description_parts else "experiment"

    X_train, y_train, X_val, y_val, X_test, y_test = load_data()

    from train import build_model, preprocess

    X_train_processed = preprocess(X_train)
    X_val_processed = preprocess(X_val)

    model = build_model()
    print(f"Model: {type(model).__name__}")
    t0 = time.time()
    model.fit(X_train_processed, y_train)
    runtime = time.time() - t0
    print(f"Training time: {runtime:.2f}s")

    val_f2, val_recall, val_precision = evaluate(model, X_val_processed, y_val)
    print(f"val_f2:        {val_f2:.6f}")
    print(f"val_recall:    {val_recall:.6f}")
    print(f"val_precision: {val_precision:.6f}")
    print(f"status:        {status}")

    experiment_id = get_next_experiment_id()
    log_result(experiment_id, val_f2, val_recall, val_precision, status, description, runtime)
    print(f"Experiment {experiment_id} logged to results.tsv (status={status})")

if __name__ == "__main__":
    main()
