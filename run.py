"""
FROZEN -- Do not modify this file.
Run one experiment: build model, train, evaluate, log result.
Automatically logs keep or discard based on whether F2 improves on current best.
Baseline flag overrides automatic decision.

Usage:
    python3 run.py "description"             # auto keep or discard based on F2
    python3 run.py "description" --baseline  # logs as baseline
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

def get_current_best():
    if not os.path.exists(RESULTS_FILE):
        return 0.0
    with open(RESULTS_FILE) as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)
    best = 0.0
    for row in rows:
        if row["status"] in ["keep", "baseline"]:
            try:
                f2 = float(row["val_f2"])
                if f2 > best:
                    best = f2
            except:
                pass
    return best

def main():
    args = sys.argv[1:]
    is_baseline = False
    description_parts = []
    for a in args:
        if a == "--baseline":
            is_baseline = True
        else:
            description_parts.append(a)
    description = " ".join(description_parts) if description_parts else "experiment"

    # 1. Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()

    # 2. Import from train.py
    from train import build_model, preprocess

    # 3. Preprocess
    X_train_processed = preprocess(X_train)
    X_val_processed = preprocess(X_val)

    # 4. Build and train
    model = build_model()
    print(f"Model: {type(model).__name__}")
    t0 = time.time()
    model.fit(X_train_processed, y_train)
    runtime = time.time() - t0
    print(f"Training time: {runtime:.2f}s")

    # 5. Evaluate
    val_f2, val_recall, val_precision = evaluate(model, X_val_processed, y_val)
    print(f"val_f2:        {val_f2:.6f}")
    print(f"val_recall:    {val_recall:.6f}")
    print(f"val_precision: {val_precision:.6f}")

    # 6. Auto determine keep or discard
    if is_baseline:
        status = "baseline"
    else:
        current_best = get_current_best()
        if val_f2 > current_best:
            status = "keep"
        else:
            status = "discard"

    print(f"status:        {status}")

    # 7. Log exactly one row
    experiment_id = get_next_experiment_id()
    log_result(experiment_id, val_f2, val_recall, val_precision, status, description, runtime)
    print(f"Experiment {experiment_id} logged to results.tsv (status={status})")

    # 8. If discarded, revert train.py automatically
    if status == "discard":
        os.system("git checkout train.py")
        print("train.py reverted automatically")

if __name__ == "__main__":
    main()
