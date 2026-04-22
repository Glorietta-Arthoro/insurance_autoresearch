"""
FROZEN -- Do not modify this file.
Run one experiment: build model, train, evaluate, log result.

Usage:
    python run.py "description"             # logs as status=keep
    python run.py "description" --baseline  # logs as status=baseline
    python run.py "description" --discard   # logs as status=discard
"""
import sys
import time
import subprocess
from prepare import load_data, evaluate, log_result


def get_experiment_id():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return str(int(time.time()))


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

    # 1. Load data (frozen)
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()

    # 2. Import model and preprocessor from train.py (editable)
    from train import build_model, preprocess

    # 3. Preprocess
    X_train_processed = preprocess(X_train)
    X_val_processed = preprocess(X_val)

    # 4. Build and train model
    model = build_model()
    print(f"Model: {model}")
    t0 = time.time()
    model.fit(X_train_processed, y_train)
    runtime = time.time() - t0
    print(f"Training time: {runtime:.2f}s")

    # 5. Evaluate (frozen metric)
    val_f2, val_recall, val_precision = evaluate(model, X_val_processed, y_val)
    print(f"val_f2:        {val_f2:.6f}")
    print(f"val_recall:    {val_recall:.6f}")
    print(f"val_precision: {val_precision:.6f}")

    # 6. Log result
    experiment_id = get_experiment_id()
    log_result(experiment_id, val_f2, val_recall, val_precision, status, description, runtime)
    print(f"Result logged to results.tsv (status={status})")


if __name__ == "__main__":
    main()
