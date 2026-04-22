# AutoResearch Agent Instructions

## Objective
Maximize validation F2 score on the Healthcare Claims Denial Prediction task.
F2 weights recall 2x over precision. Higher is better. Baseline is 0.613306.

## Rules

1. You may ONLY modify `train.py`
2. `prepare.py` and `run.py` are FROZEN — do not touch them
3. `build_model()` must return an sklearn-compatible estimator or Pipeline
4. `preprocess()` must return a numeric DataFrame with no missing values
5. Training and evaluation must complete in under 60 seconds on CPU
6. No additional data sources or external downloads

## Workflow

1. Read current train.py
2. Propose one modification
3. Edit train.py
4. Run: python3 run.py "description of change"
5. Check val_f2 in output
6. If improved: git add train.py && git commit -m "feat: description"
7. If worse: git checkout train.py
8. Repeat from step 1

## Ideas to Explore

* Different classifiers: RandomForest, XGBoost, LightGBM, GradientBoosting
* Class imbalance: try SMOTE, adjust class_weight, change decision threshold
* Feature encoding: try target encoding or binary encoding for Procedure Code and Diagnosis Code
* Feature engineering: extract month from Date of Service, compute Billed minus Paid ratio
* Hyperparameter tuning: n_estimators, max_depth, learning_rate, min_samples_leaf
* Preprocessing: try RobustScaler instead of StandardScaler

## What NOT to Do

* Do not modify `prepare.py` or `run.py`
* Do not use Outcome, Claim Status, Reason Code, or AR Status as features
* Do not access or peek at the test set
* Do not change the function signatures of `build_model()` or `preprocess()`
* Do not add dependencies not in requirements.txt
* Do not hard-code validation data into the model
