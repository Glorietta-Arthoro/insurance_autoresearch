# AutoResearch Agent Instructions

## Objective
Maximize validation F2 score on the Healthcare Claims Denial Prediction task.
F2 weights recall 2x over precision. Higher is better. Baseline is 0.613306.
Threshold is locked at 0.5. Do not tune it. Beat the baseline through genuine model improvement.

## Before Proposing Experiments
Search the web for the following before starting:
- "healthcare claims denial prediction features machine learning"
- "insurance claim denial risk factors predictive model"
Use what you find to inform which feature interactions and variable combinations are worth testing.

## Rules

1. You may ONLY modify `train.py`
2. `prepare.py` and `run.py` are FROZEN — do not touch them
3. `build_model()` must return an sklearn-compatible estimator or Pipeline
4. `preprocess()` must return a numeric DataFrame with no missing values
5. Training and evaluation must complete in under 60 seconds on CPU
6. No additional data sources or external downloads
7. The decision threshold is locked at 0.5 — do not tune it under any circumstances

## Workflow

1. Search the web for claims denial prediction research
2. Read current train.py
3. Propose one modification informed by your research
4. Edit train.py
5. Run: python3 run.py "description of change"
6. Check val_f2 in output
7. If improved: git add train.py && git commit -m "feat: description"
8. If worse: git checkout train.py && python3 run.py "description" --discard
9. Repeat from step 2

## Ideas to Explore

* Different classifiers: XGBoost, LightGBM, CatBoost
* Feature interactions: procedure code combined with insurance type, billed-to-allowed ratio
* Target encoding for Procedure Code and Diagnosis Code using denial rate per code
* Hyperparameter tuning: n_estimators, max_depth, learning_rate, min_samples_leaf
* Ensemble methods without threshold tuning
* Regularization strategies to improve generalization

## What NOT to Do

* Do not modify `prepare.py` or `run.py`
* Do not use Outcome, Claim Status, Reason Code, or AR Status as features
* Do not access or peek at the test set
* Do not change the function signatures of `build_model()` or `preprocess()`
* Do not add dependencies not in requirements.txt
* Do not tune the decision threshold — it is locked at 0.5
* Do not hard-code validation data into the model
