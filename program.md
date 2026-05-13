# AutoResearch Agent Instructions

## Objective
Maximize validation F2 score on the Healthcare Claims Denial Prediction task.
F2 weights recall 2x over precision. Higher is better.
Threshold is locked at 0.5. Do not tune it.

## Before Proposing Experiments
Search the web for:
- "healthcare claims denial prediction machine learning models"
- "best models for insurance claim denial classification"
Use what you find to inform your experiments.

## Rules

1. You may ONLY modify `train.py`
2. `prepare.py` and `run.py` are FROZEN — do not touch them
3. `build_model()` must return an sklearn-compatible estimator or Pipeline
4. `preprocess()` must return a numeric DataFrame with no missing values
5. Training and evaluation must complete in under 60 seconds on CPU
6. No additional data sources or external downloads
7. The decision threshold is locked at 0.5 — do not tune it

## Workflow — Simple and Strict

Step 1: Propose one change and explain why
Step 2: Edit train.py
Step 3exactly once: python3 run.py "short description"
Step 4: run.py automatically logs keep or discard and reverts train.py if discarded
Step 5: If kept, commit: git add train.py && git commit -m "description"
Step 6: Move to next experiment

Never call python3 run.py more than once per experiment.
Never manually revert train.py. run.py handles it automatically on discard.

## Models Available
xgboost, lightgbm, catboost, RandomForest, ExtraTrees, GradientBoosting,
HistGradientBoosting, SVM, KNN, MLP, AdaBoost, BaggingClassifier, LogisticRegression

## Ideas to Explore
- XGBoost, LightGBM, CatBoost as replacements for Random Forest
- Target encoding for Procedure Code and Diagnosis Code
- Feature interactions: procedure code combined with insurance type
- Hyperparameter tuning: n_estimators, max_depth, learning_rate
- Ensemble methods without threshold tuning

## What NOT to Do
- Do not modify prepare.py or run.py
- Do not use Outcome, Claim Status, Reason Code, or AR Status as features
- Do not access the test set
- Do not tune the decision threshold
- Do not call python3 run.py more than once per experiment
- Do not manually revert train.py
