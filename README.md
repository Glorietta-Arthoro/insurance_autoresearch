# Healthcare Claims Denial Prediction

Predicting insurance claim denial before submission using Karpathy's AutoResearch framework. An LLM agent autonomously iterates on the modeling pipeline overnight, keeping only changes that improve F2 score on the denied class.

---

## Problem

Can an autonomous ML agent predict which insurance claims will be denied before submission, allowing billing teams to fix errors upfront rather than chasing rejections after the fact?

**Metric:** F2 score on the denied class (weights recall 2x over precision)
**Data:** Kaggle synthetic healthcare claims dataset (1000 claims, 15 columns)
**Target variable:** Outcome — Denied or Partially Paid = 1, Paid = 0
**Current best F2:** 0.9015 (exp5: RF+ExtraTrees ensemble, auto-threshold, ratio features)

---

## Project Structure

    insurance_autoresearch/
    prepare.py        FROZEN   — data loading, 70/15/15 split, F2 evaluation, test set lock
    train.py          EDITABLE — only file the agent may modify
    run.py            FROZEN   — runs one experiment, logs result to results.tsv
    program.md                 — agent research agenda in plain English
    README.md                  — this file
    research_log.md            — dated weekly notes on agent behavior and decisions
    failure_log.md             — crashed experiments and lessons learned
    requirements.txt           — pinned Python dependencies
    .gitignore                 — excludes data/, results.tsv, performance.png, cache
    results.tsv       GENERATED — experiment log, one row per trial
    performance.png   GENERATED — F2 score trajectory plot over iterations
    data/             LOCAL ONLY — raw Kaggle CSV, never pushed to GitHub

Boundary principle: train.py is the only file the agent touches. prepare.py and run.py form the frozen evaluation harness. This separation makes every experiment comparable.

---

## How to Run

    pip install -r requirements.txt
    python3 run.py "baseline: logistic regression" --baseline

One command returns a single F2 score. Runtime is approximately 0.01 seconds per iteration.

---

## How the Agent Loop Works

1. Agent reads program.md for instructions and reads current train.py for context
2. Agent proposes one change to train.py
3. Agent runs python3 run.py "description of change"
4. F2 score is compared to current best
   - If improved — change is committed to git, new best recorded
   - If worse — train.py is reverted, agent tries something else
5. Repeat — approximately 80 to 100 experiments overnight

---

## Results Log

| # | Description | F2 Score | Recall | Decision |
|---|-------------|----------|--------|----------|
| 0 | baseline: logistic regression | 0.6133 | 0.6082 | baseline |
| 1 | Random Forest 300 trees balanced | 0.7256 | 0.7530 | kept |
| 2 | RF + month/ratio features + threshold 0.35 | 0.8879 | 0.9790 | kept |
| 3 | GradientBoosting + features + threshold | 0.8397 | 0.9070 | discarded |
| 4 | RF + auto-threshold CV-optimized F2 | 0.8955 | 0.9900 | kept |
| 5 | RF+ExtraTrees ensemble + richer features + auto-threshold | 0.9015 | 1.0000 | best |

---

## Status

**Week 3** — first agent loop complete. Best F2 improved from 0.613 to 0.9015, exceeding the success criterion of 0.70. Recall hit 1.0 on validation set. Ablation study planned for Week 4 to confirm stability.

See research_log.md for weekly notes and failure_log.md for experiments that did not work.
