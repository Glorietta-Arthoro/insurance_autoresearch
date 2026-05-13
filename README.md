# Healthcare Claims Denial Prediction

Predicting insurance claim denial before submission using Karpathy's AutoResearch framework. An LLM agent autonomously iterates on the modeling pipeline overnight, keeping only changes that improve F2 score on the denied class.

---

## Problem

Can an autonomous ML agent predict which insurance claims will be denied before submission, allowing billing teams to fix errors upfront rather than chasing rejections after the fact?

**Metric:** F2 score on the denied class (weights recall 2x over precision)
**Data:** Kaggle synthetic healthcare claims dataset (1000 claims, 15 columns)
**Target variable:** Outcome — Denied or Partially Paid = 1, Paid = 0
**Current best F2:** 0.9015 (exp5: RF+ET ensemble, threshold locked at 0.5 from Week 5 onward)

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

## Weekly Status

| Week | Goal | Status |
|------|------|--------|
| 1 | Project charter, research question, workflow diagram, risk list, repo structure | Complete |
| 2 | Freeze evaluation pipeline, run reproducible baseline | Complete |
| 3 | First agent loop, 5 dry run experiments | Complete |
| 4 | Controlled ablation study, error taxonomy, failure memo | Complete |
| 5 | First real overnight autonomous block, threshold locked at 0.5 | In Progress |
| 6 | Ablation table, scope lock, final story | Pending |
| 7 | Confirmation runs, final report draft | Pending |
| 8 | Final presentation and retrospective | Pending |

---

## Results Log

| # | Description | F2 | Recall | Status |
|---|-------------|-----|--------|--------|
| 1 | baseline: logistic regression | 0.6133 | 0.6082 | baseline |
| 2 | exp1: random forest 300 trees balanced | 0.7256 | 0.7526 | keep |
| 3 | exp2: RF + month/ratio features + threshold=0.35 | 0.8879 | 0.9794 | keep |
| 4 | exp3: gradient boosting + features + threshold=0.35 | 0.8397 | 0.9072 | discard |
| 5 | exp4: RF + features + auto-threshold CV F2 | 0.8955 | 0.9897 | keep |
| 6 | exp5: RF+ET voting ensemble + richer features + auto-threshold | 0.9015 | 1.0000 | keep |
| 7 | exp6: HistGradientBoosting + richer features + auto-threshold | 0.8939 | 0.9897 | discard |
| 8 | exp7: RF+ET ensemble + log transforms + OHE + auto-threshold | 0.9015 | 1.0000 | discard |
| 9 | exp8: RF+ET ensemble + SMOTE + auto-threshold | 0.8895 | 0.9794 | discard |
| 10 | exp9: stacking RF+ET base LR meta + auto-threshold | 0.9015 | 1.0000 | discard |
| 11 | ablation: baseline logistic regression | 0.6328 | 0.6289 | baseline |
| 12 | ablation: random forest only | 0.7256 | 0.7526 | keep |
| 13 | ablation: LR + month/ratio features only | 0.5814 | 0.5670 | discard |
| 14 | ablation: LR + threshold 0.35 only | 0.9015 | 1.0000 | keep |
| 15 | ablation: RF + auto-threshold only | 0.9015 | 1.0000 | keep |
| 16 | ablation: RF + features only | 0.8301 | 0.8866 | keep |
| 17 | ablation: full best model | 0.9015 | 1.0000 | keep |

---

## Status

**Week 5 prep complete.** Threshold locked at 0.5, results.tsv cleaned, run.py updated with sequential experiment numbering. Ready for first overnight autonomous block.

See research_log.md for weekly notes and failure_log.md for experiments that did not work.
