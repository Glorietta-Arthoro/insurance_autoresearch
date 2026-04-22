# Healthcare Claims Denial Prediction

Predicting insurance claim denial before submission using Karpathy's AutoResearch framework. An LLM agent autonomously iterates on the modeling pipeline overnight, keeping only changes that improve F2 score on the denied class.

---

## Problem

Can an autonomous ML agent predict which insurance claims will be denied before submission, allowing billing teams to fix errors upfront rather than chasing rejections after the fact?

**Metric:** F2 score on the denied class (weights recall 2x over precision)  
**Data:** Kaggle synthetic healthcare claims dataset  
**Target variable:** denial_status — Denied vs Paid (binary classification)  
**Current best F2:** not yet established — baseline pending Week 2

---

## Project Structure
```
insurance_autoresearch/
├── prepare.py        FROZEN   — data loading, 70/15/15 split, F2 evaluation, test set lock
├── train.py          EDITABLE — only file the agent may modify
├── run.py            FROZEN   — runs one experiment, logs result to results.tsv
├── program.md                 — agent research agenda in plain English
├── README.md                  — this file
├── research_log.md            — dated weekly notes on agent behavior and decisions
├── failure_log.md             — crashed experiments and lessons learned
├── requirements.txt           — pinned Python dependencies
├── .gitignore                 — excludes data/, results.tsv, performance.png, cache
├── results.tsv       GENERATED — experiment log, one row per trial
├── performance.png   GENERATED — F2 score trajectory plot over iterations
└── data/             LOCAL ONLY — raw Kaggle claims CSV, never pushed to GitHub
```

**Boundary principle:** `train.py` is the only file the agent touches. `prepare.py` and `run.py` form the frozen evaluation harness. This separation makes every experiment comparable.

---

## How the Agent Loop Works

1. Agent reads `program.md` for instructions and reads current `train.py` for context
2. Agent proposes one change to `train.py`
3. Agent runs `python run.py "description of change"`
4. F2 score is compared to current best
   - If improved → change is committed to git, new best recorded
   - If worse → `train.py` is reverted, agent tries something else
5. Repeat — approximately 80 to 100 experiments overnight

---

## Setup
```
pip install -r requirements.txt
python prepare.py
python run.py "baseline: logistic regression"
```
---

## Results Log

| # | Description | F2 Score | Decision |
|---|-------------|----------|----------|
| 0 | baseline: logistic regression | TBD | baseline |

---

## Status

**Week 1** — project charter, research question, workflow diagram, risk list, and repository structure complete. Data pipeline and baseline pending Week 2.

See `research_log.md` for weekly notes and `failure_log.md` for experiments that did not work.
