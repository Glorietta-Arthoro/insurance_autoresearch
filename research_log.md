# Research Log

## Week 2 — April 2026

**Goal:** Establish a fixed evaluation pipeline and run a reproducible baseline.

**What I did:**
- Rewrote prepare.py for the Kaggle claims dataset
- Built binary target from Outcome column (Denied + Partially Paid = 1, Paid = 0)
- Implemented deterministic 70/15/15 train/val/test split with seed 42
- Locked test set, agent cannot access it during search phase
- Rewrote train.py with logistic regression baseline
- Rewrote run.py to evaluate F2 score on denied class
- Ran baseline twice, confirmed identical results both times

**Baseline result:**
- val F2: 0.613306
- val Recall: 0.608247
- val Precision: 0.634409
- Runtime: 0.01 seconds per iteration

**What I observed:**
- Denial rate is 64% across all splits, well balanced, no severe class imbalance
- Dataset has 1000 rows, 8 usable features after dropping ID and leakage columns
- Dropped Claim Status, Reason Code, and AR Status as post-submission leakage
- Runtime is very fast at 0.01s, agent can run hundreds of experiments overnight

**Next steps:**
- Write program.md research agenda
- Run 3 to 5 dry run experiments manually
- Launch first agent loop

## Week 1 — April 2026

**Goal:** Set up project structure and complete Week 1 deliverables.

**What I did:**
- Defined research question and success criterion
- Wrote project charter
- Built week1_deliverables.html with workflow diagram, risk list, and repo structure
- Set up GitHub repo and README

**What I observed:**
- No experiments run yet, baseline pending Week 2

**Next steps:**
- Download Kaggle claims dataset
- Write prepare.py to load data and compute F2 score
- Run logistic regression baseline
