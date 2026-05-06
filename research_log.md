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

## Week 3 — April 2026

**Goal:** Run first agent loop, complete 3 to 5 dry run experiments, document findings.

**What I did:**
- Launched Claude Code inside Cursor on the claims project
- Let the agent run 5 experiments modifying only train.py
- Monitored F2 score after each experiment via results.tsv

**Experiment Results:**
- Baseline: Logistic Regression — F2 0.6133
- Exp 1: Random Forest 300 trees balanced — F2 0.7256 — KEPT
- Exp 2: RF + month/ratio features + threshold 0.35 — F2 0.8879 — KEPT
- Exp 3: GradientBoosting + features + threshold — F2 0.8397 — DISCARDED
- Exp 4: RF + auto-threshold CV-optimized F2 — F2 0.8955 — KEPT
- Exp 5: RF+ExtraTrees voting ensemble + richer features + auto-threshold — F2 0.9015 — BEST

**What the agent did well:**
- Immediately identified that switching from logistic regression to Random Forest was the biggest lever
- Discovered that lowering the decision threshold significantly improved recall on denied claims
- Found that adding month and billing ratio features added meaningful signal
- CV-optimized threshold tuning in exp4 and exp5 was a smart move that paid off

**What the agent did badly:**
- GradientBoosting in exp3 was worse than Random Forest, wasting one experiment slot
- Additional complexity beyond the RF+ET ensemble produced no improvement, suggesting the agent explored diminishing returns too early
- Baseline was logged three times due to duplicate runs, creating noise in results.tsv

**Current best:**
- F2: 0.9015, Recall: 1.000, Precision: 0.647
- Model: RF+ExtraTrees soft-voting ensemble with month/quarter extraction, billed-paid-allowed ratio features, and CV-optimized decision threshold

**Next steps:**
- Run ablation study to confirm which features and changes are actually driving the improvement
- Lock scope and update program.md for Week 4 controlled experiments
- Investigate whether recall of 1.0 is genuinely good or a sign of overfitting to validation set

## Week 4 — May 2026

**Goal:** Run controlled ablation experiments, build experiment result matrix, metric plot, error taxonomy, and failure analysis memo.

**What I did:**
- Ran 7 controlled ablation experiments isolating one variable at a time
- Built experiment result matrix mapping each experiment to its outcome
- Generated metric over time plot showing F2 and recall across all runs
- Wrote error taxonomy categorizing all failure types
- Wrote one page failure analysis memo identifying the dominant failure mode

**Ablation Results:**
- Exp 1: Baseline LR — F2 0.6328
- Exp 2: RF only — F2 0.7256
- Exp 3: LR + features only — F2 0.5814 — DISCARDED (features hurt LR)
- Exp 4: LR + threshold 0.35 only — F2 0.9015 — KEY FINDING
- Exp 5: RF + auto-threshold only — F2 0.9015
- Exp 6: RF + features only — F2 0.8301
- Exp 7: Full model — F2 0.9015

**Key finding:**
Threshold tuning alone is the dominant driver of improvement. A plain logistic regression with threshold 0.35 hits exactly the same F2 as the full RF+ET stacking ensemble. All additional model complexity adds zero F2 benefit. The metric has effectively saturated on this 150-row validation set.

**What I observed:**
- Recall of 1.0 is likely an artifact of low threshold on a small validation set, not genuine generalization
- Features hurt logistic regression but help Random Forest
- The current evaluation setup cannot distinguish between model architectures once threshold tuning is active
- Need to lock threshold or change metric before Week 5 overnight run

**Next steps:**
- Decide whether to lock threshold at 0.5 or change metric to precision at fixed recall
- Update program.md with new constraints before Week 5
- Run first real overnight autonomous block
