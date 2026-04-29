# Failure Log

## Week 1

No experiments run yet. Log will be updated starting Week 2 when the agent loop begins.

## Week 3 — April 2026

**Exp 3 — GradientBoosting discarded**
Tried GradientBoosting with the same feature engineering as exp2. F2 dropped from 0.8879 to 0.8397. Random Forest consistently outperformed GradientBoosting on this dataset. Reverted via git.

**Baseline logged three times**
The baseline was run multiple times across sessions, creating duplicate rows in results.tsv. Not a model failure but a logging issue. Fix: only run baseline once per session going forward.

**Recall of 1.0 flagged as potential concern**
Exp5 hit recall of 1.000 on the validation set which means the model flagged every single denied claim correctly. This could be genuine or could indicate the model is overfitting to the validation set. Need ablation study in Week 4 to confirm the result is stable.

**Additional complexity produced no improvement**
Experiments beyond exp5 tried log transforms, one-hot encoding, and stacking but none improved on 0.9015. The RF+ET ensemble with auto-threshold appears to be a local ceiling on this dataset. Suggests the 1000-row dataset may be limiting further gains.
