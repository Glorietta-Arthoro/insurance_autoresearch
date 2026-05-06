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

## Week 4 — May 2026

**Exp 3 — LR + features discarded**
Adding month extraction and billing ratio features to logistic regression dropped F2 from 0.6328 to 0.5814. The ratio features add noise that logistic regression cannot handle. Tree models can use these features but linear models cannot. Reverted.

**Threshold ceiling — dominant failure mode**
Experiments 4, 5, and 7 all hit F2 = 0.9015 with recall = 1.0 despite completely different architectures. The threshold of 0.35 flags enough claims to push recall to 1.0 on the 150-row validation set regardless of model quality. This makes the current metric unable to distinguish between architectures. The agent cannot improve further without a metric change or threshold lock.

**Evaluation logging inconsistency**
Several Week 3 experiments were logged as keep in results.tsv despite being reverted in git. This created a misleading log where discarded experiments appeared as improvements. Fix: agent should log discard status explicitly when reverting via git checkout.

**Recall ceiling concern confirmed**
The ablation confirms that recall of 1.0 is driven entirely by threshold tuning on a small validation set, not by genuine model learning. This result will almost certainly not hold on the locked test set at Week 7.
