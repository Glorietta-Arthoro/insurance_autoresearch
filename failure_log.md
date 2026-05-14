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

## Week 5 — May 2026

**All-positive ceiling — dominant failure mode**
With threshold locked at 0.5, predicting every val sample as positive gives recall=1.0 and precision=97/150=0.6467, yielding F2=0.901487. Nearly every experiment hit this ceiling exactly — the model simply assigned every sample a probability ≥0.5. This was the dominant failure mode across 40+ experiments.

**LightGBM/XGBoost with OHE — recall plateau**
All LightGBM and XGBoost variants (OHE on diagnosis code, scale_pos_weight from 1.5 to 4, frequency encoding) plateaued at recall=0.845 or fell to the all-positive ceiling. These models cannot properly handle the high-cardinality Diagnosis Code feature even with one-hot encoding. Tried 6 variants, all discarded.

**Feature engineering adds no signal**
Interaction features (diagnosis×insurance, diagnosis×procedure), smoothed target encoding for diagnosis and procedure codes, and additional ratio features all produced no improvement over base CatBoost. Near-zero feature correlations in this synthetic dataset mean engineered features carry no new information beyond what CatBoost's ordered statistics already capture.

**CatBoost Bernoulli subsampling — TN=2 but FN=1**
Exp60 (Bernoulli bootstrap, subsample=0.8) found TN=2 but introduced FN=1, giving F2=0.897 — worse than the target. The subsampling changes which training samples CatBoost sees per iteration, identifying a different set of val negatives but losing a val positive in the process. Tried subsample=0.9 and [1,5] variant — same net result.

**Ensemble averaging cancels TN signal**
Exp57 and exp58 averaged probabilities from [1,3]+[1,4]+[1,5] and [1,3]+[1,4]. Both resulted in F2=0.901487 (all positive). Root cause: [1,3] and [1,4] identify DIFFERENT val negatives as their single TN. Averaging their probabilities pushes both TN candidates back above 0.5, canceling both signals.

**Different CatBoost seeds and depths are worse**
Seed=0 (exp55) produced all-positive predictions. Depth=4 (exp49) added FN. Depth=7 (exp64) and depth=8 (exp52) both added FN=1 with no TN gain. Lossguide grow policy (exp59) dropped to F2=0.785. The depth=6 seed=42 configuration appears uniquely optimal for this dataset and threshold.
