# Protocol: direct AI detection of longitudinal braking and push-off changes during musculoskeletal rehabilitation

**Status:** frozen internal protocol, written and hashed before any analysis of this question and before any gait data from the temporal hold-out patients were analysed. It is not an independently timestamped pre-registration. This is the final planned experiment of the project. No further architectures or questions will be added in response to hold-out results.

## Background

In the shear-force study (`docs/shear_report.md`), models reconstructed each visit's braking and push-off separately. The visit-to-visit change was then obtained by subtraction. The reconstructions got the *direction* of real change right (89%), but compressed its *size* to about a third. At the 5% crossing limit, they flagged only 30% of real changes.

Two explanations are possible, and they predict different things:

- **(a) Attenuation.** Separate per-visit reconstruction shrinks toward typical values. A model that sees both visits, and is trained directly on change, should then do better.
- **(b) Missing information.** The vertical signal does not contain the information. Then no model trained on it will do much better.

This study tests which explanation holds.

## Questions

- **Primary.** Can AI detect force-plate-defined changes in push-off between visits, using vertical-force curves alone, more sensitively than conventional vertical-force methods and than the earlier reconstruction approach?
- **Secondary.** Does a paired-visit model, trained directly on change, estimate the *size* of change in braking and push-off better than reconstructing each visit separately?

The reference is biomechanical (the force-plate anterior–posterior force). The study does not address clinical deterioration, recovery or return to work.

## Data and eligibility

**Development set:** the 1,667-patient cohort of the earlier reports. Pairs are consecutive first-stay, normal-shoe visits with at least 2 valid steps per leg.

**Temporal hold-out:** the 92 patients who meet every cohort rule except the 365-day follow-up window. Their first assessment falls between September 2017 and April 2018. Of these, 81 have two or more visits, giving 271 pairs before step-count filtering.

- These patients were excluded from every earlier *analysis*.
- **Disclosed exposure.** Their first-stay visit summaries are included in `data/derived/first_stay_session_differences.csv`. That file covers all patients and was built during the first extraction. Its only use is a pooled sanity test (`test_affected_side_coding_points_the_right_way`) checking that the injured leg is weaker on average across all patients. No visit-to-visit change, no anterior–posterior change label and no model output for these patients was ever examined. They did not inform any modelling decision.
- Their gait data are not analysed further until the final evaluation.
- Before freezing, only their counts, dates, age, sex and injury group were inspected.
- `verification/check_holdout.py` confirms that none appear in any earlier result file. It also reports their age, sex and injury group against the development set, with no gait variables.
- **Expected size:** about 13 true push-off changes. The hold-out can therefore only *confirm or contradict* the cross-validated result; it cannot establish sensitivity precisely.

## Reference labels

τ for push-off and braking is estimated on the development set, using the existing estimator. A pair is a **true change** if the force-plate change in push-off is beyond 1.96 × √(SE₁² + SE₂² + τ²), in either direction.

**Known label noise:** about 5% of stable pairs cross by chance, and the labels inherit that. Braking labels are defined the same way and are secondary.

## Inputs

Inputs come from both visits:

- each leg's mean 101-point vertical curve (affected and unaffected, first and second visit);
- per-leg step counts and stance times;
- days between visits;
- body mass and injury group.

No anterior–posterior (AP) data enter any input.

## Methods compared

Each method gives every pair a score for "push-off changed". These are compared at a common operating point.

| Method | Score |
|---|---|
| V-thr: conventional vertical tracker | |z| of the change in peak-loading difference (limit with its own τ) |
| R-gb: reconstruction, as in the earlier study | |z| of the change in reconstructed push-off difference (gradient boosting) |
| R-cnn: reconstruction with the CNN | as R-gb, using the NumPy CNN |
| P-lin: paired logistic regression | probability of change, from paired-visit features (L2-penalised) |
| P-gb: paired gradient-boosting classifier | probability of change, from the same features plus the curve differences |
| P-cnn: paired-visit CNN | input: four channels (affected and unaffected, at each of the two visits) plus covariates; outputs the change in braking and push-off; score = predicted push-off change divided by the predicted limit |

For the secondary question, the regression versions estimate the change directly: P-lin uses ridge regression and P-gb a gradient-boosting regressor, and P-cnn's own output is used.

**Paired features** are the hand-crafted vertical summaries from the earlier study, for each leg and visit, plus their changes and the difference curves.

## Development (cross-validation, development set only)

- **Split:** 5-fold patient-level cross-validation (seed 0). τ, scaling and all fitting happen inside each training fold.
- **Tuning:** hyperparameters are fixed in advance at the values used in the shear study. P-cnn uses the same layer sizes, takes 4 input channels, and stops early on a patient-level 20% inner validation split.
- **Operating point:** for each method, a score threshold is chosen in the out-of-fold development scores so that 5% of *non-change* pairs are flagged.
- **Selecting the direct model:** the P-model with the highest out-of-fold sensitivity at that threshold is "the detector". It is chosen before the hold-out is opened.

## Final evaluation (temporal hold-out, once)

The models are refitted on the whole development set. The development thresholds are applied unchanged, and each method is evaluated on the hold-out pairs.

## Metrics

**Primary:** sensitivity for true push-off changes at the development-fixed 5% false-positive threshold.

- Reported for cross-validation and for the hold-out.
- 95% intervals come from 500 patient bootstrap resamples.
- The primary contrasts are the detector against V-thr and the detector against R-gb.

**Secondary:**

- specificity, precision, ROC AUC and PR-AUC;
- for the change estimates, CCC, MAE and direction accuracy;
- **calibration slope**, from regressing the true change on the predicted change. A slope above 1 means attenuation.
- results by injury group, and in the hard subset (true change, but peak loading stable).

## Interpretation, stated in advance

No numerical success threshold is set, because no clinical requirement for this measurement exists yet. The interpretation is fixed now:

- **Supports explanation (a):** in cross-validation, the detector's sensitivity exceeds both V-thr and R-gb, with the 95% bootstrap interval of each difference above zero. The paired model's calibration slope is closer to 1 than R-gb's.
- **Supports explanation (b):** sensitivity and calibration do not improve meaningfully. Under (b), the next step is a different sensor, not a different model.
- **The hold-out** is reported as consistent or inconsistent with the cross-validated result. With about 13 events it is not used on its own to claim a sensitivity.

## Failure-case analysis (development set, out-of-fold)

Missed true changes are described by:

- injury group;
- size and direction of change;
- step count;
- stance time;
- whether peak loading also changed.

This includes why knee patients are reconstructed worst. The analysis is descriptive, and results feed only into the Discussion.

## Limits

- The reference is the force-plate AP force. There are no clinical labels.
- The hold-out is small, comes from one centre and covers a later period. Patients enrolled near the end of the data may differ systematically from earlier ones.
- The inputs are force-plate vertical signals, not insoles.
