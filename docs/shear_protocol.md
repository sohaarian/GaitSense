# Protocol: AI-reconstructed shear forces for low-cost wearable monitoring of rehabilitation in injured workers

**Status:** this is a frozen internal protocol, version 2. Version 1 was hashed at 2026-10-24T20:03:22Z (SHA-256 62b96535…a1a6ce). Version 2 was amended after outside critique and before any per-step data were loaded or analysed. It is not an independently timestamped pre-registration.
**Data:** GaitRec (Horsak et al., *Sci Data* 2020), from a rehabilitation centre of the Austrian Workers' Compensation Board (AUVA). The study recruits no new participants.

## Version 2 amendments (made before any data were analysed)

1. **New primary question.** Can AI reconstruct braking and push-off from vertical force while preserving genuine within-patient changes that vertical-only monitoring would miss? The primary analysis now uses real, force-plate-measured visit-to-visit changes. Planted declines become a secondary sensitivity analysis.
2. **Hard subset, defined here before any results.** These are the test-set visit pairs in which:
   - the true push-off change is beyond its corrected limit, 1.96 × √(SE₁² + SE₂² + τ²), with τ taken from validation patients; and
   - the change in peak vertical loading is within its own corrected limit.

   Performance on this subset is reported as a co-primary result.
3. **New baseline.** M-persist, a patient-specific no-change prediction: every later visit's predicted value equals the patient's first-visit value, so the predicted change is always 0.
4. **Information test first.** Before models are compared, the study asks whether the information exists at all. It measures the share of variance in true push-off change explained by vertical-curve change, using cross-validated ridge regression on training patients.
5. **Terminology.** "False-alert rate" is replaced by **crossing rate**. Unchanged real visits may contain real change, so a crossing rate is a statistical property, not a clinical false-alarm rate.
6. **Claim boundary.** Positive results establish *computational feasibility under simulated sensing*. They do not establish that the method works on real insoles, or that it measures clinical outcomes such as pain, function, return to work or MMI.

## Purpose

Low-cost pressure insoles measure only vertical foot force. Braking and push-off forces carry information about recovery, but measuring them currently needs laboratory force plates. This study asks two things:

- Can an AI model reconstruct braking and push-off from vertical force well enough to preserve visit-to-visit change detection?
- Which failures of current measurement approaches set requirements that a future workers' compensation rehabilitation tracker must meet?

Each finding becomes a numbered **design input**, a documented requirement defined before any device is built. Human testing is future work.

## Data and cohort

- **Cohort:** the cohort and first-stay visits from the completed report (`docs/technical_report.md`). This is 1,441 patients with two or more visits, giving 4,203 consecutive visit pairs.
- **Primary population:** working-age patients, 18–65 at the first visit. All ages are reported as a sensitivity analysis.
- **Healthy controls:** self-selected-speed sessions only (GaitRec `SPEED` = 2). Controls are used only in Study 4.
- **Per-step inputs:** vertical (V) and anterior–posterior (AP) raw force, in newtons at 250 Hz. Step validity and impulse definitions are unchanged from `src/gaitrec_rehab/features.py`.
- **Vertical curve:** each step's vertical force, divided by body weight and resampled to 101 points across stance.

## Study 1 (primary): the virtual shear-force sensor

- **Input:**
  - the 101-point vertical curve;
  - stance duration;
  - body mass;
  - injury group;
  - side (affected or unaffected).
- **Output:** braking impulse and push-off impulse for that step, in body-weight-seconds.
- **Models:**
  - M-lin: ridge regression on hand-crafted vertical features (peaks, valley, loading and unloading rates, timing, stance time). This is the baseline.
  - M-gb: gradient boosting on the same features plus the curve.
  - M-cnn: a 1D convolutional network on the full curve. Where no deep-learning library is available, an MLP on the curve is used instead, and this is recorded as a deviation.
- **Splits:** patients are split 60/20/20 into train, validation and test sets with seed 0. No patient appears in more than one set. All tuning uses the validation set only.
- **Per-step accuracy on the test set:** mean absolute error, R², and Bland–Altman bias with 95% limits of agreement.

## Study 2A (primary): agreement with real visit-to-visit change

Test-set consecutive visit pairs. Each model (M-persist, M-lin, M-gb, M-cnn) and condition (c), vertical only, is compared with the force-plate change. The measures are:

- **Change agreement:** Lin's concordance correlation coefficient and Bland–Altman limits between predicted and true change in braking and push-off.
- **Direction of change:** among pairs whose true change is beyond its corrected limit, the share for which the predicted change has the right sign (improvement vs decline).
- **Detection:** sensitivity and specificity for flagging a beyond-limit true change.
- **Hard subset (co-primary):** all of the above, restricted to the subset defined in amendment 2.
- **Robustness:** results by injury group, sex, age group and visit number.

Intervals come from 500 patient bootstrap resamples.

## Study 2B (secondary): crossing rates and planted declines

The between-visit-corrected tracker from the report is run on test-set patients under three conditions:

- **(a)** true force-plate braking and push-off;
- **(b)** braking and push-off reconstructed by the AI;
- **(c)** vertical only, using peak loading with no reconstruction.

τ is estimated within each condition from separate patients (validation-set patients), then applied to test-set patients.

Measures, for each condition:

1. Crossing rate: share of consecutive-visit changes beyond the corrected limit with no change added. Target: 4–6%.
2. Detection rate for push-off declines planted in the latest visit, of 0.002, 0.004 and 0.006 body-weight-seconds (about 0.75, 1.5 and 2.2 × τ). Each decline is applied to the force-plate push-off, and the reconstruction is recomputed from a vertical curve scaled to carry the same relative change. The scaling method is described in the analysis code before it is run.
3. τ under (b) compared with τ under (a).

## Study 3 (supporting): wearable degradation

Study 2, condition (b), is repeated with the vertical input degraded:

- **Sampling rate:** 100 Hz and 25 Hz, applied by resampling the raw signal before time-normalisation.
- **Insole-scale error:** a per-session gain error with SD 5% plus white noise of 2% of body weight.
- **Steps per visit:** 3, 5 or 10 steps, sampled at random from the steps available. Visits with fewer steps are dropped.

**Output:** the least demanding combination at which the AI-assisted tracker keeps a 4–6% crossing rate and retains at least 80% of condition (a)'s detection rate at 0.004 body-weight-seconds.

## Study 4 (supporting): two further failure modes, from data already extracted

- **4a. Symmetry-only tracking.** This test counts how often the affected-minus-unaffected difference moves beyond its limit mainly because the *unaffected* leg changed, defined as the unaffected leg's change being at least half of the total. Each leg is also referenced to healthy-control norms.
- **4b. Persistence ceiling.** Next-visit loading difference is forecast by (i) carrying forward the last value and (ii) ridge regression on the patient's history. Mean absolute error is reported for both. A forecast is useful only if it beats persistence with a 95% interval that excludes zero.

## Pre-stated success criteria

The AI is judged useful only if, in held-out patients, all of these hold:

1. **Primary.** Its predicted push-off changes agree with true changes better than M-persist and condition (c): the concordance correlation coefficient difference has a 95% interval above zero.
2. **Co-primary.** On the hard subset, its direction-of-change accuracy has a 95% interval above 50%.
3. Its advantage over M-lin in concordance correlation has a 95% interval that excludes zero. If not, the conclusion is that simple models capture whatever information exists.
4. **Secondary.** Its crossing rate stays between 4% and 6%.

If the information test (amendment 4) finds little explainable variance, that is the main finding: vertical-only sensing cannot replace force plates for push-off monitoring.

Results are reported whatever they show.

## Failure register

Each study produces a requirement. Each row is filled in with the result, whether or not the result supports the requirement.

| # | Failure in current practice | Evidence | Design input |
|---|---|---|---|
| D1 | Same-day noise limits flag far more changes than intended | Report, Section 3.3 (28.1% vs 5%) | Alerts use between-visit-calibrated limits |
| D2 | Single-visit declines rarely repeat | Report, Section 3.4 | Decline alerts need confirmation across visits |
| D3 | Insoles lose braking and push-off | Studies 1–2A (real changes, hard subset) | Whether a vertical-only device may report braking and push-off change, and at what agreement level |
| D4 | Insole sampling, error and step count degrade tracking | Study 3 | Minimum sampling rate, accuracy and steps per visit |
| D5 | Symmetry hides changes in the uninjured leg | Study 4a | Each limb referenced to healthy norms |
| D6 | Forecasts may only reflect persistence | Study 4b | Forecasts shown only if they beat persistence |
| D7 | Outputs can be misread in a claim file | Workers' compensation review | Measurement-only wording: no MMI, effort, credibility or work-capacity language |

## Reporting

- Same-day noise, visit-to-visit variation and model error are reported separately.
- Results are broken down by injury group, sex and age group.
- An independent re-implementation checks τ and the crossing rate under conditions (a) and (b).
- Any change to this protocol is listed in `docs/deviations.md`.

## Limitations

- The inputs are force-plate vertical signals, not real insoles.
- The reference standard is the force plate, not a clinical outcome.
- The data come from one Austrian centre (2007–2018) and cover lower-limb injuries only.
- Walking speed was not recorded for patients.
