# GaitSense AI — frozen internal protocol (Stage 2)

**Version 1.0. Frozen and SHA-256-hashed (docs/ai_protocol.sha256) before any model in this protocol was fitted and
before any test-partition outcome was computed.** As with the earlier protocols in this repository, the freeze time is
self-recorded, not independently timestamped. Changes after freezing are listed in `docs/ai_deviations.md`.

## 0. Context that limits what is new

- The same GaitRec cohort was used by the v1.x studies. The *data* are therefore not unseen by the author, although no
  model in this protocol has been fitted before, and the test partition defined here has never been used for model choice.
- Prior result (v1.2, shear report §4b): a ridge forecast of the next visit's loading difference did **not** beat
  carrying the last value forward (MAE 0.0303 vs 0.0315; interval touched zero). The prior expectation for point
  accuracy is therefore that complex models will not clearly beat persistence. The main new question is uncertainty.

## 1. Data and populations

- **Cohort:** the 1,667 patients of the v1.x frozen protocol (`src/gaitrec_rehab/cohort.py`): first first-stay session
  unilateral, normal shoes, full 365-day window. **Visits:** their first-stay (READMISSION = 0) normal-shoe sessions in
  which **both legs have ≥ 3 valid trials**. Visit time = days since the patient's first included visit.
- **Healthy reference:** healthy-control (HC) sessions in normal shoes, both legs ≥ 3 valid trials.
- **Partition (patient level, seed 20261024):** cohort patients stratified by injury class (A ankle, K knee, H hip,
  C calcaneus/heel) into 60% training, 20% calibration, 20% test. Healthy controls are split 60/20/20 the same way.
  Assignments are stored in `results/ai/partition.csv` and verified by tests.
- **Secondary temporal test set:** patients meeting every cohort rule except the 365-day window (the v1.3
  "temporal hold-out", already evaluated once in the detection study). Reported separately and labelled.
- **Prediction targets:** every visit k ≥ 2 of a patient, predicted from visits 1..k−1 only.

## 2. Features and targets

Trial features (`src/gaitsense_ai/features.py`): `peak_v` (max vertical force over stance / BW), `braking`,
`propulsive` (BW·s), `stance_s`; secondary curve features (peak1, peak2, valley, loading rate, vertical impulse,
first-half peak). Visit summaries per leg: mean, SD, n. Visit differences = affected − unaffected mean, with
SE = sqrt(SD_a²/n_a + SD_u²/n_u).

- **Primary target Y:** affected-minus-unaffected `peak_v` (BW).
- **Secondary targets:** affected-minus-unaffected `braking`, `propulsive`, `stance_s`; affected-leg absolute `peak_v`.

## 3. Models (all fitted on training patients only)

| ID | Model | Predictive distribution |
|---|---|---|
| M0a | Last value carried forward, same-day noise only | N(y_last, SE_last² + s̄²), s̄² = mean SE² of the history visits |
| M0b | Last value carried forward, corrected | N(y_last, SE_last² + s̄² + τ²); τ by method of moments on training consecutive pairs (v1.x estimator) |
| M1 | Ridge regression on history features | Gaussian; SD = RMS of calibration residuals within prior-visit group (1, 2, ≥3) |
| M2 | Linear mixed model: fixed injury class, age, sex, body mass, time (days/30); random intercept (+ random slope if it improves training AIC) | Exact conditional Gaussian given the history; residual variance = σ² + SE_ij² |
| M3 | Longitudinal Gaussian process: fixed effects as M2 + patient GP f_i(t) with kernel chosen among {C+Matérn½, C+Matérn³⁄₂, C+Matérn⁵⁄₂, C+RBF, C+Matérn³⁄₂+linear}; noise SE_ij² + σ_w² | Exact GP conditional given the history |
| M4 | Gradient-boosted trees (HistGradientBoosting) on M1 features | As M1 |

History features for M1/M4: last value, last SE, mean and SD of history values, last-minus-first, slope of the
history (if ≥ 2 visits, else 0 with indicator), number of prior visits, days since last visit (log1p), days since first
visit, injury class, age, sex, body mass. Nothing from the target visit enters any prediction (the target's own SE is
not used for prediction; it is used only by monitoring architecture D, after measurement).

Hyperparameters: M1 alpha and M4 depth/learning rate by 5-fold patient-grouped CV within training. M2/M3 variance
parameters by maximum marginal likelihood on training patients. M3 kernel chosen by mean negative log predictive
density (NLPD) on calibration targets. The test partition is used once, after all choices are written to
`results/ai/model_selection.json`.

## 4. Evaluation (test patients)

MAE, RMSE, empirical coverage of central 50/80/90/95% intervals, mean 95% width, 95% interval score, CRPS (Gaussian
closed form), NLPD, PIT histogram. Breakdowns: injury class, prior visits (1, 2, ≥3), days since last visit (≤7,
8–14, 15–28, >28). Uncertainty: patient-clustered bootstrap (2,000 resamples of test patients); paired differences.

## 5. Hypotheses

| | Status | Outcome | Comparator | Success criterion |
|---|---|---|---|---|
| **H1** | Confirmatory | Mean CRPS of M3, primary target, test | The best of M0b, M1, M2 by calibration-set CRPS (chosen before test) | Paired difference M3 − comparator < 0 with 95% bootstrap CI excluding 0 |
| **H2** | Confirmatory | 95% interval coverage, primary target, test | M0a (same-day noise only) | M3 coverage within [0.93, 0.97] **and** absolute coverage error of M3 smaller than M0a with 95% bootstrap CI of the difference excluding 0 |
| H3 | Exploratory | Agreement of monitoring architectures A–D on test consecutive pairs | Pairwise | "Material disagreement" if any pairwise Cohen's κ < 0.60 at matched alert rate |
| H4 | Exploratory | Alert flip rate under perturbations | Across architectures | Descriptive; differences reported with bootstrap CIs |
| H5 | Exploratory | MAE, CRPS, coverage, agreement by injury class | Across classes | Descriptive |
| E1 | Exploratory | Is M3's predictive SD informative? Spearman(SD, |error|); coverage by SD quintile | — | Descriptive |

Secondary H1 statements (not success criteria): M3 vs M0a/M0b in MAE; all models on secondary targets; temporal test.

## 6. Monitoring architectures (Stage 6), common target Y between consecutive visits t−1, t

- **A — Raw asymmetry.** Limb symmetry index LSI = 100·affected/unaffected `peak_v` (unaffected ≥ 0.5 BW by the
  validity rule, so no near-zero denominator; guarded anyway). Native alert: |ΔLSI| ≥ 10 points (a round number chosen
  a priori, not a validated threshold). Score: ΔY. Which limb moved is retained (Δ affected, Δ unaffected).
- **B — Population reference.** Healthy reference from training HC sessions: signed left-minus-right `peak_v`
  difference (mean μ_H, SD σ_H). z_t = (Y_t − μ_H)/σ_H. Native alert: the visit pair crosses the healthy 95% band
  (|z| ≤ 1.96) in either direction. Score: |z_t| − |z_{t−1}|. Continuous extras reported separately: curve-level
  Mahalanobis distance of the affected leg's normalised V+AP curve (PCA fitted on training HC, components to 95%
  variance, Ledoit–Wolf covariance); injury-matched percentile among training patients of the same class and week.
- **C — Statistical change limit.** z = ΔY / sqrt(SE_t² + SE_{t−1}² + τ²), τ from training patients. Native alert
  |z| > 1.96. The v1.x original (τ = 0) is reported alongside. τ contains unidentified between-visit variation
  (measurement and genuine change); it is **not** sensor error.
- **D — Predictive.** M3 (kernel as selected) predicts visit t from visits < t; r = (Y_t − μ̂)/sqrt(σ̂²_latent + σ_w² + SE_t²).
  Native alert |r| > 1.96. A deviation is not deterioration.
- **Matched-rate comparison:** for each architecture, a two-sided threshold on its score set on calibration-patient
  pairs so that 10% of pairs alert; applied unchanged to test pairs.
- **Metrics:** pairwise agreement, Cohen's κ, positive agreement, joint alert rate, directional conflict (A, C, D:
  increase vs decrease; B is toward/away from healthy and is excluded from directional conflict), alert rates by injury
  class, visit number, days between visits; threshold proximity (share of disagreements where a score is within 10% of
  its threshold). Patient-clustered bootstrap CIs.

## 7. Perturbations and injected changes (Stages 7–8), test patients, later visit of each consecutive pair

Perturbations of the affected-leg raw trials (scales from training data: s_w = median within-visit SD of affected
`peak_v`; cv_w = median within-visit coefficient of variation of `peak_v`):
force offset ±{0.5, 1, 2}·s_w (BW·body weight in N added to vertical force); amplitude scaling 1 ± {0.5, 1, 2}·cv_w;
stance-boundary trimming of {1, 2, 4} samples at each end (4–16 ms; extension is impossible because samples outside
the 25 N stance are not in the files); trial removal (drop 1, drop 2, keep 3 — only when ≥ 3 remain); time
normalisation (features from 101-point linear, 101-point cubic and 51-point linear resampled curves instead of raw);
missing observation (drop visit t−1: C compares t with t−2, D predicts t without t−1).
Outcomes: alert flip rate per architecture (native rules), mean |Δscore|; for D also change in predictive SD and in
95% coverage. These perturbations are mathematical; they are not claimed to reproduce strap displacement, footwear or
speed changes.

Injected changes: vertical force scaled by (1+δ), δ ∈ {−0.15, −0.10, −0.05, −0.02, +0.02, +0.05, +0.10, +0.15};
positive AP force scaled by (1−δ), δ ∈ {0.05, 0.10, 0.20, 0.30} (propulsive); negative AP force scaled by (1+δ),
same δ (braking); first-peak-local reduction (Gaussian window at peak1, SD 10% of stance), δ ∈ {0.05, 0.10, 0.15};
bilateral vertical reduction (both legs), δ ∈ {0.05, 0.10}. Reported: detection rate in the injected direction,
baseline alert rate on unmodified pairs, by magnitude, class and architecture; detection vs baseline alert rate as z
varies (1.645, 1.96, 2.576, 3.0). Baseline alerts are not called false positives (an unmodified pair is not a stable
patient); none of this is clinical sensitivity, specificity or ROC.

## 8. Decisions fixed in advance

- If M3 does not beat persistence in MAE on test, the prototype labels its point forecast "not more accurate than the
  last value" (v1.x design input D6) and shows intervals only with their measured coverage.
- Negative results are reported in full. No model, threshold or kernel is changed after the test partition is opened;
  any later analysis is labelled post hoc.
