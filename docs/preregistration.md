# Analysis plan (pre-registration)

**Study:** Early gait biomechanics and subsequent rehabilitation stays after lower-extremity injury
**Author:** Kasra · **Frozen:** 2026-10-22, before any analysis relating predictors to the outcome
**Integrity record:** the SHA-256 hash of this file is stored in `docs/preregistration.sha256` and was
recorded before the outcome analysis was run. The same file is to be posted on OSF.

## Status at freezing

Done before this plan was frozen:
- the cohort audit (event counts only);
- download verification;
- trial-level feature extraction;
- validation of the extracted features against the metadata;
- a check of the affected-side coding using first-session feature averages;
- one monitoring page for one patient chosen by ID.

Not done: no predictor has been related to the outcome. Exploratory simulations used synthetic data only.

## 1. Data

GaitRec (Horsak et al., 2020), files verified against figshare MD5 checksums. Raw forces are in newtons, sampled at 250 Hz.

## 2. Cohort

**Inclusion** (all at the patient's first assessment):
- a session in the first recorded stay (READMISSION = 0); the earliest such session is the **index session**;
- AFFECTED_SIDE 0 or 1 (one leg);
- SHOD_CONDITION = 1 (normal shoes);
- index date at least 365 days before the last patient session (2018-04-10).

**Expected size:** 1,667 patients, 84 events. Any difference from these numbers will be reported.

## 3. Outcome

The first session with READMISSION > 0, within 365 days of the index date. This is a later-stay gait assessment, not an admission.

## 4. Predictors (all from the index session)

**Patient characteristics:** age, sex, body mass, injury group (A, C, H, K; three indicator terms with K as reference).

**Gait features.** For each valid trial, computed from the raw files:
- peak vertical force in the first half of stance;
- braking impulse (area under the negative anterior–posterior force);
- propulsive impulse (area under the positive anterior–posterior force).

Each feature is divided by body weight. A trial is valid if it has at least 10 samples, no missing samples, and a vertical peak above 0.5 body weight. Session means over valid trials are taken per leg. The feature is the affected-leg mean minus the unaffected-leg mean. At least 3 valid trials per leg are required.

**Reconstruction error (M1).** This reimplements the approach of Chakraborty et al. (2022) with one deviation: a dense autoencoder (scikit-learn MLPRegressor, hidden layers 64-16-64, ReLU, Adam, early stopping, seed 0) is used because PyTorch is unavailable.
- **Input:** the session-mean vertical and anterior–posterior curves of one leg, each time-normalised to 101 points and divided by body weight (202 values).
- **Training data:** healthy-control legs (both legs of every session) in the training fold. Healthy controls are split into folds by subject, alongside the patients.
- **Score:** the mean squared reconstruction error of the patient's affected-leg curves.

## 5. Models

| Model | Predictors | Parameters |
|---|---|---|
| M0 | Patient characteristics | 6 |
| M1 | M0 + reconstruction error | 7 |
| M2 | M0 + three gait features | 9 |
| M2+ (exploratory) | M0 + reconstruction error + three gait features | 10 |

**Fitting:**
- L2-penalised logistic regression.
- Continuous predictors are standardised using training-fold statistics.
- The penalty is chosen by inner 5-fold patient-level cross-validation over 10 log-spaced values of C from 0.001 to 100, minimising log loss.

## 6. Primary analysis

**Comparison:** M2 against M0.

**Point estimate:**
- The cross-validation is 5-fold, stratified and grouped by patient, repeated 10 times with seeds 0–9.
- In each repeat, out-of-fold predictions are pooled and AUROC is computed for each model.
- The estimate is the mean over repeats of the paired difference, AUROC(M2) − AUROC(M0).

**Interval:**
- 200 bootstrap resamples of patients (seed 2026).
- In each resample the complete pipeline is rerun: one 5-fold cross-validation, including penalty selection.
- The 95% interval is the 2.5th to 97.5th percentiles of the bootstrap differences. Resampled duplicates of a patient are kept in the same fold.

**Also reported** for each model, from the repeated cross-validation:
- AUROC;
- area under the precision–recall curve;
- Brier score and its paired difference;
- calibration intercept and slope;
- positive predictive value and false alerts per true alert at a **10% alert rate** (the top decile of predicted risk).

**Interpretation:**
- A gain is claimed only if the interval excludes zero and the calibration slope of M2 is between 0.7 and 1.3.
- A minimum clinically meaningful AUROC gain of 0.05 is pre-specified. An interval lying entirely below 0.05 is reported as evidence against a gain of that size.

## 7. Secondary analyses

**S1 (head-to-head):** M2 against M1, using the same measures. The interval comes from 200 bootstrap resamples of the out-of-fold predictions of one repeat (no refitting), and is labelled as such.

**S2 (later returns):** a day-90 landmark analysis.
- **Risk set:** cohort patients with no outcome session within 90 days.
- **Outcome:** an outcome session on days 91 to 365.
- **Predictors:** index-session predictors only.
- **Comparison:** M2 against M0, with the same cross-validation and an interval from 200 resamples of out-of-fold predictions.

**S3 (injury groups):** AUROC of the pooled models within each injury group, with bootstrap intervals. Descriptive only.

**S4 (insoles):** the primary comparison after excluding patients with ORTHOPEDIC_INSOLE = 1 or missing at the index session.

## 8. Exploratory analyses

**E1 (temporal hold-out):**
- Models are trained on patients whose index date falls before 2014-01-01 and tested on later patients.
- The penalty is selected within the training period.
- Reported: AUROC of M0 and M2 and their difference, with a bootstrap interval over test patients.

**E2 (combined model):** M2+ against M1.

**E3 (time to event):** a discrete-time logistic model in 30-day intervals, censored at 2018-04-10, using all first later-stay sessions. The C-index is reported for M0 and M2.

**E4 (missed-event bias analysis):**
- **Recording sensitivity:** s ∈ {0.9, 0.8, 0.6}, so round(84(1/s − 1)) ∈ {9, 21, 56} events are added to event-free patients.
- **Random mechanism:** events are added at random.
- **Gait-dependent mechanism:** the probability of an added event is proportional to exp(z). Here z is the standardised poor-gait score: minus the mean of the three standardised gait features.
- **Runs:** 200 iterations per scenario. Each iteration reruns one 5-fold cross-validation, and the paired AUROC difference M2 − M0 is recorded.

## 9. Monitoring component (descriptive, not hypothesis-tested)

**Sessions:** all first-stay sessions in normal shoes of patients meeting the inclusion criteria.

**Change limit:** a change counts as real only if it exceeds 1.96 × sqrt(SE₁² + SE₂²), where the SEs come from within-session trial variation.

**Reference distribution:** sessions from other patients with the same injury group and week of stay. At least 20 reference sessions are needed.

**Review prompts:**
1. **Persistent low loading:** at the third session or later, the peak-loading difference is below the reference 10th percentile and has not changed beyond the noise limit since the first session.
2. **Loading declined:** the latest peak-loading difference is lower than at the first session by more than the noise limit.
3. **Loading above typical range:** at the third session or later, the peak-loading difference is above the reference 90th percentile.

Rule 3 was added after viewing one example page (patient 1079) and before any outcome analysis.

**Reported:** the prevalence of each prompt by injury group. No prompt is evaluated against the outcome.

## 10. Deviations

Any departure from this plan will be listed in the technical report, with its reason.
