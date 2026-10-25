# GaitSense — Medical AI baseline technical audit (Stage 1)

Audited version: repository v1.5.0 (`GaitSense_Device_Feasibility_v0.3.zip`, identical to the unpacked
`gaitrec-rehab` folder). Audit date 2026-10-24. Every statement below was checked by reading or running the code;
nothing is assumed from file names.

## 1. Existing capabilities

| Area | What exists | Verified by |
|---|---|---|
| Cohort | Frozen v1.x rules → 1,667 patients (84 later-stay events) | `tests/test_cohort.py`; rebuilt here |
| Trial features | Weight-acceptance peak (first-half max), braking and propulsive impulse, stance duration | Re-extracted from RAW: identical to 0.0 on 151,464 trials |
| Prediction study | Later-stay prediction, logistic models M0/M1/M2 | Re-run: `results/main.json` reproduced (136 numbers, 0 differ) |
| Reliability | Within-session SE, τ (method of moments), split-half calibration, injury-group τ | Re-run: `variability.json` 0/92 differ; `group_reliability.json` see §8 |
| Missed declines, positive control | Injected declines; planted signal | Re-run: 0 differences |
| Shear/detection studies | Per-step AI reconstruction, CNN, direct detection, temporal hold-out | Independent checks pass once `steps_*.npz` are supplied |
| Device package | Bench calibration, drift, quality gate, dashboard, security, drivers | 62 tests pass; simulated data only |
| Prototype | `prototype/gait_review.html` (static, all 1,667 patients) | Opened; uses v1.x monitoring rules |

## 2. Verified source files

- `src/gaitrec_rehab/` (658 lines): `cohort.py`, `features.py`, `monitoring.py`, `sessions.py`, `safeguards.py`,
  `analysis.py`, `cnn1d.py`, `report.py`, `synthetic.py`.
- 27 analysis scripts in `scripts/`; 4 clean-room checks in `verification/`; device code in `gaitsense/`,
  `bench_analysis.py`.
- Data present: `data/GRF_metadata.csv`, `data/derived/trial_features_{left,right}.csv`, `curves_*.npz`,
  `cnn_preds.npz`, `shear_models.pkl`. **Missing from the zip:** `data/derived/steps_{left,right}.npz` (found in
  `Downloads/CLAUDE READ THIS/outputs`; SHA-256 match `docs/data_checksums.txt`). RAW force files
  (`GRF_F_V_RAW_*`, `GRF_F_AP_RAW_*`) found in the same folder.

## 3. Existing model implementations

- **Later-stay prediction** (`src/gaitrec_rehab/analysis.py`): penalised logistic regression M0 (patient
  characteristics), M2 (+3 force features), M1 (dense autoencoder on curves). Result: AUROC 0.503 (M0), 0.494 (M2),
  difference −0.009 — a negative finding, preserved.
- **Shear reconstruction / detection:** gradient boosting, a NumPy 1-D CNN, paired-visit detectors.
- **Persistence forecast** (shear report §4b): ridge on history vs last value for the loading difference — ridge
  did not beat persistence (MAE 0.0303 vs 0.0315).
- **No longitudinal probabilistic model exists**: no mixed model, no Gaussian process, no predictive intervals for
  future visits. This is the gap Stages 4–5 address.

## 4. Existing feature definitions

`peak_loading` = max vertical force in the **first half** of stance / BW (a weight-acceptance peak, not the stance
maximum); `braking_impulse` = −∫min(AP,0)dt / BW; `propulsive_impulse` = ∫max(AP,0)dt / BW; `stance_s` = samples /
250 Hz. Visit summary = affected − unaffected mean, SE from trial-to-trial variation.
The directive's Feature 1 ("maximum vertical force during stance") differs from `peak_loading`; the new pipeline
computes both (`peak_v` and `wa_peak`).

## 5. Existing validation procedures

Frozen internal protocols with SHA-256 hashes (self-recorded times); patient-level CV for prediction; split-half
cross-fitting of τ (seed 0); patient-clustered bootstraps; temporal hold-out (92 patients, used once in v1.3);
report-number check (`scripts/check_report.py`, 92/92 pass); clean-room re-implementations (all pass).

## 6. Existing software tests

`tests/` 37 tests (features, cohort, monitoring, safeguards, leakage of shear split, CNN gradients, interface text);
`tests_device/` 62 tests. All 99 pass (run with a minimal pytest-compatible runner because PyPI is unreachable in
this environment; the CI workflow `.github/workflows/tests.yml` uses real pytest).

## 7. Reproduced results

| Result | Reported | Reproduced |
|---|---|---|
| Cohort / events | 1,667 / 84 | 1,667 / 84 |
| AUROC M2 − M0 | −0.009 | −0.0089 (identical) |
| Patients with ≥2 visits / consecutive pairs | 1,441 / 4,203 | 1,441 / 4,203 (new pipeline, independently) |
| Within-session-only crossing (loading) | 28.1% | 28.05% |
| τ loading (full cohort) | 0.038 BW | 0.03808 |
| Held-out crossing with cross-fitted τ | 4.7% / 5.3% | 4.74% / 5.3% |
| Missed-decline, positive control | as reported | identical |
| Shear CCC, detection sensitivity | 0.39; 9.5% (V-thr) | identical (independent scripts) |

## 8. Identified technical limitations and defects

1. **Non-reproducible bootstrap seeds** in `scripts/group_reliability.py`: `seed=hash((g, f)) % 2**32` uses Python's
   per-process randomised string hashing, so injury-group τ confidence intervals change on every run (24 numbers
   differed in the re-run; point estimates identical). Fixed in v2.0 by a deterministic seed (see deviations).
2. Derived per-step files are not shipped and the README rebuild path requires the 1 GB RAW files.
3. Feature naming: `peak_loading` is a first-half peak, which is easy to misread as the stance maximum.
4. τ mixes measurement variability and genuine change (stated in all reports; unchanged).
5. No probabilistic forecasting, no calibration of uncertainty for future visits, no comparison of monitoring
   architectures on identical pairs.
6. `results/` would be overwritten by re-running scripts; v1.5 results are archived in `archive/v1.5.0/results/`.
7. `scripts/run_analysis.py` needs an argument (`main` / `bias`); calling it bare raises IndexError.

## 9. Data availability and missing variables

Verified from RAW files and metadata: forces in newtons, 250 Hz, stance pre-cut at 25 N (minimum sample
25.000 N in every trial), NaN padding only at the end (0 interior gaps in 151,464 trials), AP negative = braking.
Body weight equals body mass × 9.81 to within 0.1% in all 8,971 sessions. All patient sessions are recorded at
self-selected speed (SPEED = 2); healthy controls walked at three speeds (1, 2, 3) — **healthy references must use
SPEED = 2 only**. 215 consecutive patient pairs (199 in the cohort, 16 in the temporal set) are two sessions on the same day.

**Absent:** mediolateral RAW force in this analysis copy; step length, cadence, swing time, joint angles, walking
velocity; treatment details; pain, function scores, return-to-work, MMI, claim costs; reasons for readmission.

## 10. Components that can be reused

Cohort rules; validity rule; τ estimator and cross-fitting logic; visit-difference SE; safeguards; injection logic
of `missed_decline.py`; temporal hold-out definition; report-check pattern; pandoc/wkhtmltopdf rendering.

## 11. Components needing modification

Bootstrap seeding (defect 1); feature naming (add `peak_v`, keep `wa_peak`); healthy reference restricted to
self-selected speed; prototype needs model predictions, intervals and architecture comparison.

## 12. Components that should not be implemented (data absent)

Clinical recovery or deterioration labels; MMI, return-to-work, impairment or cost prediction; clinical
sensitivity/specificity or ROC; cadence/step-length/velocity features; mediolateral features (file not available
here); speed adjustment for patients (speed not varied); treatment-response modelling.
