# GaitSense: detecting meaningful change in injured workers' walking with affordable sensors

**Goal.** Make it possible to detect meaningful changes in an injured patient's walking function using affordable technology instead of laboratory force plates, for use in workers' compensation rehabilitation.

**Data.** GaitRec (Horsak et al., *Scientific Data* 2020): 1,667 patients with knee, ankle, hip or heel injuries from a rehabilitation centre of the Austrian Workers' Compensation Board (AUVA), plus a 92-patient temporal hold-out.

**Three studies, one question each**

| Study | Question | Answer |
|---|---|---|
| 1. Reliability (`docs/technical_report.pdf`) | When is a change between visits real? | Same-day noise understates visit-to-visit variation by half, so standard limits flag 28% of changes instead of 5%. A corrected limit works in held-out patients and in all four injury groups. |
| 2. AI reconstruction (`docs/shear_report.pdf`, Sections 1–4) | Can AI recover push-off and braking (force-plate-only signals) from vertical force, the signal a cheap insole measures? | Partly. Peak loading carries no push-off change information. The full curve with AI gets direction right 89% of the time, but explains about a quarter of the variation and flags about 30% of real changes. |
| 3. Direct change detection (`docs/shear_report.pdf`, Section 5) | Can AI detect real changes directly, without reconstructing their size? | Not better. Paired-visit models, including a CNN, flagged 31% against 32% for reconstruction (9% for peak loading). The temporal hold-out gave 8% (0–33%, 12 changes). |

**Bottom line.** Across the evaluated models, vertical-force measurements provided insufficient information to reliably detect most force-plate-defined push-off changes, and more sophisticated modelling did not resolve this. The results set seven device requirements (D1–D7), and they point to the next step: a portable sensor that measures shear or front–back motion directly (`docs/next_stage_sensor_proposal.md`).

**Version history.** See `CHANGELOG.md`. An earlier prediction study, also under a frozen protocol, found that first-week gait did not predict a later rehabilitation stay. That null result led to the reliability question.

## Version 3.0: one platform, two model families

| | Capability 1: biomechanical forecasting and monitoring | Capability 2: clinical recovery prediction |
|---|---|---|
| Question | Predict the next force measurement, quantify its uncertainty, judge whether a change is unusual | Do biomechanical forecasts add predictive value for independently measured recovery outcomes beyond clinical data alone? |
| Status | Validated on GaitRec (v2.0 results unchanged); post-hoc conformal option added | Built and tested on labelled synthetic controls; no clinical results yet, pending a verified linked dataset |

Both run independently and can be evaluated together through a leakage-safe landmark design once legitimately linked
patient-level biomechanical and clinical outcome data are available. A verification gate refuses unverified or
synthetic data for real analyses. Architecture: `docs/v3_architecture.md`. Dataset search: `docs/v3_dataset_assessment.md`
(no suitable open dataset; MOST and a fracture-insole study are the leads). Capability 2 protocol template:
`docs/v3_capability2_protocol.md`. Capability 1 update: `docs/v3_capability1_update.md`. Code: `src/gaitsense_platform/`,
`scripts/v3/`, tests `tests_v3/`.

Workers' compensation remains the intended application, to support clinical review of recovery, not to automate
benefit decisions.

## Version 2.0: GaitSense AI (uncertainty-aware longitudinal prediction)

| Question | Answer |
|---|---|
| Can longitudinal models predict a patient's next force measurement better than the last value? | Slightly. All history-based models cut point error by about 6%; the Gaussian process was no better than ridge regression. |
| Is their uncertainty trustworthy? | Yes on average. Same-day-noise intervals covered 74% of a nominal 95%; the GP covered 94.6% with 14% narrower intervals than a fixed between-visit correction. |
| Do monitoring rules agree on the same visits? | No. Native rules flagged 1% to 20% of identical visit pairs; the trajectory-based rule detects more injected changes but is less stable under force perturbations. Asymmetry rules miss bilateral change. |

Manuscript: `docs/gaitsense_ai_manuscript.pdf`. Protocol: `docs/ai_protocol.md` (hashed before analysis). Audit: `docs/medical_ai_baseline_audit.md`. Verification: `docs/ai_verification_report.md`. Technical summary: `docs/yale_technical_portfolio.md`. Prototype: `prototype/gaitsense_ai_review.html`.

Reproduce (raw GaitRec files in `data/raw/`):

```
python scripts/ai/s3a_extract.py && python scripts/ai/s3b_build.py
python scripts/ai/s4_predict.py select && python scripts/ai/s4_predict.py test && python scripts/ai/s5_evaluate.py
python scripts/ai/s6_monitoring.py && python scripts/ai/s6b_tau_by_gap.py
python scripts/ai/s7a_cache_raw.py && python scripts/ai/s7_perturb.py
python scripts/ai/s8_prototype.py && python scripts/ai/s9_figures.py && python scripts/ai/s9b_manifest.py
python -I verification/independent_ai.py && python scripts/ai/check_ai_docs.py && pytest tests_ai tests tests_device
```

None of these results concern clinical recovery: GaitRec has no functional outcomes. Research use only.

Version 2.0 code, analyses and documents were produced with an AI assistant (Claude, Anthropic) under the author's direction and review.

## What Study 1 shows

- Same-day trial-to-trial noise understates how much these measurements move between visits by about half. 28% of visit-to-visit changes crossed a limit meant to be crossed 5% of the time.
- A threshold that adds visit-to-visit variation (τ = 0.038 body weights for weight accepted) crossed 4.7% and 5.3% in held-out halves. It held in each injury group: 4.4% to 5.3% for knee, ankle, hip and heel. A separate re-implementation reproduces these results.
- First-week gait did not predict a later recorded rehabilitation stay: AUROC 0.49 against 0.50 for patient characteristics. The paired difference was −0.009 (95% interval −0.057 to 0.073). A planted signal was recovered (AUROC 0.82).

## What the studies do not show

- **That a crossing is clinically important.** The threshold is *statistically* calibrated: it controls how often ordinary variation crosses it. It has not been tested against clinical judgement or any outcome.
- **That the threshold catches the declines that matter.** It flags 24% of 0.05 body-weight declines and needs about 0.12 body weights to flag 80% (`scripts/missed_decline.py`). The original threshold flags more declines but prompts for one patient in five.
- **How much of τ is measurement error rather than real change.** That needs a test–retest study (see Future work).
- **Anything beyond this setting.** The results do not cover other injuries, other centres, or the years after 2007–2018.
- **That an affordable sensor can replace force plates for push-off monitoring.** It could not, with vertical force and the models evaluated (Studies 2–3).
- **Anything about real insoles.** The inputs were force-plate vertical signals.
- **Clinical benefit.** Nothing here was compared against pain, function, return to work or MMI.
- **Clinical usability or external review.** Neither has been evaluated yet (see Future work).

## Studies 2 and 3 in more detail

A second study (`docs/shear_report.pdf`, protocol `docs/shear_protocol.md`) tests whether AI can reconstruct braking and push-off forces from vertical force alone, which is what a low-cost pressure insole measures, and whether the reconstruction follows *real* visit-to-visit change in injured workers.

- **Peak vertical loading carries no push-off change information** (CCC 0.00). The full vertical curve does: it explains 32% of push-off change.
- **The AI gets the direction of real change right 89% of the time** (88% in hard cases where vertical loading did not move, against 58% for vertical only). Its predictions are calibrated but capture only about a quarter of the variation in real change, and it flags 30% of real changes.
- **Final experiment** (`docs/detection_protocol.md`): models trained directly on change (paired visits, including a CNN) detected no more real push-off changes than reconstruction: 31% vs 32% at a 5% false-positive rate in cross-validation, and 8% (0–33%) on a 12-change temporal hold-out.
- **Simulated insole conditions:** 100 Hz sampling loses nothing; insole-scale gain and noise error matters most.
- **Two more failure modes:** half of injured-minus-uninjured alerts involve real change in the uninjured leg, and history-based forecasts did not beat carrying the last value forward.
- **Output:** seven design inputs (D1–D7) for a workers' compensation rehabilitation tracker. This is computational feasibility under simulated sensing, not a validated device.

The per-step extract (`data/derived/steps_*.npz`, 82 MB each) is too large for the repository. Rebuild it with `scripts/extract_steps.py` from the raw files.

## Status

**Completed**

- Analysis, 37 tests and 92 automated checks of every reported number
- Independent clean-room re-implementation of the reliability results (`verification/`, by the project author; imports nothing from the package)
- Missed-decline analysis (report Section 3.6)
- Study 2 (AI reconstruction) and Study 3 (direct change detection) under frozen protocols, each with an independent check; the temporal hold-out was evaluated once
- v1.1.0 archived on Zenodo (record 23611093)

**Future work**

| Study | Protocol | Requirements |
|---|---|---|
| Portable shear-sensor study | `docs/next_stage_sensor_proposal.md` | Gait lab, hardware, ethics approval |
| Test–retest study | `docs/test_retest_protocol.md` | Gait lab |
| Outside statistical review | `docs/verification_guide.md` | External reviewer |
| Clinician feedback | `docs/clinician_feedback_form.md` | Clinician participants |
| Usability testing with clinicians and case managers | `docs/usability_protocol.md` | Intended users |

## Cohort (reproduced by `tests/test_cohort.py`)

| Step | Patients |
|---|---|
| Patients in GaitRec | 2,084 |
| At least one first-stay session | 2,076 |
| One affected side at first assessment | 1,975 |
| Normal shoes at first assessment | 1,759 |
| Full 365-day window before the last patient session (2018-04-10) | 1,667 |
| Later-stay gait assessment within 365 days | 84 (5.0%) |

## Layout

```
src/gaitrec_rehab/cohort.py      cohort and outcome rules (the frozen protocol's definitions)
src/gaitrec_rehab/features.py    stance-phase force features: peak loading, braking and propulsive impulse
src/gaitrec_rehab/safeguards.py  checks against merged patients, future information and train/test overlap
src/gaitrec_rehab/monitoring.py  visit-by-visit mechanics, change vs noise, injury-group percentiles, review prompts
src/gaitrec_rehab/report.py      self-contained HTML review page for one patient
src/gaitrec_rehab/synthetic.py   synthetic trials for tests and demos (never patient data)
scripts/verify_downloads.py      size and MD5 check of the GaitRec source files
scripts/extract_features.py      trial-level force features from the raw files (one leg per run)
scripts/extract_curves.py        session-mean force curves for the M1 autoencoder (one leg per run)
scripts/run_analysis.py          the protocol analyses (main; bias = missed-event analysis)
scripts/positive_control.py      planted-signal check of the pipeline
scripts/run_monitoring.py        monitoring component across the cohort
scripts/variability.py           reliability of visit-to-visit change and corrected thresholds
scripts/group_reliability.py     reliability and threshold calibration by injury group
scripts/missed_decline.py        detection of injected declines under both thresholds
scripts/build_prototype.py       builds prototype/gait_review.html from the real data
scripts/make_figures.py          report figures
scripts/reproduce_check.py       independent re-run of the prediction results against results/
scripts/check_report.py          checks every number in the report against results/
scripts/sample_size.py           Riley et al. (2020) criteria for the number of predictor parameters
scripts/power_simulation.py      preliminary power and missed-event simulation for M2 vs M0
scripts/demo_review.py           renders the review page for a synthetic patient
verification/independent_reliability.py   clean-room re-implementation (imports nothing from the package)
verification/independent_shear.py         independent check of the shear study's tau, crossing rates and CCC
scripts/extract_steps.py         per-step vertical curves (250/100/25 Hz) and AP impulses from the raw files
scripts/shear_data.py            per-step dataset for the shear study
scripts/shear_study.py           Studies 1, 2A, 2B (AI reconstruction, real change agreement, planted declines)
scripts/shear_study3.py          Study 3 (simulated wearable degradation)
scripts/shear_study4.py          Study 4 (symmetry-only tracking, persistence ceiling)
scripts/detect_study.py          final experiment: direct change detection (phases: cv, then final)
docs/detection_protocol.md       frozen protocol for the final experiment (hash in detection_protocol.sha256)
verification/independent_detect.py  independent check of labels and conventional-tracker results
verification/check_holdout.py    confirms the temporal hold-out was not used before the final evaluation
docs/shear_protocol.md           frozen protocol, version 2 (hashes in shear_protocol.sha256)
docs/shear_report.md/.pdf        Studies 2 and 3: AI reconstruction, direct detection, design inputs D1–D7
docs/next_stage_sensor_proposal.md  proposal for a portable shear-sensor study
docs/technical_report.md/.pdf    TRIPOD+AI-structured technical report
docs/preregistration.md          frozen internal analysis plan, hashed before analysis (preregistration.sha256)
docs/deviations.md               protocol audit, deviations and post-hoc additions A–H
docs/verification_guide.md       step-by-step check for an outside reviewer
docs/data_checksums.txt          SHA-256 of the metadata and derived data files
docs/test_retest_protocol.md     proposed study to separate measurement error from real change
docs/clinician_feedback_form.md  20-minute structured feedback form
docs/usability_protocol.md       protocol for a clinician / case-manager usability study
docs/model_card.md               intended use, performance and limits of the models
docs/hazard_log.md               hazards, mitigations and the test covering each
prototype/gait_review.html       interactive review prototype (open in a browser)
results/                         analysis outputs (JSON) and example review pages
```

## Data

`data/GRF_metadata.csv` is GaitRec's metadata file (CC0). The four raw force files (CC BY 4.0, about 1 GB) are not included. Download them from figshare into `data/`, then run `python scripts/verify_downloads.py`, which checks each file's size and MD5 against figshare.

The derived files in `data/derived/` are enough to rerun every analysis except feature extraction. Check them with:

```
cd data && sha256sum -c ../docs/data_checksums.txt
```

## Reproduce everything

```
pip install -r requirements.txt && pip install -e .
python scripts/verify_downloads.py                       # 1. check the source files (needs raw files)
python scripts/extract_features.py data data/derived/trial_features_left.csv left
python scripts/extract_features.py data data/derived/trial_features_right.csv right
python scripts/extract_curves.py data data/derived/curves_left.npz left
python scripts/extract_curves.py data data/derived/curves_right.npz right
pytest                                                   # 2. tests, including real-data checks
python scripts/run_analysis.py main                      # 3. protocol analyses (~15 min)
python scripts/run_analysis.py bias                      #    missed-event analysis (~1 h)
python scripts/run_monitoring.py                         # 4. monitoring component
python scripts/variability.py                            # 5. reliability and corrected thresholds
python scripts/group_reliability.py                      #    by injury group
python scripts/missed_decline.py                         #    missed declines under both thresholds
python scripts/build_prototype.py                        # 6. prototype page
python scripts/make_figures.py                           # 7. figures
mkdir -p data/outputs && python scripts/extract_steps.py data left && python scripts/extract_steps.py data right && mv data/outputs/steps_*.npz data/derived/   # per-step curves
python scripts/shear_study.py && python scripts/shear_cnn.py && python scripts/shear_study.py   # Study 2 (CNN added post hoc)
python scripts/shear_study3.py && python scripts/shear_study4.py   # Study 2 degradation and failure modes
python scripts/detect_study.py cv && python scripts/detect_study.py final   # Study 3 (hold-out opened once)
python verification/independent_reliability.py          # 8. verification
python verification/independent_shear.py && python verification/independent_detect.py && python verification/check_holdout.py
python scripts/reproduce_check.py && python scripts/check_report.py
```

## Intended use

Research only. Outputs are a prompt for clinical review. They are not for causation, compensability, impairment or work-capacity decisions.

## Citation

Software: see `CITATION.cff` (v1.1.0 archived on Zenodo, record 23611093).

Data: Horsak B, Slijepcevic D, Raberger AM, Schwab C, Worisch M, Zeppelzauer M. GaitRec, a large-scale ground reaction force dataset of healthy and impaired gait. *Scientific Data*. 2020;7:143. Data licensed CC BY 4.0 (metadata CC0).
