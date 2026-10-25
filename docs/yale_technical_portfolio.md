# GaitSense — technical portfolio (Yale MHS Medical AI, Technical Track)

**Applicant:** Kasra · **Project:** GaitSense, version 2.0.0 · **Status:** research software evaluated on a public
historical dataset; not a validated medical device.

**AI assistance.** Much of the code, analysis and drafting in this repository, including all of version 2.0, was produced with an AI assistant (Claude, Anthropic) working from my research direction, decisions and review. The implementation should not be read as written by me alone. The results were verified by automated tests, a clean-room re-implementation and full re-runs, not by an independent person.

## Version 3.0: two connected model families

GaitSense now has two capabilities that run independently and can be evaluated together. Capability 1 is the
validated biomechanical forecasting and monitoring system described below. Capability 2 asks whether those forecasts
add predictive value for independently measured recovery outcomes beyond clinical data alone. Its code (data
contracts, verification gate, leakage-safe linkage, clinical-only vs biomechanical vs combined models) is built and
passes synthetic positive and negative controls, but it has produced **no clinical results**: no open dataset links
force-plate gait to recovery outcomes, and candidate datasets (MOST; a fracture-insole study) require access
agreements. See `docs/v3_architecture.md` and `docs/v3_dataset_assessment.md`.

## Research question

Can longitudinal, uncertainty-aware models predict a rehabilitation patient's next force-plate measurement more
reliably than simple rules, and can they tell when a prediction is too uncertain to interpret? Secondary: do common
monitoring rules agree when applied to the same visits?

## Medical motivation

I worked in insurance claims. In workers' compensation, decisions about treatment authorisation, case management and
return to work lean on reports of whether an injured worker is improving. Objective movement measurements are
attractive, but a change between two visits is ambiguous: noise, day-to-day variation and genuine change overlap. A
tool that over-reads noise could push a worker back too early or flag a non-existent decline; one that under-reads
could miss a real problem. GaitRec comes from a rehabilitation centre of the Austrian Workers' Compensation Board,
so it is the right population to study this. The intended future users are treating clinicians, physical therapists,
nurse case managers and, only as consumers of clinician-reviewed information, adjusters. None of those benefits is
established by this work.

## Dataset

GaitRec (Horsak et al. 2020): bilateral ground reaction forces, 250 Hz. Cohort 1,667 patients (knee, ankle, hip,
heel), 5,870 visits, 4,203 consecutive visit pairs, 211 healthy controls. Units, sampling, stance
segmentation, sign conventions, missingness and recording speed were verified directly from the raw files.

## Technical architecture

`src/gaitsense_ai/`: ingestion with checksums → automated quality control (16 checks) → trial and curve features →
cohort and visit tables → patient-level stratified partition with leakage guards → baseline, mixed-model and Gaussian
process forecasters → probabilistic evaluation with clustered bootstrap → four monitoring architectures →
perturbation and injection engine → figures → static research prototype. Python, NumPy, pandas, SciPy,
scikit-learn, Matplotlib.

## Algorithms implemented

- Last value carried forward (same-day noise; with between-visit variance τ).
- Ridge regression and gradient-boosted trees on leakage-safe history features, with patient-grouped CV.
- Linear mixed model (random intercept/slope) and longitudinal Gaussian process (Matérn/RBF + constant + linear
  kernels) with known per-visit measurement variance, fitted by pooled maximum marginal likelihood with GLS-profiled
  fixed effects; validated against scikit-learn to machine precision.
- Monitoring: limb symmetry index, healthy-reference band and curve-level Mahalanobis (PCA + Ledoit–Wolf), variance-
  corrected change limit, GP prediction deviation.

## Validation methodology

Frozen, hashed protocol with confirmatory and exploratory hypotheses; 60/20/20 patient split; hyperparameters and
kernel chosen on calibration patients and written to file before the test set was opened (enforced in code);
prediction from earlier visits only (tested by corrupting future visits); proper scoring rules (CRPS, interval score,
log score) and coverage; 2,000-resample patient-clustered bootstrap; temporal secondary set; clean-room
re-implementation (35 checks); byte-for-byte reproducibility; 124 automated tests in v2.0 (139 with the v3.0 platform tests).

## Actual results

- Same-day-noise prediction intervals covered **74%** of measured values instead of 95%. The GP covered **94.6%**
  with intervals 14% narrower than a fixed-τ rule. Confirmatory H2 met.
- GP CRPS beat the pre-chosen mixed model by **1.5%** (H1 formally met), but not ridge regression, not the temporal
  set, and not any secondary target. This is reported as no practical advantage over simpler models.
- All history-based models cut point error by about **6%** relative to the last value.
- Monitoring rules flagged **1% to 20%** of the same visits; even at a matched alert rate, the trajectory-based rule
  disagreed with step-based rules (κ 0.59–0.68). The trajectory rule detected more injected reductions at the same
  baseline alert rate but was less stable under force perturbations. Asymmetry rules missed bilateral change.
- Post hoc: same-day repeat sessions carry only about **22%** of the weekly between-visit variance.
- Earlier v1.x results (negative later-stay prediction; corrected change limit) were reproduced exactly, and a reproducibility
  defect in the project's earlier bootstrap code was found and fixed.

## Engineering contributions

Modular package; deterministic pipeline (seeded, gzip-deterministic outputs); run manifest with SHA-256 of inputs,
code and outputs; leakage tests on real outputs; static, self-contained prototype that shows predictions with their
measured coverage, separates "statistically unusual change", "prediction deviation", "uncertain measurement" and
"insufficient evidence", and carries a fixed notice that it is not for clinical, impairment, MMI, return-to-work or
insurance decisions.

## Limitations

One centre, historical data, no functional outcomes, no treatment data, irregular visits, Gaussian assumptions
under heavier-tailed errors, cohort previously analysed by me, no external validation, no clinician or user testing.

## Future clinical development

A prospective study in injured workers with same-day and between-day repeat measurements, validated function and
pain scales, return-to-work dates, pre-registered definitions of meaningful change, multiple sites, and an
evaluation of whether calibrated ranges change clinician or case-manager decisions safely. This is the kind of
study I would like to design and carry out in the program.

## Where to look

Manuscript `docs/gaitsense_ai_manuscript.pdf` · protocol `docs/ai_protocol.md` · audit
`docs/medical_ai_baseline_audit.md` · verification `docs/ai_verification_report.md` · novelty
`docs/novelty_assessment.md` · code `src/gaitsense_ai/`, `scripts/ai/` · tests `tests_ai/` · prototype
`prototype/gaitsense_ai_review.html`.
