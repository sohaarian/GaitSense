# Changelog

## 3.0.0 (2026-10-24)

### Added
- **Integrated platform** `src/gaitsense_platform/` with two model families. Capability 1 (biomechanical forecasting and monitoring) wraps the frozen v2.0 system (`load_v2`) and adds a dataset-agnostic `BiomechForecaster` with leakage-safe landmark features. Capability 2 (clinical recovery prediction) adds data contracts, a verification gate (complete provenance manifest, checksums, synthetic data refused), landmark linkage, and Models A (clinical), B (biomechanical) and C (combined) with incremental-value evaluation (AUROC, Brier, calibration, net benefit; MAE/CRPS for continuous outcomes).
- Synthetic pipeline check (labelled, not a clinical result): no added value detected when biomechanics carry no extra outcome information (AUROC difference −0.003, 95% CI −0.011 to 0.005); added value detected when they do (+0.054, 0.024 to 0.082).
- Dataset assessment: no open dataset links force-plate gait to independent recovery outcomes; MOST (controlled access) and Warmerdam et al. 2025 (fracture insoles with PROMIS) are leads.
- Capability 2 protocol template, architecture document, 15 tests in `tests_v3/`.
- Capability 1 post-hoc conformal recalibration: better 50–80% interval calibration, but over-covers at 95% with a slightly worse interval score, so the default 95% interval stays Gaussian.

### Unchanged
- All v2.0 models, results (`results/ai/`), monitoring methods and prototype.

## 2.0.0 (2026-10-24)

### Added
- **GaitSense AI study** (`src/gaitsense_ai/`, `scripts/ai/`, `docs/ai_protocol.md`, frozen and hashed before any model fit): future-visit prediction from earlier visits with last-value baselines, ridge regression, a linear mixed model, a longitudinal Gaussian process and boosted trees; probabilistic evaluation (CRPS, interval score, coverage); four monitoring architectures on identical visit pairs; perturbation and injected-change experiments; Figures 1–8; manuscript `docs/gaitsense_ai_manuscript.pdf`.
- Results: same-day-noise intervals covered 73.8% of a nominal 95%, the GP 94.6% (H2 met). GP CRPS beat the pre-chosen mixed model by 1.5% (H1 formally met) but not ridge regression, the temporal set or any secondary target. History-based models cut point error by about 6% versus the last value. Monitoring rules flagged 1–20% of identical visits (native κ 0.03–0.60).
- Post-hoc E2: τ by gap between visits; same-day pairs carry about 22% of weekly between-visit variance.
- Raw-file feature re-extraction (identical to v1.x on 151,464 trials), 16 automated quality checks, data-quality report and feature dictionary.
- 25 tests in `tests_ai/` (124 in total), clean-room `verification/independent_ai.py` (35 checks), run manifest with checksums, verification report.
- Research prototype `prototype/gaitsense_ai_review.html` (test patients only).
- `docs/medical_ai_baseline_audit.md`, `docs/novelty_assessment.md`, `docs/yale_technical_portfolio.md`, `docs/ai_deviations.md`.

### Fixed
- `scripts/group_reliability.py` seeded its bootstrap with Python's per-process randomised `hash()`, so injury-group τ intervals were not reproducible. Now deterministic; two reported lower bounds changed in the third decimal (hip and heel 0.034 → 0.033). v1.5.0 results are archived in `archive/v1.5.0/results/`.
- Restored the missing `data/derived/steps_{left,right}.npz` checks (files supplied separately; SHA-256 match).

## 1.5.0 (2026-10-24)

### Added
- GaitSense software v0.3: reference-load drift correction, HTML dashboard, authentication/roles/encryption/access log, sensor-driver interface with framed protocol, and expanded verification tests (corruption, sensor failure, calibration change, longitudinal error rates). Quality gate now rejects empty and non-finite recordings. Simulated data only; no security review.

## 1.4.0 (2026-10-24)

### Added
- **GaitSense bench-analysis package** (`bench_analysis.py`, `tests_device/`, `README_DEVICE.md`): calibration with cross-axis terms, independent validation, hysteresis, drift, remount repeatability/MDC, Bland-Altman and a reference-load check. Tested on simulated data only (9 tests); no physical device exists yet. The GaitRec research results are unchanged.

## 1.3.0 (2026-10-24)

### Added

- **Final experiment: direct detection of push-off change** (`docs/detection_protocol.md`, frozen before analysis; report Section 5; Figure 6). Six methods were compared at a 5% false-positive rate. Paired-visit models trained on change did not beat reconstruction: the CNN detector reached 31% (24–37%) against 32% for reconstruction. On the temporal hold-out (81 patients, 12 changes, evaluated once), the detector reached 8% (0–33%). By the pre-stated rule, this supports "missing information" over "correctable attenuation".
- **Multi-channel input for the NumPy CNN**, with a gradient test (37 tests in total). Also `verification/independent_detect.py` and `verification/check_holdout.py`, which pass. The prior hold-out exposure is disclosed.

### Changed

- **Correction.** Earlier text said the shear-force AI "underestimates change size by about two-thirds". The new calibration analysis shows its predictions are calibrated (slope 0.99) but explain only about 24% of the variance in change. The flat scatter mainly reflects limited information rather than uniform shrinkage. Calibration varies by injury group (slopes 0.82–1.10). Wording was later softened to "across the evaluated models", following outside review. The report, README and changelog are updated.

## 1.2.0 (2026-10-24)

### Added

- **Shear-force reconstruction study** (`docs/shear_report.md`), under a frozen internal protocol (`docs/shear_protocol.md`, version 2, hashed before any per-step data were loaded). It tests whether AI can reconstruct braking and push-off from vertical force while following real visit-to-visit change. All four pre-stated criteria were met: CCC 0.39 with the true push-off change, and 88% direction accuracy in the hard subset. Predictions are calibrated but explain about a quarter of the variation in change (see 1.3.0 for the correction).
- **Studies 3 and 4:** simulated insole degradation, symmetry-only tracking, and the persistence ceiling.
- **Failure register:** seven design inputs (D1–D7) for a workers' compensation rehabilitation tracker.
- **Figure 5**, `verification/independent_shear.py` (all key numbers match), and deviations S1–S6.
- **A true 1D CNN** (`src/gaitrec_rehab/cnn1d.py`, NumPy, gradient-checked) as a post-hoc model. It matches gradient boosting (change CCC 0.41 vs 0.39, difference interval includes zero). An untested PyTorch version is in `scripts/shear_cnn_pytorch.py`.
- **Per-step accuracy by injury group**, and leakage tests for the patient split (`tests/test_shear_leakage.py`). The suite now has 36 tests.

## 1.1.0 (2026-10-22)

### Added

- **Missed-decline analysis** (`scripts/missed_decline.py`, report Section 3.6, Figure 4). Known declines are injected into real trajectories to estimate how often each threshold flags them. The corrected threshold flags 24% of 0.05 body-weight declines, against 60% for the original, and needs about 0.12 body weights to reach 80%.
- **Independent re-implementation** of the reliability results (`verification/independent_reliability.py`, report Section 3.7). It imports nothing from the package. It reproduces counts, τ, crossing rates and group τ exactly, and gives a pooled held-out rate of 5.3% under a different patient split.
- **Reviewer materials:** `docs/verification_guide.md` and `docs/data_checksums.txt` (SHA-256 of the metadata and derived data).
- **Study protocols and forms:** `docs/test_retest_protocol.md` (design, sample size, analysis) and `docs/clinician_feedback_form.md`.
- **Report additions:** a "What this shows / does not show" box in the report, the faculty summary and the README, and a paragraph separating statistical calibration from clinical detection.
- **Test:** `test_corrected_limit_misses_a_decline_the_original_catches` (hazard H14), bringing the suite to 32 tests.
- **Report check:** `scripts/check_report.py` now runs 92 checks, including the new results and an assertion that the independent re-implementation matched.

### Changed

- **Protocol wording.** "Pre-registered" is replaced throughout by "frozen internal protocol". The freeze time is self-recorded and no registry timestamped the plan before the analysis. `docs/preregistration.md` itself is unchanged, and its hash still matches.
- **Prototype label.** The original threshold is now labelled "Original: same-day noise only (frozen protocol)".
- **README.** Restructured around the main contribution, with a status table of what has and has not been done.

## 1.0.0 (2026-10-22)

- Prediction study with a frozen internal protocol: null result, with a positive control and missed-event analysis.
- Monitoring component, reliability analysis with split-half calibration, a corrected threshold, and reliability by injury group.
- Interactive prototype covering all 1,667 patients, 31 tests, and a technical report.
