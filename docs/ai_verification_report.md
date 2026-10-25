# GaitSense AI — verification report (Stage 9)

Experiment ID `gaitsense-ai-2.0.0-065bf3843746` (`results/ai/run_manifest.json`: software versions, seed 20261024,
configuration, SHA-256 of every input, script and output).

## What was checked

| Check | How | Result |
|---|---|---|
| Historical v1.x results | Re-ran `run_analysis.py main`, `bias`, `variability.py`, `group_reliability.py`, `missed_decline.py`, `positive_control.py` in a scratch copy and diffed every number | Identical (only runtime differs), except injury-group τ CIs, which were not reproducible (defect fixed; see `ai_deviations.md` AI-5) |
| Historical shear/detection checks | `verification/independent_*.py`, `check_holdout.py` with the recovered `steps_*.npz` (SHA-256 match) | All pass |
| Report numbers | `scripts/check_report.py` | 92/92 |
| New feature extraction | Re-extracted from RAW; compared with v1.x trial features | Identical on 151,464 trials (max abs diff 0.0) |
| Clean-room recomputation | `verification/independent_ai.py` (imports nothing from `src/`): counts, partition, visit summaries, last-value predictions, MAE/CRPS/coverage for all six models, τ, architecture C rate, C–D κ, healthy reference, order of work | 35/35 checks pass |
| GP implementation | Kernel matrices and posterior compared with scikit-learn `GaussianProcessRegressor` | Agreement to ~1e-17 |
| Mixed-model implementation | Simulation recovery (random intercept 0.05, noise 0.02) | Recovered within 15% |
| Leakage | `tests_ai/test_ai_leakage_data.py`: disjoint partitions; stored predictions unchanged when the target and all later visits are corrupted (40 sampled rows); selection finished before test opened; selection used only train/calib targets; calibration patients absent from evaluation; healthy reference = training controls at patient speed; covariate scaling from training patients | All pass |
| Determinism | Model selection re-run from scratch in a separate directory; Stages 3, 5, 6, 6b, 7 re-run in place | Identical choices, identical test predictions (max diff 0.0), identical result files (byte-for-byte after making gzip headers deterministic) |
| Test suites | `tests_ai` 25, `tests` 37, `tests_device` 62 | 124/124 pass |
| Document numbers | `scripts/ai/check_ai_docs.py`: 65 numbers in the manuscript, portfolio, novelty assessment and README recomputed from results | 65/65 match (after correcting one rounding error and several wording errors found in an accuracy review) |
| Citations | Each reference checked against PubMed, the publisher, arXiv or a library record | All found; one description (Chakraborty et al. 2022) corrected |
| Protocol integrity | SHA-256 of `docs/ai_protocol.md` vs `docs/ai_protocol.sha256` | Match |

## Limits of this verification

- All checks were written by the same author and run on the same cohort; this is not independent validation.
- pytest and statsmodels were unavailable; tests ran under a minimal pytest-compatible runner and the mixed model is
  the project's own implementation (validated as above).
- The temporal set had been opened once before (v1.3 detection study) for a different question.
