# Analysis protocol audit and deviations

**Status:** a frozen internal protocol, not an independently timestamped pre-registration.

## What can be verified

- **The plan file is unchanged.** `docs/preregistration.md` has the SHA-256 hash recorded in `docs/preregistration.sha256` (`2982e0687b4cb674b655eff8895261a1ce028c0dabc69160f8606620c86588d7`). Anyone can confirm the file is unchanged with:
  `sha256sum -c <(head -1 docs/preregistration.sha256)`

- **The freeze time is self-recorded, not independently verifiable.** The freeze time, 2026-10-22 20:24:03 UTC, was written by the author's own process; no third party witnessed it. When the plan is posted to OSF, the OSF timestamp will be later than the analysis. The OSF entry should say so plainly: "frozen and hashed before analysis; posted after analysis."
- **The order of work, as recorded in the project history:**
  1. cohort audit (event counts only);
  2. feature extraction and validation;
  3. plan frozen;
  4. waveform extraction;
  5. outcome analyses.

## Planned analyses, all run as specified

The primary analysis (M2 against M0) and secondary analyses S1–S4 were run, as were exploratory analyses E1–E4 and the monitoring component with prompt rules 1–3.

## Deviations

| # | What | Why | Effect on conclusions |
|---|---|---|---|
| 1 | E3 discrete-time model used a fixed penalty, C = 1 | The plan did not specify a penalty for E3 | None; the result was null |
| 2 | Calibration slopes reported but not interpreted | Predictions were almost constant, so the slopes are meaningless | None |
| 3 | M1 used a dense autoencoder in place of the published LSTM | PyTorch was not available | Stated in the plan before analysis |

## Post-hoc additions (not in the frozen protocol, labelled as such everywhere)

| # | Addition | Reason |
|---|---|---|
| A | Positive control (planted signal) and in-sample check | To rule out a pipeline error after the null primary result |
| B | Reliability analysis of visit-to-visit change, with split-half calibration | Monitoring showed far more "beyond noise" changes than expected |
| C | Corrected review threshold that adds between-session variation | Result of B. The protocol rule is kept and still shown in the prototype. |
| D | Alert behaviour under both thresholds: rates, overlap, repetition, sensitivity to z | To show how the corrected threshold changes prompts |
| E | Reproduction script and report-number check | Verification |
| F | Injury-specific reliability and calibration (knee, ankle, hip, heel) | To check whether one threshold fits all four injury groups |
| G | Missed-decline analysis: known declines injected into real trajectories (`scripts/missed_decline.py`) | To measure what the corrected threshold costs in missed declines |
| H | Independent re-implementation of the reliability results (`verification/independent_reliability.py`), with data checksums | To guard against coding errors before outside review |

None of the additions changes a protocol result. Additions B–D and F–H concern the descriptive monitoring component, which was never tested against the outcome.

## Shear-force study (docs/shear_protocol.md, version 2)

| # | Deviation | Reason | Effect |
|---|---|---|---|
| S1 | M-cnn replaced by an MLP on the full curve | No deep-learning library available; allowed by the protocol | The MLP did worse than gradient boosting on validation, so it was not chosen as "the AI" |
| S2 | Vertical-only condition: crossing rate, sensitivity and specificity not interpreted | Its calibration slope is near zero, so its predicted change is almost constant and its limit is tiny | CCC and direction are still reported; conclusions unchanged |
| S3 | The "noise ceiling" in the information test is reported but not interpreted | Vertical and AP noise come from the same steps and are correlated, so it is not a true ceiling | None |
| S4 | Study 3 criterion (planted-decline retention) could not be assessed | The planted-decline method was uninformative even with clean data (Study 2B) | CCC and crossing rate reported instead |
| S5 | The independent check compares τ to within 1e-6, not 1e-9 | The main script stores curves as float32; the check uses float64 | τ agrees to 3e-8; counts, crossing rates and CCC match exactly |
| S6 | A true 1D CNN (NumPy, gradient-checked) added after the main analysis; reported as post hoc | Completes the planned model comparison | Change CCC 0.41, the same as M-gb within uncertainty; "the AI" is unchanged |
