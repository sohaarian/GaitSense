# GaitSense AI — deviations from the frozen protocol and post-hoc additions

Protocol: `docs/ai_protocol.md` (SHA-256 in `docs/ai_protocol.sha256`; self-recorded freeze time).

## Deviations

| # | What | Why | When | Effect |
|---|---|---|---|---|
| AI-1 | Tables written as gzip CSV instead of Parquet | pyarrow could not be installed (no package-index access in the analysis environment) | Stage 3 | None on content |
| AI-2 | Linear mixed model fitted with the project's own maximum-likelihood code (`longitudinal.py`, random intercept/slope as a linear-kernel GP) instead of statsmodels | statsmodels not installable | Stage 4 | Validated by a simulation-recovery test and by exact agreement of the GP code with scikit-learn |
| AI-3 | `peak1`/`peak2` must be ≥ 90% of the half-stance maximum | First extraction found impact-transient "peaks" as low as 0.19 BW and loading rates up to 251 BW/s | After the first extraction, before any model or outcome; secondary features only | Secondary features more robust; primary features unchanged |
| AI-4 | Healthy reference restricted to self-selected-speed sessions (SPEED = 2) | Audit found controls walked at three speeds while all patient sessions are SPEED = 2 | Before Stage 6 was run | Removes a speed confound from architecture B |
| AI-5 | `scripts/group_reliability.py` bootstrap seed changed from `hash((g, f))` (randomised per process) to `zlib.crc32` | Audit defect: injury-group CIs were not reproducible | Stage 1 | Two reported CIs changed in the third decimal (hip 0.034→0.033, heel 0.034→0.033 lower bound); report updated; point estimates unchanged |
| AI-6 | Tests run with a minimal pytest-compatible runner | pytest not installable here | All stages | CI workflow still uses pytest |
| AI-7 | No mediolateral features | Mediolateral RAW files were not in the analysis copy | Stage 3 | Exploratory item not done |

## Post-hoc additions (labelled as such everywhere)

| # | Addition | Reason |
|---|---|---|
| E2 | τ by gap between visits, including 199 same-day session pairs in the cohort (`scripts/ai/s6b_tau_by_gap.py`) | The audit found same-day repeat sessions, which allow a partial look at how much of τ is present without time for recovery |
