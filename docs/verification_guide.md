# Guide for an independent reviewer

This guide is for a statistician or engineer who has agreed to check the work. A full check takes about two hours. No reviewer has signed off yet. Until someone does, every result in the report has been checked only by the author's own code.

## 1. Check that the files are the ones analysed

```
cd data && sha256sum -c ../docs/data_checksums.txt
sha256sum docs/preregistration.md   # expect 2982e068…88d7, as in docs/preregistration.sha256
```

The raw GaitRec files are not in the repository. `scripts/verify_downloads.py` checks them against the MD5 values published on figshare. `scripts/extract_features.py` then rebuilds `data/derived/` from those raw files.

The protocol hash shows that the file has not changed since it was hashed. It does **not** show when it was written. The freeze time is self-recorded (see `docs/deviations.md`).

## 2. Run the independent re-implementation

```
python verification/independent_reliability.py
```

This script imports nothing from the `gaitrec_rehab` package. It was written from the protocol text and rebuilds the following from the metadata and the trial-level CSVs:

- the cohort;
- visit-level affected-minus-unaffected differences and their standard errors;
- visit-to-visit variation (τ);
- held-out calibration;
- alert rates.

It splits patients by odd and even ID. The main analysis uses random halves (seed 0).

Expected output, with exit code 0:

| Quantity | Value | Match |
|---|---|---|
| Patients / visits / visit pairs | 1,667 / 5,870 / 4,203 | exact |
| Mean visit SE, loading (BW) | 0.0155 | exact |
| τ: loading, braking, push-off (BW, BW·s, BW·s) | 0.0381, 0.0028, 0.0027 | exact |
| Share of changes beyond the same-day-only limit | 28.1% | exact |
| SD of standardised change | 2.03 (nominal 1) | exact |
| τ by group: knee, ankle, hip, calcaneus | 0.0366, 0.0394, 0.0375, 0.0385 | exact |
| Held-out share beyond corrected limit, pooled | 5.3% (nominal 5%) | ±1 point |
| "Lower than first visit" rate, corrected | 5.6% | ±1 point |

**Expected discrepancy.** Each held-out half depends on the split. With the odd/even split, the halves give 6.7% and 3.9%. With the main analysis split, they give 4.7% and 5.3%. The pooled rate is close to 5% under both splits. Calibration is therefore approximately right on average, but any single estimate carries roughly ±1.5 percentage points of split-to-split uncertainty.

## 3. Run the main pipeline's own checks

```
pytest                                     # 37 tests
python scripts/reproduce_check.py          # reruns the prediction study from scratch
python scripts/check_report.py             # every number in the report against results/*.json
```

## 4. What to scrutinise

1. **Definition of τ.** τ is the variance of consecutive-visit changes minus the mean summed squared within-visit standard error. It mixes real change in the patient with day-to-day measurement variation. The corrected threshold therefore produces the right *statistical* false-alarm rate for this population. It does not show that a crossing is clinically meaningful, and it can hide genuine declines. Section 3.6 of the report estimates how often it does.
2. **Independence of trials within a visit.** The visit SE assumes independent trials. Within-visit correlation (for example, from fatigue) would make the SE too small, and τ would absorb the difference.
3. **Outcome definition in the prediction study.** READMISSION > 0 marks any later stay, planned or unplanned.
4. **Cohort rules.** These are in `src/gaitrec_rehab/cohort.py` and the protocol: first first-stay session, one affected side, normal shoes, and a full 365-day window before 2018-04-10.

## 5. Sign-off

If you agree to be named, please record in `docs/independent_review.md`:

- your name;
- the date;
- the commit hash checked;
- which steps you ran;
- any disagreements.

A disagreement is as useful as an agreement.
