# GaitSense Capability 2 — clinical recovery prediction protocol (template)

**Status: template, not frozen.** It becomes a frozen protocol only after (1) a linked dataset passes the
verification gate and (2) before any outcome in that dataset is examined. At that point the dataset-specific
choices below are filled in, the file is hashed, and deviations are logged as in earlier GaitSense protocols.

## Question

Do uncertainty-aware forecasts of biomechanics (Capability 1) improve prediction of an independently measured,
clinically meaningful recovery outcome compared with clinical information alone?

## Data requirements (all must hold)

- Repeated biomechanical measurements with dates, from which a value and within-visit SE can be computed.
- Clinical covariates with the date each became available (e.g. age, sex, injury type and side, baseline pain and
  function scores, time since injury, treatment received by the landmark).
- Outcomes measured independently of the biomechanics with dates: a validated function scale (e.g. LEFS, KOOS,
  HOOS, PROMIS Physical Function), pain, performance tests, or work status.
- Permission (licence or data-use agreement) and ethics basis recorded in the manifest.

## Design

- **Landmark:** L days after injury or rehabilitation start (to fill: e.g. 42). Predictors: data on or before L.
- **Outcome window:** first assessment after L + gap (to fill: e.g. 14 days) and before the window end.
- **Endpoints** (choose before outcomes are examined):
  - Primary, binary: meaningful improvement from baseline by the window end, defined by a published MCID for the
    chosen instrument; or delayed recovery (MCID not reached).
  - Secondary: the continuous score; return to work by a date, if recorded.
- **Models:** A (clinical covariates only), B (Capability 1 landmark features only), C (both). Penalised logistic
  (binary) or ridge (continuous), fixed hyperparameters or nested selection, patient-level cross-validation.
- **Capability 1 features:** number of visits, last value, slope, smoothed level now and its SD, forecast at the
  horizon and its SD, forecast change; forecaster fitted on visits up to L only.

## Primary analysis and decision rule

Difference in out-of-fold AUROC and Brier score, Model C minus Model A, with patient-bootstrap 95% intervals.
"Biomechanics add value" only if the AUROC difference interval excludes zero **and** the Brier score does not get
worse. Report calibration intercept and slope, and net benefit at pre-specified thresholds. Report Model B to show
what biomechanics carry on their own.

## Sample size

Before analysis, check the number of events against the number of candidate parameters for Model C (Riley et al.
2020 criteria; `scripts/sample_size.py`). If the dataset is too small, report the analysis as exploratory.

## What a result would and would not mean

- Positive: biomechanical forecasts add information about this outcome in this population. Not: that biomechanical
  change is recovery, that the model is clinically useful, or that it should influence benefits.
- Negative: no added value detected at this sample size. Report it in full.
- Either way: external validation in another population is required before any clinical use; workers'
  compensation use would additionally require a prospective study with clinician review.

## Validation already done (synthetic pipeline check, not a clinical result)

`scripts/v3/check_pipeline_synthetic.py`: in synthetic data where biomechanics carry no outcome information beyond
clinical covariates, Model C did not beat Model A; where they do, it did. Results in
`results/v3/synthetic_pipeline_check.json`, labelled as synthetic.
