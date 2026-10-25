# Test–retest study protocol (draft for discussion, not yet approved or run)

## Why this study is needed

In GaitRec, changes between routine visits varied more than same-day noise predicts. The extra variation was τ = 0.038 body weights for weight accepted, against a single-visit standard error of 0.015. Routine visits cannot show how much of τ is measurement variation (a different day, different shoes, a different walking speed) and how much is real change in the patient. The answer decides the review threshold:

- **If τ is mostly measurement variation**, the corrected threshold is right, and smaller declines simply cannot be seen with this measurement.
- **If τ is mostly real change**, the corrected threshold is too wide, and the analysis in Section 3.6 of the report shows what it misses. The threshold should then use the smaller test–retest value.

## Design

- **Participants.** Inpatients with a unilateral knee, ankle, hip or calcaneus injury, in a clinically stable phase of their stay. Stable means no change in treatment, weight-bearing status or walking aid planned between the two sessions, as judged by the treating clinician. Target: 25 per injury group, 100 in total.
- **Sessions.** Two sessions 2–4 days apart, at the same time of day, with the same footwear, the same force plates and the same operator where possible.
- **Within each session.** At least 5 valid steps per leg at self-selected speed, then a 10-minute rest, then a second block of 5 steps. This separates trial-to-trial, within-day and between-day variation.
- **Recorded alongside.** Walking speed (timing gates), pain (0–10 numeric rating scale), walking aid, footwear, and any change in treatment between sessions.
- **Exclusions.** Any planned change in treatment between the sessions, or a pain score that changes by 2 or more points. Exclusions are reported, not hidden.

## Measures

The three GaitRec features: injured-minus-uninjured peak loading, braking impulse and push-off impulse, each divided by body weight. The code is `src/gaitrec_rehab/features.py`, unchanged.

## Analysis

1. A variance-components model with patient, day-within-patient, block-within-day and trial levels, giving between-day, within-day and trial-level SDs.
2. Standard error of measurement (SEM), minimal detectable change (MDC95 = 1.96 × √2 × SEM) and ICC(2,1) for the visit mean.
3. **Primary comparison.** The between-day SD in stable patients against the routine-visit τ of 0.038 body weights. A one-sided test at α = 0.025 of whether the test–retest between-day variation is smaller.
4. Re-run the threshold calibration and the missed-decline analysis with the test–retest value, and report both thresholds side by side.
5. Adjust for walking speed as a sensitivity analysis. If speed explains much of the between-day variation, controlling speed is a cheaper fix than a wider threshold.

## Sample size

The planning values come from GaitRec: a visit SE of 0.0155 body weights and a routine τ of 0.038.

- **Power.** Suppose the true test–retest between-day SD is half the routine τ (0.019). A one-sided variance test at α = 0.025 then has 85% power with 30 patients, 95% with 40 and over 99% with 60.
- **Precision.** With 100 patients, the 95% confidence interval for the SD of between-day differences runs from about 0.88 to 1.16 times the estimate. With 25 per group, each group's interval is wide, so the study is sized for a pooled estimate with group-level estimates as descriptive.

These figures use a chi-square approximation for the variance of paired differences. Before ethics submission, a statistician should confirm them by simulation.

## Ethics and practicalities

- **Risk.** Minimal: the study adds one extra gait session per patient.
- **Approvals.** Needs ethics approval and informed consent. Minors need parental consent, or they are excluded.
- **Data.** Pseudonymised, using the GaitRec feature extraction code.
- **Requirements.** Access to a gait laboratory. This is the main obstacle, and the reason this study is proposed rather than done.

## What the result would and would not settle

- **Would settle:** how much of τ is measurement variation, and therefore which threshold to use.
- **Would not settle:** whether a decline of a given size matters clinically. That needs clinician judgement (`docs/clinician_feedback_form.md`) and, eventually, an outcome such as return-to-work status.
