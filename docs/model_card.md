# Model card

**Models:** M0, M1, M2 and M2+ (penalised logistic regression), developed on GaitRec. See `docs/preregistration.md` and `docs/technical_report.md`.

## Intended use

Research on whether early gait measurements carry information about later rehabilitation use.

**Not intended for:**

- clinical decisions;
- claim handling;
- causation, compensability, impairment or work-capacity judgements;
- any decision about an individual.

## Data

1,667 patients aged 9–80 (median 43; 35 under 18; 78% male) with one affected lower limb (hip, knee, ankle or heel), first assessed in normal shoes at an AUVA rehabilitation centre in Austria, 2007–2017. Follow-up runs to April 2018. The under-18 patients are most likely pupils or students, whom AUVA also insures.

## Outcome

A later-stay gait assessment within 365 days. The reason for the return is unknown, and returns without gait assessment are not captured.

## Performance

**Cross-validated AUROC:**

- M0: 0.50
- M1: 0.51
- M2: 0.49
- M2+: 0.51

The paired M2 minus M0 difference was −0.009 (95% interval −0.057 to 0.073).

**At a 10% alert rate,** positive predictive value was 3.8–4.4%, against a 5.0% base rate.

The models do not predict the outcome and should not be used to flag patients.

## Known limitations

- Walking speed is not recorded.
- Data come from a single centre.
- The outcome mixes planned and unplanned returns.
- M1 uses a dense autoencoder in place of the published LSTM.

## Monitoring component

The monitoring component is descriptive.

- **Protocol limit:** uses same-day noise only. It understates visit-to-visit variation: 28% of changes exceeded it, where 5% were expected.
- **Corrected limit:** adds visit-to-visit variation estimated on other patients, and was well calibrated on held-out patients (4.7%).

Both limits are research settings and have not been validated against test–retest data or clinical outcomes.


## Version 2.0 addendum: longitudinal forecasters (GaitSense AI)

- **Models:** last value (with and without between-visit variance), ridge, linear mixed model, Gaussian process (Matérn-3/2 + linear kernel for the primary target), boosted trees. Trained on 1,001 patients; tuned on 333; tested once on 333.
- **Output:** a predictive mean and 95% range for the next visit's affected-minus-unaffected force measures, from earlier visits only.
- **Performance (test):** GP MAE 0.0254 BW, 95% coverage 94.6%; same-day-noise intervals 73.8%.
- **Not for:** diagnosis, treatment, impairment rating, MMI, return-to-work or insurance decisions. A deviation from prediction is a measurement statement, not deterioration.
- **Known weaknesses:** heavier-tailed errors than Gaussian (50% intervals over-cover); predicted SD only weakly identifies which predictions will be wrong (Spearman 0.17); single centre, no outcomes.
