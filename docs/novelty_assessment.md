# GaitSense — novelty assessment

Focused search, October 2026: GaitRec machine-learning studies; Gaussian processes and mixed models for longitudinal
clinical trajectories; ground-reaction-force reliability and minimal detectable change; uncertainty quantification and
calibration in medical ML. Only sources whose details could be checked were used. A focused search is not a
systematic review, so every "not found" below means "not found in this search".

## Closest existing work

| Study | Data | Question | Models | Outcome | Validation | How GaitSense differs |
|---|---|---|---|---|---|---|
| Horsak et al. 2020, *Sci Data* 7:143 | GaitRec (2,084 patients, 211 controls; AUVA) | Dataset release; baseline classification | Standard classifiers | Gait-disorder class | Provided train/test split | Uses the same data for longitudinal prediction, not classification |
| Slijepcevic et al. 2022, *ACM Trans Comput Healthcare* (doi 10.1145/3474121) | GRF from 132 patients, 62 controls (dataset not named in the record checked) | Explain gait-disorder classifiers | Classifiers + layer-wise relevance propagation | Class, explanations | Statistical parametric mapping, expert review | Single-session classification; no repeated-visit prediction |
| Pandey et al. 2022, *PPAR Research* (doi 10.1155/2022/9355015) | Large-scale GRF dataset (title suggests GaitRec; not confirmed in the record checked) | Healthy vs impaired | Deep network vs ML classifiers | Binary class | 5-fold CV | Classification only; no uncertainty, no longitudinal design |
| Chakraborty et al. 2022, *Med Eng Phys* 103:103788 (PubMed 35500989) | GaitRec | Recovery assessment across rehabilitation sessions | LSTM autoencoder trained on healthy gait | Reconstruction error used as an indicator of recovery status (no clinical outcome labels) | Healthy-trained model applied to patient sessions | Closest prior longitudinal GaitRec work. It scores distance from healthy gait; GaitSense forecasts each patient's next measurement with calibrated uncertainty and compares monitoring rules. v1.x reimplemented a version of it as model M1 |
| Futoma et al. 2016, MLHC, PMLR 56:42–54 | Longitudinal EHR biomarkers (chronic disease) | Disease trajectory forecasting | Per-patient GPs with latent classes | Future biomarker values | Compared with a prior trajectory model on a large patient dataset | Same model family; different domain and signal |
| Peterson et al. 2017, arXiv:1712.00181 | ADNI/TADPOLE, 100 patients | One-visit-ahead cognitive scores | Personalised (domain-adaptive) GP | MMSE, ADAS-Cog13, CDR-SB | 10-fold patient-wise CV, MAE | Same idea (one-step patient GP); we add proper scoring rules, calibration, persistence baselines |
| Liang et al., arXiv:2005.11770 (AAAI 2021) | SWAN, GSS, TADPOLE | Longitudinal regression | Deep-kernel GP | Continuous scores | Held-out comparison | More complex kernel; we deliberately keep standard kernels and test against simpler models |
| Chou et al. 2026 (doi 10.1097/PXR.0000000000000572) | 10 people with transfemoral amputation, two visits | Between-session reliability of GRF impulses and loading rates | ICC, MDC, SPM | Reliability | Test–retest | Designed reliability study; ours is observational on routine visits |
| Weir 2005, *J Strength Cond Res* 19(1):231–240 | — | ICC, SEM, MDC methodology | — | — | — | Our change limit is in this tradition |
| Gneiting & Raftery 2007, *JASA* 102:359–378; Angelopoulos & Bates (arXiv:2107.07511) | — | Proper scoring rules; distribution-free uncertainty | — | — | — | Evaluation framework we apply |

## Which contributions are incremental

- Gaussian-process and mixed-model forecasting of irregular longitudinal clinical measurements (established).
- Variance-corrected change limits and MDC (established; v1.x already applied one).
- CRPS, interval score, coverage, patient-clustered bootstrap (standard).
- Limb symmetry index and healthy-reference comparisons (standard clinical practice).

## Which findings might be original (application and evidence, not method)

1. Beyond Chakraborty et al. (2022), who scored distance from healthy gait across sessions: a head-to-head, pre-specified comparison of probabilistic one-visit-ahead forecasters against persistence baselines,
   scored with proper scoring rules, for repeated ground reaction forces in rehabilitation. Not found in this search.
2. Evidence that the benefit of longitudinal models in this setting is calibration and sharpness, not point accuracy:
   same-day noise gives 74% coverage of a nominal 95%; history-aware models reach about 95% with 12–14% narrower
   intervals than a fixed between-visit variance.
3. Quantified disagreement between four monitoring rules on identical rehabilitation visits (native κ 0.03–0.60;
   matched-rate κ 0.59–0.91), with a detection-stability trade-off measured by injection and perturbation.
4. The observation that asymmetry-based monitoring cannot see bilateral change and that most symmetry-index alerts in
   this cohort involve the unaffected leg.
5. A same-day vs between-day decomposition of between-visit variation (τ) in routine rehabilitation data (post hoc).

## Novelty claims that cannot be justified

- Any new algorithm: the GP, mixed model, bootstrap and variance correction are standard.
- "First" anything: the search was focused, not systematic.
- Clinical relevance, recovery detection, deterioration detection, return-to-work or MMI prediction: no outcomes.
- External or independent validation: one centre, one author, temporal set previously opened.
- That the GP is better than simpler models: the formal H1 success is small, not seen against ridge regression, not
  replicated in the temporal set or secondary targets.

## Originality of application vs method

The method is textbook. Any originality is in the application (longitudinal GRF forecasting in workers'
compensation rehabilitation), the evaluation design (frozen protocol, persistence baselines, proper scoring rules,
monitoring-rule comparison on identical visits) and the negative and mixed findings it produced.
