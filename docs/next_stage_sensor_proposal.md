# Next-stage proposal: does a portable shear signal close the push-off gap?

**Status:** proposal only. It needs a gait laboratory, hardware, ethics approval and clinical partners. None of it has been started.

## Why this is the next step

The GaitRec studies leave one clear gap between affordable sensing and laboratory measurement:

| Signal available | Real push-off changes detected at a 5% false-positive rate |
|---|---|
| Peak vertical loading (what simple insoles report) | 9% |
| Full vertical curve with AI (best evaluated model) | 31–32% |
| Force-plate anterior–posterior force (reference) | by definition, the reference |

Across the evaluated models, vertical-force measurements did not provide enough information to detect most force-plate-defined push-off changes, and more sophisticated modelling did not resolve this. The open question is whether a *directly measured* front–back signal from portable hardware closes the gap. Candidates are a shear-sensing insole or a foot-mounted inertial sensor.

Shear-sensing insoles and inertial sensors already exist. Any novelty would have to come from a specific, demonstrated improvement for *longitudinal change detection in rehabilitation*. That is the question the GaitRec work identified, and no device study we found has addressed it.

## Question

In patients rehabilitating from lower-limb injury, does adding a portable anterior–posterior signal to vertical-force sensing increase detection of force-plate-defined push-off change between visits? The comparison is with vertical-force-only AI, at the same false-positive rate.

## Design

**Setting.** Outpatient or inpatient lower-limb rehabilitation, ideally including workers' compensation patients.

**Visits.** Each visit records simultaneously:

- force plates, the reference;
- the candidate portable sensors;
- walking speed;
- pain (numeric rating scale).

**Arms**, compared within the same steps:

- A: vertical-only AI, as in the GaitRec study;
- B: the portable shear or inertial signal alone;
- C: A and B combined.

**Labels.** Force-plate push-off change beyond the between-visit-calibrated limit (requirement D1). τ is re-estimated in the new cohort.

**Analysis.** The frozen-protocol approach of the GaitRec work:

- patient-level splits;
- a single untouched hold-out;
- sensitivity at a 5% false-positive rate as the primary measure;
- results by injury group, and declines reported separately from increases.

## Sample size (rough, from GaitRec numbers)

The study is powered to detect an improvement in sensitivity from 31% (vertical-only) to 60%, at 80% power and two-sided α = 0.05. That needs about 45 true changes. A paired comparison within the same steps would need fewer.

In GaitRec, 5.8% of visit pairs were true changes, and patients with repeat visits contributed about 2.9 pairs each. That implies about 780 visit pairs, or roughly 270 patients with repeat visits.

A statistician should confirm these figures by simulation before ethics submission.

## Device requirements carried forward (GaitRec design inputs)

- **D1** Alert limits include visit-to-visit variation, estimated on separate patients.
- **D2** Decline alerts need confirmation at a second visit.
- **D3** Record the full force curve. Do not report push-off magnitude from vertical force alone.
- **D4** Sample at 100 Hz or more. Keep per-visit gain error well under 5%. Record at least 5 steps per leg.
- **D5** Track each leg against its own history and healthy norms, not only the asymmetry.
- **D6** Show forecasts only if they beat carrying the last value forward.
- **D7** Use measurement-only wording. No MMI, effort, credibility, malingering or work-capacity language, because records can reach claim files.

## What it still would not show

Even a positive result would show better *biomechanical* change detection. It would not show:

- that the detected changes matter clinically;
- that they predict return to work;
- that the device improves outcomes.

Those need independent clinical outcomes and a prospective evaluation with clinicians.
