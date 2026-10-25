---
title: "Detecting Meaningful Change in Injured Workers' Walking with Affordable Sensors"
subtitle: "Research summary for faculty review"
author: "Kasra"
date: "October 2026"
---

# Goal

Rehabilitation teams and workers' compensation case managers follow patients through recovery from lower-limb injury. Laboratory force plates can measure how walking changes between visits, but few clinics have them. **My goal is to find out whether affordable sensing, such as a pressure insole, can detect meaningful change in an injured patient's walking, and what it would need to do so.**

**Data.** I used GaitRec (Horsak et al., *Scientific Data*, 2020), an open dataset from a rehabilitation centre of the Austrian Workers' Compensation Board. The analysis covers 1,667 patients with knee, ankle, hip or heel injuries: 5,870 visits and about 100,000 force-plate steps. A further 92 later patients were kept untouched as a temporal hold-out. I built every measure from the raw force curves.

# Study 1: when is a change real?

The usual limit for a "real" change uses same-day noise from repeated steps. With that limit, **28%** of visit-to-visit changes crossed it, where 5% were expected. Measurements move about twice as much between visits as within one.

A corrected limit adds that visit-to-visit variation, estimated on separate patients. It gave 4.7–5.3% in held-out patients, and held in each injury group. The cost is that it misses most small declines.

# Study 2: can AI recover what an insole cannot measure?

An insole measures only vertical force. Push-off and braking come from the front–back force, which needs a force plate. I trained models to reconstruct push-off and braking from vertical force: ridge regression, gradient boosting and a 1D CNN. The CNN was written in NumPy, with its gradients checked numerically.

| Signal | Agreement with real push-off change (CCC) | Direction correct |
|---|---|---|
| Peak vertical loading | 0.00 | 59% |
| Full vertical curve, gradient boosting | 0.39 | 89% |
| Full vertical curve, CNN | 0.41 | 89% |

So the **shape** of the vertical curve carries push-off information, while its peak does not. But the AI explained only about a quarter of the variation in change.

# Study 3: can AI detect change directly?

Under a protocol frozen before analysis, I trained paired-visit models, including a four-channel CNN, to detect real push-off changes directly. Results at a 5% false-positive rate:

| Method | Development set, 5-fold CV (243 changes) | Temporal hold-out (12 changes) |
|---|---|---|
| Peak vertical loading | 9% | 8% |
| AI reconstruction | 32% | 8% |
| Paired-visit CNN | 31% | 8% (95% CI 0–33%) |

Detecting change directly did not beat reconstruction.

> **What this shows.** Standard limits mistake ordinary variation for change, and a corrected limit fixes this across four injury groups. The full vertical-force curve carries push-off information that peak loading misses. Across the evaluated models, though, vertical force could not detect most force-plate-defined push-off changes, and more sophisticated modelling did not resolve this.
>
> **What it does not show.** That detected changes matter clinically, or that real insoles behave like force-plate vertical signals. Nothing was compared against pain, function or return to work. The hold-out is small. All analysis plans were frozen internally, not publicly registered. No outside reviewer has checked the work yet.

# What it means for an affordable device

The results set seven requirements for a workers' compensation rehabilitation tracker. Among them:

- use between-visit-calibrated alert limits;
- confirm declines at a second visit;
- record the full force curve;
- sample at 100 Hz or more;
- compare each leg with its own history and with healthy norms;
- use measurement-only wording, with no MMI, effort or work-capacity language.

The next step is a study testing whether a **portable shear or inertial sensor** closes the push-off gap. That study needs about 270 patients with simultaneous force plates. The proposal is drafted.

# Where I would value your guidance

1. **Sensor study.** Is a study of simultaneous force-plate and shear-insole recordings in rehabilitation patients feasible, and what would make it credible?
2. **Clinical meaning.** How should the size of a clinically meaningful decline be set?
3. **Clinical evaluation.** What would a clinician-facing version need before evaluation?

*The code (37 tests, with independent re-implementations), three reports, frozen protocols and an interactive prototype are at github.com/sohaarian/gaitsense-clinical-ai (DOI 10.5281/zenodo.23611092).*
