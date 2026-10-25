# GaitSense v3.0 — dataset assessment for Capability 2 (clinical recovery prediction)

Searched October 2026 for datasets that link patient-level biomechanical measurements to **independently measured**
clinical outcomes (function, pain, performance tests, return to work). Only details confirmed on the source pages
are listed. **No dataset below has been obtained or verified by this project; none has been analysed.**

## Requirement for Capability 2

Repeated biomechanical measurements (force plate, instrumented insole or equivalent) and, for the same patients,
outcomes measured by a separate instrument or assessor (validated function or pain scales, performance tests, work
status), with assessment dates, under a licence or data-use agreement that allows this research.

## Candidates

| Dataset | Biomechanics | Independent outcomes | Population | Access | Suitability |
|---|---|---|---|---|---|
| **GaitRec** (Horsak et al. 2020, *Sci Data* 7:143) — in use | 3D GRF, force plates, repeated visits | **None** (readmission dates only) | Lower-limb injuries, AUVA workers' compensation centre | Open (CC BY 4.0) | Capability 1 only |
| **MOST**, Multicenter Osteoarthritis Study (NIA Aging Research Biobank) | Plantar pressure and GAITRite (years 0–7); vertical GRF-derived peak impact force and loading rate reported by Sara et al. 2025 (*Osteoarthritis Cartilage*, doi 10.1016/j.joca.2025.04.007); OPAL heel-strike force (years 12–14) | WOMAC, KOOS, knee pain, 20-m walk, chair stands, employment and work limitations (per the MOST table of measures, January 2026) | 3,026 + ~1,525 adults with or at risk of knee OA, 2003–2018+ | Controlled: request through the NIA biobank, agreement required | **Strongest candidate for testing the scientific question.** Not occupational injury; gait devices differ from GaitRec; exact variables and visit alignment must be confirmed from the data dictionary |
| **Warmerdam et al. 2025**, *JMIR mHealth uHealth* 13:e71022 | OpenGO instrumented insoles (16 pressure sensors, IMU), in hospital week 1, 6 weeks, 3, 6, 9, 12 months | PROMIS Global Physical Health, Global Mental Health, Pain Interference at the same timepoints | 85 patients after tibial or malleolar fracture surgery | No data-sharing statement found on the article page; would require contacting the authors | **Closest to the intended injury-rehabilitation setting.** Insole forces, not force plates; availability unknown |
| **Bertaux et al. 2022**, *Sci Data* 9:399 | 3D GRF, AMTI force plates, before and 6 months after total hip arthroplasty | No pain or function scores; observer-coded gait disturbance categories only | 106 patients, 80 controls | figshare (file licence to confirm) | Not suitable for Capability 2; possible external check of Capability 1 monitoring rules (two timepoints only) |
| **Keilhauer et al. 2026**, *Sci Data* 13:1151 | Markerless motion capture, before and 6 weeks after knee arthroplasty | States that no pain, disability or function measures are included | 23 patients | Zenodo | Not suitable |

## Conclusion

Capability 2 cannot produce results yet. v3.0 therefore delivers the clinical model family as validated, gated
infrastructure: data contracts, a verification gate that refuses unverified or synthetic data for real analyses,
leakage-safe linkage, the Model A/B/C comparison and its evaluation, tested on clearly labelled synthetic data.

## Next steps toward real data (in order of feasibility)

1. Read the MOST data dictionary; if gait and outcome visits align, submit an NIA biobank request (requires an
   institutional affiliation and data-use agreement).
2. Contact the Warmerdam et al. authors about data sharing under an agreement.
3. Longer term: a prospective workers' compensation rehabilitation study with force or insole measurements,
   validated function and pain scales, and return-to-work dates (see `docs/v3_capability2_protocol.md`).

Sources: Horsak et al. 2020; NIA Aging Research Biobank MOST study page and BU MOST table of measures; Sara et al.
2025; Warmerdam et al. 2025; Bertaux et al. 2022; Keilhauer et al. 2026.
