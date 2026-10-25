"""Synthetic linked data for pipeline checks ONLY. Never a source of clinical findings.

Generates patients with a latent recovery rate that drives both the biomechanical trajectory and (optionally,
via `bio_effect`) a clinical outcome. With bio_effect = 0 the outcome depends on clinical covariates only, so a
correct pipeline must find no added value from biomechanics (negative control); with bio_effect > 0 it should find
some (positive control). Every manifest produced here has is_synthetic=True.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .contracts import DatasetManifest


def make(n_patients=600, bio_effect=1.0, seed=0):
    rng = np.random.default_rng(seed)
    age = rng.normal(45, 10, n_patients); severity = rng.normal(0, 1, n_patients)
    rate = rng.normal(0, 1, n_patients)                     # latent recovery speed, unknown to the clinic
    vis = []
    for i in range(n_patients):
        days = np.sort(np.r_[0, rng.choice(np.arange(5, 43), rng.integers(1, 5), replace=False)])
        for d in days:
            level = -0.12 - 0.03 * severity[i] + 0.02 * (1 + 0.8 * rate[i]) * d / 30.44
            vis.append({"patient_id": i, "visit_day": int(d), "value": level + rng.normal(0, 0.02),
                        "value_se": 0.015})
    visits = pd.DataFrame(vis).drop_duplicates(["patient_id", "visit_day"])
    cov = pd.DataFrame({"patient_id": np.arange(n_patients), "available_day": 0, "age": age, "severity": severity})
    lin = -0.4 - 0.03 * (age - 45) - 0.8 * severity + bio_effect * rate
    good = rng.random(n_patients) < 1 / (1 + np.exp(-lin))
    outcomes = pd.DataFrame({"patient_id": np.arange(n_patients), "outcome": "meaningful_improvement",
                             "instrument": "synthetic", "value": good.astype(float),
                             "assessment_day": rng.integers(80, 100, n_patients),
                             "independent_of_biomechanics": True})
    manifest = DatasetManifest(name=f"synthetic(bio_effect={bio_effect}, seed={seed})", source_url="generated",
                               citation="none", licence_or_agreement="none", access_type="open",
                               ethics_or_dua_reference="none", obtained_by="generator", obtained_on="n/a",
                               verified_by="n/a", verified_on="n/a", is_synthetic=True)
    return visits, cov, outcomes, manifest
