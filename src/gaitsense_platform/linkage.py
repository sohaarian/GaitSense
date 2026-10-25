"""Linking biomechanical and clinical data at the patient level with a landmark design that prevents leakage.

At a landmark day L (e.g. 6 weeks after injury):
* predictors use only biomechanical visits and clinical information available on or before L;
* the outcome is the first qualifying assessment after L + min_gap_days, up to the outcome window;
* patients whose outcome was assessed before that point are excluded, never back-filled.
"""
from __future__ import annotations

import pandas as pd

from .contracts import validate_covariates, validate_outcomes


def outcome_at(outcomes: pd.DataFrame, name: str, landmark_day: float, min_gap_days: float, window_end_day: float):
    """One outcome value per patient: the first assessment of `name` in (landmark + gap, window_end]."""
    validate_outcomes(outcomes)
    o = outcomes[(outcomes.outcome == name) & (outcomes.assessment_day > landmark_day + min_gap_days)
                 & (outcomes.assessment_day <= window_end_day)]
    return o.sort_values("assessment_day").groupby("patient_id").first()[["value", "assessment_day"]].reset_index()


def clinical_at(covariates: pd.DataFrame, landmark_day: float) -> pd.DataFrame:
    """Covariates available by the landmark (rows with later availability are dropped)."""
    validate_covariates(covariates)
    return covariates[covariates.available_day <= landmark_day].drop(columns="available_day")


def build_analysis_table(bio_features: pd.DataFrame, covariates: pd.DataFrame, outcome: pd.DataFrame,
                         landmark_day: float) -> tuple[pd.DataFrame, dict]:
    """Patient-level table for Models A/B/C with a linkage report. Only patients with all three parts are kept."""
    cov = clinical_at(covariates, landmark_day)
    t = cov.merge(bio_features, on="patient_id").merge(outcome.rename(columns={"value": "outcome"}), on="patient_id")
    if (t.assessment_day <= landmark_day).any():
        raise AssertionError("outcome assessed on or before the landmark: leakage")
    report = {"patients_with_covariates": int(cov.patient_id.nunique()),
              "patients_with_biomechanics": int(bio_features.patient_id.nunique()),
              "patients_with_outcome": int(outcome.patient_id.nunique()), "linked_patients": int(len(t))}
    return t, report
