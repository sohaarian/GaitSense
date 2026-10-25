"""Data contracts shared by both capabilities. Every table entering the platform is validated against one of these.

* Biomechanical visits (Capability 1): one row per patient visit with a target value, its within-visit SE and a
  visit day.
* Clinical outcomes (Capability 2): one row per outcome assessment, measured independently of the biomechanics.
* Clinical covariates (Capability 2): baseline information with the day it became available.
* Dataset manifest: provenance and permission record required by the verification gate.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields

import numpy as np
import pandas as pd

BIOMECH_COLUMNS = {"patient_id": "id", "visit_day": "number", "value": "number", "value_se": "number"}
OUTCOME_COLUMNS = {"patient_id": "id", "outcome": "text", "instrument": "text", "value": "number",
                   "assessment_day": "number", "independent_of_biomechanics": "bool"}
COVARIATE_COLUMNS = {"patient_id": "id", "available_day": "number"}


class ContractError(ValueError):
    """A table does not meet its data contract."""


def _check(df: pd.DataFrame, spec: dict, name: str):
    missing = [c for c in spec if c not in df.columns]
    if missing:
        raise ContractError(f"{name}: missing columns {missing}")
    for c, kind in spec.items():
        s = df[c]
        if s.isna().any():
            raise ContractError(f"{name}: column '{c}' has {int(s.isna().sum())} missing values")
        if kind == "number" and not pd.api.types.is_numeric_dtype(s):
            raise ContractError(f"{name}: column '{c}' must be numeric")
        if kind == "number" and not np.isfinite(s.to_numpy(float)).all():
            raise ContractError(f"{name}: column '{c}' has non-finite values")
        if kind == "bool" and not pd.api.types.is_bool_dtype(s):
            raise ContractError(f"{name}: column '{c}' must be boolean")


def validate_biomech(df: pd.DataFrame) -> pd.DataFrame:
    _check(df, BIOMECH_COLUMNS, "biomechanical visits")
    if (df.value_se < 0).any():
        raise ContractError("biomechanical visits: value_se must be non-negative")
    if df.duplicated(["patient_id", "visit_day", "visit_order"] if "visit_order" in df else ["patient_id", "visit_day"]).any():
        raise ContractError("biomechanical visits: duplicate patient visits (add a 'visit_order' column for same-day visits)")
    return df


def validate_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    _check(df, OUTCOME_COLUMNS, "clinical outcomes")
    if not df.independent_of_biomechanics.all():
        raise ContractError("clinical outcomes: every outcome must be measured independently of the biomechanics; "
                            "derived-from-gait labels cannot serve as clinical outcomes")
    return df


def validate_covariates(df: pd.DataFrame) -> pd.DataFrame:
    _check(df, COVARIATE_COLUMNS, "clinical covariates")
    if df.patient_id.duplicated().any():
        raise ContractError("clinical covariates: one row per patient")
    return df


@dataclass
class DatasetManifest:
    """Provenance record. A real (non-synthetic) analysis requires every field to be filled."""
    name: str
    source_url: str
    citation: str
    licence_or_agreement: str
    access_type: str                      # "open" | "controlled"
    ethics_or_dua_reference: str
    obtained_by: str
    obtained_on: str
    verified_by: str
    verified_on: str
    is_synthetic: bool
    file_sha256: dict = field(default_factory=dict)
    notes: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "DatasetManifest":
        allowed = {f.name for f in fields(cls)}
        unknown = set(d) - allowed
        if unknown:
            raise ContractError(f"manifest: unknown fields {sorted(unknown)}")
        return cls(**d)
