"""Integrated pipeline: Capability 1 features -> linkage -> Capability 2 Models A/B/C. Gate first, always."""
from __future__ import annotations

from dataclasses import dataclass, asdict

import pandas as pd

from . import clinical, gate, linkage
from .capability1 import BiomechForecaster
from .contracts import validate_biomech


@dataclass
class Config:
    outcome: str
    landmark_day: float
    horizon_day: float
    min_gap_days: float
    window_end_day: float
    binary: bool
    clinical_cols: tuple
    kernel: str = "matern32_lin"
    folds: int = 5
    seed: int = 20261024


def run(manifest, visits: pd.DataFrame, covariates: pd.DataFrame, outcomes: pd.DataFrame, cfg: Config,
        root=".", allow_synthetic: bool = False) -> dict:
    g = gate.check(manifest, root=root, allow_synthetic=allow_synthetic)     # raises unless verified
    validate_biomech(visits)
    pre = visits[visits.visit_day <= cfg.landmark_day]                         # nothing after the landmark is used
    fc = BiomechForecaster(cfg.kernel).fit(pre, seed=cfg.seed)
    bio = fc.landmark_features(pre, cfg.landmark_day, cfg.horizon_day)
    out = linkage.outcome_at(outcomes, cfg.outcome, cfg.landmark_day, cfg.min_gap_days, cfg.window_end_day)
    table, report = linkage.build_analysis_table(bio, covariates, out, cfg.landmark_day)
    bio_cols = [c for c in bio.columns if c != "patient_id"]
    res = clinical.compare_models(table, cfg.clinical_cols, bio_cols, cfg.binary, cfg.folds, cfg.seed)
    res.pop("oof_predictions")
    return {"gate": g, "config": asdict(cfg), "linkage": report, "biomech_hyperparameters": fc.model.hyperparameters(),
            "results": res}
