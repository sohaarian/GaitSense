"""Capability 1: biomechanical forecasting and monitoring.

Two entry points:
* `load_v2(results_dir)` gives the frozen, validated GaitRec v2.0 models exactly as evaluated (no refitting).
* `BiomechForecaster` applies the same longitudinal Gaussian model family to any dataset that meets the
  biomechanical contract, and exports landmark features for Capability 2 using only visits up to the landmark.

`ConformalCalibrator` is an optional interval recalibration (v3.0 addition; see docs/v3_capability1_update.md).
"""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from gaitsense_ai.longitudinal import MONTH, LongitudinalGP, PatientData

from .contracts import validate_biomech


def load_v2(results_dir) -> dict:
    """Frozen v2.0 artefacts: fitted models, selection record, and the covariate scaler (read-only)."""
    d = Path(results_dir)
    with open(d / "models.pkl", "rb") as f:
        pk = pickle.load(f)
    return {"fitted": pk["fitted"], "standardiser": pk["std"],
            "selection": json.loads((d / "model_selection.json").read_text())}


@dataclass
class Forecast:
    mean: float
    latent_var: float        # uncertainty about the patient's underlying trajectory value
    extra_var: float         # visit-level variation not represented by the kernel (not pure measurement error)
    measurement_var: float   # within-visit measurement variance assumed for the forecast visit

    @property
    def sd(self) -> float:
        return float(np.sqrt(self.latent_var + self.extra_var + self.measurement_var))

    def interval(self, level=0.95, calibrator: "ConformalCalibrator | None" = None):
        q = calibrator.quantile(level) if calibrator else norm.ppf(0.5 + level / 2)
        return self.mean - q * self.sd, self.mean + q * self.sd


class BiomechForecaster:
    """Longitudinal GP forecaster for contract-format visit tables (patient_id, visit_day, value, value_se, ...)."""

    def __init__(self, kernel: str = "matern32_lin", covariates: tuple = ()):
        self.kernel, self.covariates = kernel, tuple(covariates)

    def _design(self, p: pd.DataFrame):
        t = p.visit_day.to_numpy(float) / MONTH
        cov = [p[c].iloc[0] for c in self.covariates]
        X = np.array([[1.0, *[(v - self.cov_mean[i]) / self.cov_sd[i] for i, v in enumerate(cov)], ti] for ti in t])
        return t, p.value.to_numpy(float), p.value_se.to_numpy(float) ** 2, X

    def fit(self, visits: pd.DataFrame, restarts: int = 2, seed: int = 0):
        validate_biomech(visits)
        base = visits.drop_duplicates("patient_id")
        self.cov_mean = [float(base[c].mean()) for c in self.covariates]
        self.cov_sd = [float(base[c].std()) or 1.0 for c in self.covariates]
        parts = [self._design(p.sort_values("visit_day")) for _, p in visits.groupby("patient_id")]
        data = PatientData.from_lists(*zip(*parts))
        self.model = LongitudinalGP(self.kernel).fit(data, float(visits.value.std()), restarts=restarts, seed=seed)
        return self

    def forecast(self, history: pd.DataFrame, day: float) -> Forecast:
        """Forecast the measured value at `day` from one patient's history (all rows must precede `day`)."""
        h = history.sort_values("visit_day")
        if (h.visit_day > day).any():
            raise ValueError("history contains visits after the forecast day")
        t, y, se2, X = self._design(h)
        xn = X[-1].copy(); xn[-1] = day / MONTH
        mean, vl = self.model.predict_next(t, y, se2, X, day / MONTH, xn)
        return Forecast(mean, vl, self.model.sigma_w ** 2, float(se2.mean()))

    def landmark_features(self, visits: pd.DataFrame, landmark_day: float, horizon_day: float,
                          prefix: str = "bio_") -> pd.DataFrame:
        """One row per patient using ONLY visits on or before `landmark_day`.

        Features: number of visits, last value, observed slope, smoothed current level and its uncertainty, forecast
        at `horizon_day` (mean, SD) and forecast change. Patients without a visit by the landmark are omitted."""
        rows = []
        for pid, p in visits[visits.visit_day <= landmark_day].groupby("patient_id"):
            p = p.sort_values("visit_day")
            cur, fut = self.forecast(p, landmark_day), self.forecast(p, horizon_day)
            slope = np.polyfit(p.visit_day, p.value, 1)[0] * 30.44 if p.visit_day.nunique() > 1 else 0.0
            rows.append({"patient_id": pid, f"{prefix}n_visits": len(p), f"{prefix}last": float(p.value.iloc[-1]),
                         f"{prefix}slope_per_month": float(slope), f"{prefix}level_now": cur.mean,
                         f"{prefix}level_now_sd": float(np.sqrt(cur.latent_var)), f"{prefix}forecast": fut.mean,
                         f"{prefix}forecast_sd": fut.sd, f"{prefix}forecast_change": fut.mean - cur.mean})
        return pd.DataFrame(rows)


class ConformalCalibrator:
    """Split-conformal rescaling of Gaussian intervals: quantiles of |z| = |y - mean| / sd from calibration data."""

    def fit(self, y, mean, sd):
        self.absz = np.sort(np.abs((np.asarray(y) - np.asarray(mean)) / np.asarray(sd)))
        return self

    def quantile(self, level: float) -> float:
        n = self.absz.size
        k = int(np.ceil((n + 1) * level)) - 1          # finite-sample conformal rank
        return float(self.absz[min(max(k, 0), n - 1)])
