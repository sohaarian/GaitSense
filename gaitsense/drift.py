"""Drift correction that cannot absorb genuine gait change.

Principle: drift is estimated ONLY from readings of a KNOWN reference load (the built-in test load, or a bench
load), never from gait data. Gait recordings are then divided by the modelled sensitivity ratio. Because the model has
no access to gait values, a real change in push-off cannot be 'corrected away'.

Model per axis:  g(t, T) = c0 + a*(T - T_ref) + b*t,   g = estimated force / true reference force, using the
stored calibration. A term is applied only if its 95% CI excludes zero (so noise is not fitted).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import numpy as np

MIN_CHECKS = 6


@dataclass
class AxisDrift:
    c0: float; a_per_C: float; b_per_h: float
    use_temp: bool; use_time: bool
    se: List[float]
    t_range_h: List[float]; temp_range_c: List[float]; T_ref: float

    def gain(self, t_h: float, temp_c: Optional[float]) -> float:
        g = self.c0
        if self.use_temp and temp_c is not None:
            g += self.a_per_C * (temp_c - self.T_ref)
        if self.use_time and t_h is not None:
            g += self.b_per_h * t_h
        return g

    def extrapolating(self, t_h, temp_c) -> bool:
        out = False
        if self.use_time and t_h is not None:
            out |= not (self.t_range_h[0] - 1e-9 <= t_h <= self.t_range_h[1] + 1e-9)
        if self.use_temp and temp_c is not None:
            out |= not (self.temp_range_c[0] - 1e-9 <= temp_c <= self.temp_range_c[1] + 1e-9)
        return bool(out)


@dataclass
class DriftModel:
    x: AxisDrift
    z: AxisDrift


def _fit_axis(t_h, temp_c, ratio, T_ref) -> AxisDrift:
    t_h, temp_c, ratio = (np.asarray(v, float) for v in (t_h, temp_c, ratio))
    n = len(ratio)
    X = np.column_stack([np.ones(n), temp_c - T_ref, t_h])
    if np.linalg.matrix_rank(X) < 3:
        raise ValueError("reference checks do not vary independently in time and temperature")
    beta, *_ = np.linalg.lstsq(X, ratio, rcond=None)
    resid = ratio - X @ beta
    s2 = float(resid @ resid) / (n - 3)
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    tcrit = 2.0 + 6.0 / max(n - 3, 1)         # conservative t(0.975) approximation for small df
    return AxisDrift(c0=float(beta[0]), a_per_C=float(beta[1]), b_per_h=float(beta[2]),
                     use_temp=bool(abs(beta[1]) > tcrit * se[1]), use_time=bool(abs(beta[2]) > tcrit * se[2]),
                     se=[float(v) for v in se], t_range_h=[float(t_h.min()), float(t_h.max())],
                     temp_range_c=[float(temp_c.min()), float(temp_c.max())], T_ref=float(T_ref))


def fit_drift(t_h, temp_c, ratio_x, ratio_z, T_ref: float = 25.0) -> DriftModel:
    """Inputs are reference-load checks only: time (h), temperature (C), and the measured/true force ratio per axis
    (1.0 = no drift). Needs >= MIN_CHECKS checks spread over both time and temperature."""
    if len(t_h) < MIN_CHECKS:
        raise ValueError(f"need at least {MIN_CHECKS} reference checks")
    ok = np.all(np.isfinite([t_h, temp_c, ratio_x, ratio_z]), axis=0)
    if not ok.all():
        raise ValueError("non-finite reference check values")
    return DriftModel(_fit_axis(t_h, temp_c, ratio_x, T_ref), _fit_axis(t_h, temp_c, ratio_z, T_ref))


def correct_forces(fx, fz, model: DriftModel, t_h: Optional[float], temp_c: Optional[float]):
    """Divide calibrated forces by the modelled sensitivity ratio. Returns (fx, fz, warnings)."""
    warns = []
    for name, ax in (("x", model.x), ("z", model.z)):
        if (ax.use_time and t_h is None) or (ax.use_temp and temp_c is None):
            warns.append(f"drift model for {name} needs time/temperature that the recording lacks; not corrected")
        if ax.extrapolating(t_h, temp_c):
            warns.append(f"recording outside the range of reference checks for {name}; correction is an extrapolation")
    if any("not corrected" in w for w in warns):
        return np.asarray(fx, float), np.asarray(fz, float), warns
    gx, gz = model.x.gain(t_h, temp_c), model.z.gain(t_h, temp_c)
    if not (0.5 < gx < 2 and 0.5 < gz < 2):
        warns.append("modelled gain implausible (<0.5 or >2); not corrected")
        return np.asarray(fx, float), np.asarray(fz, float), warns
    return np.asarray(fx, float) / gx, np.asarray(fz, float) / gz, warns
