"""Gait features from raw ground-reaction-force curves of one stance phase.

All forces are in newtons and are divided by body weight (N) so results are in
body weights (peak force) or body-weight-seconds (impulses).
Sign convention for the anterior-posterior force: negative = braking, positive = propulsion.
"""
from __future__ import annotations

import numpy as np

MIN_VALID_TRIALS = 3


def _integrate(y: np.ndarray, fs: float) -> float:
    return float(np.trapezoid(y, dx=1.0 / fs))


def trial_features(fz: np.ndarray, fap: np.ndarray, fs: float, body_weight_n: float) -> dict:
    """Features for one stance phase of one leg."""
    fz, fap = np.asarray(fz, float), np.asarray(fap, float)
    if fz.shape != fap.shape or fz.size < 10:
        raise ValueError("vertical and AP curves must be equal-length stance phases")
    if body_weight_n <= 0:
        raise ValueError("body weight must be positive")
    half = fz.size // 2
    return {
        "peak_loading": fz[:half].max() / body_weight_n,
        "braking_impulse": -_integrate(np.minimum(fap, 0.0), fs) / body_weight_n,
        "propulsive_impulse": _integrate(np.maximum(fap, 0.0), fs) / body_weight_n,
    }


def is_valid_trial(fz: np.ndarray, body_weight_n: float) -> bool:
    """Quality rule: no missing samples and a plausible vertical peak (> 0.5 body weight)."""
    fz = np.asarray(fz, float)
    return bool(fz.size >= 10 and np.isfinite(fz).all() and fz.max() > 0.5 * body_weight_n)


def session_difference(affected: list[dict], unaffected: list[dict]) -> dict | None:
    """Affected-minus-unaffected session means; None if either leg has too few valid trials."""
    if len(affected) < MIN_VALID_TRIALS or len(unaffected) < MIN_VALID_TRIALS:
        return None
    keys = affected[0].keys()
    return {k: float(np.mean([t[k] for t in affected]) - np.mean([t[k] for t in unaffected]))
            for k in keys}
