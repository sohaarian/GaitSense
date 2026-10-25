"""Trial-level ground-reaction-force features (one stance phase of one leg).

Verified data facts (see docs/medical_ai_baseline_audit.md, section 9):
  * GaitRec RAW curves are in newtons, sampled at 250 Hz, already cut to stance at a 25 N vertical threshold,
    and NaN-padded only at the end (no interior gaps in the raw files).
  * AP force: negative = braking, positive = propulsion.
  * Mediolateral RAW files were not available in this analysis, so no mediolateral feature is computed.

Primary features (body-weight normalised; BW = SESSION BODY_WEIGHT in newtons):
  peak_v          maximum vertical force over the whole stance / BW                    [BW]
  braking         -integral of min(AP, 0) over stance / BW                            [BW*s]
  propulsive      integral of max(AP, 0) over stance / BW                              [BW*s]
  stance_s        number of stance samples / 250 Hz                                    [s]
Secondary (curve-shape) features; NaN when the curve does not have the required shape:
  peak1, peak2    first and second vertical peaks (local maxima, 5-45% and 55-95% of stance) [BW]
  valley          minimum between the two peaks                                         [BW]
  loading_rate    mean slope between 20% and 80% of peak1 on the rising edge             [BW/s]
  vert_impulse    integral of vertical force over stance / BW                            [BW*s]
  wa_peak         maximum over the first half of stance / BW (the historical 'peak_loading' of v1.x) [BW]
"""
from __future__ import annotations

import numpy as np

FS = 250.0
PRIMARY = ["peak_v", "braking", "propulsive", "stance_s"]
SECONDARY = ["peak1", "peak2", "valley", "loading_rate", "vert_impulse", "wa_peak"]
ALL = PRIMARY + SECONDARY
GRID = np.linspace(0.0, 1.0, 101)


def _trapz(y, fs=FS):
    return float(np.trapezoid(y, dx=1.0 / fs))


def _local_max(z, lo, hi):
    """Index of the largest strict local maximum of z within [lo, hi), or None."""
    lo, hi = max(lo, 1), min(hi, z.size - 1)
    best = None
    for i in range(lo, hi):
        if z[i] >= z[i - 1] and z[i] > z[i + 1] and (best is None or z[i] > z[best]):
            best = i
    return best


def trial_features(fz, fap, body_weight_n: float, fs: float = FS) -> dict:
    """Features of one stance phase. Raises ValueError on unusable input (caller records it as invalid)."""
    fz, fap = np.asarray(fz, float), np.asarray(fap, float)
    if fz.shape != fap.shape or fz.ndim != 1 or fz.size < 10:
        raise ValueError("vertical and AP curves must be equal-length 1-D stance phases (>=10 samples)")
    if not (np.isfinite(fz).all() and np.isfinite(fap).all()):
        raise ValueError("non-finite samples inside stance")
    if not (np.isfinite(body_weight_n) and body_weight_n > 0):
        raise ValueError("body weight must be positive")
    n = fz.size
    z = fz / body_weight_n
    out = {
        "peak_v": float(z.max()),
        "braking": -_trapz(np.minimum(fap, 0.0), fs) / body_weight_n,
        "propulsive": _trapz(np.maximum(fap, 0.0), fs) / body_weight_n,
        "stance_s": n / fs,
        "vert_impulse": _trapz(fz, fs) / body_weight_n,
        "wa_peak": float(z[: n // 2].max()),
        "peak1": np.nan, "peak2": np.nan, "valley": np.nan, "loading_rate": np.nan,
    }
    i1 = _local_max(z, int(0.05 * n), int(0.45 * n) + 1)
    i2 = _local_max(z, int(0.55 * n), int(0.95 * n) + 1)
    # a 'first peak' must be the dominant early maximum, not an impact transient or ripple on the rising edge
    if i1 is not None and z[i1] < 0.9 * z[: n // 2].max():
        i1 = None
    if i2 is not None and z[i2] < 0.9 * z[n // 2:].max():
        i2 = None
    if i1 is not None:
        out["peak1"] = float(z[i1])
        rise = z[: i1 + 1]
        a = np.flatnonzero(rise >= 0.2 * z[i1]); b = np.flatnonzero(rise >= 0.8 * z[i1])
        if a.size and b.size and b[0] > a[0]:
            out["loading_rate"] = float((rise[b[0]] - rise[a[0]]) / ((b[0] - a[0]) / fs))
    if i1 is not None and i2 is not None and i2 > i1 + 1:
        v = z[i1:i2 + 1].min()
        if v < min(z[i1], z[i2]):          # a genuine valley; otherwise the curve is single-humped
            out["peak2"], out["valley"] = float(z[i2]), float(v)
    return out


def is_valid(fz, body_weight_n: float) -> tuple[bool, str]:
    """Trial validity rule (same as v1.x plus explicit reasons)."""
    fz = np.asarray(fz, float)
    if fz.size < 10:
        return False, "too_few_samples"
    if not np.isfinite(fz).all():
        return False, "non_finite"
    if not (np.isfinite(body_weight_n) and body_weight_n > 0):
        return False, "no_body_weight"
    if fz.max() <= 0.5 * body_weight_n:
        return False, "peak_below_half_bw"
    return True, ""


def normalised_curve(y, body_weight_n: float, grid=GRID):
    """Stance-time-normalised curve on `grid` (fraction of stance), in body weights, by linear interpolation."""
    y = np.asarray(y, float) / body_weight_n
    t = np.linspace(0.0, 1.0, y.size)
    return np.interp(grid, t, y)
