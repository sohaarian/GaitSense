"""Mathematical perturbations and injected changes applied to raw stance-phase trials, and visit recomputation.

These are controlled transformations of recorded curves. They are NOT claimed to reproduce sensor displacement,
footwear, walking-speed or clinical changes.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline

from .features import FS, trial_features


# ---- perturbations of one trial (z: vertical N, a: AP N, bw: N) -------------------------------------------------
def offset(z, a, bw, k):          # add k body weights of force to the vertical channel
    return z + k * bw, a


def scale(z, a, bw, f):           # multiply both force channels by f
    return z * f, a * f


def trim(z, a, bw, m):            # remove m samples at each end of stance (boundary moved inward)
    return (z[m:-m], a[m:-m]) if m > 0 and z.size > 2 * m + 10 else (z, a)


# ---- injected changes --------------------------------------------------------------------------------------------
def inj_vertical(z, a, bw, d):
    return z * (1 + d), a


def inj_propulsive(z, a, bw, d):  # positive AP scaled by (1 - d)
    return z, np.where(a > 0, a * (1 - d), a)


def inj_braking(z, a, bw, d):     # negative AP scaled by (1 + d): larger braking impulse
    return z, np.where(a < 0, a * (1 + d), a)


def inj_peak1(z, a, bw, d):       # local reduction around the first vertical peak (Gaussian window, SD 10% stance)
    n = z.size; i = int(np.argmax(z[: n // 2])); x = np.arange(n)
    w = np.exp(-0.5 * ((x - i) / (0.10 * n)) ** 2)
    return z * (1 - d * w), a


# ---- feature recomputation ---------------------------------------------------------------------------------------
def resampled_features(z, a, bw, m=101, kind="linear"):
    """Features computed from a time-normalised curve instead of the raw samples."""
    n = z.size; dur = n / FS; t = np.linspace(0, 1, n); g = np.linspace(0, 1, m)
    if kind == "cubic":
        zz, aa = CubicSpline(t, z)(g), CubicSpline(t, a)(g)
    else:
        zz, aa = np.interp(g, t, z), np.interp(g, t, a)
    dt = dur / (m - 1)
    return {"peak_v": zz.max() / bw, "braking": -np.trapezoid(np.minimum(aa, 0), dx=dt) / bw,
            "propulsive": np.trapezoid(np.maximum(aa, 0), dx=dt) / bw, "stance_s": dur}


def visit_summary(trials_a, trials_u, bw, feats=("peak_v", "braking", "propulsive", "stance_s"),
                  fn_a=None, fn_u=None, featfn=None):
    """Visit-level affected/unaffected means, difference and SE after optional per-trial transforms.

    trials_*: lists of (z, a) raw arrays. fn_*: callable (z, a, bw) -> (z, a) or None. featfn: alternative feature
    function (z, a, bw) -> dict (e.g. resampled_features)."""
    def leg(trials, fn):
        F = []
        for z, a in trials:
            if fn is not None:
                z, a = fn(z, a, bw)
            F.append(featfn(z, a, bw) if featfn else trial_features(z, a, bw))
        return {f: np.array([x[f] for x in F], float) for f in feats}
    A, U = leg(trials_a, fn_a), leg(trials_u, fn_u)
    out = {}
    for f in feats:
        ma, mu = A[f].mean(), U[f].mean()
        out[f"{f}_a"], out[f"{f}_u"] = ma, mu
        out[f"d_{f}"] = ma - mu
        out[f"d_{f}_se"] = np.sqrt(A[f].var(ddof=1) / A[f].size + U[f].var(ddof=1) / U[f].size)
        out[f"{f}_a_se"] = np.sqrt(A[f].var(ddof=1) / A[f].size)
    return out
