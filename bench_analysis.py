"""GaitSense bench-analysis: verification statistics for a two-channel (vertical Fz, fore-aft Fx)
wearable force sensor, computed from physical calibration-rig measurements.

STATUS: tested on SIMULATED inputs only. No physical sensor has been built or measured.
Passing tests show the arithmetic is right, not that any device works.

Inputs are CSV files with a reference instrument (applied load) and the prototype's raw outputs.
Forces are in newtons (N) unless a column says otherwise.

Calibration file columns
    trial, ref_fx, ref_fz, raw_x, raw_z            (required)
    branch   -> "up" / "down" (loading / unloading)   (needed for hysteresis)
    time_h   -> hours since start                     (needed for drift)
    mount    -> integer id of each remount            (needed for remount repeatability)
    session  -> integer id of each test day           (needed for between-day stability)
    temp_c   -> temperature                           (optional, reported only)

Usage
    python bench_analysis.py calibrate  train.csv test.csv  --fs-x 400 --fs-z 1500 [--out out.json]
    python bench_analysis.py repeat     remount.csv         --fs-x 400
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence

import numpy as np

REQUIRED = ("ref_fx", "ref_fz", "raw_x", "raw_z")


# ----------------------------------------------------------------------------- I/O
def load_csv(path: str) -> Dict[str, np.ndarray]:
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"{path}: no data rows")
    cols: Dict[str, np.ndarray] = {}
    for key in rows[0]:
        vals = [r[key] for r in rows]
        try:
            cols[key] = np.array([float(v) for v in vals])
        except ValueError:
            cols[key] = np.array(vals)
    missing = [c for c in REQUIRED if c not in cols]
    if missing:
        raise ValueError(f"{path}: missing required columns {missing}")
    for c in REQUIRED:
        if not np.all(np.isfinite(cols[c])):
            raise ValueError(f"{path}: non-finite values in column {c}")
    return cols


# ----------------------------------------------------------------------------- calibration
@dataclass
class Calibration:
    """raw = S @ [fx, fz] + offset  (S[0,1] and S[1,0] are the cross-axis terms).
    Inverse: [fx, fz] = inv(S) @ (raw - offset)."""
    S: List[List[float]]
    offset: List[float]
    cond: float

    def predict(self, raw_x, raw_z):
        S = np.array(self.S)
        raw = np.column_stack([np.asarray(raw_x, float), np.asarray(raw_z, float)]) - np.array(self.offset)
        f = raw @ np.linalg.inv(S).T
        return f[:, 0], f[:, 1]

    @property
    def cross_axis_pct(self) -> Dict[str, float]:
        """Cross-axis sensitivity as % of the main-axis sensitivity of the *responding* channel."""
        S = np.array(self.S)
        return {"fz_into_x_channel": 100 * abs(S[0, 1] / S[0, 0]),
                "fx_into_z_channel": 100 * abs(S[1, 0] / S[1, 1])}


def fit_calibration(ref_fx, ref_fz, raw_x, raw_z) -> Calibration:
    """Least-squares fit of the full 2x2 sensitivity matrix plus offsets.
    Needs combined loading (fx and fz varied independently); otherwise cross terms are unidentifiable."""
    ref_fx, ref_fz, raw_x, raw_z = (np.asarray(a, float) for a in (ref_fx, ref_fz, raw_x, raw_z))
    if len(ref_fx) < 6:
        raise ValueError("need at least 6 calibration points")
    A = np.column_stack([ref_fx, ref_fz, np.ones_like(ref_fx)])
    if np.linalg.matrix_rank(A) < 3:
        raise ValueError("calibration loads do not vary independently in Fx and Fz; cross-axis terms unidentifiable")
    cx, *_ = np.linalg.lstsq(A, raw_x, rcond=None)
    cz, *_ = np.linalg.lstsq(A, raw_z, rcond=None)
    S = [[float(cx[0]), float(cx[1])], [float(cz[0]), float(cz[1])]]
    cond = float(np.linalg.cond(np.array(S)))
    if not np.isfinite(cond) or cond > 1e6:
        raise ValueError("sensitivity matrix is singular")
    return Calibration(S=S, offset=[float(cx[2]), float(cz[2])], cond=cond)


def _err_stats(err: np.ndarray, full_scale: float) -> Dict[str, float]:
    return {"rmse_N": float(np.sqrt(np.mean(err ** 2))),
            "max_abs_N": float(np.max(np.abs(err))),
            "bias_N": float(np.mean(err)),
            "rmse_pct_fs": float(100 * np.sqrt(np.mean(err ** 2)) / full_scale),
            "max_pct_fs": float(100 * np.max(np.abs(err)) / full_scale)}


def validate(cal: Calibration, ref_fx, ref_fz, raw_x, raw_z, fs_x: float, fs_z: float) -> Dict[str, Dict[str, float]]:
    """Independent validation: apply a calibration fitted on one data set to a different one."""
    px, pz = cal.predict(raw_x, raw_z)
    return {"fx": _err_stats(px - np.asarray(ref_fx, float), fs_x),
            "fz": _err_stats(pz - np.asarray(ref_fz, float), fs_z)}


def naive_validate(ref_fx, ref_fz, raw_x, raw_z, train, fs_x, fs_z):
    """Comparator: diagonal-only calibration (ignores cross-axis coupling), as in a vertical-only workflow.
    Shows how much the cross-axis terms matter. `train` is a dict of calibration columns."""
    ax = np.polyfit(train["ref_fx"], train["raw_x"], 1)
    az = np.polyfit(train["ref_fz"], train["raw_z"], 1)
    px = (np.asarray(raw_x, float) - ax[1]) / ax[0]
    pz = (np.asarray(raw_z, float) - az[1]) / az[0]
    return {"fx": _err_stats(px - np.asarray(ref_fx, float), fs_x),
            "fz": _err_stats(pz - np.asarray(ref_fz, float), fs_z)}


# ----------------------------------------------------------------------------- hysteresis
def hysteresis_pct_fs(ref: np.ndarray, out: np.ndarray, branch: np.ndarray, full_scale: float,
                      n_bins: int = 10) -> float:
    """Max gap between loading and unloading branches at matched loads, in % of full scale (output in N).
    Matched by interpolating each branch (sorted by load) onto a common load grid."""
    ref, out, branch = np.asarray(ref, float), np.asarray(out, float), np.asarray(branch).astype(str)
    up, dn = branch == "up", branch == "down"
    if up.sum() < 3 or dn.sum() < 3:
        raise ValueError("need >=3 points on each of the up and down branches")
    lo = max(ref[up].min(), ref[dn].min()); hi = min(ref[up].max(), ref[dn].max())
    if hi <= lo:
        raise ValueError("up and down branches do not overlap in load")
    grid = np.linspace(lo, hi, n_bins + 1)
    ou, od = np.argsort(ref[up]), np.argsort(ref[dn])
    yu = np.interp(grid, ref[up][ou], out[up][ou])
    yd = np.interp(grid, ref[dn][od], out[dn][od])
    return float(100 * np.max(np.abs(yu - yd)) / full_scale)


# ----------------------------------------------------------------------------- drift
def drift_rate(time_h: np.ndarray, force_est: np.ndarray, full_scale: float) -> Dict[str, float]:
    """Linear drift of the estimated force under a held constant reference load.
    Returns slope (N/h), % FS per hour, and a 95% CI on the slope (t-based)."""
    t, y = np.asarray(time_h, float), np.asarray(force_est, float)
    n = len(t)
    if n < 5 or np.ptp(t) <= 0:
        raise ValueError("need >=5 points spanning a nonzero time")
    X = np.column_stack([t, np.ones(n)])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    s2 = resid @ resid / (n - 2)
    se = np.sqrt(s2 / np.sum((t - t.mean()) ** 2))
    tcrit = _t975(n - 2)
    return {"slope_N_per_h": float(beta[0]), "slope_pct_fs_per_h": float(100 * beta[0] / full_scale),
            "ci95_N_per_h": [float(beta[0] - tcrit * se), float(beta[0] + tcrit * se)],
            "ci_excludes_zero": bool(abs(beta[0]) > tcrit * se)}


def _t975(df: int) -> float:
    table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262,
             10: 2.228, 12: 2.179, 15: 2.131, 20: 2.086, 30: 2.042, 60: 2.000}
    if df in table:
        return table[df]
    keys = sorted(table)
    if df > keys[-1]:
        return 1.96
    lo = max(k for k in keys if k < df); hi = min(k for k in keys if k > df)
    return table[lo] + (table[hi] - table[lo]) * (df - lo) / (hi - lo)


# ----------------------------------------------------------------------------- repeatability / remount
def repeatability(values: np.ndarray, groups: np.ndarray) -> Dict[str, float]:
    """One-way random-effects decomposition of repeated readings of the SAME applied load.
    `groups` identifies the remount (or test day). Returns within-group SD, between-group SD,
    ICC(1,1), SEM and the minimal detectable change (MDC95 = 1.96*sqrt(2)*SEM) for a single
    measurement compared across two occasions."""
    v, g = np.asarray(values, float), np.asarray(groups)
    ids = np.unique(g)
    k = len(ids)
    if k < 3:
        raise ValueError("need >=3 groups (remounts/days)")
    n_i = np.array([np.sum(g == i) for i in ids])
    if np.any(n_i < 2):
        raise ValueError("each group needs >=2 readings")
    means = np.array([v[g == i].mean() for i in ids])
    N = n_i.sum()
    grand = v.mean()
    ssb = np.sum(n_i * (means - grand) ** 2)
    ssw = sum(np.sum((v[g == i] - m) ** 2) for i, m in zip(ids, means))
    msb, msw = ssb / (k - 1), ssw / (N - k)
    n0 = (N - np.sum(n_i ** 2) / N) / (k - 1)
    var_b = max((msb - msw) / n0, 0.0)
    icc = var_b / (var_b + msw) if (var_b + msw) > 0 else float("nan")
    sem = float(np.sqrt(msw + var_b))  # total error if comparing across occasions without remount correction
    return {"within_sd": float(np.sqrt(msw)), "between_sd": float(np.sqrt(var_b)), "icc": float(icc),
            "sem_total": sem, "mdc95_across_occasions": float(1.96 * np.sqrt(2) * sem),
            "mean": float(grand), "n_groups": int(k), "n_readings": int(N)}


# ----------------------------------------------------------------------------- agreement
def bland_altman(estimate: np.ndarray, reference: np.ndarray) -> Dict[str, float]:
    d = np.asarray(estimate, float) - np.asarray(reference, float)
    m, s = float(d.mean()), float(d.std(ddof=1))
    return {"bias": m, "sd_diff": s, "loa_low": m - 1.96 * s, "loa_high": m + 1.96 * s, "n": int(len(d))}


# ----------------------------------------------------------------------------- reference-check (self-calibration)
def reference_check(cal: Calibration, raw_x_at_ref: float, raw_z_at_ref: float,
                    ref_fx: float, ref_fz: float, tol_pct: float = 3.0) -> Dict[str, object]:
    """Compare a visit's reading under the built-in KNOWN test load with the stored calibration.
    Flags the sensor as 'needs recalibration' when either axis deviates by more than tol_pct of the
    applied reference load. tol_pct must come from the bench repeatability study, not be assumed."""
    px, pz = cal.predict([raw_x_at_ref], [raw_z_at_ref])
    dev_x = float(100 * (px[0] - ref_fx) / ref_fx) if ref_fx else float("nan")
    dev_z = float(100 * (pz[0] - ref_fz) / ref_fz) if ref_fz else float("nan")
    ok = bool(abs(dev_x) <= tol_pct and abs(dev_z) <= tol_pct)
    return {"dev_fx_pct": dev_x, "dev_fz_pct": dev_z, "tol_pct": tol_pct, "within_tolerance": ok}


# ----------------------------------------------------------------------------- simulator (for tests / dry runs)
def simulate_bench(n: int = 80, seed: int = 0, S=((2.0e-3, 1.5e-4), (3.0e-4, 1.0e-3)),
                   offset=(0.02, -0.01), noise=0.002, fs_x=300.0, fs_z=1200.0) -> Dict[str, np.ndarray]:
    """Synthetic rig data from a known ground-truth sensor. NOT real measurements."""
    r = np.random.default_rng(seed)
    fx = r.uniform(-fs_x, fs_x, n)
    fz = r.uniform(0, fs_z, n)
    S = np.array(S)
    raw = np.column_stack([fx, fz]) @ S.T + np.array(offset)
    raw += r.normal(0, noise, raw.shape)
    return {"trial": np.arange(n, dtype=float), "ref_fx": fx, "ref_fz": fz,
            "raw_x": raw[:, 0], "raw_z": raw[:, 1]}


# ----------------------------------------------------------------------------- CLI
def _cmd_calibrate(a) -> Dict[str, object]:
    tr, te = load_csv(a.train), load_csv(a.test)
    cal = fit_calibration(tr["ref_fx"], tr["ref_fz"], tr["raw_x"], tr["raw_z"])
    out = {"status": "bench analysis only; not clinical evidence",
           "calibration": asdict(cal), "cross_axis_pct": cal.cross_axis_pct,
           "validation_full": validate(cal, te["ref_fx"], te["ref_fz"], te["raw_x"], te["raw_z"], a.fs_x, a.fs_z),
           "validation_diagonal_only": naive_validate(te["ref_fx"], te["ref_fz"], te["raw_x"], te["raw_z"],
                                                      tr, a.fs_x, a.fs_z)}
    if "branch" in te:
        px, pz = cal.predict(te["raw_x"], te["raw_z"])
        for ax, ref, est, fs in (("fx", te["ref_fx"], px, a.fs_x), ("fz", te["ref_fz"], pz, a.fs_z)):
            try:
                out.setdefault("hysteresis_pct_fs", {})[ax] = hysteresis_pct_fs(ref, est, te["branch"], fs)
            except ValueError as e:
                out.setdefault("hysteresis_pct_fs", {})[ax] = f"not computed: {e}"
    return out


def _cmd_repeat(a) -> Dict[str, object]:
    d = load_csv(a.data)
    if "mount" not in d:
        raise ValueError("remount.csv needs a 'mount' column")
    return {"status": "bench analysis only; not clinical evidence",
            "fx_repeatability": repeatability(d["raw_x"], d["mount"]),
            "fz_repeatability": repeatability(d["raw_z"], d["mount"])}


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("calibrate"); c.add_argument("train"); c.add_argument("test")
    c.add_argument("--fs-x", type=float, required=True, help="full-scale Fx (N)")
    c.add_argument("--fs-z", type=float, required=True, help="full-scale Fz (N)")
    c.add_argument("--out")
    r = sub.add_parser("repeat"); r.add_argument("data"); r.add_argument("--out")
    a = p.parse_args(argv)
    try:
        res = _cmd_calibrate(a) if a.cmd == "calibrate" else _cmd_repeat(a)
    except (ValueError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    txt = json.dumps(res, indent=2)
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(txt + "\n")
    print(txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
