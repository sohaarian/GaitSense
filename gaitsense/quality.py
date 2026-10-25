"""Step 19: sensor-quality and malfunction detection. Gate analysis on these before computing metrics."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List
import numpy as np

CRITICAL = {"empty", "non_finite", "non_monotonic_time", "data_gap", "saturation", "flatline", "too_few_steps"}


@dataclass
class QualityReport:
    flags: List[str] = field(default_factory=list)
    detail: dict = field(default_factory=dict)

    @property
    def usable(self) -> bool:
        return not (set(self.flags) & CRITICAL)


def check_recording(t, fx, fz, fs_nominal: float, fz_full_scale: float, fx_full_scale: float,
                    raw_x=None, raw_z=None, min_active_n: float = 20.0) -> QualityReport:
    """fx, fz are CALIBRATED forces in N. raw_x/raw_z (optional) are the uncalibrated channel outputs; the flatline
    check should use them because cross-axis terms make a dead channel look alive after calibration."""
    q = QualityReport()
    t, fx, fz = np.asarray(t, float), np.asarray(fx, float), np.asarray(fz, float)
    if len(t) < 10:
        q.flags.append("empty"); return q
    if not (np.all(np.isfinite(t)) and np.all(np.isfinite(fx)) and np.all(np.isfinite(fz))):
        q.flags.append("non_finite"); return q
    dt = np.diff(t)
    if np.any(dt <= 0):
        q.flags.append("non_monotonic_time"); return q
    med = float(np.median(dt))
    q.detail["effective_fs"] = 1 / med
    if abs(1 / med - fs_nominal) > 0.05 * fs_nominal:
        q.flags.append("sample_rate_mismatch")
    gap = float(dt.max())
    q.detail["max_gap_s"] = gap
    if gap > 2.5 * med and gap > 0.1:     # >100 ms missing would corrupt a stance phase
        q.flags.append("data_gap")
    for name, v, fs in (("z", fz, fz_full_scale), ("x", fx, fx_full_scale)):
        av = np.abs(v)
        frac = float(np.mean(av >= 0.98 * fs))
        # clipping also shows up as a flat top: a run of consecutive samples tied at the maximum
        at_max = av >= av.max() - 1e-6 * max(av.max(), 1.0)
        tied, run_ = 0, 0
        for flag_ in at_max:
            run_ = run_ + 1 if flag_ else 0
            tied = max(tied, run_)
        q.detail[f"saturated_frac_{name}"] = frac
        q.detail[f"max_tied_run_{name}"] = tied
        if frac > 0.002 or tied >= 5:
            q.flags.append("saturation")
    # flatline: a raw channel with exactly zero variance over a 1 s window in which the load is changing
    w = max(int(fs_nominal), 5)
    for name, v in (("x", raw_x), ("z", raw_z)):
        if v is None:
            continue
        v = np.asarray(v, float)
        for a in range(0, len(v) - w, w):
            if np.ptp(v[a:a + w]) == 0 and np.ptp(fz[a:a + w]) > min_active_n:
                q.flags.append("flatline"); q.detail["flatline_channel"] = name; break
    swing = np.abs(fz) <= min_active_n
    if swing.sum() > 10:
        q.detail["swing_noise_fz_N"] = float(np.std(fz[swing]))
    return q


def check_steps(steps: list, min_steps: int = 8, max_peak_cv: float = 0.25) -> QualityReport:
    q = QualityReport()
    q.detail["n_steps"] = len(steps)
    if len(steps) < min_steps:
        q.flags.append("too_few_steps"); return q
    pk = np.array([s["peak_fz"] for s in steps])
    cv = float(pk.std(ddof=1) / pk.mean())
    q.detail["peak_fz_cv"] = cv
    if cv > max_peak_cv:
        q.flags.append("unstable_steps")
    return q
