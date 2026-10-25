"""Steps 17-20 glue: raw recording -> calibrated forces -> quality gate -> steps -> metrics."""
from __future__ import annotations
from typing import Optional
import numpy as np
from .acquisition import Recording
from . import quality, metrics


def analyze(rec: Recording, calibration, body_weight_N: float, fs_nominal: float,
            fx_full_scale: float, fz_full_scale: float, audit=None, drift_model=None) -> dict:
    """`calibration` is a bench_analysis.Calibration (raw -> N). Returns quality, steps and a `usable` verdict.
    Metrics are withheld (steps=[]) when the recording fails the quality gate."""
    warnings = []
    finite = bool(np.all(np.isfinite(rec.raw_x)) and np.all(np.isfinite(rec.raw_z)))
    if finite and len(rec.t):
        fx, fz = calibration.predict(rec.raw_x, rec.raw_z)
        if drift_model is not None:
            from . import drift
            fx, fz, warnings = drift.correct_forces(fx, fz, drift_model, rec.start_time_h, rec.temp_c)
    else:                                   # the quality gate rejects this; never push NaN through calibration
        fx = fz = np.full(len(rec.t), np.nan)
    q = quality.check_recording(rec.t, fx, fz, fs_nominal, fz_full_scale, fx_full_scale,
                                 raw_x=rec.raw_x, raw_z=rec.raw_z)
    steps = []
    if q.usable:
        steps = metrics.segment_steps(rec.t, fx, fz, body_weight_N)
        q2 = quality.check_steps(steps)
        q.flags += q2.flags; q.detail.update(q2.detail)
        if not q.usable:
            steps = []
    out = {"usable": q.usable, "flags": q.flags, "quality": q.detail, "steps": steps, "side": rec.side,
           "warnings": warnings}
    if audit is not None:
        audit.append("analysis", source=rec.source, side=rec.side, usable=q.usable, flags=q.flags, n_steps=len(steps))
    return out
