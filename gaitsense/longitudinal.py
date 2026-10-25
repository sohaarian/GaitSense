"""Steps 21-22: visit summaries, visit-to-visit change with uncertainty, minimum detectable change, bilateral view."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class VisitSummary:
    n: int
    median: float
    mean: float
    sd: float
    se: float


def summarize(values) -> VisitSummary:
    v = np.asarray(values, float)
    if len(v) < 2:
        raise ValueError("need >=2 steps for a visit summary")
    sd = float(v.std(ddof=1))
    return VisitSummary(len(v), float(np.median(v)), float(v.mean()), sd, sd / np.sqrt(len(v)))


def change(v1: VisitSummary, v2: VisitSummary, remount_sd: float, z: float = 1.96) -> dict:
    """Change in mean between two visits. Each visit carries independent sampling error (se) AND an
    independent remount/calibration error (remount_sd, from the bench repeatability study, in metric units).
    remount_sd=0 would claim a perfect remount; callers must supply a measured value."""
    if remount_sd is None or remount_sd < 0:
        raise ValueError("remount_sd must be a measured non-negative number")
    delta = v2.mean - v1.mean
    sd_delta = float(np.sqrt(v1.se ** 2 + v2.se ** 2 + 2 * remount_sd ** 2))
    mdc = z * sd_delta
    return {"delta": float(delta), "sd_delta": sd_delta, "mdc95": float(mdc),
            "ci95": [float(delta - mdc), float(delta + mdc)],
            "exceeds_measurement_error": bool(abs(delta) > mdc)}


def bilateral_ratio(affected: VisitSummary, other: VisitSummary) -> dict:
    """Affected/other ratio of means. The 'other' limb also changes between visits in the GaitRec data, so this
    is reported as context and must not be used alone to judge recovery."""
    if other.mean == 0:
        raise ValueError("other-limb mean is zero")
    r = affected.mean / other.mean
    # delta-method SE for a ratio of independent means
    se = abs(r) * np.sqrt((affected.se / affected.mean) ** 2 + (other.se / other.mean) ** 2)
    return {"ratio": float(r), "se": float(se)}
