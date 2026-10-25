"""Step 20: stance segmentation and braking / push-off impulse, normalised by body weight."""
from __future__ import annotations
from typing import List
import numpy as np


def _trapz(y, t):
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(t))) if len(y) > 1 else 0.0


def segment_steps(t, fx, fz, body_weight_N: float, thresh_bw: float = 0.05,
                  min_stance_s: float = 0.3, max_stance_s: float = 1.5) -> List[dict]:
    """Stance = contiguous run with Fz above thresh_bw*BW. Runs outside the duration limits are dropped."""
    t, fx, fz = np.asarray(t, float), np.asarray(fx, float), np.asarray(fz, float)
    on = fz > thresh_bw * body_weight_N
    edges = np.flatnonzero(np.diff(on.astype(int)))
    starts = list(edges[np.diff(on.astype(int))[edges] == 1] + 1)
    ends = list(edges[np.diff(on.astype(int))[edges] == -1] + 1)
    if on[0]:
        starts.insert(0, 0)
    if on[-1]:
        ends.append(len(on))
    steps = []
    for a, b in zip(starts, ends):
        # a stance touching either end of the record is truncated: skip it
        if a == 0 or b == len(on) or b - a < 3:
            continue
        dur = t[b - 1] - t[a]
        if not (min_stance_s <= dur <= max_stance_s) or np.ptp(np.diff(t[a:b])) > 0.5 * np.median(np.diff(t[a:b])):
            continue
        tt, x, z = t[a:b], fx[a:b], fz[a:b]
        steps.append({"t0": float(tt[0]), "dur": float(dur), "peak_fz": float(z.max()),
                      "brake_imp": float(-_trapz(np.minimum(x, 0), tt)),   # N*s, positive number
                      "push_imp": float(_trapz(np.maximum(x, 0), tt)),
                      "net_imp": float(_trapz(x, tt)),
                      "vert_imp": float(_trapz(z, tt))})
    for s in steps:
        s["brake_imp_bw"] = s["brake_imp"] / body_weight_N      # units: BW*s
        s["push_imp_bw"] = s["push_imp"] / body_weight_N
    return steps
