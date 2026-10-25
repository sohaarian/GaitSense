"""Step 16: synchronized two-channel acquisition interface, with a simulator and a CSV reader."""
from __future__ import annotations
import csv
from dataclasses import dataclass
from typing import Optional, Protocol
import numpy as np


@dataclass
class Recording:
    t: np.ndarray       # seconds, one clock shared by both channels (synchronization is by construction)
    raw_x: np.ndarray   # raw fore-aft channel output
    raw_z: np.ndarray   # raw vertical channel output
    side: str = "unknown"   # "L" / "R"
    source: str = ""
    start_time_h: Optional[float] = None   # hours since the device calibration (for drift correction)
    temp_c: Optional[float] = None         # sensor temperature during the recording

    def __post_init__(self):
        self.t, self.raw_x, self.raw_z = (np.asarray(a, float) for a in (self.t, self.raw_x, self.raw_z))
        if not (len(self.t) == len(self.raw_x) == len(self.raw_z)):
            raise ValueError("t, raw_x and raw_z must have equal length")


class Source(Protocol):
    def read(self) -> Recording: ...


class CsvSource:
    """Columns: t, raw_x, raw_z. A future hardware driver would implement the same `read()`."""
    def __init__(self, path: str, side: str = "unknown"):
        self.path, self.side = path, side

    def read(self) -> Recording:
        with open(self.path, newline="") as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            raise ValueError(f"{self.path}: empty")
        try:
            cols = {k: np.array([float(r[k]) for r in rows]) for k in ("t", "raw_x", "raw_z")}
        except KeyError as e:
            raise ValueError(f"{self.path}: missing column {e}") from None
        return Recording(side=self.side, source=self.path, **cols)


@dataclass
class SimConfig:
    fs: float = 200.0
    n_steps: int = 30
    stride_s: float = 1.1
    stance_s: float = 0.65
    body_weight_N: float = 750.0
    push_amp_bw: float = 0.20      # peak propulsive Fx / BW
    brake_amp_bw: float = 0.18
    step_cv: float = 0.04          # step-to-step amplitude variability
    noise_N: float = 2.0
    mount_gain: float = 1.0        # multiplicative sensitivity error from a remount (1.0 = none)
    dropout_at_s: Optional[float] = None   # a gap of dropout_len_s starts here
    dropout_len_s: float = 0.5
    saturate_z_N: Optional[float] = None   # clip Fz here
    flatline_x_from_s: Optional[float] = None
    seed: int = 0
    start_time_h: Optional[float] = None
    temp_c: Optional[float] = None
    # sensor model (same form as bench_analysis.Calibration): raw = S @ [fx, fz] + offset
    S: tuple = ((2.0e-3, 1.5e-4), (3.0e-4, 1.0e-3))
    offset: tuple = (0.02, -0.01)


class SimulatedSource:
    """Synthetic walking. True per-step impulses are kept in `.truth` for testing."""
    def __init__(self, cfg: SimConfig = SimConfig(), side: str = "R"):
        self.cfg, self.side, self.truth = cfg, side, []

    def read(self) -> Recording:
        c = self.cfg
        r = np.random.default_rng(c.seed)
        n = int((c.n_steps + 1) * c.stride_s * c.fs)
        t = np.arange(n) / c.fs
        fx, fz = np.zeros(n), np.zeros(n)
        self.truth = []
        for k in range(c.n_steps):
            i0 = int((0.3 + k * c.stride_s) * c.fs)
            m = int(c.stance_s * c.fs)
            u = np.arange(m) / m
            g = 1 + c.step_cv * r.normal()
            fz[i0:i0 + m] = c.body_weight_N * g * (1.0 * np.sin(np.pi * u) + 0.15 * np.sin(3 * np.pi * u))
            A_push, A_brake = c.push_amp_bw * c.body_weight_N * g, c.brake_amp_bw * c.body_weight_N * g
            sh = np.sin(2 * np.pi * u)
            fx[i0:i0 + m] = np.where(u < 0.5, -A_brake * sh, -A_push * sh)  # braking first (Fx<0), then propulsion (Fx>0)
            T = c.stance_s
            self.truth.append({"brake_imp": A_brake * T / np.pi, "push_imp": A_push * T / np.pi})
        fx += r.normal(0, c.noise_N, n); fz += r.normal(0, c.noise_N, n)
        if c.saturate_z_N is not None:
            fz = np.minimum(fz, c.saturate_z_N)
        S = np.array(c.S) * c.mount_gain
        raw = np.column_stack([fx, fz]) @ S.T + np.array(c.offset)
        if c.flatline_x_from_s is not None:
            j = int(c.flatline_x_from_s * c.fs); raw[j:, 0] = raw[j, 0]
        t_out, raw_x, raw_z = t, raw[:, 0], raw[:, 1]
        if c.dropout_at_s is not None:
            a = int(c.dropout_at_s * c.fs); b = a + int(c.dropout_len_s * c.fs)
            keep = np.ones(n, bool); keep[a:b] = False
            t_out, raw_x, raw_z = t[keep], raw_x[keep], raw_z[keep]
        return Recording(t_out, raw_x, raw_z, side=self.side, source="simulated",
                         start_time_h=c.start_time_h, temp_c=c.temp_c)
