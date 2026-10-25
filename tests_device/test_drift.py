import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bench_analysis as ba
from gaitsense import acquisition as acq, drift, pipeline

BW, FSX, FSZ, FS = 750.0, 300.0, 1200.0, 200.0
A_T, B_T = -0.003, -0.0005     # true sensitivity drift: per degC and per hour


def gain(t, T): return 1 + A_T * (T - 25) + B_T * t


def cal():
    S = np.array(acq.SimConfig().S)
    return ba.Calibration(S=S.tolist(), offset=list(acq.SimConfig().offset), cond=1.0)


def ref_checks(seed=0, noise=0.002, n=10, drifting=True):
    r = np.random.default_rng(seed)
    t = np.linspace(0, 48, n); T = 22 + 12 * r.permutation(n) / (n - 1)       # temperature unrelated to time
    g = gain(t, T) if drifting else np.ones(n)
    return t, T, g + r.normal(0, noise, n), g + r.normal(0, noise, n)


def visit(amp, t, T, seed):
    cfg = acq.SimConfig(push_amp_bw=amp, mount_gain=gain(t, T), start_time_h=t, temp_c=T, seed=seed, n_steps=40, noise_N=0.0, step_cv=0.0)
    src = acq.SimulatedSource(cfg); return src, src.read()


def mean_push(res): return np.mean([s["push_imp"] for s in res["steps"]])


def test_drift_model_recovers_true_coefficients():
    m = drift.fit_drift(*ref_checks())
    assert m.z.a_per_C == pytest.approx(A_T, abs=0.0006) and m.z.b_per_h == pytest.approx(B_T, abs=0.0002)
    assert m.z.use_temp and m.z.use_time


def test_correction_removes_drift_bias_but_keeps_a_genuine_change():
    m = drift.fit_drift(*ref_checks())
    s1, r1 = visit(0.20, 2.0, 21.0, 1)
    s2, r2 = visit(0.23, 40.0, 33.0, 2)        # genuinely +15% push-off, at a hotter, later time
    true1, true2 = s1.truth[0]["push_imp"], s2.truth[0]["push_imp"]
    raw1, raw2 = pipeline.analyze(r1, cal(), BW, FS, FSX, FSZ), pipeline.analyze(r2, cal(), BW, FS, FSX, FSZ)
    c1, c2 = (pipeline.analyze(r, cal(), BW, FS, FSX, FSZ, drift_model=m) for r in (r1, r2))
    err = lambda res, tr: abs(mean_push(res) / tr - 1)
    assert err(raw2, true2) > 0.03                       # uncorrected visit 2 is biased by several %
    assert err(c1, true1) < 0.015 and err(c2, true2) < 0.015
    assert mean_push(c2) / mean_push(c1) == pytest.approx(1.15, abs=0.02)     # the real +15% survives correction


def test_no_drift_means_values_are_essentially_unchanged():
    m = drift.fit_drift(*ref_checks(seed=3, drifting=False))
    assert not m.z.use_temp and not m.z.use_time and not m.x.use_temp and not m.x.use_time
    _, r = visit(0.2, 10.0, 28.0, 5)
    a = pipeline.analyze(r, cal(), BW, FS, FSX, FSZ)
    b = pipeline.analyze(r, cal(), BW, FS, FSX, FSZ, drift_model=m)
    assert mean_push(b) == pytest.approx(mean_push(a), rel=0.01)


def test_fit_refuses_bad_inputs():
    t, T, gx, gz = ref_checks()
    with pytest.raises(ValueError, match="at least"):
        drift.fit_drift(t[:4], T[:4], gx[:4], gz[:4])
    with pytest.raises(ValueError, match="independently"):
        drift.fit_drift(t, 20 + 0.1 * t, gx, gz)             # temperature is just time: cannot separate
    with pytest.raises(ValueError, match="non-finite"):
        drift.fit_drift(t, T, np.where(np.arange(len(t)) == 2, np.nan, gx), gz)


def test_missing_temperature_or_extrapolation_is_reported_not_silent():
    m = drift.fit_drift(*ref_checks())
    fx, fz = np.ones(5), np.ones(5)
    out = drift.correct_forces(fx, fz, m, 10.0, None)
    assert np.array_equal(out[1], fz) and any("not corrected" in w for w in out[2])
    out = drift.correct_forces(fx, fz, m, 500.0, 25.0)
    assert any("extrapolation" in w for w in out[2])
