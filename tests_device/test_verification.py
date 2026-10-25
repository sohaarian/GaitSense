"""Expanded verification: corrupted recordings, sensor failures, calibration changes, longitudinal error rates.
All SIMULATED. These check that the software fails safe, not that any physical sensor behaves."""
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bench_analysis as ba
from gaitsense import acquisition as acq, longitudinal as lg, pipeline

BW, FS, FSX, FSZ = 750.0, 200.0, 300.0, 1200.0


def cal():
    c = acq.SimConfig(); S = np.array(c.S)
    return ba.Calibration(S.tolist(), list(c.offset), 1.0)


def analyze(rec): return pipeline.analyze(rec, cal(), BW, FS, FSX, FSZ)


def clean(**kw): return acq.SimulatedSource(acq.SimConfig(**kw)).read()


# ---------------- corrupted recordings
@pytest.mark.parametrize("mutate,flag", [
    (lambda r: r.raw_z.__setitem__(500, np.nan), "non_finite"),
    (lambda r: r.raw_x.__setitem__(10, np.inf), "non_finite"),
    (lambda r: r.t.__setitem__(300, r.t[200]), "non_monotonic_time"),
    (lambda r: r.t.__setitem__(slice(0, None), r.t[::-1].copy()), "non_monotonic_time"),
])
def test_corrupt_values_fail_safe(mutate, flag):
    r = clean(); mutate(r)
    res = analyze(r)
    assert flag in res["flags"] and not res["usable"] and res["steps"] == []


def test_empty_tiny_and_mismatched_recordings():
    assert "empty" in analyze(acq.Recording([], [], []))["flags"]
    assert "empty" in analyze(acq.Recording([0, 1], [0, 0], [0, 0]))["flags"]
    with pytest.raises(ValueError):
        acq.Recording([0, 1, 2], [0, 0], [0, 0, 0])


def test_all_zero_and_constant_signals_yield_no_metrics():
    n = 2000; t = np.arange(n) / FS
    for z in (np.zeros(n), np.full(n, 0.5)):
        res = analyze(acq.Recording(t, np.zeros(n), z))
        assert not res["usable"] and res["steps"] == []


# ---------------- sensor failures
def test_dead_vertical_channel_is_not_reported_as_zero_loading():
    r = clean(); r.raw_z[:] = acq.SimConfig().offset[1]      # vertical channel stuck at its offset
    res = analyze(r)
    assert not res["usable"] and res["steps"] == []


def test_noise_burst_does_not_create_phantom_steps():
    r = clean(n_steps=30); rng = np.random.default_rng(0)
    a, b = int(2.0 * FS), int(2.0 * FS) + 40
    r.raw_z[a:b] += rng.normal(0, 0.5, b - a)                # large burst during swing
    res = analyze(r)
    assert len(res["steps"]) <= 30                           # duration limits reject short phantom 'stances'


def test_high_noise_sensor_is_still_not_silently_accepted_with_wrong_numbers():
    clean_res = analyze(clean(seed=2, n_steps=40))
    noisy = analyze(clean(seed=2, n_steps=40, noise_N=25.0))
    if noisy["usable"]:                                      # if the gate passes it, the answer must still be right
        a = np.mean([s["push_imp"] for s in clean_res["steps"]]); b = np.mean([s["push_imp"] for s in noisy["steps"]])
        assert b == pytest.approx(a, rel=0.08)


# ---------------- calibration changes
def test_calibration_sensitivity_change_is_caught_by_reference_check():
    c = acq.SimConfig(); S = np.array(c.S); off = np.array(c.offset)
    for g, expect_ok in ((1.01, True), (0.93, False), (1.08, False)):
        raw = S @ np.array([100.0, 500.0]) * g + off
        chk = ba.reference_check(cal(), raw[0], raw[1], 100.0, 500.0, tol_pct=3.0)
        assert chk["within_tolerance"] is expect_ok


def test_cross_axis_coupling_change_is_caught_by_reference_check():
    c = acq.SimConfig(); S = np.array(c.S).copy(); S[0, 1] *= 4                # coupling Fz->x channel quadruples
    raw = S @ np.array([100.0, 500.0]) + np.array(c.offset)
    assert not ba.reference_check(cal(), raw[0], raw[1], 100.0, 500.0, tol_pct=3.0)["within_tolerance"]


# ---------------- longitudinal measurement error
def _visit_summary(seed, gain):
    r = analyze(clean(seed=seed, n_steps=30, mount_gain=gain))
    return lg.summarize([s["push_imp_bw"] for s in r["steps"]])


def test_false_alarm_rate_with_and_without_remount_error_in_the_test():
    """Identical true gait at both visits; only remount sensitivity differs (SD 5%)."""
    rng = np.random.default_rng(42); N = 120
    base = _visit_summary(0, 1.0).mean
    naive = honest = 0
    for k in range(N):
        v1 = _visit_summary(2 * k + 1, 1 + rng.normal(0, 0.05)); v2 = _visit_summary(2 * k + 2, 1 + rng.normal(0, 0.05))
        naive += lg.change(v1, v2, 0.0)["exceeds_measurement_error"]
        honest += lg.change(v1, v2, 0.05 * base)["exceeds_measurement_error"]
    assert naive / N > 0.30                    # ignoring remount error: many false 'changes'
    assert honest / N < 0.12                   # nominal 5%, allowing sampling slack


def test_power_to_detect_a_true_change_given_remount_error():
    rng = np.random.default_rng(7); N = 80; base = _visit_summary(0, 1.0).mean
    for frac, lo, hi in ((0.30, 0.8, 1.0), (0.05, 0.0, 0.3)):
        hits = 0
        for k in range(N):
            a = analyze(clean(seed=3 * k + 1, n_steps=30, mount_gain=1 + rng.normal(0, 0.05)))
            b = analyze(clean(seed=3 * k + 2, n_steps=30, push_amp_bw=0.20 * (1 + frac), mount_gain=1 + rng.normal(0, 0.05)))
            s = lambda r: lg.summarize([x["push_imp_bw"] for x in r["steps"]])
            hits += lg.change(s(a), s(b), 0.05 * base)["exceeds_measurement_error"]
        assert lo <= hits / N <= hi, (frac, hits / N)


def test_longitudinal_input_validation():
    with pytest.raises(ValueError):
        lg.summarize([1.0])
    a = lg.summarize([1, 1.1, 0.9])
    with pytest.raises(ValueError):
        lg.change(a, a, None)
    with pytest.raises(ValueError):
        lg.bilateral_ratio(a, lg.summarize([0.0, 0.0, 0.0]))


def test_non_finite_is_critical_on_its_own():
    from gaitsense import quality
    t = np.arange(100) / FS; f = np.ones(100); f[5] = np.nan
    q = quality.check_recording(t, f, f, FS, FSZ, FSX)
    assert q.flags == ["non_finite"] and not q.usable
