"""Tests use SIMULATED sensors with known ground truth. They verify the analysis arithmetic only."""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bench_analysis as ba

S_TRUE = ((2.0e-3, 1.5e-4), (3.0e-4, 1.0e-3))


def test_calibration_recovers_known_matrix_and_cross_axis():
    d = ba.simulate_bench(n=200, seed=1, noise=1e-5)
    cal = ba.fit_calibration(d["ref_fx"], d["ref_fz"], d["raw_x"], d["raw_z"])
    assert np.allclose(cal.S, S_TRUE, rtol=1e-2, atol=1e-6)
    assert np.allclose(cal.offset, (0.02, -0.01), atol=1e-3)
    assert cal.cross_axis_pct["fz_into_x_channel"] == pytest.approx(100 * 1.5e-4 / 2e-3, rel=0.02)


def test_independent_validation_beats_diagonal_only_when_coupling_exists():
    tr, te = ba.simulate_bench(300, seed=2), ba.simulate_bench(100, seed=3)
    cal = ba.fit_calibration(tr["ref_fx"], tr["ref_fz"], tr["raw_x"], tr["raw_z"])
    full = ba.validate(cal, te["ref_fx"], te["ref_fz"], te["raw_x"], te["raw_z"], 300, 1200)
    diag = ba.naive_validate(te["ref_fx"], te["ref_fz"], te["raw_x"], te["raw_z"], tr, 300, 1200)
    assert full["fx"]["rmse_N"] < 0.2 * diag["fx"]["rmse_N"]
    assert full["fx"]["rmse_pct_fs"] < 1.0


def test_calibration_rejects_non_independent_loads():
    fz = np.linspace(0, 1000, 20)
    with pytest.raises(ValueError, match="independently"):
        ba.fit_calibration(2 * fz, fz, fz, fz)  # fx perfectly collinear with fz
    with pytest.raises(ValueError, match="at least 6"):
        ba.fit_calibration([1, 2], [3, 4], [1, 2], [3, 4])


def test_hysteresis_recovers_injected_gap():
    ref = np.concatenate([np.linspace(0, 100, 11), np.linspace(100, 0, 11)])
    branch = np.array(["up"] * 11 + ["down"] * 11)
    out = ref + np.where(branch == "down", 2.0, 0.0)   # 2 N offset on unloading
    assert ba.hysteresis_pct_fs(ref, out, branch, 100.0) == pytest.approx(2.0, abs=1e-9)
    with pytest.raises(ValueError):
        ba.hysteresis_pct_fs(ref[:11], out[:11], branch[:11], 100.0)


def test_drift_detects_real_slope_and_not_noise():
    r = np.random.default_rng(0); t = np.linspace(0, 8, 40)
    d = ba.drift_rate(t, 100 + 0.5 * t + r.normal(0, 0.05, 40), 100.0)
    assert d["slope_N_per_h"] == pytest.approx(0.5, abs=0.05) and d["ci_excludes_zero"]
    flat = ba.drift_rate(t, 100 + r.normal(0, 0.5, 40), 100.0)
    assert not flat["ci_excludes_zero"]


def test_repeatability_separates_remount_error_from_within_session_noise():
    r = np.random.default_rng(4)
    k, n = 30, 6
    mount_off = r.normal(0, 3.0, k)                      # remount SD = 3
    g = np.repeat(np.arange(k), n)
    v = 100 + mount_off[g] + r.normal(0, 1.0, k * n)     # within SD = 1
    res = ba.repeatability(v, g)
    assert res["within_sd"] == pytest.approx(1.0, abs=0.2)
    assert res["between_sd"] == pytest.approx(3.0, abs=0.9)
    assert res["icc"] == pytest.approx(9 / 10, abs=0.08)
    assert res["mdc95_across_occasions"] > 1.96 * np.sqrt(2) * res["within_sd"]  # remount error widens MDC


def test_bland_altman_known_bias():
    ref = np.linspace(10, 100, 50)
    ba_ = ba.bland_altman(ref + 2.0, ref)
    assert ba_["bias"] == pytest.approx(2.0) and ba_["sd_diff"] == pytest.approx(0.0, abs=1e-12)


def test_reference_check_flags_sensitivity_change():
    d = ba.simulate_bench(200, seed=5, noise=1e-5)
    cal = ba.fit_calibration(d["ref_fx"], d["ref_fz"], d["raw_x"], d["raw_z"])
    S, off = np.array(S_TRUE), np.array((0.02, -0.01))
    ok_raw = S @ np.array([100.0, 500.0]) + off
    assert ba.reference_check(cal, ok_raw[0], ok_raw[1], 100.0, 500.0)["within_tolerance"]
    drifted = off + (S * np.array([[1.0, 1.0], [0.9, 0.9]])) @ np.array([100.0, 500.0])  # z channel 10% low
    res = ba.reference_check(cal, drifted[0], drifted[1], 100.0, 500.0)
    assert not res["within_tolerance"] and res["dev_fz_pct"] < -5


def test_cli_end_to_end_and_bad_input(tmp_path, capsys):
    def write(p, d, extra=None):
        with open(p, "w", newline="") as fh:
            cols = ["trial", "ref_fx", "ref_fz", "raw_x", "raw_z"]
            w = csv.writer(fh); w.writerow(cols)
            for i in range(len(d["trial"])):
                w.writerow([d[c][i] for c in cols])
    write(tmp_path / "tr.csv", ba.simulate_bench(200, seed=7)); write(tmp_path / "te.csv", ba.simulate_bench(80, seed=8))
    out = tmp_path / "o.json"
    rc = ba.main(["calibrate", str(tmp_path / "tr.csv"), str(tmp_path / "te.csv"), "--fs-x", "300", "--fs-z", "1200", "--out", str(out)])
    assert rc == 0 and json.loads(out.read_text())["validation_full"]["fz"]["rmse_pct_fs"] < 1.0
    (tmp_path / "bad.csv").write_text("a,b\n1,2\n")
    assert ba.main(["calibrate", str(tmp_path / "bad.csv"), str(tmp_path / "te.csv"), "--fs-x", "1", "--fs-z", "1"]) == 2
