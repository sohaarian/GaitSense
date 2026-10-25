"""GaitSense software tests on SIMULATED signals with known ground truth."""
import sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import bench_analysis as ba
from gaitsense import acquisition as acq, audit, longitudinal as lg, metrics, pipeline, quality

BW, FSX, FSZ, FS = 750.0, 300.0, 1200.0, 200.0


def true_cal(cfg=None):
    cfg = cfg or acq.SimConfig()
    S = np.array(cfg.S)
    return ba.Calibration(S=S.tolist(), offset=list(cfg.offset), cond=float(np.linalg.cond(S)))


def run(cfg, side="R"):
    src = acq.SimulatedSource(cfg, side)
    rec = src.read()
    return src, pipeline.analyze(rec, true_cal(), BW, FS, FSX, FSZ)


def test_impulses_match_analytic_truth():
    src, res = run(acq.SimConfig(noise_N=0.0, step_cv=0.0))
    assert res["usable"] and len(res["steps"]) >= 28
    t = src.truth[0]
    got = res["steps"][5]
    assert got["push_imp"] == pytest.approx(t["push_imp"], rel=0.02)
    assert got["brake_imp"] == pytest.approx(t["brake_imp"], rel=0.02)
    assert got["push_imp"] > 0 and got["brake_imp"] > 0 and got["net_imp"] == pytest.approx(got["push_imp"] - got["brake_imp"], rel=0.05, abs=0.2)


def test_noise_does_not_break_segmentation_or_bias_impulse():
    src, res = run(acq.SimConfig(noise_N=3.0, seed=4))
    est = np.mean([s["push_imp"] for s in res["steps"]])
    truth = np.mean([s["push_imp"] for s in src.truth])
    assert est == pytest.approx(truth, rel=0.03)


@pytest.mark.parametrize("cfg,flag", [
    (acq.SimConfig(dropout_at_s=10.0, dropout_len_s=0.4), "data_gap"),
    (acq.SimConfig(saturate_z_N=600.0), "saturation"),
    (acq.SimConfig(flatline_x_from_s=5.0), "flatline"),
    (acq.SimConfig(n_steps=4), "too_few_steps"),
])
def test_malfunctions_are_flagged_and_metrics_withheld(cfg, flag):
    _, res = run(cfg)
    assert flag in res["flags"] and not res["usable"] and res["steps"] == []


def test_clean_recording_has_no_critical_flags():
    _, res = run(acq.SimConfig())
    assert res["usable"] and not res["flags"]


def test_non_monotonic_time_flagged():
    rec = acq.SimulatedSource().read(); rec.t[100] = rec.t[90]
    q = quality.check_recording(rec.t, rec.raw_x, rec.raw_z, FS, FSZ, FSX, raw_x=rec.raw_x, raw_z=rec.raw_z)
    assert "non_monotonic_time" in q.flags and not q.usable


def test_truncated_edge_stance_is_dropped():
    t = np.arange(0, 1.0, 0.005); fz = np.full_like(t, 400.0)   # load on for the whole record
    assert metrics.segment_steps(t, np.zeros_like(t), fz, BW) == []


def test_remount_gain_error_is_a_real_confound_and_change_test_accounts_for_it():
    """Same true gait, two visits with a 6% sensitivity difference from remounting."""
    vals = {}
    for k, g in (("v1", 1.00), ("v2", 0.94)):
        _, res = run(acq.SimConfig(mount_gain=g, seed=11 if k == "v1" else 12, n_steps=40))
        vals[k] = lg.summarize([s["push_imp_bw"] for s in res["steps"]])
    # the calibration assumed gain 1.0, so v2 reads ~6% low: a spurious 'change'
    spurious = vals["v2"].mean / vals["v1"].mean - 1
    assert spurious == pytest.approx(-0.06, abs=0.02)
    mean = vals["v1"].mean
    naive = lg.change(vals["v1"], vals["v2"], remount_sd=0.0)
    honest = lg.change(vals["v1"], vals["v2"], remount_sd=0.05 * mean)   # measured remount SD of 5%
    assert naive["exceeds_measurement_error"] and not honest["exceeds_measurement_error"]
    with pytest.raises(ValueError):
        lg.change(vals["v1"], vals["v2"], remount_sd=-1)


def test_a_large_true_change_is_still_detected():
    _, r1 = run(acq.SimConfig(push_amp_bw=0.20, seed=1, n_steps=40))
    _, r2 = run(acq.SimConfig(push_amp_bw=0.30, seed=2, n_steps=40))
    s1, s2 = (lg.summarize([s["push_imp_bw"] for s in r["steps"]]) for r in (r1, r2))
    c = lg.change(s1, s2, remount_sd=0.05 * s1.mean)
    assert c["delta"] > 0 and c["exceeds_measurement_error"]


def test_bilateral_ratio():
    a, b = lg.summarize([0.8, 0.82, 0.78, 0.81]), lg.summarize([1.0, 1.02, 0.98, 1.01])
    r = lg.bilateral_ratio(a, b)
    assert r["ratio"] == pytest.approx(a.mean / b.mean) and r["se"] > 0


def test_audit_log_detects_tampering(tmp_path):
    log = audit.AuditLog(str(tmp_path / "a.log"))
    for i in range(3):
        log.append("analysis", i=i)
    assert log.verify()
    p = tmp_path / "a.log"
    lines = p.read_text().splitlines()
    p.write_text("\n".join([lines[0], lines[1].replace('"i": 1', '"i": 9'), lines[2]]) + "\n")
    assert not log.verify()
    p.write_text("\n".join([lines[0], lines[2]]) + "\n")   # removed entry
    assert not log.verify()


def test_pipeline_writes_audit_entry(tmp_path):
    log = audit.AuditLog(str(tmp_path / "b.log"))
    rec = acq.SimulatedSource(acq.SimConfig(saturate_z_N=600.0)).read()
    pipeline.analyze(rec, true_cal(), BW, FS, FSX, FSZ, audit=log)
    assert log.verify() and "saturation" in (tmp_path / "b.log").read_text()


def test_csv_source_roundtrip_and_bad_file(tmp_path):
    rec = acq.SimulatedSource(acq.SimConfig(n_steps=3)).read()
    p = tmp_path / "r.csv"
    p.write_text("t,raw_x,raw_z\n" + "\n".join(f"{a},{b},{c}" for a, b, c in zip(rec.t, rec.raw_x, rec.raw_z)))
    back = acq.CsvSource(str(p), "L").read()
    assert np.allclose(back.raw_z, rec.raw_z) and back.side == "L"
    (tmp_path / "bad.csv").write_text("t,x\n1,2\n")
    with pytest.raises(ValueError):
        acq.CsvSource(str(tmp_path / "bad.csv")).read()
