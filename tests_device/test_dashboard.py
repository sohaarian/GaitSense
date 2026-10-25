import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bench_analysis as ba
from gaitsense import acquisition as acq, dashboard, pipeline

BW = 750.0


def cal():
    c = acq.SimConfig(); S = np.array(c.S)
    return ba.Calibration(S.tolist(), list(c.offset), 1.0)


def visit(label, **kw):
    r = acq.SimulatedSource(acq.SimConfig(n_steps=40, **kw)).read()
    return dashboard.build_visit(label, pipeline.analyze(r, cal(), BW, 200, 300, 1200))


def test_dashboard_shows_trend_changes_and_failures_and_escapes_labels():
    visits = [visit("V1", seed=1), visit("V2<script>alert(1)</script>", seed=2, push_amp_bw=0.32),
              visit("V3", seed=3, saturate_z_N=600.0), visit("V4", seed=4)]
    out = dashboard.render("Test <b>patient</b>", visits, remount_sd_push=0.003, remount_sd_brake=0.003)
    assert "<script" not in out and "&lt;script&gt;" in out and "&lt;b&gt;patient" in out
    assert "No usable data: saturation" in out                       # failed visit shown as a failure, not a number
    assert "exceeds measurement error" in out                        # the +60% push-off visit
    assert "within measurement error" in out                         # V4 ~ baseline
    assert "Not a medical device" in out and "<svg" in out and "http" not in out.split("</style>")[1]


def test_dashboard_with_no_usable_visits_does_not_crash():
    v = [visit("V1", saturate_z_N=600.0)]
    out = dashboard.render("P", v, 0.003, 0.003)
    assert "no usable visits" in out
