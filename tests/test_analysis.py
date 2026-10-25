"""Model-validation code: metrics, C-index, fold handling and the healthy-control split for M1."""
import numpy as np
import pandas as pd
import pytest

import gaitrec_rehab.analysis as an


def test_metrics_on_known_cases():
    y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
    perfect = np.array([.1] * 8 + [.9, .9])
    m = an.metrics(y, perfect, alert_rate=0.2)
    assert m["auroc"] == 1.0 and m["ppv_at_10pct"] == 1.0 and m["events_caught_at_10pct"] == 2
    assert an.metrics(y, np.full(10, .2))["auroc"] == 0.5


def test_calibration_slope_near_one_for_calibrated_predictions():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.01, 0.5, 20000); y = (rng.random(20000) < p).astype(int)
    a, s = an.calibration(y, p)
    assert abs(s - 1) < 0.1 and abs(a) < 0.1


def test_c_index():
    t = np.array([1, 2, 3, 4.]); e = np.array([1, 1, 1, 0]); r = np.array([4, 3, 2, 1.])
    assert an.c_index(t, e, r) == 1.0 and an.c_index(t, e, -r) == 0.0


def _toy(n=300, seed=0):
    rng = np.random.default_rng(seed)
    c = pd.DataFrame({k: rng.normal(size=n) for k in an.M0 + an.GAIT})
    c["SUBJECT_ID"] = np.arange(n); c["event"] = (rng.random(n) < 0.1).astype(int)
    return an.Data(c, rng.normal(size=(80, 202)), np.repeat(np.arange(1000, 1040), 2), rng.normal(size=(n, 202)))


def test_cv_predict_covers_every_row_once():
    d = _toy(); y = d.cohort.event.to_numpy(); rows = np.arange(len(y))
    o = an.cv_predict(d, y, rows, d.cohort.SUBJECT_ID.to_numpy(), ["M0", "M2"], 0, False)
    assert all(len(v) == len(y) and np.all((v > 0) & (v < 1)) for v in o.values())


def test_autoencoder_never_sees_held_out_healthy_controls(monkeypatch):
    seen = []
    real = an.recon_scores
    def spy(d, mask, rows):
        seen.append(set(d.hc_subject[mask])); return real(d, mask, rows)
    monkeypatch.setattr(an, "recon_scores", spy)
    d = _toy(); y = d.cohort.event.to_numpy()
    an.cv_predict(d, y, np.arange(len(y)), d.cohort.SUBJECT_ID.to_numpy(), ["M1"], 0, True)
    all_hc = set(d.hc_subject)
    assert len(seen) == 5
    held_out = [all_hc - s for s in seen]
    assert all(held_out) and set().union(*held_out) == all_hc
    assert all(not (a & b) for i, a in enumerate(held_out) for b in held_out[i + 1:])
