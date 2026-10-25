"""Unit tests on synthetic inputs (no patient data needed)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from gaitsense_ai import evaluation as E
from gaitsense_ai import features as F
from gaitsense_ai import monitoring as M
from gaitsense_ai import perturb as PT
from gaitsense_ai import prediction as P
from gaitsense_ai import splits
from gaitsense_ai.longitudinal import LongitudinalGP, PatientData, kernel_matrix

BW = 800.0


def two_hump(n=185, p1=1.1, p2=1.05, valley=0.8):
    t = np.linspace(0, 1, n)
    z = BW * (p1 * np.exp(-((t - 0.25) / 0.12) ** 2) + p2 * np.exp(-((t - 0.75) / 0.12) ** 2))
    z = z - z.min() + 25.0
    a = -0.2 * BW * np.sin(np.pi * t / 0.5) * (t < 0.5) + 0.2 * BW * np.sin(np.pi * (t - 0.5) / 0.5) * (t >= 0.5)
    return z, a


def test_primary_features_and_sign_convention():
    z, a = two_hump()
    f = F.trial_features(z, a, BW)
    assert f["peak_v"] == pytest.approx(z.max() / BW)
    # half-sine of amplitude 0.2 BW over half the stance: impulse = 0.2 * (2/pi) * (T/2)
    T = (z.size - 1) / F.FS
    assert f["braking"] == pytest.approx(0.2 * 2 / np.pi * T / 2, rel=0.02)
    assert f["propulsive"] == pytest.approx(0.2 * 2 / np.pi * T / 2, rel=0.02)
    assert f["braking"] > 0 and f["propulsive"] > 0
    assert f["stance_s"] == pytest.approx(z.size / 250.0)
    assert np.isfinite(f["peak2"]) and f["valley"] < min(f["peak1"], f["peak2"])


def test_single_hump_curve_has_no_forced_second_peak():
    t = np.linspace(0, 1, 180); z = BW * 1.1 * np.sin(np.pi * t) + 25; a = np.zeros_like(z)
    f = F.trial_features(z, a, BW)
    assert np.isnan(f["peak2"]) and np.isnan(f["valley"])


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        F.trial_features(np.ones(5), np.ones(5), BW)
    with pytest.raises(ValueError):
        F.trial_features(np.r_[np.ones(20), np.nan], np.ones(21), BW)
    assert F.is_valid(np.full(50, 0.3 * BW), BW) == (False, "peak_below_half_bw")


def test_gp_kernels_match_scikit_learn():
    from sklearn.gaussian_process.kernels import ConstantKernel as C, Matern, RBF
    t = np.array([0.0, 0.3, 1.1, 2.0])
    for name, nu in (("matern12", .5), ("matern32", 1.5), ("matern52", 2.5), ("rbf", np.inf)):
        p = np.log([0.03, 0.05, 0.8])
        k = C(0.03 ** 2) + C(0.05 ** 2) * (RBF(0.8) if nu == np.inf else Matern(0.8, nu=nu))
        assert np.allclose(kernel_matrix(name, p, t, t), k(t[:, None]), atol=1e-14)


def test_gp_posterior_matches_scikit_learn():
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel as C, Matern, WhiteKernel
    t = np.array([0.0, 0.25, 0.5, 1.4]); y = np.array([0.01, -0.02, 0.03, 0.0])
    g = LongitudinalGP("matern32"); g.p = np.log([0.02, 0.04, 0.6, 0.015]); g.beta = np.zeros(1)
    m, v = g.predict_next(t, y, np.zeros(4), np.zeros((4, 1)), 2.0, np.zeros(1))
    sk = GaussianProcessRegressor(C(0.02 ** 2) + C(0.04 ** 2) * Matern(0.6, nu=1.5) + WhiteKernel(0.015 ** 2),
                                  optimizer=None, alpha=1e-10).fit(t[:, None], y)
    mm, ss = sk.predict(np.array([[2.0]]), return_std=True)
    assert m == pytest.approx(mm[0], abs=1e-12) and v + 0.015 ** 2 == pytest.approx(ss[0] ** 2, abs=1e-12)


def test_random_intercept_model_recovers_simulated_variances():
    rng = np.random.default_rng(3); ts, ys, ss, Xs = [], [], [], []
    for i in range(400):
        n = rng.integers(2, 6); t = np.sort(rng.uniform(0, 2, n)); b = rng.normal(0, 0.05)
        ys.append(0.1 + b + rng.normal(0, 0.02, n)); ts.append(t); ss.append(np.zeros(n)); Xs.append(np.ones((n, 1)))
    g = LongitudinalGP("lmm_int").fit(PatientData.from_lists(ts, ys, ss, Xs), 0.05, restarts=1)
    h = g.hyperparameters()
    assert h["s_intercept"] == pytest.approx(0.05, rel=0.15) and h["sigma_w"] == pytest.approx(0.02, rel=0.15)
    assert g.beta[0] == pytest.approx(0.1, abs=0.01)


def test_crps_interval_score_and_coverage():
    rng = np.random.default_rng(0); y = rng.normal(size=20000)
    df = pd.DataFrame({"y": y, "mu": 0.0, "sd": 1.0}); s = E.summarise(E.pointwise(df))
    assert s["CRPS"] == pytest.approx(1 / np.sqrt(np.pi), rel=0.02)   # E[CRPS] of a calibrated N(0,1) forecast
    assert s["cov95"] == pytest.approx(0.95, abs=0.01) and s["cov50"] == pytest.approx(0.5, abs=0.015)
    y0 = 0.7     # CRPS by numerical integration of (F - 1{x>=y})^2, split at the jump
    from scipy.stats import norm
    lo, hi = np.linspace(-10, y0, 20001), np.linspace(y0, 10, 20001)
    num = np.trapezoid(norm.cdf(lo) ** 2, lo) + np.trapezoid((norm.cdf(hi) - 1) ** 2, hi)
    assert E.crps_gaussian(y0, 0.0, 1.0) == pytest.approx(num, rel=1e-4)


def test_cluster_bootstrap_resamples_patients_not_rows():
    pid = np.repeat(np.arange(10), [1] * 9 + [91]); v = np.r_[np.zeros(9), np.ones(91)]
    d = E.cluster_boot_mean(v, pid, B=4000, seed=1)
    assert 0.3 < d.std() < 0.6        # one patient dominates rows; row bootstrap would give a tiny SD


def synthetic_visits(n_pat=30, seed=0):
    rng = np.random.default_rng(seed); rows = []
    for p in range(n_pat):
        n = rng.integers(1, 6); days = np.cumsum(np.r_[0, rng.integers(3, 15, n - 1)])
        for k in range(n):
            rows.append({"SUBJECT_ID": p, "SESSION_ID": 1000 * p + k, "visit_no": k + 1, "day": days[k],
                         "split": ["train", "calib", "test"][p % 3], "CLASS_LABEL": "AKHC"[p % 4], "AGE": 40 + p,
                         "SEX": 1 + p % 2, "BODY_MASS": 70 + p, "y": rng.normal(), "y_se": 0.1})
    return pd.DataFrame(rows)


def test_targets_use_only_earlier_visits():
    V = synthetic_visits(); std = P.Standardiser().fit(V)
    T1 = P.build_targets(P.patient_arrays(V, "y", "y_se", std), std)
    V2 = V.copy(); last = V2.groupby("SUBJECT_ID").visit_no.transform("max") == V2.visit_no
    V2.loc[last, "y"] += 100.0                       # change only each patient's final visit
    T2 = P.build_targets(P.patient_arrays(V2, "y", "y_se", std), std)
    hist = [c for c in P.HIST]
    assert np.allclose(T1[hist].to_numpy(), T2[hist].to_numpy())       # no history feature saw the future
    assert not np.allclose(T1.y, T2.y)


def test_gp_predictions_unchanged_by_future_visits():
    V = synthetic_visits(); std = P.Standardiser().fit(V)
    g = LongitudinalGP("matern32"); g.p = np.log([0.5, 0.5, 1.0, 0.3]); g.beta = np.zeros(8)
    arr = P.patient_arrays(V, "y", "y_se", std); T = P.build_targets(arr, std)
    m1, v1 = P.gp_predict_rows(g, arr, T)
    V2 = V.copy(); last = V2.groupby("SUBJECT_ID").visit_no.transform("max") == V2.visit_no
    V2.loc[last, "y"] -= 50
    arr2 = P.patient_arrays(V2, "y", "y_se", std); m2, v2 = P.gp_predict_rows(g, arr2, T)
    assert np.allclose(m1, m2) and np.allclose(v1, v2)


def test_partition_is_disjoint_stratified_and_deterministic():
    pts = pd.DataFrame({"SUBJECT_ID": range(500), "role": "cohort", "CLASS_LABEL": np.array(list("AKHC"))[np.arange(500) % 4]})
    a, b = splits.make_partition(pts), splits.make_partition(pts)
    pd.testing.assert_frame_equal(a, b)
    splits.assert_disjoint(a)
    share = a.groupby("CLASS_LABEL").split.value_counts(normalize=True).unstack()
    assert np.allclose(share["train"], 0.6, atol=0.01) and np.allclose(share["test"], 0.2, atol=0.01)
    with pytest.raises(AssertionError):
        splits.assert_disjoint(pd.concat([a, a.head(1).assign(split="test")]))
    with pytest.raises(AssertionError):
        splits.assert_only(pd.DataFrame({"SUBJECT_ID": a[a.split == "test"].SUBJECT_ID.head(3)}), a, {"train"}, "x")


def test_tau_moments_estimator():
    rng = np.random.default_rng(5); arrays = {}
    for p in range(3000):
        y = rng.normal(0, 1) + rng.normal(0, 0.03, 4) + rng.normal(0, 0.02, 4)   # 0.03 extra, 0.02 known noise
        arrays[p] = (np.arange(4.0), y, np.full(4, 0.02 ** 2))
    # var(diff) = 2*(0.03^2 + 0.02^2); minus mean(se1^2+se2^2)=2*0.02^2 -> tau = sqrt(2)*0.03
    assert P.tau_moments(arrays, list(arrays)) == pytest.approx(np.sqrt(2) * 0.03, rel=0.08)


def test_monitoring_scores():
    Pp = pd.DataFrame({"Y_prev": [-0.10, 0.0], "Y_cur": [0.0, 0.0], "se_prev": [0.01, 0.01], "se_cur": [0.01, 0.01],
                       "pa_prev": [1.0, 1.1], "pu_prev": [1.1, 1.1], "pa_cur": [1.1, 1.1], "pu_cur": [1.1, 1.1],
                       "mu_D": [-0.1, 0.0], "var_D": [0.0004, 0.0004]})
    S = M.scores(Pp, tau=0.03, mu_h=0.0, sd_h=0.04)
    assert S.score_C.iat[0] == pytest.approx(0.1 / np.sqrt(0.0002 + 0.0009))
    assert bool(S.alert_C.iat[0]) and not bool(S.alert_C.iat[1])
    assert S.score_D.iat[0] == pytest.approx(0.1 / np.sqrt(0.0004 + 0.0001))
    assert S.dLSI.iat[0] == pytest.approx(100 - 100 / 1.1 * 1.0)
    assert bool(S.alert_B.iat[0]) and S.score_B.iat[0] < 0       # moved into the healthy band
    assert np.isnan(M.lsi([1.0], [0.1])[0])                      # implausible denominator guarded


def test_matched_threshold_gives_requested_rate():
    rng = np.random.default_rng(0)
    S = pd.DataFrame({f"score_{a}": rng.normal(size=5000) for a in M.ARCH})
    thr = M.matched_thresholds(S, 0.10); al = M.matched_alerts(S, thr)
    assert np.allclose(al.mean(), 0.10, atol=0.002)


def test_agreement_statistics():
    x = np.array([1, 1, 0, 0, 1, 0, 0, 0], bool); y = np.array([1, 0, 0, 0, 1, 1, 0, 0], bool)
    s = M.pair_stats(x, y)
    assert s["agreement"] == pytest.approx(6 / 8) and s["joint_alert"] == pytest.approx(2 / 8)
    assert s["kappa"] == pytest.approx((0.75 - (3 / 8 * 3 / 8 + 5 / 8 * 5 / 8)) / (1 - (3 / 8 * 3 / 8 + 5 / 8 * 5 / 8)))
    assert M.pair_stats(x, x)["kappa"] == pytest.approx(1.0)


def test_perturbations_change_features_as_designed():
    z, a = two_hump(); base = F.trial_features(z, a, BW)
    f = F.trial_features(*PT.offset(z, a, BW, 0.02), BW); assert f["peak_v"] == pytest.approx(base["peak_v"] + 0.02)
    f = F.trial_features(*PT.scale(z, a, BW, 0.95), BW)
    assert f["peak_v"] == pytest.approx(0.95 * base["peak_v"]) and f["propulsive"] == pytest.approx(0.95 * base["propulsive"])
    f = F.trial_features(*PT.trim(z, a, BW, 2), BW); assert f["stance_s"] == pytest.approx(base["stance_s"] - 4 / 250)
    f = F.trial_features(*PT.inj_propulsive(z, a, BW, 0.2), BW)
    assert f["propulsive"] == pytest.approx(0.8 * base["propulsive"]) and f["braking"] == pytest.approx(base["braking"])
    f = F.trial_features(*PT.inj_braking(z, a, BW, 0.1), BW); assert f["braking"] == pytest.approx(1.1 * base["braking"])
    f = F.trial_features(*PT.inj_peak1(z, a, BW, 0.1), BW); assert f["peak1"] < base["peak1"]
    r = PT.resampled_features(z, a, BW, 101, "linear"); assert r["peak_v"] == pytest.approx(base["peak_v"], rel=0.01)


def test_visit_summary_difference_and_se():
    z, a = two_hump()
    ta = [(z * s, a) for s in (1.0, 1.02, 0.98)]; tu = [(z * s, a) for s in (1.1, 1.12, 1.08)]
    v = PT.visit_summary(ta, tu, BW)
    pa = np.array([F.trial_features(*t, BW)["peak_v"] for t in ta]); pu = np.array([F.trial_features(*t, BW)["peak_v"] for t in tu])
    assert v["d_peak_v"] == pytest.approx(pa.mean() - pu.mean())
    assert v["d_peak_v_se"] == pytest.approx(np.sqrt(pa.var(ddof=1) / 3 + pu.var(ddof=1) / 3))
