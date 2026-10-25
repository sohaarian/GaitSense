"""Four monitoring architectures applied to the same consecutive visit pairs (t-1, t).

Common target Y = affected-minus-unaffected visit mean of peak vertical force (BW). Every output describes a
measurement; none is a diagnosis, a recovery judgement or a work-capacity judgement.

Input pair columns: Y_prev, Y_cur, se_prev, se_cur (SE of Y), pa_prev, pu_prev, pa_cur, pu_cur (affected / unaffected
peak_v visit means), mu_D, var_D (predictive mean and latent+extra variance of Y_cur from visits < t).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA

Z = norm.ppf(0.975)
ARCH = ["A", "B", "C", "D"]
LSI_POINTS = 10.0


def lsi(pa, pu):
    """Limb symmetry index (affected / unaffected x 100). NaN if the denominator is implausibly small."""
    pu = np.asarray(pu, float)
    return np.where(pu > 0.5, 100.0 * np.asarray(pa, float) / np.where(pu > 0.5, pu, 1.0), np.nan)


def scores(P: pd.DataFrame, tau: float, mu_h: float, sd_h: float, z: float = Z) -> pd.DataFrame:
    """Continuous scores and native alerts of A-D (plus C0 = v1.x original limit with tau = 0)."""
    o = pd.DataFrame(index=P.index)
    dY = P.Y_cur - P.Y_prev
    # A: raw asymmetry
    o["dLSI"] = lsi(P.pa_cur, P.pu_cur) - lsi(P.pa_prev, P.pu_prev)
    o["score_A"] = dY
    o["alert_A"] = np.abs(o.dLSI) >= LSI_POINTS
    o["dir_A"] = np.sign(dY)
    o["d_affected"] = P.pa_cur - P.pa_prev; o["d_unaffected"] = P.pu_cur - P.pu_prev
    # B: healthy population reference
    zp, zc = (P.Y_prev - mu_h) / sd_h, (P.Y_cur - mu_h) / sd_h
    o["zB_prev"], o["zB_cur"] = zp, zc
    o["score_B"] = np.abs(zc) - np.abs(zp)                      # > 0: moved away from the healthy mean
    o["alert_B"] = (np.abs(zp) <= z) != (np.abs(zc) <= z)        # crossed the healthy 95% band
    o["dir_B"] = np.sign(o.score_B)
    # C: statistical change limit (tau cross-fitted on training patients); C0: same-day noise only
    o["score_C"] = dY / np.sqrt(P.se_cur ** 2 + P.se_prev ** 2 + tau ** 2)
    o["alert_C"] = np.abs(o.score_C) > z; o["dir_C"] = np.sign(o.score_C)
    o["score_C0"] = dY / np.sqrt(P.se_cur ** 2 + P.se_prev ** 2)
    o["alert_C0"] = np.abs(o.score_C0) > z
    # D: deviation from the longitudinal model's prediction (target's own SE known after measurement)
    o["score_D"] = (P.Y_cur - P.mu_D) / np.sqrt(P.var_D + P.se_cur ** 2)
    o["alert_D"] = np.abs(o.score_D) > z; o["dir_D"] = np.sign(o.score_D)
    return o


def matched_thresholds(S_cal: pd.DataFrame, rate: float = 0.10) -> dict:
    """Two-sided thresholds on |score| giving `rate` alerts in calibration pairs."""
    return {a: float(np.quantile(np.abs(S_cal[f"score_{a}"].dropna()), 1 - rate)) for a in ARCH}


def matched_alerts(S: pd.DataFrame, thr: dict) -> pd.DataFrame:
    return pd.DataFrame({a: np.abs(S[f"score_{a}"]) > thr[a] for a in ARCH}, index=S.index)


def pair_stats(x, y):
    """Agreement statistics for two boolean alert vectors."""
    x, y = np.asarray(x, bool), np.asarray(y, bool)
    a = np.mean(x & y); b = np.mean(x & ~y); c = np.mean(~x & y); d = np.mean(~x & ~y)
    po = a + d; pe = (a + b) * (a + c) + (c + d) * (b + d)
    kappa = (po - pe) / (1 - pe) if pe < 1 else np.nan
    pos = 2 * a / (2 * a + b + c) if (2 * a + b + c) > 0 else np.nan
    return {"agreement": po, "disagreement": 1 - po, "kappa": kappa, "positive_agreement": pos, "joint_alert": a,
            "only_first": b, "only_second": c}


def fit_healthy_reference(hc_visits: pd.DataFrame):
    """Mean and SD of signed (random-leg minus other-leg) peak_v difference in healthy controls."""
    y = hc_visits.d_peak_v.to_numpy()
    return float(y.mean()), float(y.std(ddof=1))


class CurveReference:
    """Curve-level healthy reference: PCA (to 95% variance) + Ledoit-Wolf covariance, fitted on training HC only."""
    def fit(self, curves):
        self.pca = PCA(n_components=0.95, svd_solver="full").fit(curves)
        self.cov = LedoitWolf().fit(self.pca.transform(curves)); return self

    def distance(self, curves):
        return np.sqrt(self.cov.mahalanobis(self.pca.transform(curves)))
