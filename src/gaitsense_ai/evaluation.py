"""Probabilistic forecast evaluation and patient-clustered bootstrap."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

LEVELS = [0.5, 0.8, 0.9, 0.95]


def crps_gaussian(y, mu, sd):
    z = (y - mu) / sd
    return sd * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))


def interval_score(y, mu, sd, level=0.95):
    a = 1 - level; q = norm.ppf(1 - a / 2); lo, hi = mu - q * sd, mu + q * sd
    return (hi - lo) + (2 / a) * (lo - y) * (y < lo) + (2 / a) * (y - hi) * (y > hi)


def nlpd(y, mu, sd):
    return -norm.logpdf(y, mu, sd)


def pointwise(df: pd.DataFrame) -> pd.DataFrame:
    """df columns y, mu, sd -> per-target scores."""
    y, mu, sd = df.y.to_numpy(), df.mu.to_numpy(), df.sd.to_numpy()
    out = pd.DataFrame({"abs_err": np.abs(y - mu), "sq_err": (y - mu) ** 2, "crps": crps_gaussian(y, mu, sd),
                        "is95": interval_score(y, mu, sd), "nlpd": nlpd(y, mu, sd), "width95": 2 * norm.ppf(0.975) * sd,
                        "pit": norm.cdf((y - mu) / sd)}, index=df.index)
    for lv in LEVELS:
        out[f"cov{int(lv * 100)}"] = (np.abs(y - mu) <= norm.ppf(0.5 + lv / 2) * sd).astype(float)
    return out


def summarise(scores: pd.DataFrame) -> dict:
    s = {"n": int(len(scores)), "MAE": scores.abs_err.mean(), "RMSE": float(np.sqrt(scores.sq_err.mean())),
         "CRPS": scores.crps.mean(), "IS95": scores.is95.mean(), "NLPD": scores.nlpd.mean(),
         "width95": scores.width95.mean()}
    for lv in LEVELS:
        s[f"cov{int(lv * 100)}"] = scores[f"cov{int(lv * 100)}"].mean()
    return {k: float(v) for k, v in s.items()}


def bootstrap_weights(patients, B=2000, seed=0):
    """Multinomial patient weights (B x n_patients) for a cluster bootstrap."""
    rng = np.random.default_rng(seed); u = np.unique(patients)
    return u, rng.multinomial(len(u), np.full(len(u), 1 / len(u)), size=B)


def cluster_boot_mean(values, patients, B=2000, seed=0, weights=None):
    """Bootstrap distribution of the target-level mean of `values`, resampling patients."""
    if weights is None:
        u, W = bootstrap_weights(patients, B, seed)
    else:
        u, W = weights
    idx = pd.Index(u).get_indexer(patients)
    s = np.bincount(idx, weights=values, minlength=len(u)); c = np.bincount(idx, minlength=len(u))
    return (W @ s) / (W @ c)


def ci(dist, level=0.95):
    a = (1 - level) / 2
    return [float(np.quantile(dist, a)), float(np.quantile(dist, 1 - a))]


def sd_informativeness(scores: pd.DataFrame, sd) -> dict:
    sd = np.asarray(sd); q = pd.qcut(sd, 5, labels=False, duplicates="drop")
    rho = spearmanr(sd, scores.abs_err).statistic
    by = scores.assign(q=q).groupby("q").agg(mean_sd=("width95", lambda w: (w / 3.92).mean()),
                                             mae=("abs_err", "mean"), cov95=("cov95", "mean"), n=("abs_err", "size"))
    return {"spearman_sd_abs_error": float(rho), "by_sd_quintile": by.reset_index().to_dict(orient="records")}
