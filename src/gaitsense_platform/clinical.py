"""Capability 2: clinical recovery prediction and the incremental value of biomechanics.

Model A: clinical covariates only. Model B: biomechanical features only. Model C: both.
The central comparison is C versus A: do uncertainty-aware biomechanical forecasts add predictive value for an
independently measured outcome beyond clinical data alone? All estimates come from patient-level cross-validation
(out-of-fold predictions) with patient-clustered bootstrap intervals for paired differences.

Binary endpoints (e.g. meaningful improvement, delayed recovery, return to work by a date): penalised logistic
regression; AUROC, Brier score, calibration intercept and slope, net benefit. Continuous endpoints (e.g. a function
score): ridge regression with a Gaussian predictive SD from out-of-fold residuals; MAE and CRPS.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from gaitsense_ai.evaluation import crps_gaussian


def meaningful_improvement(baseline: pd.Series, followup: pd.Series, mcid: float, higher_is_better: bool = True):
    """Binary endpoint: change from baseline reaches a pre-specified minimal clinically important difference."""
    d = (followup - baseline) if higher_is_better else (baseline - followup)
    return (d >= mcid).astype(int)


def _oof(X, y, binary, folds, seed):
    split = (StratifiedKFold if binary else KFold)(folds, shuffle=True, random_state=seed)
    pred = np.empty(len(y))
    for tr, te in split.split(X, y):
        m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000) if binary else Ridge(alpha=1.0))
        m.fit(X[tr], y[tr])
        pred[te] = m.predict_proba(X[te])[:, 1] if binary else m.predict(X[te])
    return pred


def calibration_intercept_slope(y, p):
    lp = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
    m = LogisticRegression(C=1e6, max_iter=2000).fit(lp[:, None], y)
    return float(m.intercept_[0]), float(m.coef_[0, 0])


def net_benefit(y, p, threshold):
    n = len(y); tp = np.sum((p >= threshold) & (y == 1)); fp = np.sum((p >= threshold) & (y == 0))
    return float(tp / n - fp / n * threshold / (1 - threshold))


def compare_models(table: pd.DataFrame, clinical_cols, bio_cols, binary: bool, folds: int = 5, seed: int = 20261024,
                   B: int = 1000, thresholds=(0.1, 0.2, 0.3)) -> dict:
    """Out-of-fold Models A/B/C on a patient-level table (one row per patient, column 'outcome')."""
    y = table.outcome.to_numpy(float if not binary else int)
    sets = {"A_clinical": list(clinical_cols), "B_biomech": list(bio_cols), "C_combined": list(clinical_cols) + list(bio_cols)}
    preds = {k: _oof(table[c].to_numpy(float), y, binary, folds, seed) for k, c in sets.items()}
    rng = np.random.default_rng(seed); n = len(y)
    idx = [rng.integers(0, n, n) for _ in range(B)]
    res = {"n_patients": int(n), "binary": binary, "models": {}, "C_minus_A": {}, "B_minus_A": {}}
    if binary:
        res["event_rate"] = float(y.mean())
        for k, p in preds.items():
            ci, cs = calibration_intercept_slope(y, p)
            res["models"][k] = {"auroc": float(roc_auc_score(y, p)), "brier": float(brier_score_loss(y, p)),
                                "calibration_intercept": ci, "calibration_slope": cs,
                                "net_benefit": {str(t): net_benefit(y, p, t) for t in thresholds}}

        def stat(fn, a, b):
            est = fn(y, preds[a]) - fn(y, preds[b]); d = []
            for ix in idx:
                if 0 < y[ix].sum() < n:
                    d.append(fn(y[ix], preds[a][ix]) - fn(y[ix], preds[b][ix]))
            return {"estimate": float(est), "ci95": [float(np.quantile(d, .025)), float(np.quantile(d, .975))]}
        for a, key in (("C_combined", "C_minus_A"), ("B_biomech", "B_minus_A")):
            res[key] = {"auroc": stat(roc_auc_score, a, "A_clinical"), "brier": stat(brier_score_loss, a, "A_clinical")}
    else:
        for k, p in preds.items():
            sd = float(np.sqrt(np.mean((y - p) ** 2)))
            res["models"][k] = {"mae": float(np.mean(np.abs(y - p))), "crps": float(np.mean(crps_gaussian(y, p, sd)))}
        for a, key in (("C_combined", "C_minus_A"), ("B_biomech", "B_minus_A")):
            ea, eb = np.abs(y - preds[a]), np.abs(y - preds["A_clinical"])
            d = [ea[ix].mean() - eb[ix].mean() for ix in idx]
            res[key] = {"mae": {"estimate": float(ea.mean() - eb.mean()),
                                "ci95": [float(np.quantile(d, .025)), float(np.quantile(d, .975))]}}
    res["oof_predictions"] = {k: v.tolist() for k, v in preds.items()}
    return res
