"""Model fitting and evaluation, implementing docs/preregistration.md."""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from .cohort import build_cohort, load_metadata

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

GAIT = ["peak_loading", "braking_impulse", "propulsive_impulse"]
CS = np.logspace(-3, 2, 10)


@dataclass
class Data:
    cohort: pd.DataFrame        # one row per patient: predictors, event, groups
    hc_curves: np.ndarray       # healthy-control leg curves (n_legs x 202)
    hc_subject: np.ndarray      # subject id per healthy-control leg
    pt_curves: np.ndarray       # affected-leg curves aligned with cohort rows


def load_data(root: str | Path) -> Data:
    root = Path(root)
    meta = load_metadata(str(root / "GRF_metadata.csv"))
    cohort, _ = build_cohort(meta)
    t = pd.concat([pd.read_csv(root / "derived" / f"trial_features_{s}.csv") for s in ("left", "right")])
    t = t[t["valid"] & t["SESSION_ID"].isin(cohort["index_session"])]
    side = cohort.set_index("index_session")["AFFECTED_SIDE"].map({0: "left", 1: "right"})
    t["leg"] = np.where(t["side"] == t["SESSION_ID"].map(side), "affected", "unaffected")
    means = t.groupby(["SESSION_ID", "leg"])[GAIT].mean().unstack("leg")
    diff = pd.DataFrame({f: means[(f, "affected")] - means[(f, "unaffected")] for f in GAIT})
    cohort = cohort.merge(diff, left_on="index_session", right_index=True, how="left")
    for g in ("A", "C", "H"):
        cohort[f"inj_{g}"] = (cohort["CLASS_LABEL"] == g).astype(float)
    cohort["male"] = (cohort["SEX"] == 1).astype(float)

    curves = {s: np.load(root / "derived" / f"curves_{s}.npz") for s in ("left", "right")}
    lookup = {s: dict(zip(curves[s]["session_id"], curves[s]["curves"])) for s in curves}
    pt = np.stack([lookup[side[s]][s] for s in cohort["index_session"]]).astype(float)
    hc_sess = meta.loc[meta["CLASS_LABEL"] == "HC", ["SESSION_ID", "SUBJECT_ID"]]
    hc_x, hc_g = [], []
    for s, sid in hc_sess.itertuples(index=False):
        for leg in ("left", "right"):
            if s in lookup[leg]:
                hc_x.append(lookup[leg][s]); hc_g.append(sid)
    return Data(cohort.reset_index(drop=True), np.array(hc_x, float), np.array(hc_g), pt)


M0 = ["AGE", "male", "BODY_MASS", "inj_A", "inj_C", "inj_H"]
MODELS = {"M0": M0, "M1": M0 + ["recon"], "M2": M0 + GAIT, "M2+": M0 + ["recon"] + GAIT}


def recon_scores(d: Data, hc_train_mask: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Autoencoder trained on training-fold healthy-control legs; reconstruction MSE of affected legs."""
    sc = StandardScaler().fit(d.hc_curves[hc_train_mask])
    X = sc.transform(d.hc_curves[hc_train_mask])
    ae = MLPRegressor(hidden_layer_sizes=(64, 16, 64), activation="relu", early_stopping=True,
                      max_iter=500, random_state=0).fit(X, X)
    P = sc.transform(d.pt_curves[rows])
    return ((ae.predict(P) - P) ** 2).mean(axis=1)


def _fit_predict(X_tr, y_tr, X_te, seed):
    sc = StandardScaler().fit(X_tr)
    inner = StratifiedKFold(5, shuffle=True, random_state=seed)
    m = LogisticRegressionCV(Cs=CS, cv=inner, scoring="neg_log_loss", max_iter=5000)
    m.fit(sc.transform(X_tr), y_tr)
    return m.predict_proba(sc.transform(X_te))[:, 1]


def cv_predict(d: Data, y: np.ndarray, rows: np.ndarray, groups: np.ndarray, models, seed: int,
               with_recon: bool) -> dict[str, np.ndarray]:
    """One repeat of 5-fold patient-level CV; returns out-of-fold predictions per model."""
    out = {m: np.zeros(len(rows)) for m in models}
    folds = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
    hc_subj = np.unique(d.hc_subject)
    hc_fold = dict(zip(np.random.default_rng(seed).permutation(hc_subj), np.arange(len(hc_subj)) % 5))
    for k, (tr, te) in enumerate(folds.split(np.zeros(len(rows)), y, groups)):
        X = d.cohort.iloc[rows].reset_index(drop=True)
        if with_recon:
            mask = np.array([hc_fold[s] != k for s in d.hc_subject])
            X = X.assign(recon=recon_scores(d, mask, rows))
        for m in models:
            cols = MODELS[m]
            out[m][te] = _fit_predict(X.loc[tr, cols].to_numpy(float), y[tr], X.loc[te, cols].to_numpy(float), seed)
    return out


def calibration(y, p):
    lp = np.log(np.clip(p, 1e-9, 1 - 1e-9) / np.clip(1 - p, 1e-9, 1))
    slope = LogisticRegression(C=1e6, max_iter=1000).fit(lp[:, None], y).coef_[0, 0]
    # calibration-in-the-large: intercept with the linear predictor as an offset (Newton steps)
    a = 0.0
    for _ in range(50):
        q = 1 / (1 + np.exp(-(a + lp)))
        a += (y - q).sum() / max((q * (1 - q)).sum(), 1e-9)
    return float(a), float(slope)


def metrics(y, p, alert_rate=0.10):
    k = int(np.ceil(alert_rate * len(p)))
    top = np.argsort(-p)[:k]
    tp = int(y[top].sum())
    a, s = calibration(y, p)
    return {"auroc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p),
            "brier": brier_score_loss(y, p), "cal_intercept": a, "cal_slope": s,
            "ppv_at_10pct": tp / k, "false_alerts_per_true": (k - tp) / tp if tp else np.inf,
            "events_caught_at_10pct": tp}


def c_index(time, event, risk):
    """Harrell's C for right-censored data."""
    n = c = 0
    for i in np.where(event == 1)[0]:
        at_risk = time > time[i]
        n += at_risk.sum()
        c += (risk[i] > risk[at_risk]).sum() + 0.5 * (risk[i] == risk[at_risk]).sum()
    return c / n
