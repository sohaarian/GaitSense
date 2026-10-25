"""Future-visit prediction: target construction (history only) and models M0a, M0b, M1, M2, M3, M4."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .longitudinal import MONTH, LongitudinalGP, PatientData

TARGETS = {"d_peak_v": "d_peak_v_se", "d_braking": "d_braking_se", "d_propulsive": "d_propulsive_se",
           "d_stance_s": "d_stance_s_se", "peak_v_a": "peak_v_a_se"}
CLASSES = ["K", "H", "C"]          # reference class: A (ankle)
HIST = ["last", "last_se", "hist_mean", "hist_sd", "last_minus_first", "hist_slope", "has_slope", "n_prior",
        "log_gap", "t_last_m", "t_target_m", "cls_K", "cls_H", "cls_C", "age_z", "sex", "mass_z"]


def n_prior_group(n):
    return np.where(n >= 3, "3+", n.astype(int).astype(str))


class Standardiser:
    """Covariate scaling fitted on training patients only."""
    def fit(self, V):
        b = V.drop_duplicates("SUBJECT_ID")
        self.age = (b.AGE.mean(), b.AGE.std()); self.mass = (b.BODY_MASS.mean(), b.BODY_MASS.std()); return self

    def fixed_effects(self, r, t_months):
        """Fixed-effect design row: intercept, injury class dummies, age, sex, body mass, time (months)."""
        return np.array([1.0] + [float(r.CLASS_LABEL == c) for c in CLASSES]
                        + [(r.AGE - self.age[0]) / self.age[1], float(r.SEX == 1),
                           (r.BODY_MASS - self.mass[0]) / self.mass[1], t_months])


def patient_arrays(V, y, se, std: Standardiser):
    """Per patient: (t months, y, se^2, X) arrays in visit order, plus static row."""
    out = {}
    for pid, p in V.sort_values(["SUBJECT_ID", "visit_no"]).groupby("SUBJECT_ID"):
        t = p.day.to_numpy() / MONTH; r = p.iloc[0]
        X = np.array([std.fixed_effects(r, ti) for ti in t])
        out[pid] = (t, p[y].to_numpy(float), p[se].to_numpy(float) ** 2, X, r, p)
    return out


def build_targets(arrays, std: Standardiser) -> pd.DataFrame:
    """One row per visit k >= 2 with history features computed from visits 1..k-1 only."""
    rows = []
    for pid, (t, y, se2, X, r, p) in arrays.items():
        for k in range(1, len(t)):
            h = slice(0, k); th, yh = t[h], y[h]
            slope = np.polyfit(th, yh, 1)[0] if k >= 2 and np.ptp(th) > 0 else 0.0
            rows.append({"SUBJECT_ID": pid, "split": p.split.iat[0], "CLASS_LABEL": r.CLASS_LABEL,
                         "SESSION_ID": int(p.SESSION_ID.iat[k]), "k": k + 1, "y": y[k], "se2_target": se2[k],
                         "last": yh[-1], "last_se": np.sqrt(se2[k - 1]), "hist_mean": yh.mean(),
                         "hist_sd": yh.std(ddof=1) if k >= 2 else 0.0, "last_minus_first": yh[-1] - yh[0],
                         "hist_slope": slope, "has_slope": float(k >= 2 and np.ptp(th) > 0), "n_prior": k,
                         "gap_days": (t[k] - t[k - 1]) * MONTH, "log_gap": np.log1p((t[k] - t[k - 1]) * MONTH),
                         "t_last_m": th[-1], "t_target_m": t[k],
                         "cls_K": float(r.CLASS_LABEL == "K"), "cls_H": float(r.CLASS_LABEL == "H"),
                         "cls_C": float(r.CLASS_LABEL == "C"),
                         "age_z": (r.AGE - std.age[0]) / std.age[1], "sex": float(r.SEX == 1),
                         "mass_z": (r.BODY_MASS - std.mass[0]) / std.mass[1], "sbar2": se2[h].mean()})
    T = pd.DataFrame(rows)
    T["n_prior_grp"] = n_prior_group(T.n_prior.to_numpy())
    return T


def tau_moments(arrays, pids):
    """v1.x estimator: sqrt(max(0, var(consecutive change) - mean(SE1^2 + SE2^2))) on the given patients."""
    d, s = [], []
    for pid in pids:
        t, y, se2 = arrays[pid][:3]
        if len(y) > 1:
            d.extend(np.diff(y)); s.extend(se2[1:] + se2[:-1])
    return float(np.sqrt(max(np.var(d, ddof=1) - np.mean(s), 0.0)))


def resid_sd_by_group(T, mu):
    """RMS residual by prior-visit group (used to give M1/M4 a Gaussian predictive SD; fitted on calibration)."""
    r = (T.y - mu) ** 2
    return {g: float(np.sqrt(r[T.n_prior_grp == g].mean())) for g in sorted(T.n_prior_grp.unique())}


def cv_select(make, grid, Xtr, ytr, groups, seed):
    best = None
    for params in grid:
        err = []
        for a, b in GroupKFold(5, shuffle=True, random_state=seed).split(Xtr, ytr, groups):
            m = make(**params).fit(Xtr[a], ytr[a]); err.append(np.mean(np.abs(m.predict(Xtr[b]) - ytr[b])))
        if best is None or np.mean(err) < best[0]:
            best = (float(np.mean(err)), params)
    return best[1], best[0]


def make_ridge(alpha):
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha))


def make_hgb(max_depth, learning_rate):
    return HistGradientBoostingRegressor(max_depth=max_depth, learning_rate=learning_rate, max_iter=300,
                                         early_stopping=False, random_state=0)


RIDGE_GRID = [{"alpha": a} for a in np.logspace(-3, 3, 13)]
HGB_GRID = [{"max_depth": d, "learning_rate": lr} for d in (2, 3) for lr in (0.03, 0.1)]


def gp_predict_rows(model: LongitudinalGP, arrays, T):
    """Predictive mean and latent variance for each target row from that patient's earlier visits."""
    mu, vl = np.empty(len(T)), np.empty(len(T))
    for i, (pid, k) in enumerate(zip(T.SUBJECT_ID.to_numpy(), T.k.to_numpy())):
        t, y, se2, X = arrays[pid][:4]; j = k - 1
        mu[i], vl[i] = model.predict_next(t[:j], y[:j], se2[:j], X[:j], t[j], X[j])
    return mu, vl


def training_data(arrays, pids):
    sel = [arrays[p] for p in pids]
    return PatientData.from_lists([a[0] for a in sel], [a[1] for a in sel], [a[2] for a in sel], [a[3] for a in sel])
