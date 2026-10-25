"""Study 4 of docs/shear_protocol.md: symmetry-only tracking (4a) and the persistence ceiling (4b).

Uses true force-plate peak loading only (no reconstruction). Same patient split as shear_study.py (seed 0).
Writes results/shear_study4.json.
"""
import json, sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parents[1]; sys.path.insert(0, str(ROOT / "scripts"))
from shear_data import load_steps

Z = 1.96
st, _, _, _ = load_steps()
HC = st[st.role == "healthy"]; st = st[st.role != "healthy"]
rng = np.random.default_rng(0)
ids = np.sort(st.SUBJECT_ID.unique()); perm = rng.permutation(ids)
split = {s: ("train" if i < 0.6 * len(ids) else "val" if i < 0.8 * len(ids) else "test") for i, s in enumerate(perm)}

g = st.groupby(["SESSION_ID", "role"]).peak_loading.agg(["mean", "var", "count"]).unstack("role")
g = g[(g[("count", "affected")] >= 2) & (g[("count", "unaffected")] >= 2)]
v = pd.DataFrame({"a": g[("mean", "affected")], "u": g[("mean", "unaffected")],
                  "se2_a": g[("var", "affected")] / g[("count", "affected")], "se2_u": g[("var", "unaffected")] / g[("count", "unaffected")]})
info = st.drop_duplicates("SESSION_ID").set_index("SESSION_ID")[["SUBJECT_ID", "DATE", "CLASS_LABEL", "age_first"]]
v = v.join(info).reset_index().sort_values(["SUBJECT_ID", "DATE", "SESSION_ID"]).reset_index(drop=True)
v["diff"] = v.a - v.u; v["se2"] = v.se2_a + v.se2_u
v["split"] = v.SUBJECT_ID.map(split)
prev = v.groupby("SUBJECT_ID").shift(1)
p = v[prev.SESSION_ID.notna()].copy()
for c in ("a", "u", "diff"):
    p["d_" + c] = v[c] - prev[c]
p["s2"] = v.se2 + prev.se2; p["s2_u"] = v.se2_u + prev.se2_u
res = {}

# ---- 4a: symmetry-only tracking ----
tau = float(np.sqrt(max(p.d_diff.var() - p.s2.mean(), 0)))
tau_u = float(np.sqrt(max(p.d_u.var() - p.s2_u.mean(), 0)))
beyond = (p.d_diff.abs() > Z * np.sqrt(p.s2 + tau ** 2))
driven_u = p.d_u.abs() >= 0.5 * p.d_diff.abs()
u_beyond = p.d_u.abs() > Z * np.sqrt(p.s2_u + tau_u ** 2)
hcs = HC.groupby("SESSION_ID").peak_loading.mean()
lo, hi = float(hcs.quantile(0.025)), float(hcs.quantile(0.975))
first = v.groupby("SUBJECT_ID").first()
res["4a"] = {"pairs": int(len(p)), "tau_diff": tau, "tau_unaffected_leg": tau_u,
             "pairs_beyond_limit": int(beyond.sum()),
             "share_of_beyond_pairs_where_unaffected_change_is_half_or_more": float(driven_u[beyond].mean()),
             "share_of_beyond_pairs_where_unaffected_leg_itself_changed_beyond_its_limit": float(u_beyond[beyond].mean()),
             "share_of_all_pairs_where_unaffected_leg_changed_beyond_its_limit": float(u_beyond.mean()),
             "healthy_sessions": int(len(hcs)), "healthy_range_95": [lo, hi],
             "patients_first_visit_unaffected_outside_healthy_range": float(((first.u < lo) | (first.u > hi)).mean()),
             "patients_first_visit_affected_outside_healthy_range": float(((first.a < lo) | (first.a > hi)).mean()),
             "note": "Healthy controls walked at self-selected speed; patient speed was not recorded."}

# ---- 4b: persistence ceiling, forecasting the next visit's loading difference ----
rows = []
for sid, h in v.groupby("SUBJECT_ID"):
    d = h["diff"].to_numpy(); t = (h.DATE - h.DATE.iat[0]).dt.days.to_numpy()
    for k in range(1, len(h)):
        hist = d[:k]
        slope = np.polyfit(t[:k], hist, 1)[0] if k >= 2 and t[k - 1] > t[0] else 0.0
        rows.append({"SUBJECT_ID": sid, "split": h.split.iat[0], "age_first": h.age_first.iat[0], "grp": h.CLASS_LABEL.iat[0],
                     "y": d[k], "last": hist[-1], "first": hist[0], "mean": hist.mean(), "slope": slope,
                     "gap": t[k] - t[k - 1], "since_first": t[k], "k": k})
f = pd.DataFrame(rows)
for gname in ("K", "A", "H", "C"):
    f["g_" + gname] = (f.grp == gname).astype(float)
X = ["last", "first", "mean", "slope", "gap", "since_first", "k", "g_K", "g_A", "g_H", "g_C"]
tr = f.split == "train"
m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 13))).fit(f.loc[tr, X], f.loc[tr, "y"])
te = f[(f.split == "test") & f.age_first.between(18, 65)].copy()
te["pred_ridge"] = m.predict(te[X]); te["pred_persist"] = te["last"]
ae_p, ae_r = (te.y - te.pred_persist).abs().to_numpy(), (te.y - te.pred_ridge).abs().to_numpy()
pid = te.SUBJECT_ID.to_numpy(); uid = np.unique(pid); idx = {u: np.where(pid == u)[0] for u in uid}
r = np.random.default_rng(1); diffs = []
for _ in range(500):
    ii = np.concatenate([idx[u] for u in r.choice(uid, len(uid))])
    diffs.append(ae_p[ii].mean() - ae_r[ii].mean())
res["4b"] = {"test_forecasts_18_65": int(len(te)), "mae_persistence": float(ae_p.mean()), "mae_ridge_history": float(ae_r.mean()),
             "persistence_minus_ridge_mae_ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
             "ridge_beats_persistence": bool(np.percentile(diffs, 2.5) > 0),
             "mae_ridge_relative_to_persistence": float(ae_r.mean() / ae_p.mean())}
(ROOT / "results" / "shear_study4.json").write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))
