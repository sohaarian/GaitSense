"""Study 3 of docs/shear_protocol.md: wearable degradation of the vertical input.

The AI chosen in shear_study.py (trained on clean 250 Hz force-plate curves) is applied to degraded vertical
input for validation and test patients:
  sampling 250 / 100 / 25 Hz  x  insole error none / (per-visit-leg gain SD 5% + white noise 2% BW)
  x  steps per visit: all / 10 / 5 / 3 per leg (random subset; visits with fewer steps dropped).
tau is re-estimated on validation patients under the same degradation; test patients aged 18-65 are evaluated.
Criterion: crossing rate 4-6% and detection of a planted 0.004 BW*s push-off decline >= 80% of condition (a),
the force-plate signal with all steps.
Writes results/shear_study3.json.
"""
import json, pickle, sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parents[1]; sys.path.insert(0, str(ROOT / "scripts"))
from shear_data import load_steps

Z = 1.96; GROUPS = ["K", "A", "H", "C"]
st, C250, C100, C25 = load_steps()
keep = (st.role != "healthy").to_numpy()
st, CURVES = st[keep].reset_index(drop=True), {250: C250[keep], 100: C100[keep], 25: C25[keep]}
rng = np.random.default_rng(0)
ids = np.sort(st.SUBJECT_ID.unique()); perm = rng.permutation(ids)
split = {s: ("train" if i < 0.6 * len(ids) else "val" if i < 0.8 * len(ids) else "test") for i, s in enumerate(perm)}
st["split"] = st.SUBJECT_ID.map(split)
M = pickle.load(open(ROOT / "data" / "derived" / "shear_models.pkl", "rb"))
AI, models, ysc = M["AI"], M["models"], M["ysc"]
base = json.loads((ROOT / "results" / "shear_study.json").read_text())


def handcrafted(c, s):
    t1 = c[:, :51].argmax(1); p1 = c[:, :51].max(1)
    t2 = 50 + c[:, 50:].argmax(1); p2 = c[:, 50:].max(1)
    val = np.array([c[i, t1[i]:t2[i] + 1].min() for i in range(len(c))]); tv = np.array([t1[i] + c[i, t1[i]:t2[i] + 1].argmin() for i in range(len(c))])
    lr = p1 / np.maximum(t1, 1) * 100 / s; ur = p2 / np.maximum(100 - t2, 1) * 100 / s
    m = c.mean(1)
    return np.column_stack([p1, p2, val, t1, t2, tv, lr, ur, m, m * s, s, p2 - p1, p1 - val])


def design(df, c):
    s = df.stance_s.to_numpy()
    Cv = np.column_stack([df.BODY_MASS.to_numpy(), (df.role == "affected").to_numpy(float)] + [(df.CLASS_LABEL == g).to_numpy(float) for g in GROUPS])
    H = handcrafted(c, s)
    return {"lin": np.column_stack([H, Cv]), "gb": np.column_stack([H, Cv, c]), "mlp": np.column_stack([c, s, Cv])}


def predict_push(df, c):
    X = design(df, c)
    if AI == "mlp":
        return models["mlp"].predict(X["mlp"])[:, 1] * ysc[1]
    return models[AI][1].predict(X[AI])


def ccc(a, b):
    return float(2 * np.cov(a, b, ddof=0)[0, 1] / (a.var() + b.var() + (a.mean() - b.mean()) ** 2))


def visit_pairs(df, pred):
    d = df[["SUBJECT_ID", "SESSION_ID", "DATE", "role", "age_first", "split", "push"]].assign(pred=pred)
    agg = d.groupby(["SESSION_ID", "role"])[["pred", "push"]].agg(["mean", "var", "count"]).unstack("role")
    ok = (agg[("pred", "count", "affected")] >= 2) & (agg[("pred", "count", "unaffected")] >= 2)
    agg = agg[ok]
    v = pd.DataFrame(index=agg.index)
    for c in ("pred", "push"):
        v[c] = agg[(c, "mean", "affected")] - agg[(c, "mean", "unaffected")]
        v[c + "_se2"] = (agg[(c, "var", "affected")] / agg[(c, "count", "affected")] +
                         agg[(c, "var", "unaffected")] / agg[(c, "count", "unaffected")])
    v["aff_push_mean"] = agg[("push", "mean", "affected")]
    v = v.join(d.drop_duplicates("SESSION_ID").set_index("SESSION_ID")[["SUBJECT_ID", "DATE", "age_first", "split"]])
    v = v.reset_index().sort_values(["SUBJECT_ID", "DATE", "SESSION_ID"]).reset_index(drop=True)
    prev = v.groupby("SUBJECT_ID").shift(1)
    p = v[prev.SESSION_ID.notna()].copy()
    for c in ("pred", "push"):
        p["d_" + c] = v[c] - prev[c]; p["s2_" + c] = v[c + "_se2"] + prev[c + "_se2"]
    return p


def tau(p, c):
    return float(np.sqrt(max(p["d_" + c].var() - p["s2_" + c].mean(), 0.0)))


ev = st[st.split.isin(["val", "test"])].reset_index(drop=True)
ev_idx = np.where(st.split.isin(["val", "test"]).to_numpy())[0]
r = np.random.default_rng(7)
# per (visit, leg) insole gain, fixed across conditions
gain_key = ev.SESSION_ID.astype(str) + ev.role
gains = dict(zip(gain_key.unique(), r.normal(1.0, 0.05, gain_key.nunique())))
noise = r.normal(0, 0.02, (len(ev), 101)).astype(np.float32)
order = r.permutation(len(ev))

out = {"ai": AI, "conditions": []}
ref_detect = None
for fs in (250, 100, 25):
    for err in ("none", "insole"):
        c = CURVES[fs][ev_idx].copy()
        if err == "insole":
            c = c * gain_key.map(gains).to_numpy()[:, None].astype(np.float32) + noise
        for k in ("all", 10, 5, 3):
            if k == "all":
                sel = np.arange(len(ev))
            else:
                rank = pd.Series(np.empty(len(ev)), dtype=float)
                rank[order] = np.arange(len(ev))
                kk = ev.assign(rk=rank.to_numpy()).groupby(["SESSION_ID", "role"]).rk.rank(method="first")
                cnt = ev.groupby(["SESSION_ID", "role"]).push.transform("size")
                sel = np.where((kk <= k) & (cnt >= k))[0]
            df = ev.iloc[sel].reset_index(drop=True); cc = c[sel]
            pred = predict_push(df, cc)
            p = visit_pairs(df, pred)
            pv, pt = p[p.split == "val"], p[(p.split == "test") & p.age_first.between(18, 65)]
            tp, tt = tau(pv, "pred"), tau(pv, "push")
            cross = float((pt.d_pred.abs() > Z * np.sqrt(pt.s2_pred + tp ** 2)).mean())
            # planted 0.004 decline on each test patient's latest pair: scale affected curve after midstance
            last = pt.groupby("SUBJECT_ID").tail(1)
            det_ai, det_true = [], []
            for _, q in last.iterrows():
                rows = np.where((df.SESSION_ID == q.SESSION_ID).to_numpy())[0]
                rr = df.iloc[rows]; aff = (rr.role == "affected").to_numpy()
                c2 = cc[rows].copy(); c2[aff, 50:] *= (1 - 0.004 / q.aff_push_mean)
                p2 = predict_push(rr, c2)
                new_diff = p2[aff].mean() - p2[~aff].mean()
                old_diff = pred[rows][aff].mean() - pred[rows][~aff].mean()
                det_ai.append(q.d_pred + (new_diff - old_diff) < -Z * np.sqrt(q.s2_pred + tp ** 2))
                det_true.append(q.d_push - 0.004 < -Z * np.sqrt(q.s2_push + tt ** 2))
            row = {"fs_hz": fs, "insole_error": err, "steps_per_leg": k, "test_pairs": int(len(pt)),
                   "tau_pred": tp, "crossing_rate": cross, "ccc_push_change": ccc(pt.d_pred.to_numpy(), pt.d_push.to_numpy()),
                   "detect_0.004": float(np.mean(det_ai)), "detect_0.004_forceplate_same_steps": float(np.mean(det_true))}
            if fs == 250 and err == "none" and k == "all":
                ref_detect = row["detect_0.004_forceplate_same_steps"]
            row["retention_vs_forceplate_all_steps"] = row["detect_0.004"] / ref_detect if ref_detect else None
            row["meets_criterion"] = bool(0.04 <= cross <= 0.06 and row["retention_vs_forceplate_all_steps"] is not None
                                          and row["retention_vs_forceplate_all_steps"] >= 0.8)
            out["conditions"].append(row)
            print(json.dumps({k2: (round(v, 4) if isinstance(v, float) else v) for k2, v in row.items()}), flush=True)
ok = [c for c in out["conditions"] if c["meets_criterion"]]
out["any_condition_meets_criterion"] = bool(ok)
(ROOT / "results" / "shear_study3.json").write_text(json.dumps(out, indent=2))
