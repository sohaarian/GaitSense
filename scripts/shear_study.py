"""Shear-force reconstruction study, following docs/shear_protocol.md (version 2).

Writes results/shear_study.json and results/shear_pairs.csv.

Choices made in code, all before any test-set result was seen:
- Patients are split 60/20/20 (train/validation/test) with seed 0. Models train on all ages; the primary
  evaluation uses test patients aged 18-65 at their first visit, and all ages are a sensitivity analysis.
- "The AI" is whichever of M-gb and M-mlp has the higher push-off change CCC on VALIDATION patients.
- A visit needs >= 2 valid steps on each leg, so that a standard error can be computed.
- M-mlp replaces the 1D CNN because no deep-learning library is installed (protocol deviation, as allowed).
- Condition (c), vertical only: the predicted push-off change is a linear calibration of the change in peak
  vertical loading difference, fitted on training pairs.
- Planted declines (Study 2B) scale the affected leg's vertical curve after midstance (points 50-100) by
  (1 - delta / that visit's mean affected push-off), at the latest visit. The vertical-curve change this
  implies is an assumption, not a physical model; this analysis is secondary for that reason.
"""
import json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parents[1]; sys.path.insert(0, str(ROOT / "scripts"))
from shear_data import load_steps, patient_split

T0 = time.time()
Z = 1.96
TARGETS = ["brake", "push"]
GROUPS = ["K", "A", "H", "C"]
rng = np.random.default_rng(0)
log = lambda *a: print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)

st, C250, C100, C25 = load_steps()
hc = st.role == "healthy"
HC, HC_C = st[hc].reset_index(drop=True), C250[hc.to_numpy()]
st, C250, C100, C25 = (st[~hc].reset_index(drop=True), C250[~hc.to_numpy()], C100[~hc.to_numpy()], C25[~hc.to_numpy()])

# ---------- split ----------
st["split"] = st.SUBJECT_ID.map(patient_split(st.SUBJECT_ID))


# ---------- features ----------
def handcrafted(c, s):
    t1 = c[:, :51].argmax(1); p1 = c[:, :51].max(1)
    t2 = 50 + c[:, 50:].argmax(1); p2 = c[:, 50:].max(1)
    val = np.array([c[i, t1[i]:t2[i] + 1].min() for i in range(len(c))]); tv = np.array([t1[i] + c[i, t1[i]:t2[i] + 1].argmin() for i in range(len(c))])
    lr = p1 / np.maximum(t1, 1) * 100 / s; ur = p2 / np.maximum(100 - t2, 1) * 100 / s
    m = c.mean(1)
    return np.column_stack([p1, p2, val, t1, t2, tv, lr, ur, m, m * s, s, p2 - p1, p1 - val])


def covariates(df):
    return np.column_stack([df.BODY_MASS.to_numpy(), (df.role == "affected").to_numpy(float)] +
                           [(df.CLASS_LABEL == g).to_numpy(float) for g in GROUPS])


def design(df, c):
    s = df.stance_s.to_numpy()
    H = handcrafted(c, s); Cv = covariates(df)
    return {"lin": np.column_stack([H, Cv]), "gb": np.column_stack([H, Cv, c]), "mlp": np.column_stack([c, s, Cv])}


log("features")
X = design(st, C250)
tr = (st.split == "train").to_numpy()
Y = st[TARGETS].to_numpy()

models = {}
log("fit M-lin")
models["lin"] = [make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13))).fit(X["lin"][tr], Y[tr, j]) for j in range(2)]
log("fit M-gb")
models["gb"] = [HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06, random_state=0).fit(X["gb"][tr], Y[tr, j]) for j in range(2)]
log("fit M-mlp")
ysc = Y[tr].std(0)
models["mlp"] = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(128, 64), alpha=1e-3, early_stopping=True,
                                                              max_iter=300, random_state=0)).fit(X["mlp"][tr], Y[tr] / ysc)


def predict(name, Xd):
    if name == "mlp":
        return models["mlp"].predict(Xd["mlp"]) * ysc
    return np.column_stack([m.predict(Xd[name]) for m in models[name]])


P = {m: predict(m, X) for m in ("lin", "gb", "mlp")}
MODELS = ["lin", "gb", "mlp"]
CNN = ROOT / "data" / "derived" / "cnn_preds.npz"
if CNN.exists():  # post-hoc CNN (scripts/shear_cnn.py), aligned to the same step rows
    z = np.load(CNN, allow_pickle=True)
    assert (z["session"] == st.SESSION_ID.to_numpy()).all() and (z["trial"] == st.TRIAL_ID.to_numpy()).all()
    P["cnn"] = z["pred"]; MODELS.append("cnn")


# ---------- per-step accuracy (Study 1) ----------
def ccc(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(2 * np.cov(a, b, ddof=0)[0, 1] / (a.var() + b.var() + (a.mean() - b.mean()) ** 2))


def step_metrics(mask):
    out = {}
    for m in P:
        out[m] = {}
        for j, t in enumerate(TARGETS):
            e = P[m][mask, j] - Y[mask, j]
            out[m][t] = {"mae": float(np.abs(e).mean()), "r2": float(1 - (e ** 2).sum() / ((Y[mask, j] - Y[mask, j].mean()) ** 2).sum()),
                         "ba_bias": float(e.mean()), "ba_loa": [float(e.mean() - Z * e.std()), float(e.mean() + Z * e.std())]}
    return out


age_ok = st.age_first.between(18, 65).to_numpy()
test = (st.split == "test").to_numpy()
res = {"n_patients": {k: int(st[st.split == k].SUBJECT_ID.nunique()) for k in ("train", "val", "test")},
       "n_steps": {k: int((st.split == k).sum()) for k in ("train", "val", "test")},
       "test_patients_18_65": int(st[test & age_ok].SUBJECT_ID.nunique()),
       "study1_step_accuracy_test_18_65": step_metrics(test & age_ok),
       "study1_step_accuracy_test_all_ages": step_metrics(test),
       "study1_step_accuracy_by_group_test_18_65": {g: step_metrics(test & age_ok & (st.CLASS_LABEL == g).to_numpy()) for g in GROUPS}}
log("study1 done", json.dumps({m: res["study1_step_accuracy_test_18_65"][m]["push"]["r2"] for m in P}))


# ---------- visit level ----------
def visit_table(df, vals, cols):
    """Affected-minus-unaffected visit means and SEs for each column of vals."""
    d = df[["SUBJECT_ID", "SESSION_ID", "DATE", "CLASS_LABEL", "SEX", "age_first", "split", "role"]].copy()
    for k, c in enumerate(cols):
        d[c] = vals[:, k]
    g = d.groupby(["SESSION_ID", "role"])
    agg = g[cols].agg(["mean", "var", "count"])
    out = []
    a = agg.xs("affected", level="role"); u = agg.xs("unaffected", level="role")
    common = a.index.intersection(u.index)
    a, u = a.loc[common], u.loc[common]
    ok = (a[(cols[0], "count")] >= 2) & (u[(cols[0], "count")] >= 2)
    a, u = a[ok], u[ok]
    t = pd.DataFrame(index=a.index)
    for c in cols:
        t[c] = a[(c, "mean")] - u[(c, "mean")]
        t[c + "_se2"] = a[(c, "var")] / a[(c, "count")] + u[(c, "var")] / u[(c, "count")]
        t[c + "_a"] = a[(c, "mean")]; t[c + "_u"] = u[(c, "mean")]
    info = d.drop_duplicates("SESSION_ID").set_index("SESSION_ID")[["SUBJECT_ID", "DATE", "CLASS_LABEL", "SEX", "age_first", "split"]]
    return t.join(info).reset_index().sort_values(["SUBJECT_ID", "DATE", "SESSION_ID"]).reset_index(drop=True)


def pairs_from(v, cols):
    prev = v.groupby("SUBJECT_ID").shift(1)
    p = v[prev.SESSION_ID.notna()].copy()
    p["visit_no"] = v.groupby("SUBJECT_ID").cumcount()[p.index] + 1
    for c in cols:
        p["d_" + c] = v[c] - prev[c]
        p["s2_" + c] = v[c + "_se2"] + prev[c + "_se2"]
    return p.reset_index(drop=True)


def tau(p, c):
    return float(np.sqrt(max(p["d_" + c].var() - p["s2_" + c].mean(), 0.0)))


def signals(df, c, preds):
    """True signals plus predicted push/brake for each model, at step level."""
    cols = ["peak_loading", "brake", "push"]
    vals = [df.peak_loading.to_numpy(), df.brake.to_numpy(), df.push.to_numpy()]
    for m, pr in preds.items():
        cols += [f"{m}_brake", f"{m}_push"]; vals += [pr[:, 0], pr[:, 1]]
    return cols, np.column_stack(vals)


cols, vals = signals(st, C250, P)
V = visit_table(st, vals, cols)
PR = pairs_from(V, cols)
res["visits_with_2plus_steps_each_leg"] = int(len(V)); res["visit_pairs"] = int(len(PR))
log("visits", len(V), "pairs", len(PR))

# condition (c): linear calibration of loading change to push/brake change, fitted on training pairs
trp = PR.split == "train"
for t in TARGETS:
    lr = LinearRegression().fit(PR.loc[trp, ["d_peak_loading"]], PR.loc[trp, "d_" + t])
    PR[f"d_vert_{t}"] = lr.predict(PR[["d_peak_loading"]]); PR[f"s2_vert_{t}"] = lr.coef_[0] ** 2 * PR.s2_peak_loading
    res.setdefault("vertical_only_calibration", {})[t] = {"slope": float(lr.coef_[0]), "intercept": float(lr.intercept_)}
    PR[f"d_persist_{t}"] = 0.0; PR[f"s2_persist_{t}"] = 0.0

valp = PR[PR.split == "val"]
TAU = {"true": {t: tau(valp, t) for t in TARGETS + ["peak_loading"]}}
for m in MODELS:
    TAU[m] = {t: tau(valp, f"{m}_{t}") for t in TARGETS}
TAU["vert"] = {t: float(np.sqrt(max(valp[f"d_vert_{t}"].var() - valp[f"s2_vert_{t}"].mean(), 0))) for t in TARGETS}
TAU["persist"] = {t: 0.0 for t in TARGETS}
res["tau_from_validation"] = TAU

# choose "the AI" on validation
val_ccc = {m: ccc(valp[f"d_{m}_push"], valp["d_push"]) for m in ("gb", "mlp")}
AI = max(val_ccc, key=val_ccc.get)
res["ai_choice"] = {"validation_push_change_ccc": val_ccc, "chosen": AI}
log("AI =", AI, val_ccc)


def cond_cols(cond, t):
    if cond == "true":
        return "d_" + t, "s2_" + t
    if cond in ("lin", "gb", "mlp"):
        return f"d_{cond}_{t}", f"s2_{cond}_{t}"
    return f"d_{cond}_{t}", f"s2_{cond}_{t}"


def flags(p, cond, t):
    d, s2 = cond_cols(cond, t)
    lim = Z * np.sqrt(p[s2] + TAU[cond][t] ** 2)
    return p[d].to_numpy(), (p[d].abs() > lim).to_numpy()


CONDS = ["persist", "vert"] + MODELS


def change_metrics(p, t):
    dt, ft = flags(p, "true", t)
    out = {"n_pairs": int(len(p)), "n_true_beyond": int(ft.sum()), "crossing_rate_true": float(ft.mean())}
    for c in CONDS:
        dp, fp = flags(p, c, t)
        e = dp - dt
        sig = np.sign(dp) == np.sign(dt)
        out[c] = {"ccc": ccc(dp, dt), "pearson": float(np.corrcoef(dp, dt)[0, 1]) if dp.std() > 0 else 0.0,
                  "ba_bias": float(e.mean()), "ba_loa": [float(e.mean() - Z * e.std()), float(e.mean() + Z * e.std())],
                  "direction_acc_true_beyond": float(sig[ft].mean()) if ft.any() and c != "persist" else None,
                  "sensitivity": float((fp & sig)[ft].mean()) if ft.any() else None,
                  "specificity": float((~fp)[~ft].mean()) if (~ft).any() else None,
                  "crossing_rate": float(fp.mean())}
    return out


def hard_mask(p):
    _, fpush = flags(p, "true", "push")
    lim_l = Z * np.sqrt(p.s2_peak_loading + TAU["true"]["peak_loading"] ** 2)
    return fpush & (p.d_peak_loading.abs() <= lim_l).to_numpy()


def evaluate(p):
    out = {t: change_metrics(p, t) for t in TARGETS}
    h = hard_mask(p)
    out["hard_subset"] = {"n": int(h.sum())}
    dt = p.d_push.to_numpy()
    for c in CONDS:
        dp, fp = flags(p, c, "push")
        sig = np.sign(dp) == np.sign(dt)
        out["hard_subset"][c] = {"ccc": ccc(dp[h], dt[h]) if h.sum() > 2 else None,
                                 "direction_acc": float(sig[h].mean()) if h.any() and c != "persist" else None,
                                 "sensitivity": float((fp & sig)[h].mean()) if h.any() else None}
    return out


TP = PR[PR.split == "test"]
TP18 = TP[TP.age_first.between(18, 65)]
res["study2A_test_18_65"] = evaluate(TP18)
res["study2A_test_all_ages"] = evaluate(TP)
log("study2A", json.dumps({c: round(res["study2A_test_18_65"]["push"][c]["ccc"], 3) for c in CONDS}),
    "hard n", res["study2A_test_18_65"]["hard_subset"]["n"])


# ---------- bootstrap (500 patient resamples) ----------
def boot(p, B=500):
    pid = p.SUBJECT_ID.to_numpy(); uid = np.unique(pid)
    idx = {u: np.where(pid == u)[0] for u in uid}
    dt = p.d_push.to_numpy(); h = hard_mask(p)
    D = {c: flags(p, c, "push") for c in CONDS}
    r = np.random.default_rng(1)
    rec = {"ai_minus_persist": [], "ai_minus_vert": [], "ai_minus_lin": [], "hard_dir_ai": [], "crossing_ai": [],
           "hard_dir_lin": [], "hard_dir_vert": []}
    if "cnn" in MODELS:
        rec.update({"cnn_minus_gb": [], "cnn_minus_lin": [], "hard_dir_cnn": []})
    for _ in range(B):
        ii = np.concatenate([idx[u] for u in r.choice(uid, len(uid))])
        cc = {c: ccc(D[c][0][ii], dt[ii]) for c in ("persist", "vert", "lin", AI)}
        rec["ai_minus_persist"].append(cc[AI] - cc["persist"]); rec["ai_minus_vert"].append(cc[AI] - cc["vert"])
        rec["ai_minus_lin"].append(cc[AI] - cc["lin"])
        hh = ii[h[ii]]
        for c, k in ((AI, "hard_dir_ai"), ("lin", "hard_dir_lin"), ("vert", "hard_dir_vert")):
            rec[k].append(float((np.sign(D[c][0][hh]) == np.sign(dt[hh])).mean()) if len(hh) else np.nan)
        rec["crossing_ai"].append(float(D[AI][1][ii].mean()))
        if "cnn" in MODELS:
            ccn = ccc(D["cnn"][0][ii], dt[ii])
            rec["cnn_minus_gb"].append(ccn - ccc(D["gb"][0][ii], dt[ii])); rec["cnn_minus_lin"].append(ccn - cc["lin"])
            rec["hard_dir_cnn"].append(float((np.sign(D["cnn"][0][hh]) == np.sign(dt[hh])).mean()) if len(hh) else np.nan)
    return {k: {"ci95": [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]} for k, v in rec.items()}


log("bootstrap")
res["bootstrap_test_18_65"] = boot(TP18)
e = res["study2A_test_18_65"]; b = res["bootstrap_test_18_65"]
res["success_criteria"] = {
    "1_ccc_beats_persist_and_vertical": bool(b["ai_minus_persist"]["ci95"][0] > 0 and b["ai_minus_vert"]["ci95"][0] > 0),
    "2_hard_subset_direction_above_50pct": bool(b["hard_dir_ai"]["ci95"][0] > 0.5),
    "3_beats_linear_baseline": bool(b["ai_minus_lin"]["ci95"][0] > 0),
    "4_crossing_rate_4_to_6pct": bool(0.04 <= e["push"][AI]["crossing_rate"] <= 0.06),
}
log("criteria", res["success_criteria"])

# ---------- robustness ----------
rob = {}
for name, grp in [("group", TP18.CLASS_LABEL), ("sex", TP18.SEX.map({0: "female", 1: "male"})),
                  ("age", pd.cut(TP18.age_first, [17, 40, 65], labels=["18-40", "41-65"]).astype(str)),
                  ("visit_no", np.minimum(TP18.visit_no, 3).astype(str))]:
    rob[name] = {}
    for g, p in TP18.groupby(grp.to_numpy()):
        rob[name][str(g)] = {"n_pairs": int(len(p)), "ccc_push": {c: ccc(flags(p, c, "push")[0], p.d_push) for c in ["vert"] + MODELS}}
res["robustness_test_18_65"] = rob

# ---------- information test (amendment 4), training patients ----------
log("information test")
d = st[st.split == "train"]
cm = pd.DataFrame(C250[(st.split == "train").to_numpy()]); cm["SESSION_ID"] = d.SESSION_ID.to_numpy(); cm["role"] = d.role.to_numpy()
cm["stance"] = d.stance_s.to_numpy()
mc = cm.groupby(["SESSION_ID", "role"]).mean()
diffc = (mc.xs("affected", level="role") - mc.xs("unaffected", level="role"))
trpairs = PR[PR.split == "train"]
prevs = V.groupby("SUBJECT_ID").SESSION_ID.shift(1)
prev_map = dict(zip(V.SESSION_ID, prevs))
s_now = trpairs.SESSION_ID.to_numpy(); s_prev = np.array([prev_map[s] for s in s_now]).astype(int)
ok = np.isin(s_now, diffc.index) & np.isin(s_prev, diffc.index)
Xi = diffc.loc[s_now[ok]].to_numpy() - diffc.loc[s_prev[ok]].to_numpy()
gi = trpairs.SUBJECT_ID.to_numpy()[ok]
info = {}
for t in TARGETS:
    y = trpairs["d_" + t].to_numpy()[ok]
    for nm, Xu in (("full_curve_change", Xi), ("peak_loading_change_only", trpairs[["d_peak_loading"]].to_numpy()[ok])):
        pr = cross_val_predict(make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 13))), Xu, y, groups=gi, cv=GroupKFold(5))
        info.setdefault(t, {})[nm] = float(1 - ((y - pr) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    info[t]["noise_ceiling_share"] = float(1 - trpairs["s2_" + t].to_numpy()[ok].mean() / y.var())
res["information_test_train"] = info
log("info", info)

# ---------- Study 2B planted declines (secondary) ----------
log("planted declines")
te_last = TP18.groupby("SUBJECT_ID").tail(1)
res["study2B_planted_test_18_65"] = {}
step_idx = {s: np.where((st.SESSION_ID == s).to_numpy())[0] for s in te_last.SESSION_ID}


def visit_mean_diff(rows, pred):
    r = st.iloc[rows]; a = (r.role == "affected").to_numpy()
    return pred[a].mean() - pred[~a].mean(), pred[a].var(ddof=1) / a.sum() + pred[~a].var(ddof=1) / (~a).sum()


for delta in (0.0, 0.002, 0.004, 0.006):
    det = {"true": [], AI: [], "lin": [], "vert": []}
    for _, p in te_last.iterrows():
        rows = step_idx[p.SESSION_ID]; r = st.iloc[rows]; aff = (r.role == "affected").to_numpy()
        c = C250[rows].copy()
        mean_push = r.push[aff].mean()
        c[aff, 50:] *= (1 - delta / mean_push)
        Xd = design(r, c)
        new = {m: predict(m, Xd)[:, 1] for m in (AI, "lin")}
        lim_t = Z * np.sqrt(p.s2_push + TAU["true"]["push"] ** 2)
        det["true"].append((p.d_push - delta) < -lim_t)
        for m in (AI, "lin"):
            old = p[f"{m}_push"]; nd, _ = visit_mean_diff(rows, new[m])
            dnew = p[f"d_{m}_push"] + (nd - old)
            det[m].append(dnew < -Z * np.sqrt(p[f"s2_{m}_push"] + TAU[m]["push"] ** 2))
        det["vert"].append(p.d_vert_push < -Z * np.sqrt(p.s2_vert_push + TAU["vert"]["push"] ** 2))
    res["study2B_planted_test_18_65"][str(delta)] = {k: float(np.mean(v)) for k, v in det.items()}
log("2B", res["study2B_planted_test_18_65"])

# save for Study 3 and verification
keep = ["SUBJECT_ID", "SESSION_ID", "CLASS_LABEL", "SEX", "age_first", "split", "visit_no", "d_peak_loading", "s2_peak_loading",
        "d_push", "s2_push", "d_brake", "s2_brake", "d_vert_push", "s2_vert_push"] + \
       [f"{x}_{m}_push" for m in MODELS for x in ("d", "s2")]
TP.assign(hard=hard_mask(TP))[keep + ["hard"]].to_csv(ROOT / "results" / "shear_pairs_test.csv", index=False)
(ROOT / "results" / "shear_study.json").write_text(json.dumps(res, indent=2, default=float))
import pickle
pickle.dump({"models": models, "ysc": ysc, "AI": AI, "TAU": TAU}, open(ROOT / "data" / "derived" / "shear_models.pkl", "wb"))
log("done")
