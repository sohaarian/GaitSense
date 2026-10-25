"""Direct detection of longitudinal push-off change (docs/detection_protocol.md, frozen).

    python scripts/detect_study.py cv      development set only: 5-fold patient-level CV, thresholds, detector choice
    python scripts/detect_study.py final   refit on the development set, evaluate the temporal hold-out once

The cv phase never loads hold-out rows (load_steps(include_holdout=False)). The final phase reads the frozen
thresholds and detector choice from results/detect_cv.json and refuses to run if that file is missing.

Implementation details fixed in code (not specified further by the protocol):
- tau for reconstruction scores (R-gb, R-cnn) is estimated on the training fold's pairs using in-sample predictions.
- P-cnn's "predicted limit" is the training fold's residual SD of the push-off change (one value per fold), because
  no AP-free, pair-specific limit exists; its score therefore ranks pairs by |predicted change|.
- Paired features: per leg and visit, step means of the 13 hand-crafted curve summaries; their changes between
  visits; step counts; days between visits; body mass; injury group. P-gb also gets both legs' change curves.
"""
import json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parents[1]; sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
from shear_data import load_steps
from gaitrec_rehab.cnn1d import CNN1D

PHASE = sys.argv[1] if len(sys.argv) > 1 else "cv"
Z = 1.96; GROUPS = list("KAHC"); T0 = time.time()
log = lambda *a: print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


def handcrafted(c, s):
    t1 = c[:, :51].argmax(1); p1 = c[:, :51].max(1)
    t2 = 50 + c[:, 50:].argmax(1); p2 = c[:, 50:].max(1)
    val = np.array([c[i, t1[i]:t2[i] + 1].min() for i in range(len(c))]); tv = np.array([t1[i] + c[i, t1[i]:t2[i] + 1].argmin() for i in range(len(c))])
    lr = p1 / np.maximum(t1, 1) * 100 / s; ur = p2 / np.maximum(100 - t2, 1) * 100 / s
    m = c.mean(1)
    return np.column_stack([p1, p2, val, t1, t2, tv, lr, ur, m, m * s, s, p2 - p1, p1 - val])


def ccc(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(2 * np.cov(a, b, ddof=0)[0, 1] / (a.var() + b.var() + (a.mean() - b.mean()) ** 2))


def tau(d, s2):
    return float(np.sqrt(max(np.var(d, ddof=1) - np.mean(s2), 0.0)))


# ---------------- data ----------------
st, C, _, _ = load_steps(include_holdout=(PHASE == "final"))
keep = (st.role != "healthy").to_numpy()
st, C = st[keep].reset_index(drop=True), C[keep].astype(np.float64)
if PHASE == "cv":
    assert not st.holdout.any()
H = handcrafted(C, st.stance_s.to_numpy())
STEP_X = np.column_stack([H, st.BODY_MASS, (st.role == "affected").astype(float)] + [(st.CLASS_LABEL == g).astype(float) for g in GROUPS] + [C])
STEP_COV = np.column_stack([st.stance_s, st.BODY_MASS, (st.role == "affected").astype(float)] + [(st.CLASS_LABEL == g).astype(float) for g in GROUPS])
log("steps", len(st), "holdout steps", int(st.holdout.sum()))

# visit x leg aggregates
key = st.SESSION_ID.astype(str) + "|" + st.role
codes, uniq = pd.factorize(key)
nvl = len(uniq); cnt = np.bincount(codes)


def gmean(x):
    x = np.asarray(x, float)
    if x.ndim == 1:
        return np.bincount(codes, x, nvl) / cnt
    return np.vstack([np.bincount(codes, x[:, j], nvl) for j in range(x.shape[1])]).T / cnt[:, None]


def gvar(x):
    m = gmean(x); return (gmean(np.asarray(x, float) ** 2) - m ** 2) * cnt / np.maximum(cnt - 1, 1)


VL = pd.DataFrame({"SESSION_ID": [int(u.split("|")[0]) for u in uniq], "role": [u.split("|")[1] for u in uniq], "n": cnt})
for c in ("peak_loading", "push", "brake"):
    VL[c] = gmean(st[c]); VL[c + "_v"] = gvar(st[c]) / cnt
VL_curve = gmean(C); VL_hc = gmean(H)
first = st.drop_duplicates("SESSION_ID").set_index("SESSION_ID")
VL = VL.join(first[["SUBJECT_ID", "DATE", "CLASS_LABEL", "BODY_MASS", "age_first", "holdout"]], on="SESSION_ID")
VL["row"] = np.arange(len(VL))
A = VL[VL.role == "affected"].set_index("SESSION_ID"); U = VL[VL.role == "unaffected"].set_index("SESSION_ID")
both = A.index.intersection(U.index)
both = both[(A.loc[both, "n"] >= 2).to_numpy() & (U.loc[both, "n"] >= 2).to_numpy()]
V = A.loc[both, ["SUBJECT_ID", "DATE", "CLASS_LABEL", "BODY_MASS", "age_first", "holdout"]].copy()
V["ra"], V["ru"] = A.loc[both, "row"].to_numpy(), U.loc[both, "row"].to_numpy()
for c in ("peak_loading", "push", "brake"):
    V[c] = A.loc[both, c].to_numpy() - U.loc[both, c].to_numpy()
    V[c + "_s2"] = A.loc[both, c + "_v"].to_numpy() + U.loc[both, c + "_v"].to_numpy()
V = V.reset_index().sort_values(["SUBJECT_ID", "DATE", "SESSION_ID"]).reset_index(drop=True)
prev = V.groupby("SUBJECT_ID").shift(1)
pi = np.where(prev.SESSION_ID.notna())[0]
P = V.iloc[pi].reset_index(drop=True); Q = V.iloc[pi - 1].reset_index(drop=True)   # Q = earlier visit
for c in ("peak_loading", "push", "brake"):
    P["d_" + c] = P[c] - Q[c]; P["s2_" + c] = P[c + "_s2"] + Q[c + "_s2"]
P["days"] = (P.DATE - Q.DATE).dt.days.to_numpy()
P["sess_prev"] = Q.SESSION_ID.to_numpy()
log("pairs", len(P), "holdout pairs", int(P.holdout.sum()))

# paired features
ra1, ru1, ra2, ru2 = Q.ra.to_numpy(), Q.ru.to_numpy(), P.ra.to_numpy(), P.ru.to_numpy()
hc = [VL_hc[r] for r in (ra1, ru1, ra2, ru2)]
nn_ = [VL.n.to_numpy()[r] for r in (ra1, ru1, ra2, ru2)]
cov_pair = np.column_stack([P.days, P.BODY_MASS] + [(P.CLASS_LABEL == g).astype(float) for g in GROUPS])
XF = np.column_stack(hc + [hc[2] - hc[0], hc[3] - hc[1]] + nn_ + [cov_pair])
dcur = np.column_stack([VL_curve[ra2] - VL_curve[ra1], VL_curve[ru2] - VL_curve[ru1]])
XG = np.column_stack([XF, dcur])
XC = np.stack([VL_curve[r] for r in (ra1, ru1, ra2, ru2)], axis=2)
XCcov = np.column_stack([cov_pair] + nn_)
PID = P.SUBJECT_ID.to_numpy()


# ---------------- model pieces ----------------
def label(p_idx, tau_push):
    return (np.abs(P.d_push.to_numpy()[p_idx]) > Z * np.sqrt(P.s2_push.to_numpy()[p_idx] + tau_push ** 2)).astype(int)


def step_rows_for(pidx):
    s = set(P.SESSION_ID.to_numpy()[pidx]) | set(P.sess_prev.to_numpy()[pidx])
    return np.where(st.SESSION_ID.isin(s).to_numpy())[0]


def recon_change(pred_step, pidx):
    """Visit change in predicted affected-minus-unaffected push-off, and its same-day variance."""
    m = gmean(pred_step); v = gvar(pred_step) / cnt
    d = (m[ra2] - m[ru2]) - (m[ra1] - m[ru1]); s2 = v[ra2] + v[ru2] + v[ra1] + v[ru1]
    return d[pidx], s2[pidx]


def fit_predict(tr_pairs, te_pairs, seed=0, with_cnn=True):
    """Fit every method on training pairs (and their patients' steps); score test pairs."""
    out = {}
    tr_pat = set(PID[tr_pairs])
    tr_steps = np.where(st.SUBJECT_ID.isin(tr_pat).to_numpy())[0]
    tau_push = tau(P.d_push.to_numpy()[tr_pairs], P.s2_push.to_numpy()[tr_pairs])
    tau_pl = tau(P.d_peak_loading.to_numpy()[tr_pairs], P.s2_peak_loading.to_numpy()[tr_pairs])
    ytr = label(tr_pairs, tau_push)
    out["tau_push"], out["tau_pl"] = tau_push, tau_pl
    out["label"] = label(te_pairs, tau_push)
    # V-thr
    out["V-thr"] = {"score": np.abs(P.d_peak_loading.to_numpy()[te_pairs]) / np.sqrt(P.s2_peak_loading.to_numpy()[te_pairs] + tau_pl ** 2)}
    out["hard"] = (out["label"] == 1) & (out["V-thr"]["score"] <= Z)
    # reconstruction models (per step)
    Ys = st[["brake", "push"]].to_numpy()
    gb = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06, random_state=seed).fit(STEP_X[tr_steps], Ys[tr_steps, 1])
    pred = np.zeros(len(st)); allrows = np.union1d(tr_steps, step_rows_for(te_pairs)); pred[allrows] = gb.predict(STEP_X[allrows])
    recon = {"R-gb": pred}
    if with_cnn:
        sc_c = (C[tr_steps].mean(0), C[tr_steps].std(0)); sc_v = (STEP_COV[tr_steps].mean(0), STEP_COV[tr_steps].std(0))
        ym, ys = Ys[tr_steps].mean(0), Ys[tr_steps].std(0)
        pats = np.array(sorted(tr_pat)); r = np.random.default_rng(seed); inner = set(r.choice(pats, len(pats) // 5, replace=False))
        iv = st.SUBJECT_ID.iloc[tr_steps].isin(inner).to_numpy()
        f = lambda rows: ((C[rows] - sc_c[0]) / sc_c[1], (STEP_COV[rows] - sc_v[0]) / sc_v[1])
        a_, b_ = f(tr_steps[~iv]); c_, d_ = f(tr_steps[iv])
        net = CNN1D(n_cov=STEP_COV.shape[1], n_out=2, seed=seed).fit(a_, b_, (Ys[tr_steps[~iv]] - ym) / ys,
                                                                   val=(c_, d_, (Ys[tr_steps[iv]] - ym) / ys), epochs=40, patience=5)
        pc = np.zeros(len(st)); x1, x2 = f(allrows); pc[allrows] = net.predict(x1, x2)[:, 1] * ys[1] + ym[1]
        recon["R-cnn"] = pc
    for name, pr in recon.items():
        dtr, s2tr = recon_change(pr, tr_pairs); t_ = tau(dtr, s2tr)
        d, s2 = recon_change(pr, te_pairs)
        out[name] = {"score": np.abs(d) / np.sqrt(s2 + t_ ** 2), "delta": d}
    # paired models
    dy = P[["d_brake", "d_push"]].to_numpy()
    lin_c = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(XF[tr_pairs], ytr)
    lin_r = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13))).fit(XF[tr_pairs], dy[tr_pairs, 1])
    out["P-lin"] = {"score": lin_c.predict_proba(XF[te_pairs])[:, 1], "delta": lin_r.predict(XF[te_pairs])}
    gbc = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06, random_state=seed).fit(XG[tr_pairs], ytr)
    gbr = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06, random_state=seed).fit(XG[tr_pairs], dy[tr_pairs, 1])
    out["P-gb"] = {"score": gbc.predict_proba(XG[te_pairs])[:, 1], "delta": gbr.predict(XG[te_pairs])}
    if with_cnn:
        cm, cs = XC[tr_pairs].mean(0), XC[tr_pairs].std(0); vm, vs = XCcov[tr_pairs].mean(0), XCcov[tr_pairs].std(0) + 1e-9
        dm, ds = dy[tr_pairs].mean(0), dy[tr_pairs].std(0)
        pats = np.array(sorted(set(PID[tr_pairs]))); r = np.random.default_rng(seed + 1); inner = set(r.choice(pats, len(pats) // 5, replace=False))
        iv = np.isin(PID[tr_pairs], list(inner)); a_, b_ = tr_pairs[~iv], tr_pairs[iv]
        g = lambda idx: ((XC[idx] - cm) / cs, (XCcov[idx] - vm) / vs)
        net = CNN1D(n_cov=XCcov.shape[1], n_out=2, seed=seed, in_ch=4).fit(*g(a_), (dy[a_] - dm) / ds, val=(*g(b_), (dy[b_] - dm) / ds),
                                                                           epochs=60, patience=6)
        d_tr = net.predict(*g(tr_pairs))[:, 1] * ds[1] + dm[1]; res_sd = float(np.std(dy[tr_pairs, 1] - d_tr))
        d_te = net.predict(*g(te_pairs))[:, 1] * ds[1] + dm[1]
        out["P-cnn"] = {"score": np.abs(d_te) / res_sd, "delta": d_te}
    return out


METHODS = ["V-thr", "R-gb", "R-cnn", "P-lin", "P-gb", "P-cnn"]
P_MODELS = ["P-lin", "P-gb", "P-cnn"]


def metrics(lab, score, thr, hard=None):
    flag = score > thr
    m = {"sensitivity": float(flag[lab == 1].mean()) if lab.sum() else None, "specificity": float((~flag)[lab == 0].mean()),
         "precision": float(lab[flag].mean()) if flag.any() else None, "n_true": int(lab.sum()), "n": int(len(lab))}
    if 0 < lab.sum() < len(lab):
        m["roc_auc"] = float(roc_auc_score(lab, score)); m["pr_auc"] = float(average_precision_score(lab, score))
    if hard is not None and hard.any():
        m["hard_sensitivity"] = float(flag[hard].mean()); m["hard_n"] = int(hard.sum())
    return m


def change_metrics(d_true, d_pred, lab):
    sig = np.sign(d_pred) == np.sign(d_true)
    slope = float(np.cov(d_true, d_pred)[0, 1] / np.var(d_pred, ddof=1)) if np.var(d_pred) > 0 else None
    return {"ccc": ccc(d_pred, d_true), "pearson": float(np.corrcoef(d_pred, d_true)[0, 1]), "mae": float(np.abs(d_pred - d_true).mean()),
            "direction_acc_true_change": float(sig[lab == 1].mean()) if lab.sum() else None, "calibration_slope": slope}


def boot(pid, lab, scores, thr, B=500, seed=1):
    u = np.unique(pid); idx = {k: np.where(pid == k)[0] for k in u}; r = np.random.default_rng(seed)
    out = {m: [] for m in scores}; diffs = {}
    for _ in range(B):
        ii = np.concatenate([idx[k] for k in r.choice(u, len(u))])
        pos = lab[ii] == 1
        sens = {m: (scores[m][ii][pos] > thr[m]).mean() if pos.any() else np.nan for m in scores}
        for m in scores:
            out[m].append(sens[m])
        for a in scores:
            for b in ("V-thr", "R-gb"):
                if a != b:
                    diffs.setdefault(f"{a} minus {b}", []).append(sens[a] - sens[b])
    ci = lambda v: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
    return {"sensitivity_ci95": {m: ci(v) for m, v in out.items()}, "difference_ci95": {k: ci(v) for k, v in diffs.items()}}


# ---------------- phases ----------------
if PHASE == "cv":
    dev = np.arange(len(P))
    oof = {m: np.full(len(P), np.nan) for m in METHODS}; dl = {m: np.full(len(P), np.nan) for m in METHODS if m != "V-thr"}
    lab = np.zeros(len(P), int); hard = np.zeros(len(P), bool); taus = []
    for k, (tr, te) in enumerate(GroupKFold(5).split(dev, groups=PID)):
        log(f"fold {k + 1}"); o = fit_predict(tr, te, seed=k)
        lab[te] = o["label"]; hard[te] = o["hard"]; taus.append((o["tau_push"], o["tau_pl"]))
        for m in METHODS:
            oof[m][te] = o[m]["score"]
            if m in dl:
                dl[m][te] = o[m]["delta"]
    thr = {m: float(np.quantile(oof[m][lab == 0], 0.95)) for m in METHODS}
    res = {"pairs": int(len(P)), "patients": int(len(np.unique(PID))), "true_changes": int(lab.sum()),
           "fold_tau_push_and_peak_loading": taus, "thresholds_at_5pct_fpr": thr,
           "cv": {m: metrics(lab, oof[m], thr[m], hard) for m in METHODS}}
    sens = {m: res["cv"][m]["sensitivity"] for m in P_MODELS}
    res["detector"] = max(sens, key=sens.get)
    res["cv_bootstrap"] = boot(PID, lab, oof, thr)
    dtrue = P.d_push.to_numpy()
    res["cv_change_size"] = {m: change_metrics(dtrue, dl[m], lab) for m in dl}
    res["cv_by_group"] = {g: {m: metrics(lab[P.CLASS_LABEL == g], oof[m][P.CLASS_LABEL == g], thr[m]) for m in METHODS} | {
        "calibration_slope": {m: change_metrics(dtrue[P.CLASS_LABEL == g], dl[m][P.CLASS_LABEL == g], lab[P.CLASS_LABEL == g])["calibration_slope"] for m in dl}}
        for g in GROUPS}
    # failure cases for the detector (descriptive)
    det = res["detector"]; flag = oof[det] > thr[det]; tc = lab == 1
    fc = pd.DataFrame({"detected": flag[tc], "group": P.CLASS_LABEL.to_numpy()[tc], "abs_change": np.abs(dtrue[tc]),
                       "decline": dtrue[tc] < 0, "min_steps": np.min(np.column_stack(nn_), 1)[tc],
                       "stance_now": VL_hc[ra2][tc, 10], "peak_loading_changed": oof["V-thr"][tc] > Z})
    res["failure_cases"] = {"n_true": int(tc.sum()), "detected_share": float(fc.detected.mean()),
                            "by_group": fc.groupby("group").detected.mean().to_dict(),
                            "detected_by_direction": {"decline": float(fc[fc.decline].detected.mean()), "increase": float(fc[~fc.decline].detected.mean())},
                            "median_abs_change": {"detected": float(fc[fc.detected].abs_change.median()), "missed": float(fc[~fc.detected].abs_change.median())},
                            "median_min_steps": {"detected": float(fc[fc.detected].min_steps.median()), "missed": float(fc[~fc.detected].min_steps.median())},
                            "detected_when_peak_loading_also_changed": float(fc[fc.peak_loading_changed].detected.mean()) if fc.peak_loading_changed.any() else None,
                            "detected_when_peak_loading_stable": float(fc[~fc.peak_loading_changed].detected.mean())}
    (ROOT / "results" / "detect_cv.json").write_text(json.dumps(res, indent=2))
    pd.DataFrame({"SUBJECT_ID": PID, "CLASS_LABEL": P.CLASS_LABEL, "label": lab, "hard": hard, "d_push": dtrue} |
                 {f"score_{m}": oof[m] for m in METHODS} | {f"delta_{m}": dl[m] for m in dl}).to_csv(ROOT / "results" / "detect_cv_pairs.csv", index=False)
    log("CV done; detector =", res["detector"], json.dumps({m: round(res["cv"][m]["sensitivity"], 3) for m in METHODS}))

elif PHASE == "final":
    cvf = ROOT / "results" / "detect_cv.json"
    if not cvf.exists():
        sys.exit("run the cv phase first: thresholds and the detector must be frozen before the hold-out is opened")
    cv = json.loads(cvf.read_text()); thr = cv["thresholds_at_5pct_fpr"]
    hold = P.holdout.to_numpy()
    tr, te = np.where(~hold)[0], np.where(hold)[0]
    assert not set(PID[tr]) & set(PID[te])
    log("final fit on development pairs", len(tr), "-> hold-out pairs", len(te))
    o = fit_predict(tr, te, seed=0)
    lab, hard = o["label"], o["hard"]
    scores = {m: o[m]["score"] for m in METHODS}
    res = {"holdout_pairs": int(len(te)), "holdout_patients": int(len(np.unique(PID[te]))), "true_changes": int(lab.sum()),
           "detector": cv["detector"], "thresholds_used": thr,
           "holdout": {m: metrics(lab, scores[m], thr[m], hard) for m in METHODS},
           "holdout_bootstrap": boot(PID[te], lab, scores, thr),
           "holdout_change_size": {m: change_metrics(P.d_push.to_numpy()[te], o[m]["delta"], lab) for m in METHODS if m != "V-thr"}}
    (ROOT / "results" / "detect_holdout.json").write_text(json.dumps(res, indent=2))
    log("hold-out done", json.dumps({m: res["holdout"][m]["sensitivity"] for m in METHODS}))
