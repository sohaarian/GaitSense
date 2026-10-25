"""Stages 7-8: perturbation robustness and injected-change detection on test-partition visit pairs.

Every transform is applied to the raw stance trials of the LATER visit of each consecutive pair; the history used by
architecture D is unchanged except in the 'missing observation' experiment. Writes results/ai/perturbation_results.json
and results/ai/injection_results.json.
"""
import json, pickle, sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parents[2]; sys.path.insert(0, str(ROOT / "src"))
from gaitsense_ai import SEED
from gaitsense_ai import monitoring as M
from gaitsense_ai import perturb as PT
from gaitsense_ai import prediction as P

A = ROOT / "results" / "ai"
sel = json.loads((A / "model_selection.json").read_text())
mon = json.loads((A / "monitoring_results.json").read_text())
with open(A / "models.pkl", "rb") as f:
    pk = pickle.load(f)
with open(ROOT / "data" / "derived" / "ai_raw_test.pkl", "rb") as f:
    RAW = pickle.load(f)
std = pk["std"]
V = pd.read_csv(A / "visit_level_features.csv.gz", parse_dates=["SESSION_DATE"])
vi = V.set_index("SESSION_ID")
S0 = pd.read_csv(A / "monitoring_pairs.csv.gz")
tau, mu_h, sd_h = mon["tau"], mon["healthy_reference"]["mu"], mon["healthy_reference"]["sd"]
rng = np.random.default_rng(SEED)

# perturbation scales from TRAINING visits only
tr = V[(V.role == "cohort") & (V.split == "train")]
s_w = float(tr.peak_v_a_sd.median()); cv_w = float((tr.peak_v_a_sd / tr.peak_v_a).median())

PV = V[V.role.isin(["cohort", "temporal"])]
EXTRA = {"d_propulsive": "propulsive", "d_braking": "braking", "peak_v_a": "peak_v_a"}
arrs, gps, taus = {}, {}, {}
for tgt in ["d_peak_v"] + list(EXTRA):
    arrs[tgt] = P.patient_arrays(PV, tgt, P.TARGETS[tgt], std)
    gps[tgt] = pk["fitted"][tgt][sel["targets"][tgt]["M3_chosen"]]; taus[tgt] = sel["targets"][tgt]["tau_train"]


def legs(sess):
    side = vi.at[sess, "affected_side"]; other = "right" if side == "left" else "left"
    return RAW[sess][side], RAW[sess][other], float(vi.at[sess, "BODY_WEIGHT"])


def D_pred(tgt, rows):
    mu, vl = P.gp_predict_rows(gps[tgt], arrs[tgt], rows)
    return mu, vl + gps[tgt].sigma_w ** 2


# predictions for secondary targets at each test pair's later visit (history < t)
for tgt in EXTRA:
    S0[f"mu_D_{tgt}"], S0[f"var_D_{tgt}"] = D_pred(tgt, S0[["SUBJECT_ID", "k"]])


def recompute(fn_a=None, fn_u=None, featfn=None, trial_filter=None):
    """Later-visit summaries after a transform. Returns DataFrame aligned with S0."""
    out = []
    for sess in S0.SESSION_ID_cur:
        ta, tu, bw = legs(sess)
        if trial_filter is not None:
            ta = trial_filter(ta, sess)
        out.append(PT.visit_summary(ta, tu, bw, fn_a=fn_a, fn_u=fn_u, featfn=featfn))
    return pd.DataFrame(out)


def arch_scores(vis, z=M.Z):
    X = S0[["Y_prev", "se_prev", "pa_prev", "pu_prev", "mu_D", "var_D"]].copy()
    X["Y_cur"], X["se_cur"] = vis.d_peak_v.to_numpy(), vis.d_peak_v_se.to_numpy()
    X["pa_cur"], X["pu_cur"] = vis.peak_v_a.to_numpy(), vis.peak_v_u.to_numpy()
    return M.scores(X, tau, mu_h, sd_h, z)


def cd_scores(vis, tgt, z=M.Z):
    """C and D (and an LSI-style A) for a secondary target."""
    if tgt == "peak_v_a":
        y_cur, se_cur = vis.peak_v_a.to_numpy(), vis.peak_v_a_se.to_numpy()
        y_prev, se_prev = S0.SESSION_ID_prev.map(vi.peak_v_a).to_numpy(), S0.SESSION_ID_prev.map(vi.peak_v_a_se).to_numpy()
        dA = None
    else:
        f = EXTRA[tgt]
        y_cur, se_cur = vis[f"d_{f}"].to_numpy(), vis[f"d_{f}_se"].to_numpy()
        y_prev, se_prev = S0.SESSION_ID_prev.map(vi[f"d_{f}"]).to_numpy(), S0.SESSION_ID_prev.map(vi[f"d_{f}_se"]).to_numpy()
        lsi_c = 100 * vis[f"{f}_a"].to_numpy() / vis[f"{f}_u"].to_numpy()
        lsi_p = 100 * S0.SESSION_ID_prev.map(vi[f"{f}_a"]).to_numpy() / S0.SESSION_ID_prev.map(vi[f"{f}_u"]).to_numpy()
        dA = lsi_c - lsi_p
    sc = (y_cur - y_prev) / np.sqrt(se_cur ** 2 + se_prev ** 2 + taus[tgt] ** 2)
    sd = (y_cur - S0[f"mu_D_{tgt}"].to_numpy()) / np.sqrt(S0[f"var_D_{tgt}"].to_numpy() + se_cur ** 2)
    return {"C": sc, "D": sd, "A": dA}


base_vis = recompute()
chk = np.abs(base_vis.d_peak_v.to_numpy() - S0.Y_cur.to_numpy()).max()
assert chk < 1e-9, f"recomputed visits differ from the visit table ({chk})"
B0 = arch_scores(base_vis)
assert all((B0[f"alert_{a}"].to_numpy() == S0[f"alert_{a}"].to_numpy()).all() for a in M.ARCH)

pids = S0.SUBJECT_ID.unique(); groups = {p: np.flatnonzero(S0.SUBJECT_ID.to_numpy() == p) for p in pids}
boot = [np.concatenate([groups[p] for p in rng.choice(pids, len(pids))]) for _ in range(500)]


def ci_mean(x):
    x = np.asarray(x, float); d = [np.nanmean(x[ix]) for ix in boot]
    return {"estimate": float(np.nanmean(x)), "ci95": [float(np.nanquantile(d, .025)), float(np.nanquantile(d, .975))]}


def compare(Bp, extra=None):
    r = {}
    for a in M.ARCH:
        flip = (Bp[f"alert_{a}"].to_numpy() != B0[f"alert_{a}"].to_numpy()).astype(float)
        r[a] = {"flip_rate": ci_mean(flip), "mean_abs_delta_score": float(np.nanmean(np.abs(Bp[f"score_{a}"] - B0[f"score_{a}"]))),
                "alert_rate": float(Bp[f"alert_{a}"].mean()),
                "flip_rate_by_injury": {c: float(flip[(S0.CLASS_LABEL == c).to_numpy()].mean()) for c in "AKHC"}}
    r["D_cov95_change"] = float((~Bp.alert_D).mean() - (~B0.alert_D).mean())
    return r


# ---------------- Stage 7: perturbations ----------------
pert = {"scales_from_training": {"s_w_peak_v_BW": s_w, "cv_w_peak_v": cv_w},
        "baseline_alert_rate": {a: float(B0[f"alert_{a}"].mean()) for a in M.ARCH}, "experiments": {}}
EXP = []
for m in (0.5, 1, 2):
    for sgn in (+1, -1):
        EXP.append((f"offset_{'+' if sgn > 0 else '-'}{m}sw", dict(fn_a=lambda z, a, bw, k=sgn * m * s_w: PT.offset(z, a, bw, k))))
        EXP.append((f"scale_{'+' if sgn > 0 else '-'}{m}cv", dict(fn_a=lambda z, a, bw, f=1 + sgn * m * cv_w: PT.scale(z, a, bw, f))))
for m in (1, 2, 4):
    EXP.append((f"trim_{m}samples", dict(fn_a=lambda z, a, bw, m=m: PT.trim(z, a, bw, m))))
for kind, mm in (("linear", 101), ("cubic", 101), ("linear", 51)):
    EXP.append((f"timenorm_{kind}{mm}", dict(featfn=lambda z, a, bw, kind=kind, mm=mm: PT.resampled_features(z, a, bw, mm, kind))))


def drop(k):
    def f(trials, sess):
        if len(trials) - k < 3:
            return trials
        r = np.random.default_rng(int(sess) + 17 * k); keep = np.sort(r.choice(len(trials), len(trials) - k, replace=False))
        return [trials[i] for i in keep]
    return f


def keep3(trials, sess):
    r = np.random.default_rng(int(sess) + 3); keep = np.sort(r.choice(len(trials), min(3, len(trials)), replace=False))
    return [trials[i] for i in keep]


EXP += [("trials_drop1", dict(trial_filter=drop(1))), ("trials_drop2", dict(trial_filter=drop(2))),
        ("trials_keep3", dict(trial_filter=keep3))]
for name, kw in EXP:
    vis = recompute(**kw); Bp = arch_scores(vis); r = compare(Bp)
    r["mean_delta_Y"] = float(np.mean(vis.d_peak_v - base_vis.d_peak_v)); r["mean_delta_se"] = float(np.mean(vis.d_peak_v_se - base_vis.d_peak_v_se))
    pert["experiments"][name] = r
    print(name, {a: round(r[a]["flip_rate"]["estimate"], 3) for a in M.ARCH}, flush=True)
# trial-count sensitivity: flip under keep3 by original number of affected-leg trials
na = np.array([len(legs(s)[0]) for s in S0.SESSION_ID_cur])
vis = recompute(trial_filter=keep3); Bp = arch_scores(vis)
pert["keep3_flip_by_original_trial_count"] = {
    lab: {a: float((Bp[f"alert_{a}"] != B0[f"alert_{a}"])[(na >= lo) & (na <= hi)].mean()) for a in M.ARCH} | {"n": int(((na >= lo) & (na <= hi)).sum())}
    for lab, lo, hi in (("3-5", 3, 5), ("6-8", 6, 8), ("9+", 9, 999))}

# missing observation: drop visit t-1 (pairs whose later visit is >= 3)
m3 = (S0.k >= 3).to_numpy(); sub = S0[m3].reset_index(drop=True)
Vp = V.set_index("SESSION_ID")
prev2 = []
for pid, k in zip(sub.SUBJECT_ID, sub.k):
    pv = PV[PV.SUBJECT_ID == pid].sort_values("visit_no"); prev2.append(pv.iloc[k - 3].SESSION_ID)
prev2 = np.array(prev2)
g = gps["d_peak_v"]; arr = arrs["d_peak_v"]; mu_m, var_m = [], []
for pid, k in zip(sub.SUBJECT_ID, sub.k):
    t, y, se2, X = arr[pid][:4]; idx = [i for i in range(k - 1) if i != k - 2]
    mu_, vl_ = g.predict_next(t[idx], y[idx], se2[idx], X[idx], t[k - 1], X[k - 1]); mu_m.append(mu_); var_m.append(vl_ + g.sigma_w ** 2)
Xm = pd.DataFrame({"Y_prev": Vp.loc[prev2, "d_peak_v"].to_numpy(), "se_prev": Vp.loc[prev2, "d_peak_v_se"].to_numpy(),
                   "pa_prev": Vp.loc[prev2, "peak_v_a"].to_numpy(), "pu_prev": Vp.loc[prev2, "peak_v_u"].to_numpy(),
                   "Y_cur": sub.Y_cur, "se_cur": sub.se_cur, "pa_cur": sub.pa_cur, "pu_cur": sub.pu_cur,
                   "mu_D": mu_m, "var_D": var_m})
Bm = M.scores(Xm, tau, mu_h, sd_h); B0s = B0[m3].reset_index(drop=True)
pert["missing_previous_visit"] = {
    "n_pairs": int(m3.sum()),
    **{a: {"flip_rate": float((Bm[f"alert_{a}"] != B0s[f"alert_{a}"]).mean()), "alert_rate": float(Bm[f"alert_{a}"].mean())} for a in M.ARCH},
    "D_mean_abs_change_in_prediction": float(np.mean(np.abs(np.array(mu_m) - sub.mu_D))),
    "D_mean_pred_sd_full_history": float(np.sqrt(sub.var_D + sub.se_cur ** 2).mean()),
    "D_mean_pred_sd_missing_t_minus_1": float(np.sqrt(np.array(var_m) + sub.se_cur ** 2).mean()),
    "D_cov95_full": float((~B0s.alert_D).mean()), "D_cov95_missing": float((~Bm.alert_D).mean())}
(A / "perturbation_results.json").write_text(json.dumps(pert, indent=2, default=float))

# ---------------- Stage 8: injected changes ----------------
inj = {"baseline": {}, "experiments": {}, "tradeoff": {}}
dirs = {"d_peak_v": -1}
base_cd = {t: cd_scores(base_vis, t) for t in EXTRA}


def rate(alert, direction=None, score=None):
    a = np.asarray(alert, bool)
    if direction is not None:
        a = a & (np.sign(score) == direction)
    return float(a.mean())


for a in M.ARCH:
    inj["baseline"][f"Y_{a}_any"] = rate(B0[f"alert_{a}"])
    inj["baseline"][f"Y_{a}_down"] = rate(B0[f"alert_{a}"], -1, B0[f"score_{a}"])
    inj["baseline"][f"Y_{a}_up"] = rate(B0[f"alert_{a}"], +1, B0[f"score_{a}"])
for t in EXTRA:
    for k in ("C", "D"):
        inj["baseline"][f"{t}_{k}_any"] = rate(np.abs(base_cd[t][k]) > M.Z)
        inj["baseline"][f"{t}_{k}_down"] = rate(np.abs(base_cd[t][k]) > M.Z, -1, base_cd[t][k])
        inj["baseline"][f"{t}_{k}_up"] = rate(np.abs(base_cd[t][k]) > M.Z, +1, base_cd[t][k])
    if base_cd[t]["A"] is not None:
        inj["baseline"][f"{t}_A_LSI10_any"] = rate(np.abs(base_cd[t]["A"]) >= 10)
        inj["baseline"][f"{t}_A_LSI10_down"] = rate(np.abs(base_cd[t]["A"]) >= 10, -1, base_cd[t]["A"])
        inj["baseline"][f"{t}_A_LSI10_up"] = rate(np.abs(base_cd[t]["A"]) >= 10, +1, base_cd[t]["A"])

INJ = [("vertical", d, dict(fn_a=lambda z, a, bw, d=d: PT.inj_vertical(z, a, bw, d))) for d in (-0.15, -0.10, -0.05, -0.02, 0.02, 0.05, 0.10, 0.15)]
INJ += [("propulsive", d, dict(fn_a=lambda z, a, bw, d=d: PT.inj_propulsive(z, a, bw, d))) for d in (0.05, 0.10, 0.20, 0.30)]
INJ += [("braking", d, dict(fn_a=lambda z, a, bw, d=d: PT.inj_braking(z, a, bw, d))) for d in (0.05, 0.10, 0.20, 0.30)]
INJ += [("peak1_local", d, dict(fn_a=lambda z, a, bw, d=d: PT.inj_peak1(z, a, bw, d))) for d in (0.05, 0.10, 0.15)]
INJ += [("bilateral_vertical", -d, dict(fn_a=lambda z, a, bw, d=d: PT.inj_vertical(z, a, bw, -d),
                                        fn_u=lambda z, a, bw, d=d: PT.inj_vertical(z, a, bw, -d))) for d in (0.05, 0.10)]
for kind, d, kw in INJ:
    vis = recompute(**kw); Bp = arch_scores(vis); key = f"{kind}_{d:+.2f}"
    r = {"mean_injected_change": {"d_peak_v": float(np.mean(vis.d_peak_v - base_vis.d_peak_v)),
                                  "peak_v_a": float(np.mean(vis.peak_v_a - base_vis.peak_v_a)),
                                  "d_propulsive": float(np.mean(vis.d_propulsive - base_vis.d_propulsive)),
                                  "d_braking": float(np.mean(vis.d_braking - base_vis.d_braking))}}
    if kind in ("vertical", "peak1_local", "bilateral_vertical"):
        direction = -1 if (kind != "vertical" or d < 0) else +1
        r["Y"] = {a: (rate(Bp[f"alert_{a}"]) if a == "B" else rate(Bp[f"alert_{a}"], direction, Bp[f"score_{a}"])) for a in M.ARCH}
        r["Y_by_injury"] = {c: {a: rate(Bp[f"alert_{a}"][(S0.CLASS_LABEL == c).to_numpy()], direction,
                                        Bp[f"score_{a}"][(S0.CLASS_LABEL == c).to_numpy()]) for a in ("C", "D")} for c in "AKHC"}
        cd = cd_scores(vis, "peak_v_a")
        r["affected_abs_peak_v"] = {k: rate(np.abs(cd[k]) > M.Z, direction, cd[k]) for k in ("C", "D")}
    if kind in ("propulsive", "braking"):
        t = "d_propulsive" if kind == "propulsive" else "d_braking"; direction = -1 if kind == "propulsive" else +1
        cd = cd_scores(vis, t)
        r[t] = {"C": rate(np.abs(cd["C"]) > M.Z, direction, cd["C"]), "D": rate(np.abs(cd["D"]) > M.Z, direction, cd["D"]),
                "A_LSI10": rate(np.abs(cd["A"]) >= 10, direction, cd["A"])}
        r["Y_peak_v_architectures"] = {a: rate(Bp[f"alert_{a}"]) for a in M.ARCH}
    inj["experiments"][key] = r
    print(key, {k: v for k, v in r.items() if k in ("Y", "d_propulsive", "d_braking", "affected_abs_peak_v")}, flush=True)

# detection vs baseline alert rate as the z threshold varies (C and D on Y); not an ROC curve
vis05 = recompute(fn_a=lambda z, a, bw: PT.inj_vertical(z, a, bw, -0.05))
vis10 = recompute(fn_a=lambda z, a, bw: PT.inj_vertical(z, a, bw, -0.10))
for zz in (1.645, 1.96, 2.576, 3.0):
    b, d5, d10 = arch_scores(base_vis, zz), arch_scores(vis05, zz), arch_scores(vis10, zz)
    inj["tradeoff"][str(zz)] = {a: {"baseline_any": rate(b[f"alert_{a}"]),
                                    "detect_-0.05": rate(d5[f"alert_{a}"], -1, d5[f"score_{a}"]),
                                    "detect_-0.10": rate(d10[f"alert_{a}"], -1, d10[f"score_{a}"])} for a in ("C", "D")}
inj["note"] = ("Detection = share of test pairs alerting in the injected direction after a known mathematical change to "
               "the later visit's affected-leg trials. Baseline alerts on unmodified pairs are not false positives; "
               "none of these rates is clinical sensitivity or specificity.")
(A / "injection_results.json").write_text(json.dumps(inj, indent=2, default=float))
print("done")
