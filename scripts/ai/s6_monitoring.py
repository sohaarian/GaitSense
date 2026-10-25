"""Stage 6: compare monitoring architectures A-D on identical consecutive visit pairs (test patients).

Thresholds for the matched-rate comparison are set on calibration pairs. Writes results/ai/monitoring_pairs.csv.gz
and results/ai/monitoring_results.json.
"""
import itertools, json, pickle, sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).parents[2]; sys.path.insert(0, str(ROOT / "src"))
from gaitsense_ai import SEED, splits
from gaitsense_ai import monitoring as M
from gaitsense_ai import prediction as P

A = ROOT / "results" / "ai"
sel = json.loads((A / "model_selection.json").read_text())
with open(A / "models.pkl", "rb") as f:
    pk = pickle.load(f)
std = pk["std"]
V = pd.read_csv(A / "visit_level_features.csv.gz", parse_dates=["SESSION_DATE"])
part = pd.read_csv(A / "partition.csv")
TGT = "d_peak_v"; s = sel["targets"][TGT]
g3 = pk["fitted"][TGT][s["M3_chosen"]]; tau = s["tau_train"]

# healthy reference: training controls, self-selected speed only (patients were all recorded at SPEED = 2)
hc_tr = V[(V.role == "hc") & (V.split == "train") & (V.SPEED == 2)]
mu_h, sd_h = M.fit_healthy_reference(hc_tr)
cz = np.load(ROOT / "data" / "derived" / "ai_visit_curves.npz")
curve = np.hstack([cz["v_a"], cz["ap_a"]]); cidx = pd.Index(cz["session_id"])
cref = M.CurveReference().fit(curve[cidx.get_indexer(hc_tr.SESSION_ID)])
dist = pd.Series(cref.distance(curve), index=cz["session_id"])

PV = V[V.role.isin(["cohort", "temporal"])]
arrays = P.patient_arrays(PV, TGT, P.TARGETS[TGT], std)
tr_vis = V[(V.role == "cohort") & (V.split == "train")].assign(week=lambda d: d.day // 7 + 1)


def build(split):
    Pp = pd.read_csv(A / "eligible_visit_pairs.csv.gz")
    Pp = Pp[Pp.split == split].reset_index(drop=True)
    X = pd.DataFrame({"SUBJECT_ID": Pp.SUBJECT_ID, "CLASS_LABEL": Pp.CLASS_LABEL, "k": Pp.visit_no_cur.astype(int),
                      "visit_no": Pp.visit_no_cur.astype(int), "gap_days": Pp.gap_days,
                      "SESSION_ID_prev": Pp.SESSION_ID_prev.astype(int), "SESSION_ID_cur": Pp.SESSION_ID_cur.astype(int),
                      "Y_prev": Pp.d_peak_v_prev, "Y_cur": Pp.d_peak_v_cur, "se_prev": Pp.d_peak_v_se_prev,
                      "se_cur": Pp.d_peak_v_se_cur, "pa_prev": Pp.peak_v_a_prev, "pu_prev": Pp.peak_v_u_prev,
                      "pa_cur": Pp.peak_v_a_cur, "pu_cur": Pp.peak_v_u_cur, "day_cur": Pp.day_cur})
    mu, vl = P.gp_predict_rows(g3, arrays, X)
    X["mu_D"], X["var_D"] = mu, vl + g3.sigma_w ** 2
    S = pd.concat([X, M.scores(X, tau, mu_h, sd_h)], axis=1)
    S["curve_dist_prev"] = S.SESSION_ID_prev.map(dist).to_numpy(); S["curve_dist_cur"] = S.SESSION_ID_cur.map(dist).to_numpy()
    S["d_curve_dist"] = S.curve_dist_cur - S.curve_dist_prev
    wk = S.day_cur // 7 + 1; pct = []
    for c, w, y in zip(S.CLASS_LABEL, wk, S.Y_cur):
        pool = tr_vis.loc[(tr_vis.CLASS_LABEL == c) & (tr_vis.week == w), "d_peak_v"]
        pct.append(np.nan if len(pool) < 20 else float((pool < y).mean() * 100))
    S["injury_matched_pct"] = pct
    return S


Scal, Ste = build("calib"), build("test")
splits.assert_only(Scal, part, {"calib"}, "calibration pairs"); splits.assert_only(Ste, part, {"test"}, "test pairs")
thr = M.matched_thresholds(Scal, 0.10)
MA = M.matched_alerts(Ste, thr)
for a in M.ARCH:
    Ste[f"matched_{a}"] = MA[a].to_numpy()
Ste.to_csv(A / "monitoring_pairs.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})

rng = np.random.default_rng(SEED); pids = Ste.SUBJECT_ID.unique(); B = 1000
groups = {p: np.flatnonzero(Ste.SUBJECT_ID.to_numpy() == p) for p in pids}
boot_idx = [np.concatenate([groups[p] for p in rng.choice(pids, len(pids))]) for _ in range(B)]


def with_ci(fn):
    est = fn(np.arange(len(Ste))); dist = np.array([fn(ix) for ix in boot_idx])
    return {"estimate": float(est), "ci95": [float(np.nanquantile(dist, .025)), float(np.nanquantile(dist, .975))]}


res = {"n_pairs_test": len(Ste), "n_patients_test": int(len(pids)), "n_pairs_calib": len(Scal), "tau": tau,
       "healthy_reference": {"mu": mu_h, "sd": sd_h, "n_sessions": len(hc_tr), "n_controls": int(hc_tr.SUBJECT_ID.nunique()),
                             "curve_pca_components": int(cref.pca.n_components_)},
       "matched_rate": 0.10, "matched_thresholds_from_calibration": thr, "D_model": s["M3_chosen"]}
for mode, col in (("native", "alert_{}"), ("matched", "matched_{}")):
    al = {a: Ste[col.format(a)].to_numpy(bool) for a in M.ARCH}
    r = {"alert_rate": {a: with_ci(lambda ix, a=a: al[a][ix].mean()) for a in M.ARCH}, "pairwise": {}}
    if mode == "native":
        r["alert_rate"]["C0_original"] = with_ci(lambda ix: Ste.alert_C0.to_numpy()[ix].mean())
        r["calibration_alert_rate"] = {a: float(Scal[f"alert_{a}"].mean()) for a in M.ARCH}
    for a, b in itertools.combinations(M.ARCH, 2):
        r["pairwise"][f"{a}-{b}"] = {k: with_ci(lambda ix, k=k: M.pair_stats(al[a][ix], al[b][ix])[k])
                                     for k in ("agreement", "kappa", "positive_agreement", "joint_alert")}
    # directional conflict among A, C, D (same sign semantics); B is toward/away from healthy and is excluded
    dc = {}
    for a, b in itertools.combinations(["A", "C", "D"], 2):
        both = al[a] & al[b]
        conf = both & (Ste[f"dir_{a}"].to_numpy() != Ste[f"dir_{b}"].to_numpy())
        dc[f"{a}-{b}"] = {"both_alert": int(both.sum()), "opposite_direction": int(conf.sum())}
    r["directional_conflict"] = dc
    for by, key in (("CLASS_LABEL", "by_injury"), ("visit_no", "by_visit_number"), ("gap_bin", "by_gap")):
        col_by = Ste[by] if by != "gap_bin" else pd.cut(Ste.gap_days, [-1, 7, 14, 28, 1e4], labels=["<=7d", "8-14d", "15-28d", ">28d"]).astype(str)
        if by == "visit_no":
            col_by = Ste.visit_no.clip(upper=5).astype(str).replace({"5": "5+"})
        r[key] = {str(lv): {"n": int((col_by == lv).sum()), **{a: float(al[a][(col_by == lv).to_numpy()].mean()) for a in M.ARCH}}
                  for lv in sorted(col_by.unique())}
    if mode == "matched":
        prox = {}
        for a, b in itertools.combinations(M.ARCH, 2):
            dis = al[a] != al[b]
            near = lambda x: (np.abs(Ste[f"score_{x}"]) >= 0.9 * thr[x]) & (np.abs(Ste[f"score_{x}"]) <= 1.1 * thr[x])
            prox[f"{a}-{b}"] = {"disagreements": int(dis.sum()),
                                "share_with_a_score_within_10pct_of_threshold": float((near(a) | near(b))[dis].mean()) if dis.any() else None}
        r["threshold_proximity"] = prox
        kap = {k: v["kappa"]["estimate"] for k, v in r["pairwise"].items()}
        r["H3_min_kappa"] = min(kap.values()); r["H3_material_disagreement"] = bool(min(kap.values()) < 0.60)
    res[mode] = r
sc = Ste[[f"score_{a}" for a in M.ARCH] + ["d_curve_dist", "injury_matched_pct"]]
res["continuous_spearman"] = sc.corr(method="spearman").round(4).to_dict()
alA = Ste.alert_A.to_numpy(bool)
res["A_which_limb"] = {"A_alerts": int(alA.sum()),
                       "share_unaffected_change_at_least_half_of_dY": float((np.abs(Ste.d_unaffected) >= 0.5 * np.abs(Ste.Y_cur - Ste.Y_prev))[alA].mean())}
res["B_continuous"] = {"curve_distance_median_patients_first": float(Ste.curve_dist_prev.median()),
                       "curve_distance_median_hc_test": float(dist.loc[V[(V.role == 'hc') & (V.split == 'test') & (V.SPEED == 2)].SESSION_ID].median()),
                       "share_pairs_moving_toward_healthy_curve": float((Ste.d_curve_dist < 0).mean())}
(A / "monitoring_results.json").write_text(json.dumps(res, indent=2, default=float))
print(json.dumps({k: res[k] for k in ("healthy_reference", "matched_thresholds_from_calibration")}, indent=1))
print("native rates", {a: round(v["estimate"], 3) for a, v in res["native"]["alert_rate"].items()})
print("matched kappa", {k: round(v["kappa"]["estimate"], 3) for k, v in res["matched"]["pairwise"].items()})
print("native kappa", {k: round(v["kappa"]["estimate"], 3) for k, v in res["native"]["pairwise"].items()})
