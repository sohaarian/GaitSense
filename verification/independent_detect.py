"""Independent check of the detection study's reference labels and conventional-tracker results.

Shares no code with scripts/ or the package (sklearn's GroupKFold is used to reproduce the fold assignment).
Rebuilds cohorts, visit pairs, fold-wise tau, true-change labels, the V-thr score, its 5%-FPR threshold and
sensitivity, for the development CV and (if present) the hold-out. Compares with results/detect_*.json.
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

R = Path(__file__).parents[1]
m = pd.read_csv(R / "data" / "GRF_metadata.csv"); m["date"] = pd.to_datetime(m.SESSION_DATE, format="%d.%m.%Y")
pt = m[m.CLASS_LABEL != "HC"]; end = pt.date.max(); fs_ = pt[pt.READMISSION == 0]
idx = fs_.sort_values(["date", "SESSION_ID"]).drop_duplicates("SUBJECT_ID")
idx = idx[idx.AFFECTED_SIDE.isin([0, 1]) & (idx.SHOD_CONDITION == 1)]
dev = set(idx[(end - idx.date).dt.days >= 365].SUBJECT_ID); hold = set(idx.SUBJECT_ID) - dev
aff = dict(zip(idx.SUBJECT_ID, idx.AFFECTED_SIDE.map({0: "left", 1: "right"})))
visits = fs_[fs_.SUBJECT_ID.isin(aff) & (fs_.SHOD_CONDITION == 1)].set_index("SESSION_ID")
rows = []
for side in ("left", "right"):
    z = np.load(R / "data" / "derived" / f"steps_{side}.npz"); k = z["keys"]
    f = pd.DataFrame({"SUBJECT_ID": k[:, 0], "SESSION_ID": k[:, 1], "TRIAL_ID": k[:, 2], "push": z["push_Ns"]})
    tf = pd.read_csv(R / "data" / "derived" / f"trial_features_{side}.csv")[["SESSION_ID", "TRIAL_ID", "valid", "peak_loading"]]
    f = f.merge(tf, on=["SESSION_ID", "TRIAL_ID"]); f = f[f.valid & f.SESSION_ID.isin(visits.index)].copy()
    f["push"] = f.push / visits.loc[f.SESSION_ID, "BODY_WEIGHT"].to_numpy()
    f["aff"] = [aff[s] == side for s in f.SUBJECT_ID]
    rows.append(f)
st = pd.concat(rows)
vt = []
for sess, g in st.groupby("SESSION_ID"):
    a, u = g[g.aff], g[~g.aff]
    if len(a) < 2 or len(u) < 2:
        continue
    r = {"SESSION_ID": sess, "SUBJECT_ID": g.SUBJECT_ID.iat[0], "date": visits.loc[sess, "date"]}
    for c in ("push", "peak_loading"):
        r[c] = a[c].mean() - u[c].mean(); r[c + "_v"] = a[c].var() / len(a) + u[c].var() / len(u)
    vt.append(r)
vt = pd.DataFrame(vt).sort_values(["SUBJECT_ID", "date", "SESSION_ID"]).reset_index(drop=True)
pv = vt.groupby("SUBJECT_ID").shift(1); pr = vt[pv.SESSION_ID.notna()].copy()
for c in ("push", "peak_loading"):
    pr["d" + c] = vt[c] - pv[c]; pr["v" + c] = vt[c + "_v"] + pv[c + "_v"]
tau = lambda d, v: float(np.sqrt(max(np.var(d, ddof=1) - np.mean(v), 0)))


def evaluate(train, test):
    tp, tl = tau(train.dpush, train.vpush), tau(train.dpeak_loading, train.vpeak_loading)
    lab = (test.dpush.abs() > 1.96 * np.sqrt(test.vpush + tp ** 2)).to_numpy()
    score = (test.dpeak_loading.abs() / np.sqrt(test.vpeak_loading + tl ** 2)).to_numpy()
    return lab, score


D = pr[pr.SUBJECT_ID.isin(dev)].reset_index(drop=True)
lab = np.zeros(len(D), bool); score = np.zeros(len(D))
for tr, te in GroupKFold(5).split(D, groups=D.SUBJECT_ID.to_numpy()):
    lab[te], score[te] = evaluate(D.iloc[tr], D.iloc[te])
thr = float(np.quantile(score[~lab], 0.95))
out = {"dev_pairs": len(D), "dev_true": int(lab.sum()), "vthr_threshold": thr, "vthr_cv_sensitivity": float((score[lab] > thr).mean())}
cv = json.loads((R / "results" / "detect_cv.json").read_text())
cmp = [("dev pairs", out["dev_pairs"], cv["pairs"]), ("dev true changes", out["dev_true"], cv["true_changes"]),
       ("V-thr threshold", thr, cv["thresholds_at_5pct_fpr"]["V-thr"]), ("V-thr CV sensitivity", out["vthr_cv_sensitivity"], cv["cv"]["V-thr"]["sensitivity"])]
hf = R / "results" / "detect_holdout.json"
if hf.exists():
    Hh = pr[pr.SUBJECT_ID.isin(hold)].reset_index(drop=True)
    hl, hs = evaluate(D, Hh); ho = json.loads(hf.read_text())
    out.update({"holdout_pairs": len(Hh), "holdout_true": int(hl.sum()), "vthr_holdout_sensitivity": float((hs[hl] > cv["thresholds_at_5pct_fpr"]["V-thr"]).mean()) if hl.any() else None})
    cmp += [("hold-out pairs", out["holdout_pairs"], ho["holdout_pairs"]), ("hold-out true changes", out["holdout_true"], ho["true_changes"]),
            ("V-thr hold-out sensitivity", out["vthr_holdout_sensitivity"], ho["holdout"]["V-thr"]["sensitivity"])]
bad = 0
for n_, a_, b_ in cmp:
    ok = abs(a_ - b_) < 1e-9; bad += not ok
    print(f"{n_:28s} {a_:.6f} {b_:.6f} {'OK' if ok else 'MISMATCH'}")
out["comparison_ok"] = bad == 0
(R / "results" / "independent_detect.json").write_text(json.dumps(out, indent=2))
sys.exit(1 if bad else 0)
