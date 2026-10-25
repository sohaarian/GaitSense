"""Independent check of the shear study's key numbers (tau and crossing rates for true and AI push-off).

Shares no code with scripts/shear_*.py or the package. It rebuilds the cohort from the metadata, the steps
from steps_*.npz, the patient split from its definition (seed 0, 60/20/20 of sorted IDs), and the AI's
step predictions from the saved model, then recomputes everything else here.
"""
import json, pickle, sys
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(__file__).parents[1]
m = pd.read_csv(R / "data" / "GRF_metadata.csv"); m["date"] = pd.to_datetime(m.SESSION_DATE, format="%d.%m.%Y")
pt = m[m.CLASS_LABEL != "HC"]; end = pt.date.max(); fs_ = pt[pt.READMISSION == 0]
idx = fs_.sort_values(["date", "SESSION_ID"]).drop_duplicates("SUBJECT_ID")
idx = idx[idx.AFFECTED_SIDE.isin([0, 1]) & (idx.SHOD_CONDITION == 1) & ((end - idx.date).dt.days >= 365)]
aff = dict(zip(idx.SUBJECT_ID, idx.AFFECTED_SIDE.map({0: "left", 1: "right"}))); age = dict(zip(idx.SUBJECT_ID, idx.AGE))
visits = fs_[fs_.SUBJECT_ID.isin(aff) & (fs_.SHOD_CONDITION == 1)].set_index("SESSION_ID")
ids = np.sort(list(aff)); perm = np.random.default_rng(0).permutation(ids)
split = {s: ("train" if i < 0.6 * len(ids) else "val" if i < 0.8 * len(ids) else "test") for i, s in enumerate(perm)}

M = pickle.load(open(R / "data" / "derived" / "shear_models.pkl", "rb")); gb = M["models"]["gb"][1]
frames = []
for side in ("left", "right"):
    z = np.load(R / "data" / "derived" / f"steps_{side}.npz"); k = z["keys"]
    f = pd.DataFrame({"SUBJECT_ID": k[:, 0], "SESSION_ID": k[:, 1], "TRIAL_ID": k[:, 2]})
    v = pd.read_csv(R / "data" / "derived" / f"trial_features_{side}.csv").set_index(["SESSION_ID", "TRIAL_ID"]).valid
    ok = f.SESSION_ID.isin(visits.index).to_numpy() & v.reindex(pd.MultiIndex.from_frame(f[["SESSION_ID", "TRIAL_ID"]])).fillna(False).to_numpy(bool)
    f = f[ok].copy(); r = np.where(ok)[0]
    bw = visits.loc[f.SESSION_ID, "BODY_WEIGHT"].to_numpy(); c = z["v250"][r] / bw[:, None]
    s = z["n"][r] / 250.0
    is_aff = np.array([aff[a] == side for a in f.SUBJECT_ID])
    # same input layout the model was trained with: 13 curve summaries, mass, affected flag, 4 groups, curve
    t1 = c[:, :51].argmax(1); p1 = c[:, :51].max(1); t2 = 50 + c[:, 50:].argmax(1); p2 = c[:, 50:].max(1)
    vl = np.array([c[i, t1[i]:t2[i] + 1].min() for i in range(len(c))]); tv = np.array([t1[i] + c[i, t1[i]:t2[i] + 1].argmin() for i in range(len(c))])
    mu = c.mean(1); grp = visits.loc[f.SESSION_ID, "CLASS_LABEL"].to_numpy()
    X = np.column_stack([p1, p2, vl, t1, t2, tv, p1 / np.maximum(t1, 1) * 100 / s, p2 / np.maximum(100 - t2, 1) * 100 / s, mu, mu * s, s,
                         p2 - p1, p1 - vl, visits.loc[f.SESSION_ID, "BODY_MASS"].to_numpy(), is_aff.astype(float)] +
                        [(grp == g).astype(float) for g in "KAHC"] + [c])
    f["true"] = z["push_Ns"][r] / bw; f["ai"] = gb.predict(X); f["aff"] = is_aff
    frames.append(f)
st = pd.concat(frames)
rows = []
for sess, g in st.groupby("SESSION_ID"):
    a, u = g[g.aff], g[~g.aff]
    if len(a) < 2 or len(u) < 2:
        continue
    r = {"SESSION_ID": sess, "SUBJECT_ID": g.SUBJECT_ID.iat[0], "date": visits.loc[sess, "date"]}
    for c in ("true", "ai"):
        r[c] = a[c].mean() - u[c].mean(); r[c + "_v"] = a[c].var() / len(a) + u[c].var() / len(u)
    rows.append(r)
vt = pd.DataFrame(rows).sort_values(["SUBJECT_ID", "date", "SESSION_ID"]).reset_index(drop=True)
pv = vt.groupby("SUBJECT_ID").shift(1); pr = vt[pv.SESSION_ID.notna()].copy()
for c in ("true", "ai"):
    pr["d" + c] = vt[c] - pv[c]; pr["v" + c] = vt[c + "_v"] + pv[c + "_v"]
pr["split"] = pr.SUBJECT_ID.map(split); pr["age"] = pr.SUBJECT_ID.map(age)
val = pr[pr.split == "val"]; te = pr[(pr.split == "test") & pr.age.between(18, 65)]
out = {}
for c in ("true", "ai"):
    t = float(np.sqrt(max(val["d" + c].var() - val["v" + c].mean(), 0)))
    out[c] = {"tau": t, "crossing": float((te["d" + c].abs() > 1.96 * np.sqrt(te["v" + c] + t ** 2)).mean())}
a, b = te.dai.to_numpy(), te.dtrue.to_numpy()
out["ccc"] = float(2 * np.cov(a, b, ddof=0)[0, 1] / (a.var() + b.var() + (a.mean() - b.mean()) ** 2))
out["n_test_pairs"] = int(len(te))

main = json.loads((R / "results" / "shear_study.json").read_text())
e = main["study2A_test_18_65"]["push"]
cmp = [("test pairs", out["n_test_pairs"], e["n_pairs"]), ("tau true", out["true"]["tau"], main["tau_from_validation"]["true"]["push"]),
       ("tau AI", out["ai"]["tau"], main["tau_from_validation"]["gb"]["push"]),
       ("crossing true", out["true"]["crossing"], e["crossing_rate_true"]), ("crossing AI", out["ai"]["crossing"], e["gb"]["crossing_rate"]),
       ("CCC AI", out["ccc"], e["gb"]["ccc"])]
bad = 0
for n_, a_, b_ in cmp:
    ok_ = abs(a_ - b_) < 1e-6; bad += not ok_   # main script stores curves as float32, this one in float64
    print(f"{n_:15s} {a_:.6f} {b_:.6f} {'OK' if ok_ else 'MISMATCH'}")
out["comparison_ok"] = bad == 0
(R / "results" / "independent_shear.json").write_text(json.dumps(out, indent=2))
sys.exit(1 if bad else 0)
