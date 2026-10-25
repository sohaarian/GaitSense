"""Session-mean, time-normalised (101-point) vertical and AP curves divided by body weight, one leg per run.

Exports cohort index sessions and all healthy-control sessions (input for the M1 autoencoder).
Usage:  python scripts/extract_curves.py DATA_DIR OUT_NPZ left|right
"""
import sys, numpy as np, pandas as pd
D, OUT, SIDE = sys.argv[1], sys.argv[2], sys.argv[3]
m = pd.read_csv(f"{D}/GRF_metadata.csv"); m["D"] = pd.to_datetime(m.SESSION_DATE, format="%d.%m.%Y")
p = m[m.CLASS_LABEL != "HC"]; end = p.D.max()
f = p[p.READMISSION == 0].sort_values(["SUBJECT_ID", "D", "SESSION_ID"]).groupby("SUBJECT_ID").first().reset_index()
f = f[f.AFFECTED_SIDE.isin([0, 1]) & (f.SHOD_CONDITION == 1) & ((end - f.D).dt.days >= 365)]
keep = set(f.SESSION_ID) | set(m.loc[m.CLASS_LABEL == "HC", "SESSION_ID"])
bw = m.set_index("SESSION_ID").BODY_WEIGHT
g = np.linspace(0, 1, 101); acc = {}
V = pd.read_csv(f"{D}/GRF_F_V_RAW_{SIDE}.csv", chunksize=4000); A = pd.read_csv(f"{D}/GRF_F_AP_RAW_{SIDE}.csv", chunksize=4000)
for cv, ca in zip(V, A):
    sel = cv.SESSION_ID.isin(keep).values
    if not sel.any(): continue
    fv, fa = cv.iloc[:, 3:].to_numpy(float)[sel], ca.iloc[:, 3:].to_numpy(float)[sel]
    for s, z, a in zip(cv.SESSION_ID.values[sel], fv, fa):
        mk = np.isfinite(z) & np.isfinite(a); z, a = z[mk], a[mk]; w = bw[s]
        if z.size < 10 or z.max() <= 0.5 * w: continue
        x = np.linspace(0, 1, z.size)
        acc.setdefault(int(s), []).append(np.r_[np.interp(g, x, z), np.interp(g, x, a)] / w)
ids = np.array(sorted(acc)); X = np.stack([np.mean(acc[s], 0) for s in ids]).astype(np.float32)
np.savez_compressed(OUT, session_id=ids, curves=X, n_trials=np.array([len(acc[s]) for s in ids]))
print("DONE", SIDE, len(ids), X.shape)
