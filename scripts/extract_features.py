"""Trial-level force features from the GaitRec raw files, one leg per run.

Usage:  python scripts/extract_features.py DATA_DIR OUT_CSV left|right
Verified on 2026-10-22: raw forces are in newtons sampled at 250 Hz (median stance 185 samples
= 0.74 s); AP force is negative during braking and positive during push-off; vertical and AP
files share row order (checked chunk by chunk below).
"""
import sys
import numpy as np
import pandas as pd

D, OUT, SIDE = sys.argv[1], sys.argv[2], sys.argv[3]
FS = 250.0
bw = pd.read_csv(f"{D}/GRF_metadata.csv").set_index("SESSION_ID")["BODY_WEIGHT"]
rows = []
V = pd.read_csv(f"{D}/GRF_F_V_RAW_{SIDE}.csv", chunksize=4000)
A = pd.read_csv(f"{D}/GRF_F_AP_RAW_{SIDE}.csv", chunksize=4000)
for cv, ca in zip(V, A):
    if not (cv.iloc[:, :3].values == ca.iloc[:, :3].values).all():
        raise SystemExit("row keys differ between vertical and AP files")
    fv, fa = cv.iloc[:, 3:].to_numpy(float), ca.iloc[:, 3:].to_numpy(float)
    for k in range(len(cv)):
        sid, sess, trial = (int(x) for x in cv.iloc[k, :3])
        w = float(bw.get(sess, np.nan))
        m = np.isfinite(fv[k]) & np.isfinite(fa[k])
        z, a = fv[k][m], fa[k][m]
        n = z.size
        valid = bool(n >= 10 and np.isfinite(w) and w > 0 and z.max() > 0.5 * w)
        r = dict(SUBJECT_ID=sid, SESSION_ID=sess, TRIAL_ID=trial, side=SIDE, n_samples=n, valid=valid,
                 peak_loading=np.nan, braking_impulse=np.nan, propulsive_impulse=np.nan, stance_s=n / FS)
        if valid:
            r["peak_loading"] = z[: n // 2].max() / w
            r["braking_impulse"] = -np.trapezoid(np.minimum(a, 0), dx=1 / FS) / w
            r["propulsive_impulse"] = np.trapezoid(np.maximum(a, 0), dx=1 / FS) / w
        rows.append(r)
pd.DataFrame(rows).to_csv(OUT, index=False)
print("DONE", len(rows))
