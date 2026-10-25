"""Per-step vertical curves (250, 100 and 25 Hz, time-normalised to 101 points, newtons) and AP impulses (N·s).

Usage: python scripts/extract_steps.py DATA_DIR left|right   -> DATA_DIR/outputs/steps_{side}.npz
Run on the machine holding the raw GaitRec files (verified against figshare MD5s on 2026-10-24).
Lower sampling rates are simulated by point-sampling the 250 Hz signal at 1/fs before time-normalisation.
"""
import sys, numpy as np, pandas as pd
D, SIDE = sys.argv[1], sys.argv[2]
FS = 250.0
g101 = np.linspace(0, 1, 101)
def curve(z, fs_new):
    n = z.size; t = np.arange(n) / FS
    if fs_new < FS:
        ts = np.arange(0, t[-1] + 1e-9, 1 / fs_new); z = np.interp(ts, t, z); t = ts
    return np.interp(g101 * t[-1], t, z).astype(np.float32)
keys, c250, c100, c25, brk, prp, ns = [], [], [], [], [], [], []
V = pd.read_csv(f"{D}/GRF_F_V_RAW_{SIDE}.csv", chunksize=4000)
A = pd.read_csv(f"{D}/GRF_F_AP_RAW_{SIDE}.csv", chunksize=4000)
for cv, ca in zip(V, A):
    assert (cv.iloc[:, :3].values == ca.iloc[:, :3].values).all()
    fv, fa = cv.iloc[:, 3:].to_numpy(float), ca.iloc[:, 3:].to_numpy(float)
    k3 = cv.iloc[:, :3].to_numpy(int)
    for k in range(len(cv)):
        m = np.isfinite(fv[k]) & np.isfinite(fa[k]); z, a = fv[k][m], fa[k][m]
        keys.append(k3[k]); ns.append(z.size)
        if z.size < 10:
            nan = np.full(101, np.nan, np.float32); c250.append(nan); c100.append(nan); c25.append(nan)
            brk.append(np.nan); prp.append(np.nan); continue
        c250.append(curve(z, 250)); c100.append(curve(z, 100)); c25.append(curve(z, 25))
        brk.append(-np.trapezoid(np.minimum(a, 0), dx=1 / FS)); prp.append(np.trapezoid(np.maximum(a, 0), dx=1 / FS))
np.savez_compressed(f"{D}/outputs/steps_{SIDE}.npz", keys=np.array(keys), n=np.array(ns), v250=np.array(c250),
                    v100=np.array(c100), v25=np.array(c25), brake_Ns=np.array(brk), push_Ns=np.array(prp))
print("DONE", SIDE, len(keys))
