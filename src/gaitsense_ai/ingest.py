"""Data ingestion: GaitRec RAW force files -> trial-level feature table and time-normalised curves."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from . import features as F

KEYS = ["SUBJECT_ID", "SESSION_ID", "TRIAL_ID"]


def sha256(path, chunk=1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def load_metadata(path) -> pd.DataFrame:
    m = pd.read_csv(path)
    if m.SESSION_ID.duplicated().any():
        raise ValueError("SESSION_ID must be unique in metadata")
    m["SESSION_DATE"] = pd.to_datetime(m.SESSION_DATE, format="%d.%m.%Y")
    return m


def iter_raw(raw_dir, side, chunksize=4000):
    """Yield (keys[n,3], vertical[n,L], ap[n,L]) chunks, checking that the V and AP files share row order."""
    V = pd.read_csv(Path(raw_dir) / f"GRF_F_V_RAW_{side}.csv", chunksize=chunksize)
    A = pd.read_csv(Path(raw_dir) / f"GRF_F_AP_RAW_{side}.csv", chunksize=chunksize)
    for cv, ca in zip(V, A):
        k = cv.iloc[:, :3].to_numpy(int)
        if not (k == ca.iloc[:, :3].to_numpy(int)).all():
            raise ValueError("row keys differ between vertical and AP files")
        yield k, cv.iloc[:, 3:].to_numpy(float), ca.iloc[:, 3:].to_numpy(float)


def stance(row_v, row_a):
    """Strip trailing NaN padding; report interior gaps (a QC failure)."""
    fin = np.isfinite(row_v) & np.isfinite(row_a)
    n = int(fin.sum())
    last = int(np.flatnonzero(fin)[-1]) + 1 if n else 0
    return row_v[:last], row_a[:last], (last - n)


def extract_side(raw_dir, side, body_weight: pd.Series):
    """Return (trial DataFrame, curves dict) for one leg file."""
    rows, cv_list, ca_list = [], [], []
    for keys, fv, fa in iter_raw(raw_dir, side):
        for k in range(len(keys)):
            sid, sess, trial = (int(x) for x in keys[k])
            z, a, gaps = stance(fv[k], fa[k])
            bw = float(body_weight.get(sess, np.nan))
            r = dict(SUBJECT_ID=sid, SESSION_ID=sess, TRIAL_ID=trial, side=side, n_samples=int(z.size),
                     interior_gaps=int(gaps), min_v_N=float(np.nanmin(z)) if z.size else np.nan,
                     max_abs_ap_N=float(np.nanmax(np.abs(a))) if a.size else np.nan)
            ok, why = F.is_valid(z, bw) if gaps == 0 else (False, "interior_gap")
            r["valid"], r["invalid_reason"] = ok, why
            feats = dict.fromkeys(F.ALL, np.nan)
            cz = ca = np.full(F.GRID.size, np.nan, np.float32)
            if ok:
                feats = F.trial_features(z, a, bw)
                cz = F.normalised_curve(z, bw).astype(np.float32)
                ca = F.normalised_curve(a, bw).astype(np.float32)
            r.update(feats); rows.append(r); cv_list.append(cz); ca_list.append(ca)
    t = pd.DataFrame(rows)
    curves = {"keys": t[KEYS].to_numpy(np.int64), "v": np.vstack(cv_list), "ap": np.vstack(ca_list)}
    return t, curves
