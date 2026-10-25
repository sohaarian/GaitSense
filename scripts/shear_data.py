"""Per-step dataset for the shear-force study (docs/shear_protocol.md).

Input: data/derived/steps_{left,right}.npz, made on the data owner's machine from the raw GaitRec files by
scripts/extract_steps.py, plus data/GRF_metadata.csv and data/derived/trial_features_*.csv (validity).
Output: data/derived/shear_steps.parquet-like npz with one row per valid step of a cohort first-stay visit
and of healthy-control self-selected-speed sessions.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parents[1]; sys.path.insert(0, str(ROOT / "src"))
from gaitrec_rehab.cohort import CohortRules, build_cohort, load_metadata

D = ROOT / "data" / "derived"


def load_steps(include_holdout=False):
    """Valid steps of cohort first-stay visits (+ healthy controls).

    include_holdout=True also returns the temporal hold-out of docs/detection_protocol.md: patients meeting every
    cohort rule except the 365-day window. Their rows carry holdout=True. The default excludes them.
    """
    meta = load_metadata(str(ROOT / "data" / "GRF_metadata.csv"))
    cohort, _ = build_cohort(meta)
    dev_ids = set(cohort.SUBJECT_ID)
    if include_holdout:
        cohort, _ = build_cohort(meta, CohortRules(window_days=0))
    side_aff = cohort.set_index("SUBJECT_ID").AFFECTED_SIDE.map({0: "left", 1: "right"})
    first_age = cohort.set_index("SUBJECT_ID").AGE
    parts = []
    for side in ("left", "right"):
        z = np.load(D / f"steps_{side}.npz")
        k = z["keys"]
        df = pd.DataFrame({"SUBJECT_ID": k[:, 0], "SESSION_ID": k[:, 1], "TRIAL_ID": k[:, 2], "side": side,
                           "brake_Ns": z["brake_Ns"], "push_Ns": z["push_Ns"], "n": z["n"]})
        df["row"] = np.arange(len(df))
        tf = pd.read_csv(D / f"trial_features_{side}.csv")[["SESSION_ID", "TRIAL_ID", "valid", "peak_loading",
                                                             "braking_impulse", "propulsive_impulse"]]
        df = df.merge(tf, on=["SESSION_ID", "TRIAL_ID"], how="left", validate="1:1")
        parts.append((df, z))
    rows, c250, c100, c25 = [], [], [], []
    m = meta.set_index("SESSION_ID")
    for df, z in parts:
        sess = m.loc[df.SESSION_ID]
        bw = sess.BODY_WEIGHT.to_numpy()
        df = df.assign(BW=bw, BODY_MASS=sess.BODY_MASS.to_numpy(), CLASS_LABEL=sess.CLASS_LABEL.to_numpy(),
                       SEX=sess.SEX.to_numpy(), SPEED=sess.SPEED.to_numpy(), READMISSION=sess.READMISSION.to_numpy(),
                       SHOD=sess.SHOD_CONDITION.to_numpy(), DATE=sess.SESSION_DATE.to_numpy())
        keep_pt = (df.SUBJECT_ID.isin(cohort.SUBJECT_ID) & (df.READMISSION == 0) & (df.SHOD == 1))
        keep_hc = (df.CLASS_LABEL == "HC") & (df.SPEED == 2)
        keep = df.valid.fillna(False).astype(bool) & (keep_pt | keep_hc)
        df = df[keep].copy()
        r = df.row.to_numpy(); w = df.BW.to_numpy()[:, None]
        c250.append(z["v250"][r] / w); c100.append(z["v100"][r] / w); c25.append(z["v25"][r] / w)
        df["brake"] = df.brake_Ns / df.BW; df["push"] = df.push_Ns / df.BW
        df["stance_s"] = df.n / 250.0
        rows.append(df)
    out = pd.concat(rows, ignore_index=True)
    out["role"] = np.where(out.CLASS_LABEL == "HC", "healthy",
                           np.where(out.side == out.SUBJECT_ID.map(side_aff), "affected", "unaffected"))
    out["age_first"] = out.SUBJECT_ID.map(first_age)
    out["holdout"] = (out.role != "healthy") & ~out.SUBJECT_ID.isin(dev_ids)
    return out, np.vstack(c250).astype(np.float32), np.vstack(c100).astype(np.float32), np.vstack(c25).astype(np.float32)


if __name__ == "__main__":
    st, a, b, c = load_steps()
    pt = st[st.role != "healthy"]
    print("steps", len(st), "patient steps", len(pt), "patients", pt.SUBJECT_ID.nunique(),
          "sessions", pt.SESSION_ID.nunique(), "HC steps", (st.role == "healthy").sum())
    print("max |brake - earlier braking_impulse|", float(np.nanmax(np.abs(st.brake - st.braking_impulse))))
    print("max |push - earlier propulsive_impulse|", float(np.nanmax(np.abs(st.push - st.propulsive_impulse))))
    print("max |curve first-half peak - earlier peak_loading| (101-pt resample)",
          float(np.nanmax(np.abs(a[:, :51].max(1) - st.peak_loading))))
    print("NaN curves", int(np.isnan(a).any(1).sum()), int(np.isnan(c).any(1).sum()))


def patient_split(ids, seed=0):
    """Patient-level 60/20/20 train/validation/test split used by every shear-study script."""
    ids = np.sort(np.unique(ids)); perm = np.random.default_rng(seed).permutation(ids)
    return {s: ("train" if i < 0.6 * len(ids) else "val" if i < 0.8 * len(ids) else "test") for i, s in enumerate(perm)}
