"""Automated data-quality checks. Each check returns a row for results/data_quality_report.csv."""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd


def _row(check, level, flagged, total, action, detail=""):
    return {"check": check, "level": level, "n_flagged": int(flagged), "n_total": int(total),
            "share": float(flagged) / total if total else np.nan, "action": action, "detail": detail}


def run_checks(trials: pd.DataFrame, meta: pd.DataFrame, curves: dict, visits: pd.DataFrame,
               partition: pd.DataFrame, stance_limits=(0.4, 1.5), peak_limits=(0.8, 2.0)) -> pd.DataFrame:
    R = []
    n = len(trials)
    R.append(_row("missing force values inside stance", "trial", (trials.interior_gaps > 0).sum(), n, "trial invalid"))
    R.append(_row("invalid trial (validity rule)", "trial", (~trials.valid).sum(), n, "trial excluded",
                  "; ".join(f"{k}:{v}" for k, v in trials.loc[~trials.valid, "invalid_reason"].value_counts().items())))
    ms = meta.set_index("SESSION_ID").SUBJECT_ID
    R.append(_row("session missing from metadata", "trial", (~trials.SESSION_ID.isin(ms.index)).sum(), n, "trial excluded"))
    bad_id = trials.SUBJECT_ID.to_numpy() != trials.SESSION_ID.map(ms).to_numpy()
    R.append(_row("patient id disagrees with metadata", "trial", bad_id.sum(), n, "trial excluded"))
    pts = meta[(meta.CLASS_LABEL != "HC") & (meta.READMISSION == 0)]
    side_var = pts.groupby("SUBJECT_ID").AFFECTED_SIDE.nunique()
    R.append(_row("affected side changes within first stay", "patient", (side_var > 1).sum(), len(side_var),
                  "index-session side used (v1.x rule); patients with side 2 (both) excluded by cohort rule"))
    bwr = meta.BODY_WEIGHT / (meta.BODY_MASS * 9.81)
    R.append(_row("body weight != body mass x g (>2%)", "session", ((bwr - 1).abs() > 0.02).sum(), len(meta),
                  "flag only", f"ratio range {bwr.min():.3f}-{bwr.max():.3f}"))
    v = trials[trials.valid]
    R.append(_row(f"peak_v outside {peak_limits} BW", "trial", (~v.peak_v.between(*peak_limits)).sum(), len(v), "flag only"))
    R.append(_row(f"stance duration outside {stance_limits} s", "trial", (~v.stance_s.between(*stance_limits)).sum(),
                  len(v), "flag only"))
    R.append(_row("curve does not start/end at the 25 N stance threshold", "trial",
                  (v.min_v_N < 24.0).sum(), len(v), "flag only"))
    R.append(_row("|AP force| > 0.6 BW (implausible)", "trial",
                  (v.max_abs_ap_N / v.SESSION_ID.map(meta.set_index('SESSION_ID').BODY_WEIGHT) > 0.6).sum(), len(v), "flag only"))
    R.append(_row("no distinct second peak (atypical vertical shape)", "trial", v.peak2.isna().sum(), len(v),
                  "peak2/valley set to NaN; never forced"))
    R.append(_row("no dominant first peak", "trial", v.peak1.isna().sum(), len(v), "peak1/loading rate NaN"))
    # duplicates: identical normalised vertical curves (possible copy errors)
    hv = [hashlib.md5(np.round(c, 5).tobytes()).hexdigest() for c in curves["v"]]
    dup = pd.Series(hv).duplicated(keep=False) & np.isfinite(curves["v"]).all(1)
    R.append(_row("duplicated recordings (identical curves)", "trial", int(dup.sum()), len(hv), "flag only"))
    # visit-level
    nv = visits[visits.role != "hc"]
    same_day = nv.groupby(["SUBJECT_ID", "SESSION_DATE"]).size()
    R.append(_row("more than one included session on the same day", "visit", int((same_day > 1).sum()), len(same_day),
                  "kept; ordered by SESSION_ID, gap 0 days"))
    order = nv.sort_values(["SUBJECT_ID", "SESSION_DATE", "SESSION_ID"]).groupby("SUBJECT_ID").SESSION_DATE.apply(
        lambda s: s.is_monotonic_increasing)
    R.append(_row("visit order not monotone in date", "patient", int((~order).sum()), len(order), "sorted by date"))
    c = partition.groupby("SUBJECT_ID").split.nunique()
    R.append(_row("patient in more than one partition", "patient", int((c > 1).sum()), len(c), "abort if > 0"))
    return pd.DataFrame(R)
