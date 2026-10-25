"""Cohort construction: visit-level table (one row per session) and consecutive visit pairs.

Uses the v1.x frozen cohort rules (src/gaitrec_rehab/cohort.py) so that the patient population is unchanged.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))
from gaitrec_rehab.cohort import CohortRules, build_cohort  # noqa: E402

from . import SEED
from .features import ALL

MIN_TRIALS = 3
CLASS_NAMES = {"A": "ankle", "K": "knee", "H": "hip", "C": "heel"}


def patient_roles(meta: pd.DataFrame) -> pd.DataFrame:
    """One row per patient/control: role in {'cohort', 'temporal', 'hc'}, affected side, injury class."""
    cohort, _ = build_cohort(meta)
    allrules, _ = build_cohort(meta, CohortRules(window_days=0))
    temporal = allrules[~allrules.SUBJECT_ID.isin(cohort.SUBJECT_ID)]
    rows = [pd.DataFrame({"SUBJECT_ID": d.SUBJECT_ID, "role": r, "affected_side": d.AFFECTED_SIDE.map({0: "left", 1: "right"}),
                          "CLASS_LABEL": d.CLASS_LABEL}) for d, r in ((cohort, "cohort"), (temporal, "temporal"))]
    hc = meta[meta.CLASS_LABEL == "HC"].drop_duplicates("SUBJECT_ID")
    rng = np.random.default_rng(SEED)
    # healthy controls have no affected leg; a seeded random leg plays that role for reference statistics
    rows.append(pd.DataFrame({"SUBJECT_ID": hc.SUBJECT_ID, "role": "hc",
                              "affected_side": rng.choice(["left", "right"], len(hc)), "CLASS_LABEL": "HC"}))
    return pd.concat(rows, ignore_index=True)


def visit_table(trials: pd.DataFrame, meta: pd.DataFrame, roles: pd.DataFrame) -> pd.DataFrame:
    """Visit-level summaries. Sessions: READMISSION == 0 (patients), normal shoes, both legs >= MIN_TRIALS valid."""
    m = meta.merge(roles, on=["SUBJECT_ID", "CLASS_LABEL"])
    m = m[(m.SHOD_CONDITION == 1) & ((m.role == "hc") | (m.READMISSION == 0))]
    t = trials[trials.valid & trials.SESSION_ID.isin(m.SESSION_ID)].merge(
        m[["SESSION_ID", "affected_side"]], on="SESSION_ID")
    t["leg"] = np.where(t.side == t.affected_side, "a", "u")
    g = t.groupby(["SESSION_ID", "leg"])[ALL]
    mean, sd, n = g.mean(), g.std(ddof=1), g.count()
    wide = []
    for leg in ("a", "u"):
        part = pd.concat([mean.xs(leg, level="leg").add_suffix(f"_{leg}"),
                          sd.xs(leg, level="leg").add_suffix(f"_{leg}_sd"),
                          n.xs(leg, level="leg")[["peak_v"]].rename(columns={"peak_v": f"n_{leg}"})], axis=1)
        wide.append(part)
    W = pd.concat(wide, axis=1, join="inner").reset_index()
    W = W[(W.n_a >= MIN_TRIALS) & (W.n_u >= MIN_TRIALS)]
    for f in ALL:
        W[f"d_{f}"] = W[f"{f}_a"] - W[f"{f}_u"]
        # SE of a difference of two independent trial means (NaN if a curve feature is missing in all trials)
        na = t[t.leg == "a"].groupby("SESSION_ID")[f].count(); nu = t[t.leg == "u"].groupby("SESSION_ID")[f].count()
        W[f"d_{f}_se"] = np.sqrt(W[f"{f}_a_sd"] ** 2 / W.SESSION_ID.map(na) + W[f"{f}_u_sd"] ** 2 / W.SESSION_ID.map(nu))
        W[f"{f}_a_se"] = W[f"{f}_a_sd"] / np.sqrt(W.SESSION_ID.map(na))
    keep = ["SESSION_ID", "SUBJECT_ID", "SESSION_DATE", "CLASS_LABEL", "role", "affected_side", "AGE", "SEX",
            "BODY_MASS", "BODY_WEIGHT", "SPEED", "ORTHOPEDIC_INSOLE"]
    V = m[keep].merge(W, on="SESSION_ID").sort_values(["SUBJECT_ID", "SESSION_DATE", "SESSION_ID"])
    first = V.groupby("SUBJECT_ID").SESSION_DATE.transform("min")
    V["day"] = (V.SESSION_DATE - first).dt.days.astype(int)
    V["visit_no"] = V.groupby("SUBJECT_ID").cumcount() + 1
    V["n_visits"] = V.groupby("SUBJECT_ID").SESSION_ID.transform("size")
    return V.reset_index(drop=True)


def consecutive_pairs(V: pd.DataFrame, cols) -> pd.DataFrame:
    """Consecutive visit pairs (t-1, t) within each patient. Columns `cols` are carried as *_prev / *_cur."""
    V = V.sort_values(["SUBJECT_ID", "visit_no"])
    prev = V.groupby("SUBJECT_ID")[cols + ["SESSION_ID", "day", "visit_no"]].shift()
    P = pd.concat([V[["SUBJECT_ID", "CLASS_LABEL", "role"]], V[cols + ["SESSION_ID", "day", "visit_no"]].add_suffix("_cur"),
                   prev.add_suffix("_prev")], axis=1)
    P = P[P.SESSION_ID_prev.notna()].copy()
    P["gap_days"] = (P.day_cur - P.day_prev).astype(int)
    return P.reset_index(drop=True)
