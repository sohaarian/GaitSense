"""Patient-level partitioning (60/20/20, stratified by injury class) and leakage guards."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import SEED

FRACTIONS = {"train": 0.6, "calib": 0.2, "test": 0.2}


def make_partition(patients: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """patients: one row per SUBJECT_ID with CLASS_LABEL (strata) and role. Temporal patients are not split."""
    rng = np.random.default_rng(seed)
    out = []
    pool = patients[patients.role.isin(["cohort", "hc"])].drop_duplicates("SUBJECT_ID")
    for (role, cls), g in pool.groupby(["role", "CLASS_LABEL"], sort=True):
        ids = rng.permutation(np.sort(g.SUBJECT_ID.to_numpy()))
        n = len(ids); a = int(round(FRACTIONS["train"] * n)); b = a + int(round(FRACTIONS["calib"] * n))
        out.append(pd.DataFrame({"SUBJECT_ID": ids, "role": role, "CLASS_LABEL": cls,
                                 "split": ["train"] * a + ["calib"] * (b - a) + ["test"] * (n - b)}))
    tmp = patients[patients.role == "temporal"].drop_duplicates("SUBJECT_ID")
    out.append(pd.DataFrame({"SUBJECT_ID": tmp.SUBJECT_ID, "role": "temporal", "CLASS_LABEL": tmp.CLASS_LABEL,
                             "split": "temporal"}))
    return pd.concat(out, ignore_index=True)


def assert_disjoint(partition: pd.DataFrame):
    """Every SUBJECT_ID appears in exactly one split."""
    c = partition.groupby("SUBJECT_ID").split.nunique()
    if (c > 1).any() or partition.SUBJECT_ID.duplicated().any():
        raise AssertionError(f"patient leakage across partitions: {list(c[c > 1].index[:5])}")


def assert_only(df: pd.DataFrame, partition: pd.DataFrame, allowed: set, what: str):
    """Raise if any row of df belongs to a patient outside the allowed splits."""
    s = df.SUBJECT_ID.map(partition.set_index("SUBJECT_ID").split)
    bad = ~s.isin(allowed)
    if bad.any():
        raise AssertionError(f"{what}: {int(bad.sum())} rows from splits {sorted(set(s[bad].astype(str)))}, "
                             f"allowed {sorted(allowed)}")
