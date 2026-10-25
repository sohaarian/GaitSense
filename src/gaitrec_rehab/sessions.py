"""First-stay session table for the cohort (shared by monitoring scripts and the prototype)."""
from pathlib import Path

import numpy as np
import pandas as pd

from .cohort import build_cohort, load_metadata
from .monitoring import session_table


def cohort_session_table(root) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (session_table, cohort) for all first-stay, normal-shoe sessions of cohort patients."""
    root = Path(root)
    meta = load_metadata(str(root / "GRF_metadata.csv"))
    cohort, _ = build_cohort(meta)
    first = meta[(meta.READMISSION == 0) & meta.SUBJECT_ID.isin(cohort.SUBJECT_ID) & (meta.SHOD_CONDITION == 1)]
    side = cohort.set_index("SUBJECT_ID").AFFECTED_SIDE.map({0: "left", 1: "right"})
    t = pd.concat([pd.read_csv(root / "derived" / f"trial_features_{s}.csv") for s in ("left", "right")])
    t = t[t.valid & t.SESSION_ID.isin(first.SESSION_ID)]
    t["leg"] = np.where(t.side == t.SUBJECT_ID.map(side), "affected", "unaffected")
    t = t.merge(first[["SESSION_ID", "SESSION_DATE", "CLASS_LABEL"]], on="SESSION_ID")
    return session_table(t), cohort
