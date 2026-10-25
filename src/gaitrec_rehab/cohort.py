"""Cohort and outcome definition for the GaitRec subsequent-stay study.

Every rule here is the frozen protocol's definition. Changing one changes the study,
so each rule is covered by a test in tests/test_cohort.py.
"""
from __future__ import annotations

from dataclasses import dataclass
import pandas as pd

REQUIRED = [
    "SUBJECT_ID", "SESSION_ID", "CLASS_LABEL", "SEX", "AGE", "BODY_MASS",
    "AFFECTED_SIDE", "SHOD_CONDITION", "ORTHOPEDIC_INSOLE", "READMISSION", "SESSION_DATE",
]
HEALTHY = "HC"
NORMAL_SHOES = 1            # GaitRec SHOD_CONDITION: 0 barefoot/socks, 1 normal shoes, 2 orthopaedic shoes
UNILATERAL = {0, 1}         # AFFECTED_SIDE: 0 left, 1 right, 2 both


@dataclass(frozen=True)
class CohortRules:
    window_days: int = 365
    end_date: pd.Timestamp | None = None   # default: last patient session in the data


def load_metadata(path: str) -> pd.DataFrame:
    meta = pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in meta.columns]
    if missing:
        raise ValueError(f"metadata missing columns: {missing}")
    if meta["SESSION_ID"].duplicated().any():
        raise ValueError("SESSION_ID must be unique: one metadata row per session")
    meta["SESSION_DATE"] = pd.to_datetime(meta["SESSION_DATE"], format="%d.%m.%Y")
    return meta


def build_cohort(meta: pd.DataFrame, rules: CohortRules = CohortRules()):
    """Return (cohort, flow).

    cohort: one row per eligible patient, with the index (first first-stay) session,
            baseline predictors, `event` (later-stay gait assessment within the window)
            and `days_to_event` (NaN when no later stay is recorded).
    flow:   list of (step, patients remaining) for the patient-flow table.
    """
    pts = meta[meta["CLASS_LABEL"] != HEALTHY]
    end = rules.end_date or pts["SESSION_DATE"].max()
    flow = [("Patients in GaitRec", pts["SUBJECT_ID"].nunique())]

    first = pts[pts["READMISSION"] == 0].sort_values(["SUBJECT_ID", "SESSION_DATE", "SESSION_ID"])
    index = first.groupby("SUBJECT_ID", as_index=False).first()
    flow.append(("At least one first-stay session", len(index)))

    index = index[index["AFFECTED_SIDE"].isin(UNILATERAL)]
    flow.append(("One affected side at first assessment", len(index)))

    index = index[index["SHOD_CONDITION"] == NORMAL_SHOES]
    flow.append(("Normal shoes at first assessment", len(index)))

    index = index[(end - index["SESSION_DATE"]).dt.days >= rules.window_days]
    flow.append((f"Full {rules.window_days}-day window before {end.date()}", len(index)))

    later = (pts[pts["READMISSION"] > 0].groupby("SUBJECT_ID")["SESSION_DATE"].min()
             .rename("first_later_date"))
    cohort = index.merge(later, left_on="SUBJECT_ID", right_index=True, how="left")
    cohort["days_to_event"] = (cohort["first_later_date"] - cohort["SESSION_DATE"]).dt.days
    cohort["event"] = cohort["days_to_event"].le(rules.window_days).astype(int)
    cohort = cohort.rename(columns={"SESSION_ID": "index_session", "SESSION_DATE": "index_date"})
    flow.append(("Events within window", int(cohort["event"].sum())))
    return cohort.reset_index(drop=True), flow
