"""Eligibility check for the temporal hold-out (docs/detection_protocol.md). Uses no gait variables.

1. Hold-out = patients meeting every cohort rule except the 365-day follow-up window.
2. Confirms none of them appear in any earlier result file that lists patients or sessions.
3. Compares age, sex, injury group and visit counts with the development cohort.
Writes results/holdout_check.json.
"""
import json, sys
from pathlib import Path
import pandas as pd

R = Path(__file__).parents[1]; sys.path.insert(0, str(R / "src"))
from gaitrec_rehab.cohort import CohortRules, build_cohort, load_metadata

m = load_metadata(str(R / "data" / "GRF_metadata.csv"))
dev, _ = build_cohort(m); allw, _ = build_cohort(m, CohortRules(window_days=0))
hold = sorted(set(allw.SUBJECT_ID) - set(dev.SUBJECT_ID))
hold_sessions = set(m[m.SUBJECT_ID.isin(hold)].SESSION_ID)

found = {}
for f in sorted((R / "results").glob("*")):
    if f.suffix == ".csv":
        d = pd.read_csv(f)
        hits = (d.SUBJECT_ID.isin(hold).sum() if "SUBJECT_ID" in d else 0) + (d.SESSION_ID.isin(hold_sessions).sum() if "SESSION_ID" in d else 0)
        found[f.name] = int(hits)
sess_csv = R / "data" / "derived" / "first_stay_session_differences.csv"
if sess_csv.exists():
    found[sess_csv.name] = int(pd.read_csv(sess_csv).SUBJECT_ID.isin(hold).sum())


def describe(ids):
    c = allw.set_index("SUBJECT_ID").loc[list(ids)]
    v = m[m.SUBJECT_ID.isin(ids) & (m.READMISSION == 0) & (m.SHOD_CONDITION == 1)].groupby("SUBJECT_ID").size()
    return {"patients": len(c), "age_median": float(c.AGE.median()), "male_share": float((c.SEX == 1).mean()),
            "injury_group_share": {g: float((c.CLASS_LABEL == g).mean()) for g in "KAHC"},
            "patients_2plus_visits": int((v >= 2).sum()), "first_visit_range": [str(c.index_date.min().date()), str(c.index_date.max().date())]}


DISCLOSED = {"first_stay_session_differences.csv": "all-patient session table; used only in a pooled affected-side sanity test (see protocol)"}
out = {"holdout_patients": len(hold), "holdout_hits_in_earlier_files": found, "disclosed_exposures": DISCLOSED,
       "holdout_untouched": all(v == 0 for k, v in found.items() if k not in DISCLOSED),
       "development": describe(dev.SUBJECT_ID), "holdout": describe(hold)}
(R / "results" / "holdout_check.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
sys.exit(0 if out["holdout_untouched"] else 1)
