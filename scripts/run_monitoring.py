"""Monitoring component across the cohort (descriptive; see the frozen protocol, docs/preregistration.md section 9)."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).parents[1]; sys.path.insert(0, str(ROOT / "src"))
from gaitrec_rehab.cohort import build_cohort, load_metadata
from gaitrec_rehab.monitoring import changes_from_first, reference_percentiles, review_flags, session_table
from gaitrec_rehab.report import render

meta = load_metadata(str(ROOT / "data" / "GRF_metadata.csv")); cohort, _ = build_cohort(meta)
first = meta[(meta.READMISSION == 0) & meta.SUBJECT_ID.isin(cohort.SUBJECT_ID) & (meta.SHOD_CONDITION == 1)]
side = cohort.set_index("SUBJECT_ID").AFFECTED_SIDE.map({0: "left", 1: "right"})
t = pd.concat([pd.read_csv(ROOT / "data" / "derived" / f"trial_features_{s}.csv") for s in ("left", "right")])
t = t[t.valid & t.SESSION_ID.isin(first.SESSION_ID)]
t["leg"] = np.where(t.side == t.SUBJECT_ID.map(side), "affected", "unaffected")
t = t.merge(first[["SESSION_ID", "SESSION_DATE", "CLASS_LABEL"]], on="SESSION_ID")
st = session_table(t)
counts = st.groupby("SUBJECT_ID").size()
res = {"patients": int(st.SUBJECT_ID.nunique()), "sessions": len(st),
       "sessions_per_patient_median": float(counts.median()), "patients_3plus_sessions": int((counts >= 3).sum())}
rows, examples = [], {}
for sid, p in st.groupby("SUBJECT_ID"):
    if len(p) < 2: continue
    ref = st[st.SUBJECT_ID != sid]
    c = changes_from_first(p); f = review_flags(c, reference_percentiles(ref, p))
    rows.append({"SUBJECT_ID": sid, "CLASS_LABEL": p.CLASS_LABEL.iat[0], "sessions": len(p),
                 "loading_change_beyond_noise": bool(c.iloc[-1].peak_loading_beyond_noise),
                 "loading_change": float(c.iloc[-1].peak_loading_change), **{k: k in f for k in
                 ("persistent_loading_deficit", "loading_declined", "loading_above_typical_range")}})
    for k in f:
        if k not in examples:
            examples[k] = sid
            (ROOT / "results").mkdir(exist_ok=True)
            (ROOT / "results" / f"review_{k}.html").write_text(render(p, c, f, ref), encoding="utf-8")
df = pd.DataFrame(rows)
res["patients_2plus_sessions"] = len(df)
res["by_group"] = df.groupby("CLASS_LABEL")[["persistent_loading_deficit", "loading_declined",
                  "loading_above_typical_range", "loading_change_beyond_noise"]].mean().round(3).to_dict()
res["overall"] = df[["persistent_loading_deficit", "loading_declined", "loading_above_typical_range",
                     "loading_change_beyond_noise"]].mean().round(3).to_dict()
res["improved_beyond_noise"] = float(((df.loading_change > 0) & df.loading_change_beyond_noise).mean())
res["examples"] = {k: int(v) for k, v in examples.items()}
(ROOT / "results" / "monitoring.json").write_text(json.dumps(res, indent=2)); print(json.dumps(res, indent=2))
