"""Stage 3b: visit-level table, consecutive pairs, patient partition, QC report, feature dictionary.

Parquet is the protocol's format, but pyarrow is not installable in the analysis environment (no package index
access); tables are written as gzip CSV with identical content (deviation AI-1 in docs/ai_deviations.md).
"""
import json, sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parents[2]; sys.path.insert(0, str(ROOT / "src"))
from gaitsense_ai import cohort, ingest, qc, splits
from gaitsense_ai.features import ALL, PRIMARY

R = ROOT / "results"; A = R / "ai"; A.mkdir(parents=True, exist_ok=True)
meta = ingest.load_metadata(ROOT / "data" / "GRF_metadata.csv")
trials = pd.read_csv(ROOT / "data" / "derived" / "ai_trials.csv.gz")
roles = cohort.patient_roles(meta)
V = cohort.visit_table(trials, meta, roles)
part = splits.make_partition(roles)
splits.assert_disjoint(part)
V = V.merge(part[["SUBJECT_ID", "split"]], on="SUBJECT_ID", how="left")
assert V.split.notna().all()

# session-mean normalised curves (affected and unaffected leg) for the population-reference architecture
cur = {}
for side in ("left", "right"):
    z = np.load(ROOT / "data" / "derived" / f"ai_curves_{side}.npz")
    k = pd.DataFrame(z["keys"], columns=ingest.KEYS); k["side"] = side
    cur[side] = (k, z["v"], z["ap"])
sess = V[["SESSION_ID", "affected_side"]]
out_c = {}
for leg in ("a", "u"):
    vs, aps = [], []
    for sid, aside in sess.itertuples(index=False):
        s = aside if leg == "a" else ("right" if aside == "left" else "left")
        k, cv, ca = cur[s]; m = (k.SESSION_ID == sid).to_numpy()
        vs.append(np.nanmean(cv[m], 0)); aps.append(np.nanmean(ca[m], 0))
    out_c[f"v_{leg}"], out_c[f"ap_{leg}"] = np.array(vs, np.float32), np.array(aps, np.float32)
np.savez_compressed(ROOT / "data" / "derived" / "ai_visit_curves.npz", session_id=V.SESSION_ID.to_numpy(), **out_c)

V.to_csv(A / "visit_level_features.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
tl = trials.merge(V[["SESSION_ID", "split", "role"]], on="SESSION_ID", how="left")
tl.to_csv(A / "trial_level_features.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
pcols = [c for c in V.columns if c.startswith("d_") or c.endswith(("_a", "_u", "_a_se"))]
P = cohort.consecutive_pairs(V[V.role != "hc"], pcols)
P = P.merge(part[["SUBJECT_ID", "split"]], on="SUBJECT_ID")
P.to_csv(A / "eligible_visit_pairs.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
part.to_csv(A / "partition.csv", index=False)

curves = {"v": np.vstack([cur["left"][1], cur["right"][1]])}
Q = qc.run_checks(trials, meta, curves, V, part)
Q.to_csv(R / "data_quality_report.csv", index=False)

units = {"peak_v": "BW", "braking": "BW*s", "propulsive": "BW*s", "stance_s": "s", "peak1": "BW", "peak2": "BW",
         "valley": "BW", "loading_rate": "BW/s", "vert_impulse": "BW*s", "wa_peak": "BW"}
desc = {"peak_v": "maximum vertical GRF over stance / body weight",
        "braking": "-integral of negative AP GRF over stance / body weight",
        "propulsive": "integral of positive AP GRF over stance / body weight",
        "stance_s": "stance samples / 250 Hz (stance cut at 25 N by the dataset)",
        "peak1": "dominant local maximum of vertical GRF in 5-45% stance (>=90% of first-half max), else NaN",
        "peak2": "dominant local maximum in 55-95% stance with a valley between peaks, else NaN",
        "valley": "minimum between peak1 and peak2, else NaN",
        "loading_rate": "mean slope between 20% and 80% of peak1 on the rising edge",
        "vert_impulse": "integral of vertical GRF over stance / body weight",
        "wa_peak": "maximum vertical GRF in the first half of stance (v1.x 'peak_loading')"}
rows = []
for f in ALL:
    for lvl, col, d in (("trial", f, desc[f]), ("visit", f"{f}_a", "affected-leg visit mean of " + f),
                        ("visit", f"{f}_u", "unaffected-leg visit mean of " + f),
                        ("visit", f"d_{f}", "affected minus unaffected visit mean of " + f),
                        ("visit", f"d_{f}_se", "SE of the difference from within-visit trial variation")):
        rows.append({"feature": col, "level": lvl, "unit": units[f], "primary": f in PRIMARY, "definition": d,
                     "source": "GaitRec GRF_F_V_RAW / GRF_F_AP_RAW, BODY_WEIGHT from GRF_metadata.csv"})
pd.DataFrame(rows).to_csv(R / "feature_dictionary.csv", index=False)

pt = V[V.role == "cohort"]
summary = {
    "visits_by_role": V.role.value_counts().to_dict(),
    "patients_by_role": V.groupby("role").SUBJECT_ID.nunique().to_dict(),
    "cohort_patients_with_2plus_visits": int((pt.groupby("SUBJECT_ID").size() >= 2).sum()),
    "pairs_by_split": P.split.value_counts().to_dict(),
    "patients_by_split_cohort": part[part.role == "cohort"].split.value_counts().to_dict(),
    "hc_by_split": part[part.role == "hc"].split.value_counts().to_dict(),
    "targets_by_split": V[V.visit_no >= 2].groupby("split").size().to_dict(),
    "visits_per_patient_cohort": pt.groupby("SUBJECT_ID").size().describe().round(2).to_dict(),
    "gap_days_quantiles": P.gap_days.quantile([0, .1, .25, .5, .75, .9, 1]).to_dict(),
}
(A / "stage3_summary.json").write_text(json.dumps(summary, indent=2, default=str))
print(json.dumps(summary, indent=1, default=str)); print(Q.to_string())
