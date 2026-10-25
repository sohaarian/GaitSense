"""Clean-room re-implementation of the monitoring reliability results.

Imports nothing from gaitrec_rehab. Rebuilds the cohort, visit-level differences, standard errors,
visit-to-visit variation (tau), held-out calibration and the 'lower than first visit' rate directly
from data/GRF_metadata.csv and data/derived/trial_features_*.csv, using its own code and a
DIFFERENT patient split (odd vs even SUBJECT_ID) from the main analysis (random halves, seed 0).
Matching results support the main implementation; small differences from the different split are expected.

Usage: python verification/independent_reliability.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parents[1]
FEAT = ["peak_loading", "braking_impulse", "propulsive_impulse"]

# --- cohort, written from the protocol text, not from the package ---
m = pd.read_csv(ROOT / "data" / "GRF_metadata.csv")
m["date"] = pd.to_datetime(m["SESSION_DATE"], format="%d.%m.%Y")
pt = m[m["CLASS_LABEL"] != "HC"]
end = pt["date"].max()
first_stay = pt[pt["READMISSION"] == 0]
idx = first_stay.sort_values(["date", "SESSION_ID"]).drop_duplicates("SUBJECT_ID", keep="first")
idx = idx[idx["AFFECTED_SIDE"].isin([0, 1]) & (idx["SHOD_CONDITION"] == 1)]
idx = idx[(end - idx["date"]).dt.days >= 365]
cohort_ids = set(idx["SUBJECT_ID"])
affected = dict(zip(idx["SUBJECT_ID"], idx["AFFECTED_SIDE"].map({0: "left", 1: "right"})))

# --- visit-level affected-minus-unaffected differences and their standard errors ---
visits = first_stay[first_stay["SUBJECT_ID"].isin(cohort_ids) & (first_stay["SHOD_CONDITION"] == 1)]
tr = pd.concat([pd.read_csv(ROOT / "data" / "derived" / f"trial_features_{s}.csv") for s in ("left", "right")])
tr = tr[tr["valid"] & tr["SESSION_ID"].isin(visits["SESSION_ID"])].copy()
tr["is_aff"] = [affected[s] == side for s, side in zip(tr["SUBJECT_ID"], tr["side"])]
rows = []
for sess, g in tr.groupby("SESSION_ID"):
    a, u = g[g["is_aff"]], g[~g["is_aff"]]
    if len(a) == 0 or len(u) == 0:
        continue
    r = {"SESSION_ID": sess, "SUBJECT_ID": g["SUBJECT_ID"].iat[0]}
    for f in FEAT:
        r[f] = a[f].mean() - u[f].mean()
        r[f + "_se"] = np.sqrt(a[f].var(ddof=1) / len(a) + u[f].var(ddof=1) / len(u))
    rows.append(r)
v = pd.DataFrame(rows).merge(visits[["SESSION_ID", "date", "CLASS_LABEL"]], on="SESSION_ID")
v = v.sort_values(["SUBJECT_ID", "date", "SESSION_ID"]).reset_index(drop=True)

# --- consecutive pairs ---
prev = v.groupby("SUBJECT_ID").shift(1)
pairs = v[prev["date"].notna()].copy()
for f in FEAT:
    pairs[f + "_d"] = v[f] - prev[f]
    pairs[f + "_se2"] = v[f + "_se"] ** 2 + prev[f + "_se"] ** 2
pairs = pairs.dropna(subset=[FEAT[0] + "_d"])


def tau(df, f):
    return float(np.sqrt(max(df[f + "_d"].var(ddof=1) - df[f + "_se2"].mean(), 0)))


def beyond(df, f, t):
    z = df[f + "_d"] / np.sqrt(df[f + "_se2"] + t ** 2)
    return float((z.abs() > 1.96).mean())


odd = pairs["SUBJECT_ID"] % 2 == 1
out = {"patients": len(cohort_ids), "visits": len(v), "pairs": len(pairs),
       "mean_visit_se_loading": float(v["peak_loading_se"].mean()),
       "tau_full": {f: tau(pairs, f) for f in FEAT},
       "beyond_same_day_only_loading": beyond(pairs, "peak_loading", 0.0),
       "sd_standardised_same_day_loading": float((pairs["peak_loading_d"] / np.sqrt(pairs["peak_loading_se2"])).std()),
       "heldout_beyond_loading": {"odd_with_tau_from_even": beyond(pairs[odd], "peak_loading", tau(pairs[~odd], "peak_loading")),
                                  "even_with_tau_from_odd": beyond(pairs[~odd], "peak_loading", tau(pairs[odd], "peak_loading"))},
       "tau_by_group_loading": {g: tau(pairs[pairs["CLASS_LABEL"] == g], "peak_loading") for g in ("K", "A", "H", "C")}}

h = out["heldout_beyond_loading"]
h["pooled"] = (h["odd_with_tau_from_even"] * odd.sum() + h["even_with_tau_from_odd"] * (~odd).sum()) / len(pairs)
h["note"] = "Per-half rates depend on the split; the pooled held-out rate is the comparison target (nominal 5%)."

# 'lower than first visit' rate: latest vs first visit, corrected limit with tau from the other parity
dec = n = 0
t_other = {1: tau(pairs[~odd], "peak_loading"), 0: tau(pairs[odd], "peak_loading")}
for sid, g in v.groupby("SUBJECT_ID"):
    if len(g) < 2:
        continue
    a, b = g.iloc[0], g.iloc[-1]
    lim = 1.96 * np.sqrt(a["peak_loading_se"] ** 2 + b["peak_loading_se"] ** 2 + t_other[sid % 2] ** 2)
    dec += (b["peak_loading"] - a["peak_loading"]) < -lim
    n += 1
out["lower_than_first_rate_corrected"] = dec / n
out["patients_2plus_visits"] = n

# --- compare with the main implementation's stored results ---
var = json.loads((ROOT / "results" / "variability.json").read_text())
grp = json.loads((ROOT / "results" / "group_reliability.json").read_text())
cmp = [
    ("patients", out["patients"], 1667, 0),
    ("visits", out["visits"], var["sessions"], 0),
    ("visit pairs", out["pairs"], var["within_session_only"]["peak_loading"]["pairs"], 0),
    ("mean visit SE (loading)", out["mean_visit_se_loading"], var["within_session_only"]["peak_loading"]["mean_within_session_se"], 1e-9),
    ("tau loading", out["tau_full"]["peak_loading"], var["extra_between_session_sd_full_cohort"]["peak_loading"], 1e-9),
    ("tau braking", out["tau_full"]["braking_impulse"], var["extra_between_session_sd_full_cohort"]["braking_impulse"], 1e-9),
    ("tau push-off", out["tau_full"]["propulsive_impulse"], var["extra_between_session_sd_full_cohort"]["propulsive_impulse"], 1e-9),
    ("beyond same-day limit", out["beyond_same_day_only_loading"],
     var["within_session_only"]["peak_loading"]["share_beyond_up"] + var["within_session_only"]["peak_loading"]["share_beyond_down"], 1e-9),
    ("SD standardised change", out["sd_standardised_same_day_loading"], var["within_session_only"]["peak_loading"]["sd_of_standardised_change"], 1e-9),
    ("held-out beyond (pooled)", out["heldout_beyond_loading"]["pooled"], 0.05, 0.01),
    ("lower-than-first rate", out["lower_than_first_rate_corrected"], var["alert_rates"]["corrected_between_session"]["loading_declined"], 0.01),
] + [(f"tau group {g}", out["tau_by_group_loading"][g], grp["tau_by_group"][n]["peak_loading"]["tau"], 1e-9)
     for g, n in (("K", "Knee"), ("A", "Ankle"), ("H", "Hip"), ("C", "Heel"))]
bad = 0
print(f"{'quantity':28s} {'independent':>12s} {'main/target':>12s}  tolerance")
for name, a, b, tol in cmp:
    ok = abs(a - b) <= tol; bad += not ok
    print(f"{name:28s} {a:12.5f} {b:12.5f}  {'exact' if tol < 1e-6 else '±' + str(tol)}  {'OK' if ok else 'MISMATCH'}")
out["comparison_ok"] = bad == 0
(ROOT / "results" / "independent_reliability.json").write_text(json.dumps(out, indent=2))
raise SystemExit(1 if bad else 0)
