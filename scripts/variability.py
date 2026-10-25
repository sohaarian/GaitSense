"""Reliability of session-to-session change and behaviour of corrected review thresholds.

Exploratory, added after the protocol analysis (see docs/deviations.md).
Writes results/variability.json.
"""
import json, sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parents[1]; sys.path.insert(0, str(ROOT / "src"))
from gaitrec_rehab.monitoring import (FEATURES, between_session_sd, changes_from_first,
                                      reference_percentiles, review_flags)
from gaitrec_rehab.sessions import cohort_session_table

st, _ = cohort_session_table(ROOT / "data")
st = st.sort_values(["SUBJECT_ID", "SESSION_DATE"]).reset_index(drop=True)
res = {"patients": int(st.SUBJECT_ID.nunique()), "sessions": len(st)}


def pair_stats(t, tau=None):
    g = t.groupby("SUBJECT_ID"); out = {}
    for f in FEATURES:
        d = g[f].diff(); se2 = t[f + "_se"] ** 2 + g[f + "_se"].shift() ** 2
        if tau: se2 = se2 + tau[f] ** 2
        ok = d.notna(); z = d[ok] / np.sqrt(se2[ok])
        out[f] = {"pairs": int(ok.sum()), "mean_change": float(d[ok].mean()),
                  "mean_within_session_se": float(t[f + "_se"].mean()),
                  "sd_of_standardised_change": float(z.std()),
                  "share_beyond_up": float((z > 1.96).mean()), "share_beyond_down": float((z < -1.96).mean())}
    return out


# 1. Verification of the reported numbers (within-session limit only)
res["within_session_only"] = pair_stats(st)
res["extra_between_session_sd_full_cohort"] = between_session_sd(st)

# 2. Split-half calibration of the corrected limit
rng = np.random.default_rng(0)
ids = st.SUBJECT_ID.unique(); half = set(rng.permutation(ids)[: len(ids) // 2])
A, B = st[st.SUBJECT_ID.isin(half)], st[~st.SUBJECT_ID.isin(half)]
tauA, tauB = between_session_sd(A), between_session_sd(B)
res["split_half"] = {"tau_from_A": tauA, "tau_from_B": tauB,
                     "B_with_tau_from_A": pair_stats(B, tauA), "A_with_tau_from_B": pair_stats(A, tauB)}

# 3. Alert behaviour: original (within-session) vs corrected (tau cross-fitted from the other half)
def flags_for(t, tau):
    rows = []
    for sid, p in t.groupby("SUBJECT_ID"):
        if len(p) < 2: continue
        c = changes_from_first(p, tau); pc = reference_percentiles(st[st.SUBJECT_ID != sid], p)
        f = review_flags(c, pc)
        rows.append({"SUBJECT_ID": sid, "CLASS_LABEL": p.CLASS_LABEL.iat[0], "flags": f})
    return rows

orig = flags_for(st, None)
corr = flags_for(A, tauB) + flags_for(B, tauA)
names = ["persistent_loading_deficit", "loading_declined", "loading_above_typical_range"]
def rates(rows): return {k: float(np.mean([k in r["flags"] for r in rows])) for k in names} | \
                        {"any": float(np.mean([bool(r["flags"]) for r in rows]))}
res["alert_rates"] = {"original_within_session": rates(orig), "corrected_between_session": rates(corr),
                      "patients_with_2plus_sessions": len(orig)}
o = {r["SUBJECT_ID"]: set(r["flags"]) for r in orig}; c = {r["SUBJECT_ID"]: set(r["flags"]) for r in corr}
res["alert_overlap"] = {k: {"original": sum(k in v for v in o.values()), "corrected": sum(k in v for v in c.values()),
                            "both": sum(k in o[s] and k in c[s] for s in o)} for k in names}

# 4. Repetition: how often a beyond-limit decline between consecutive visits is followed by another
def repeat_rate(t, tau, z=1.96):
    g = t.groupby("SUBJECT_ID"); d = g["peak_loading"].diff()
    se2 = t["peak_loading_se"] ** 2 + g["peak_loading_se"].shift() ** 2 + (tau["peak_loading"] ** 2 if tau else 0)
    down = (d < -z * np.sqrt(se2)) & d.notna()
    nxt = down.groupby(t.SUBJECT_ID).shift(-1)
    has_next = d.groupby(t.SUBJECT_ID).shift(-1).notna()
    m = down & has_next
    return {"declines": int(down.sum()), "followed_by_another_decline": float(nxt[m].mean()) if m.any() else None}
res["repeat"] = {"original": repeat_rate(st, None),
                 "corrected": {"A": repeat_rate(A, tauB), "B": repeat_rate(B, tauA)}}

# 5. Threshold sensitivity for the 'loading declined' prompt (corrected limit, different z)
sens = {}
for z in (1.645, 1.96, 2.576):
    cnt = tot = 0
    for t, tau in ((A, tauB), (B, tauA)):
        for sid, p in t.groupby("SUBJECT_ID"):
            if len(p) < 2: continue
            a, b = p.iloc[0], p.iloc[-1]
            lim = z * np.sqrt(a.peak_loading_se ** 2 + b.peak_loading_se ** 2 + tau["peak_loading"] ** 2)
            cnt += (b.peak_loading - a.peak_loading) < -lim; tot += 1
    sens[str(z)] = float(cnt / tot)
res["declined_rate_by_z_corrected"] = sens

(ROOT / "results" / "variability.json").write_text(json.dumps(res, indent=2, default=float))
print(json.dumps({k: res[k] for k in ("extra_between_session_sd_full_cohort", "alert_rates", "repeat",
                                      "declined_rate_by_z_corrected")}, indent=2, default=float))
print("within-only loading:", json.dumps(res["within_session_only"]["peak_loading"], default=float))
print("B with tau(A) loading:", json.dumps(res["split_half"]["B_with_tau_from_A"]["peak_loading"], default=float))
print("A with tau(B) loading:", json.dumps(res["split_half"]["A_with_tau_from_B"]["peak_loading"], default=float))
