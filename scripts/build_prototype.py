"""Build the interactive review prototype (prototype/gait_review.html) from the real GaitRec data.

Review prompts are computed twice: with the protocol's within-session limit, and with the
corrected limit that adds session-to-session variation estimated on the other half of patients
(cross-fitted, so no patient's own data sets their limit).
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).parents[1]; sys.path.insert(0, str(ROOT / "src"))
from gaitrec_rehab.monitoring import FEATURES, between_session_sd, changes_from_first, reference_percentiles, review_flags
from gaitrec_rehab.sessions import cohort_session_table

st, cohort = cohort_session_table(ROOT / "data")
info = cohort.set_index("SUBJECT_ID"); side = info.AFFECTED_SIDE.map({0: "left", 1: "right"})
rng = np.random.default_rng(0); ids = st.SUBJECT_ID.unique(); halfA = set(rng.permutation(ids)[: len(ids) // 2])
tau = [between_session_sd(st[~st.SUBJECT_ID.isin(halfA)]), between_session_sd(st[st.SUBJECT_ID.isin(halfA)])]  # [for A, for B]
var = json.loads((ROOT / "results" / "variability.json").read_text())
group_rel = json.loads((ROOT / "results" / "group_reliability.json").read_text())

bands = {}
for (cls, wk), grp in st.groupby(["CLASS_LABEL", "week"]):
    if len(grp) >= 20:
        bands.setdefault(cls, {})[int(wk)] = {f: [round(float(np.percentile(grp[f], 10)), 4),
                                                  round(float(np.percentile(grp[f], 90)), 4)] for f in FEATURES}
patients = []
for sid, p in st.groupby("SUBJECT_ID"):
    h = 0 if sid in halfA else 1
    fo, fc = [], []
    if len(p) >= 2:
        pc = reference_percentiles(st[st.SUBJECT_ID != sid], p)
        fo = review_flags(changes_from_first(p), pc); fc = review_flags(changes_from_first(p, tau[h]), pc)
    patients.append({"id": int(sid), "g": p.CLASS_LABEL.iat[0], "age": int(info.loc[sid, "AGE"]),
                     "sex": "M" if info.loc[sid, "SEX"] == 1 else "F", "side": side[sid], "h": h,
                     "s": [[int(r.week), int(r.day)] + [round(float(r[f]), 4) for f in FEATURES]
                           + [round(float(r[f + "_se"]), 4) for f in FEATURES] for _, r in p.iterrows()],
                     "fo": fo, "fc": fc})
wl, sh = var["within_session_only"]["peak_loading"], var["split_half"]["B_with_tau_from_A"]["peak_loading"]
data = {"patients": patients, "bands": bands, "sessions": len(st),
        "tau": [{k: round(v, 5) for k, v in t.items()} for t in tau],
        "why": {"sd_within": round(wl["sd_of_standardised_change"], 2), "sd_corr": round(sh["sd_of_standardised_change"], 2),
                "beyond_within": round(100 * (wl["share_beyond_up"] + wl["share_beyond_down"]), 1),
                "beyond_corr": round(100 * (sh["share_beyond_up"] + sh["share_beyond_down"]), 1),
                "dec_o": round(100 * var["alert_rates"]["original_within_session"]["loading_declined"], 1),
                "dec_c": round(100 * var["alert_rates"]["corrected_between_session"]["loading_declined"], 1),
                "extra_sd": round(var["extra_between_session_sd_full_cohort"]["peak_loading"], 3),
                "se_session": round(wl["mean_within_session_se"], 3),
                "groups": {n: round(100 * group_rel["calibration_by_group"][n]["peak_loading"]["pooled_threshold_beyond"], 1)
                           for n in ("Knee", "Ankle", "Hip", "Heel")}}}
md = json.loads((ROOT / "results" / "missed_decline.json").read_text())
data["why"]["miss"] = {"o": round(100 * md["flagged_at"]["0.05"]["original"]),
                       "c": round(100 * md["flagged_at"]["0.05"]["corrected"]),
                       "d80": md["delta_for_80pct"]["corrected"]}
tpl = (ROOT / "prototype" / "template.html").read_text(encoding="utf-8")
out = ROOT / "prototype" / "gait_review.html"
out.write_text(tpl.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":"))), encoding="utf-8")
print("patients", len(patients), "corrected prompts", sum(bool(p["fc"]) for p in patients),
      "original prompts", sum(bool(p["fo"]) for p in patients), "KB", out.stat().st_size // 1024)
