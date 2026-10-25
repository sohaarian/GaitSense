"""How often does each threshold miss a real decline? (exploratory; deviation G in docs/deviations.md)

Method: for every cohort patient with 2+ first-stay visits, take the real first and latest visits and
subtract a known decline delta (body weights) from the latest visit's loading difference, i.e. the
patient is made to load the injured leg delta BW less than they actually did. The 'loading declined'
prompt fires if (latest - first) < -limit. We report the share of patients flagged at each delta for
  original : limit = 1.96 * sqrt(SE1^2 + SE2^2)
  corrected: limit = 1.96 * sqrt(SE1^2 + SE2^2 + tau^2), tau cross-fitted from the other half (seed 0)
At delta = 0 the share is the alert rate with no injected decline. Because real trajectories already
contain real change in both directions, a positive share at delta = 0 is not purely false alarm,
and detection at delta > 0 is detection of a decline on top of whatever really happened.
Writes results/missed_decline.json and docs/figures/fig4_detection.png.
"""
import json, sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parents[1]; sys.path.insert(0, str(ROOT / "src"))
from gaitrec_rehab.monitoring import between_session_sd
from gaitrec_rehab.sessions import cohort_session_table

st, _ = cohort_session_table(ROOT / "data")
st = st.sort_values(["SUBJECT_ID", "SESSION_DATE"]).reset_index(drop=True)
rng = np.random.default_rng(0)
ids = st.SUBJECT_ID.unique(); half = set(rng.permutation(ids)[: len(ids) // 2])
A, B = st[st.SUBJECT_ID.isin(half)], st[~st.SUBJECT_ID.isin(half)]
tauA, tauB = between_session_sd(A)["peak_loading"], between_session_sd(B)["peak_loading"]

rows = []
for t, tau in ((A, tauB), (B, tauA)):
    for sid, p in t.groupby("SUBJECT_ID"):
        if len(p) < 2:
            continue
        a, b = p.iloc[0], p.iloc[-1]
        se2 = a.peak_loading_se ** 2 + b.peak_loading_se ** 2
        rows.append({"SUBJECT_ID": sid, "group": a.CLASS_LABEL, "change": b.peak_loading - a.peak_loading,
                     "lim_orig": 1.96 * np.sqrt(se2), "lim_corr": 1.96 * np.sqrt(se2 + tau ** 2)})
d = pd.DataFrame(rows)

deltas = np.round(np.arange(0, 0.2001, 0.01), 2)
curve = {"delta_bw": deltas.tolist(),
         "original": [float(((d.change - x) < -d.lim_orig).mean()) for x in deltas],
         "corrected": [float(((d.change - x) < -d.lim_corr).mean()) for x in deltas]}


def delta_for(share, key):
    """Smallest delta on a fine grid at which the share flagged reaches `share`."""
    for x in np.arange(0, 0.5, 0.001):
        lim = d.lim_orig if key == "original" else d.lim_corr
        if ((d.change - x) < -lim).mean() >= share:
            return round(float(x), 3)
    return None


first = st.groupby("SUBJECT_ID").first()
res = {
    "patients": len(d),
    "tau_from_A": tauA, "tau_from_B": tauB,
    "median_limit_bw": {"original": float(d.lim_orig.median()), "corrected": float(d.lim_corr.median())},
    "curve": curve,
    "delta_for_50pct": {k: delta_for(0.5, k) for k in ("original", "corrected")},
    "delta_for_80pct": {k: delta_for(0.8, k) for k in ("original", "corrected")},
    "flagged_at": {str(x): {"original": curve["original"][i], "corrected": curve["corrected"][i]}
                   for i, x in enumerate(deltas) if x in (0.0, 0.05, 0.1, 0.15)},
    "by_group_at_0.10": {g: {"n": int(len(s)),
                             "original": float(((s.change - 0.1) < -s.lim_orig).mean()),
                             "corrected": float(((s.change - 0.1) < -s.lim_corr).mean())}
                         for g, s in d.groupby("group")},
    "scale_reference_bw": {
        "median_real_change_first_to_latest": float(d.change.median()),
        "sd_first_visit_loading_difference_between_patients": float(first.peak_loading.std()),
    },
}
(ROOT / "results" / "missed_decline.json").write_text(json.dumps(res, indent=2))
print(json.dumps({k: v for k, v in res.items() if k != "curve"}, indent=2))

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=200)
ax.plot(deltas, np.array(curve["original"]) * 100, color="#b5523b", lw=2, label="Original (same-day noise only)")
ax.plot(deltas, np.array(curve["corrected"]) * 100, color="#2f6f8f", lw=2, label="Corrected (adds visit-to-visit variation)")
ax.axhline(80, color="#999", lw=0.8, ls=":")
ax.set_xlabel("Decline added to the latest visit (body weights)")
ax.set_ylabel("Patients flagged 'loading declined' (%)")
ax.set_ylim(0, 100); ax.set_xlim(0, 0.2)
ax.set_title("Figure 4. Declines flagged under each threshold (1,441 patients)", fontsize=10, loc="left")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, fontsize=8, loc="lower right")
fig.tight_layout()
fig.savefig(ROOT / "docs" / "figures" / "fig4_detection.png")
