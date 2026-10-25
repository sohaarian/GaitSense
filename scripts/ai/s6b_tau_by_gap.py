"""Post-hoc exploratory analysis E2 (not in the frozen protocol; see docs/ai_deviations.md).

Question: how much of tau (extra between-visit variation) is present when two sessions are recorded on the SAME day?
Same-day pairs leave little room for genuine recovery, so their tau approximates session-level measurement and
day-state variation; the excess at longer gaps is a mixture that includes genuine change. This is a bound-style
decomposition on the existing data, not a substitute for a designed test-retest study.
Writes results/ai/tau_by_gap.json.
"""
import json, sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parents[2]
A = ROOT / "results" / "ai"
P = pd.read_csv(A / "eligible_visit_pairs.csv.gz")
P = P[P.role == "cohort"]
FEATS = ["d_peak_v", "d_wa_peak", "d_braking", "d_propulsive", "d_stance_s"]
bins = [(-1, 0, "same day"), (0, 7, "1-7 d"), (7, 14, "8-14 d"), (14, 10_000, ">14 d")]
rng = np.random.default_rng(20261024)


def tau(df, f):
    d = df[f + "_cur"] - df[f + "_prev"]; s = df[f + "_se_cur"] ** 2 + df[f + "_se_prev"] ** 2
    return float(np.sqrt(max(d.var(ddof=1) - s.mean(), 0.0))), float(np.sqrt(s.mean() / 2))


res = {"n_pairs": len(P), "features": {}}
for f in FEATS:
    res["features"][f] = {}
    for lo, hi, lab in bins:
        g = P[(P.gap_days > lo) & (P.gap_days <= hi)]
        t, se = tau(g, f)
        pids = g.SUBJECT_ID.unique(); idx = {p: np.flatnonzero(g.SUBJECT_ID.to_numpy() == p) for p in pids}
        boots = []
        for _ in range(1000):
            ix = np.concatenate([idx[p] for p in rng.choice(pids, len(pids))]); boots.append(tau(g.iloc[ix], f)[0])
        res["features"][f][lab] = {"pairs": len(g), "patients": int(len(pids)), "tau": t,
                                   "tau_ci95": [float(np.quantile(boots, .025)), float(np.quantile(boots, .975))],
                                   "rms_within_visit_se": se}
    print(f, {k: round(v["tau"], 4) for k, v in res["features"][f].items()})
(A / "tau_by_gap.json").write_text(json.dumps(res, indent=2))
