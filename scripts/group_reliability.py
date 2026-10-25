"""Injury-specific reliability of visit-to-visit change (exploratory, post hoc).

Questions:
1. Does the extra visit-to-visit variation (tau) differ between knee, ankle, hip and heel patients?
2. Is a single pooled corrected threshold calibrated within each injury group, or do groups need their own?
Uses the same split halves as scripts/variability.py (seed 0). Writes results/group_reliability.json.
"""
import json, sys, zlib
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).parents[1]; sys.path.insert(0, str(ROOT / "src"))
from gaitrec_rehab.monitoring import FEATURES, between_session_sd
from gaitrec_rehab.sessions import cohort_session_table

GROUPS = {"K": "Knee", "A": "Ankle", "H": "Hip", "C": "Heel"}
st, _ = cohort_session_table(ROOT / "data"); st = st.sort_values(["SUBJECT_ID", "SESSION_DATE"]).reset_index(drop=True)
rng = np.random.default_rng(0); ids = st.SUBJECT_ID.unique(); halfA = set(rng.permutation(ids)[: len(ids) // 2])
inA = st.SUBJECT_ID.isin(halfA)


def pairs(t):
    g = t.groupby("SUBJECT_ID")
    out = {"cls": t.CLASS_LABEL.to_numpy(), "sid": t.SUBJECT_ID.to_numpy()}
    for f in FEATURES:
        out[f + "_d"] = g[f].diff().to_numpy()
        out[f + "_se2"] = (t[f + "_se"] ** 2 + g[f + "_se"].shift() ** 2).to_numpy()
    df = pd.DataFrame(out); return df[df[FEATURES[0] + "_d"].notna()].reset_index(drop=True)


P = pairs(st); PA = pairs(st[inA]); PB = pairs(st[~inA])


def tau_from_pairs(df, f):
    return float(np.sqrt(max(df[f + "_d"].var() - df[f + "_se2"].mean(), 0.0)))


def boot_tau(df, f, n=500, seed=1):
    r = np.random.default_rng(seed); subj = df.sid.unique(); by = {s: g for s, g in df.groupby("sid")}
    vals = []
    for _ in range(n):
        pick = r.choice(subj, len(subj), replace=True)
        vals.append(tau_from_pairs(pd.concat([by[s] for s in pick]), f))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))], np.array(vals)


def exceed(df, f, tau):
    z = df[f + "_d"] / np.sqrt(df[f + "_se2"] + tau ** 2)
    return float((z.abs() > 1.96).mean()), float((z < -1.96).mean())


res = {"pairs_by_group": {GROUPS[g]: int((P.cls == g).sum()) for g in GROUPS},
       "patients_with_pairs_by_group": {GROUPS[g]: int(P[P.cls == g].sid.nunique()) for g in GROUPS}}
pooled_tau = {f: tau_from_pairs(P, f) for f in FEATURES}
res["pooled_tau"] = pooled_tau

# 1. tau by group with bootstrap intervals (patients resampled)
res["tau_by_group"] = {}; boots = {}
for g, name in GROUPS.items():
    res["tau_by_group"][name] = {}
    for f in FEATURES:
        ci, b = boot_tau(P[P.cls == g], f, seed=zlib.crc32(f"{g}|{f}".encode()))  # v2.0: deterministic (was hash(), randomised per process)
        res["tau_by_group"][name][f] = {"tau": tau_from_pairs(P[P.cls == g], f), "ci95": ci}
        boots[(g, f)] = b

# permutation test: is the spread of group taus (max - min) larger than expected if groups were exchangeable?
def spread(df, f, labels):
    t = [tau_from_pairs(df[labels == g], f) for g in GROUPS]
    return max(t) - min(t)
res["heterogeneity_permutation_p"] = {}
r = np.random.default_rng(7)
subj_cls = P.groupby("sid").cls.first()
for f in FEATURES:
    obs = spread(P, f, P.cls.to_numpy()); cnt = 0; N = 1000
    for _ in range(N):
        perm = pd.Series(r.permutation(subj_cls.to_numpy()), index=subj_cls.index)
        cnt += spread(P, f, P.sid.map(perm).to_numpy()) >= obs
    res["heterogeneity_permutation_p"][f] = {"observed_spread": obs, "p": (cnt + 1) / (N + 1)}

# 2. calibration within each group: pooled tau (from the other half) vs group tau (from the other half, same group)
tA = {f: tau_from_pairs(PA, f) for f in FEATURES}; tB = {f: tau_from_pairs(PB, f) for f in FEATURES}
res["calibration_by_group"] = {}
for g, name in GROUPS.items():
    res["calibration_by_group"][name] = {}
    for f in FEATURES:
        a, b = PA[PA.cls == g], PB[PB.cls == g]
        ga, gb = tau_from_pairs(a, f), tau_from_pairs(b, f)
        n = len(a) + len(b)
        pooled = [exceed(b, f, tA[f]), exceed(a, f, tB[f])]
        group = [exceed(b, f, ga), exceed(a, f, gb)]
        w = lambda e: ((e[0][0] * len(b) + e[1][0] * len(a)) / n, (e[0][1] * len(b) + e[1][1] * len(a)) / n)
        pw, gw = w(pooled), w(group)
        res["calibration_by_group"][name][f] = {"pairs": n, "pooled_threshold_beyond": pw[0], "pooled_threshold_declines": pw[1],
                                                "group_threshold_beyond": gw[0], "group_threshold_declines": gw[1]}
(ROOT / "results" / "group_reliability.json").write_text(json.dumps(res, indent=2))

print("pairs", res["pairs_by_group"])
for f in FEATURES:
    print(f"\n{f}: pooled tau {pooled_tau[f]:.4f}; heterogeneity p = {res['heterogeneity_permutation_p'][f]['p']:.3f}")
    for name in GROUPS.values():
        t = res["tau_by_group"][name][f]; c = res["calibration_by_group"][name][f]
        print(f"  {name:6s} tau {t['tau']:.4f} ({t['ci95'][0]:.4f}-{t['ci95'][1]:.4f})  beyond: pooled {100*c['pooled_threshold_beyond']:.1f}%  group {100*c['group_threshold_beyond']:.1f}%")
