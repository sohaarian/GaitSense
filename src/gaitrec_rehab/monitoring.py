"""Longitudinal monitoring of injured-leg mechanics during a rehabilitation stay.

Input: one row per valid trial per leg, with columns
    SUBJECT_ID, SESSION_ID, SESSION_DATE, CLASS_LABEL, leg ('affected' | 'unaffected'),
    peak_loading, braking_impulse, propulsive_impulse
(as produced by gaitrec_rehab.features, in body weights / body-weight-seconds).

Everything here describes measurements. Nothing here diagnoses, predicts recovery,
or judges work capacity. Flags are prompts for clinical review.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = ["peak_loading", "braking_impulse", "propulsive_impulse"]
Z95 = 1.96


def session_table(trials: pd.DataFrame) -> pd.DataFrame:
    """Affected-minus-unaffected difference per session, with its standard error.

    The standard error comes from trial-to-trial variation within the session, so it
    reflects measurement noise on that day only, not day-to-day variation.
    """
    g = trials.groupby(["SUBJECT_ID", "SESSION_ID", "SESSION_DATE", "CLASS_LABEL", "leg"])[FEATURES]
    mean, sd, n = g.mean(), g.std(ddof=1), g.count()
    out = []
    for key in mean.index.droplevel("leg").unique():
        a, u = key + ("affected",), key + ("unaffected",)
        if a not in mean.index or u not in mean.index:
            continue
        row = dict(zip(["SUBJECT_ID", "SESSION_ID", "SESSION_DATE", "CLASS_LABEL"], key))
        for f in FEATURES:
            row[f] = mean.loc[a, f] - mean.loc[u, f]
            row[f + "_se"] = float(np.sqrt(sd.loc[a, f] ** 2 / n.loc[a, f] + sd.loc[u, f] ** 2 / n.loc[u, f]))
        out.append(row)
    table = pd.DataFrame(out).sort_values(["SUBJECT_ID", "SESSION_DATE"])
    first = table.groupby("SUBJECT_ID")["SESSION_DATE"].transform("min")
    table["day"] = (table["SESSION_DATE"] - first).dt.days
    table["week"] = table["day"] // 7 + 1
    return table.reset_index(drop=True)


def between_session_sd(table: pd.DataFrame) -> dict[str, float]:
    """Extra variation in a session-to-session change beyond within-session noise.

    For consecutive sessions of the same patient, var(change) = within-session part
    (SE1^2 + SE2^2) + extra part. Returns sqrt(max(0, var(change) - mean(SE1^2 + SE2^2)))
    per measure. Estimate it on patients other than those it is applied to.
    """
    t = table.sort_values(["SUBJECT_ID", "SESSION_DATE"])
    g = t.groupby("SUBJECT_ID")
    out = {}
    for f in FEATURES:
        d = g[f].diff()
        se2 = t[f + "_se"] ** 2 + g[f + "_se"].shift() ** 2
        ok = d.notna()
        out[f] = float(np.sqrt(max(d[ok].var() - se2[ok].mean(), 0.0)))
    return out


def changes_from_first(table: pd.DataFrame, tau: dict[str, float] | None = None) -> pd.DataFrame:
    """Change in each difference since the first session, and whether it exceeds a 95% limit.

    Without `tau` the limit uses within-session noise only (the frozen protocol's rule).
    With `tau` (from between_session_sd, estimated on other patients) the limit is
    1.96 * sqrt(SE1^2 + SE2^2 + tau^2), which also covers session-to-session variation."""
    rows = []
    for sid, p in table.groupby("SUBJECT_ID"):
        base = p.iloc[0]
        for _, s in p.iterrows():
            r = {"SUBJECT_ID": sid, "SESSION_ID": s["SESSION_ID"], "week": s["week"]}
            for f in FEATURES:
                r[f + "_change"] = s[f] - base[f]
                extra = (tau or {}).get(f, 0.0) if s["SESSION_ID"] != base["SESSION_ID"] else 0.0
                r[f + "_noise_limit"] = Z95 * float(np.sqrt(s[f + "_se"] ** 2 + base[f + "_se"] ** 2 + extra ** 2))
                r[f + "_beyond_noise"] = abs(r[f + "_change"]) > r[f + "_noise_limit"]
            rows.append(r)
    return pd.DataFrame(rows)


def reference_percentiles(reference: pd.DataFrame, patient: pd.DataFrame) -> pd.DataFrame:
    """Percentile of each patient session among reference sessions from the same injury
    group and week. `reference` must exclude the patient (e.g. training patients only)."""
    if set(patient["SUBJECT_ID"]) & set(reference["SUBJECT_ID"]):
        raise ValueError("reference distribution must not include the patient being described")
    out = patient[["SUBJECT_ID", "SESSION_ID", "week"]].copy()
    for f in FEATURES:
        pct = []
        for _, s in patient.iterrows():
            pool = reference.loc[(reference["CLASS_LABEL"] == s["CLASS_LABEL"])
                                 & (reference["week"] == s["week"]), f]
            pct.append(np.nan if len(pool) < 20 else float((pool < s[f]).mean() * 100))
        out[f + "_pct"] = pct
    return out


def review_flags(changes: pd.DataFrame, pcts: pd.DataFrame, low_pct: float = 10.0) -> list[str]:
    """Pre-specified review prompts for one patient (research definitions, not diagnoses).

    persistent_loading_deficit: at the third session or later, injured-leg loading relative to
        the other leg is below the injury group's 10th percentile for that week, and has not
        changed beyond measurement noise since the first session.
    loading_declined: the latest session shows injured-leg loading lower than at the first
        session by more than measurement noise.
    loading_above_typical_range: at the third session or later, injured-leg loading relative to
        the other leg is above the injury group's 90th percentile for that week (added after
        viewing one example page, before any outcome analysis; see docs/preregistration.md).
    """
    flags = []
    last_c, last_p = changes.iloc[-1], pcts.iloc[-1]
    if (len(changes) >= 3 and last_p["peak_loading_pct"] < low_pct
            and not last_c["peak_loading_beyond_noise"]):
        flags.append("persistent_loading_deficit")
    if last_c["peak_loading_beyond_noise"] and last_c["peak_loading_change"] < 0:
        flags.append("loading_declined")
    if len(changes) >= 3 and last_p["peak_loading_pct"] > 100 - low_pct:
        flags.append("loading_above_typical_range")
    return flags
