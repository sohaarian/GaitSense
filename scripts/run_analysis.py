"""Run the analyses in the frozen protocol (docs/preregistration.md). Writes results/<part>.json.

Usage:  python scripts/run_analysis.py main      # primary, S1-S4, E1-E3
        python scripts/run_analysis.py bias      # E4 missed-event analysis (slow)
"""
import json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gaitrec_rehab.analysis import (GAIT, MODELS, _fit_predict, c_index, cv_predict, load_data, metrics)
from gaitrec_rehab.cohort import load_metadata

OUT = ROOT / "results"; OUT.mkdir(exist_ok=True)
REPEATS, BOOT = 10, 200


def ci(x):
    x = np.asarray(x); return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def summarize(per_repeat):
    """per_repeat: list of {model: metrics}; returns mean over repeats."""
    models = per_repeat[0].keys()
    return {m: {k: float(np.mean([r[m][k] for r in per_repeat])) for k in per_repeat[0][m]} for m in models}


def repeated(d, y, rows, models, with_recon):
    preds, mets = [], []
    for s in range(REPEATS):
        o = cv_predict(d, y, rows, d.cohort["SUBJECT_ID"].to_numpy()[rows], models, s, with_recon)
        preds.append(o); mets.append({m: metrics(y, o[m]) for m in models})
    return preds, mets


def boot_full(d, y, rows, a="M2", b="M0", seed=2026):
    """Full-pipeline bootstrap: resample patients, rerun one 5-fold CV (penalty selection included)."""
    rng = np.random.default_rng(seed); diffs, dbrier = [], []
    for i in range(BOOT):
        idx = rows[rng.integers(0, len(rows), len(rows))]
        yy = y[np.searchsorted(rows, idx)] if False else d.cohort["event"].to_numpy()[idx]
        groups = d.cohort["SUBJECT_ID"].to_numpy()[idx]
        o = cv_predict(d, yy, idx, groups, [a, b], seed=i, with_recon=False)
        ma, mb = metrics(yy, o[a]), metrics(yy, o[b])
        diffs.append(ma["auroc"] - mb["auroc"]); dbrier.append(ma["brier"] - mb["brier"])
    return {"auroc_diff_ci": ci(diffs), "brier_diff_ci": ci(dbrier)}


def boot_oof(y, pa, pb, seed=2026):
    rng = np.random.default_rng(seed); diffs = []
    from sklearn.metrics import roc_auc_score
    for _ in range(BOOT):
        i = rng.integers(0, len(y), len(y))
        if y[i].sum() == 0: continue
        diffs.append(roc_auc_score(y[i], pa[i]) - roc_auc_score(y[i], pb[i]))
    return ci(diffs)


def main():
    t0 = time.time(); d = load_data(ROOT / "data"); c = d.cohort
    y = c["event"].to_numpy(); rows = np.arange(len(c)); res = {"n": len(c), "events": int(y.sum())}

    # Primary + S1 + E2 (repeated CV with reconstruction score)
    preds, mets = repeated(d, y, rows, ["M0", "M1", "M2", "M2+"], with_recon=True)
    res["models"] = summarize(mets)
    res["primary_auroc_diff_M2_M0"] = float(np.mean([m["M2"]["auroc"] - m["M0"]["auroc"] for m in mets]))
    res["primary_brier_diff_M2_M0"] = float(np.mean([m["M2"]["brier"] - m["M0"]["brier"] for m in mets]))
    res["primary_boot"] = boot_full(d, y, rows); print("primary done", time.time() - t0, flush=True)
    res["S1_auroc_diff_M2_M1"] = float(np.mean([m["M2"]["auroc"] - m["M1"]["auroc"] for m in mets]))
    res["S1_ci_oof"] = boot_oof(y, preds[0]["M2"], preds[0]["M1"])
    res["E2_auroc_diff_M2plus_M1"] = float(np.mean([m["M2+"]["auroc"] - m["M1"]["auroc"] for m in mets]))
    res["E2_ci_oof"] = boot_oof(y, preds[0]["M2+"], preds[0]["M1"])
    # coefficients of M2 fitted on all data (standardised), for interpretation only
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.model_selection import StratifiedKFold
    X = c[MODELS["M2"]].to_numpy(float); sc = StandardScaler().fit(X)
    m2 = LogisticRegressionCV(Cs=np.logspace(-3, 2, 10), cv=StratifiedKFold(5, shuffle=True, random_state=0),
                              scoring="neg_log_loss", max_iter=5000).fit(sc.transform(X), y)
    res["M2_standardised_coefficients"] = dict(zip(MODELS["M2"], map(float, m2.coef_[0])))
    res["M2_chosen_C"] = float(m2.C_[0])

    # S3 injury groups (pooled models, repeat 0 predictions)
    res["S3"] = {}
    for g in ("K", "A", "H", "C"):
        m = (c["CLASS_LABEL"] == g).to_numpy()
        from sklearn.metrics import roc_auc_score
        res["S3"][g] = {"n": int(m.sum()), "events": int(y[m].sum()),
                        **{k: float(roc_auc_score(y[m], preds[0][k][m])) for k in ("M0", "M2")},
                        "diff_ci_oof": boot_oof(y[m], preds[0]["M2"][m], preds[0]["M0"][m])}

    # S2 day-90 landmark
    keep = ~(c["days_to_event"] <= 90).to_numpy()
    rows2 = rows[keep]; y2 = ((c["days_to_event"] > 90) & (c["days_to_event"] <= 365)).to_numpy().astype(int)[keep]
    p2, m2s = [], []
    for s in range(REPEATS):
        o = cv_predict(d, y2, rows2, c["SUBJECT_ID"].to_numpy()[rows2], ["M0", "M2"], s, False)
        p2.append(o); m2s.append({k: metrics(y2, o[k]) for k in o})
    res["S2"] = {"n": int(keep.sum()), "events": int(y2.sum()), "models": summarize(m2s),
                 "auroc_diff": float(np.mean([m["M2"]["auroc"] - m["M0"]["auroc"] for m in m2s])),
                 "ci_oof": boot_oof(y2, p2[0]["M2"], p2[0]["M0"])}
    print("S2 done", time.time() - t0, flush=True)

    # S4 insoles
    ins = (c["ORTHOPEDIC_INSOLE"] == 0).to_numpy(); rows4 = rows[ins]; y4 = y[ins]
    _, m4 = repeated(d, y4, rows4, ["M0", "M2"], with_recon=False)
    res["S4"] = {"n": int(ins.sum()), "events": int(y4.sum()),
                 "auroc_diff": float(np.mean([m["M2"]["auroc"] - m["M0"]["auroc"] for m in m4])),
                 "boot": boot_full(d, y, rows4)}
    print("S4 done", time.time() - t0, flush=True)

    # E1 temporal hold-out
    early = (c["index_date"] < "2014-01-01").to_numpy()
    e1 = {"dev_n": int(early.sum()), "dev_events": int(y[early].sum()),
          "test_n": int((~early).sum()), "test_events": int(y[~early].sum())}
    pt = {}
    for k in ("M0", "M2"):
        cols = MODELS[k]
        pt[k] = _fit_predict(c.loc[early, cols].to_numpy(float), y[early], c.loc[~early, cols].to_numpy(float), 0)
        e1[k] = metrics(y[~early], pt[k])
    yt = y[~early]; e1["auroc_diff"] = e1["M2"]["auroc"] - e1["M0"]["auroc"]
    e1["ci"] = boot_oof(yt, pt["M2"], pt["M0"]); res["E1"] = e1

    # E3 discrete-time model over all eligible patients with censoring (no 365-day rule)
    meta = load_metadata(str(ROOT / "data" / "GRF_metadata.csv"))
    pts = meta[meta["CLASS_LABEL"] != "HC"]; end = pts["SESSION_DATE"].max()
    first = pts[pts["READMISSION"] == 0].sort_values(["SUBJECT_ID", "SESSION_DATE", "SESSION_ID"]).groupby("SUBJECT_ID").first().reset_index()
    first = first[first["AFFECTED_SIDE"].isin([0, 1]) & (first["SHOD_CONDITION"] == 1)]
    later = pts[pts["READMISSION"] > 0].groupby("SUBJECT_ID")["SESSION_DATE"].min()
    first["t_event"] = (first["SUBJECT_ID"].map(later) - first["SESSION_DATE"]).dt.days
    first["event"] = first["t_event"].notna().astype(int)
    first["time"] = first["t_event"].fillna((end - first["SESSION_DATE"]).dt.days)
    tf = pd.concat([pd.read_csv(ROOT / "data" / "derived" / f"trial_features_{s}.csv") for s in ("left", "right")])
    tf = tf[tf["valid"] & tf["SESSION_ID"].isin(first["SESSION_ID"])]
    sd = first.set_index("SESSION_ID")["AFFECTED_SIDE"].map({0: "left", 1: "right"})
    tf["leg"] = np.where(tf["side"] == tf["SESSION_ID"].map(sd), "affected", "unaffected")
    mm = tf.groupby(["SESSION_ID", "leg"])[GAIT].mean().unstack("leg")
    for f in GAIT:
        first[f] = first["SESSION_ID"].map(mm[(f, "affected")] - mm[(f, "unaffected")])
    for g in ("A", "C", "H"):
        first[f"inj_{g}"] = (first["CLASS_LABEL"] == g).astype(float)
    first["male"] = (first["SEX"] == 1).astype(float)
    first = first.dropna(subset=GAIT).reset_index(drop=True)
    from sklearn.model_selection import StratifiedKFold
    risk = {k: np.zeros(len(first)) for k in ("M0", "M2")}
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(first, first["event"]):
        trd = first.iloc[tr]
        pp = []  # person-period rows, 30-day intervals
        for _, r in trd.iterrows():
            n_int = int(np.ceil(max(r["time"], 1) / 30))
            for j in range(n_int):
                pp.append({**{col: r[col] for col in MODELS["M2"]}, "logt": np.log(30 * j + 15),
                           "y": int(r["event"] == 1 and j == n_int - 1)})
        pp = pd.DataFrame(pp)
        for k in ("M0", "M2"):
            cols = MODELS[k] + ["logt"]
            from sklearn.preprocessing import StandardScaler
            from sklearn.linear_model import LogisticRegression
            sc = StandardScaler().fit(pp[cols]); lr = LogisticRegression(C=1.0, max_iter=5000).fit(sc.transform(pp[cols]), pp["y"])
            Xte = first.iloc[te][MODELS[k]].assign(logt=np.log(15.0))[cols]
            risk[k][te] = lr.decision_function(sc.transform(Xte))
    res["E3"] = {"n": len(first), "events": int(first["event"].sum()),
                 **{k: c_index(first["time"].to_numpy(), first["event"].to_numpy(), risk[k]) for k in risk}}
    res["runtime_s"] = time.time() - t0
    (OUT / "main.json").write_text(json.dumps(res, indent=2))
    print("DONE main", res["runtime_s"])


def bias():
    d = load_data(ROOT / "data"); c = d.cohort
    y0 = c["event"].to_numpy(); rows = np.arange(len(c))
    Z = (c[GAIT] - c[GAIT].mean()) / c[GAIT].std()
    z = -Z.mean(axis=1).to_numpy()
    out = {}
    for s in (0.9, 0.8, 0.6):
        add = int(round(84 * (1 / s - 1)))
        for mech in ("random", "gait"):
            rng = np.random.default_rng(int(s * 100) + (mech == "gait"))
            diffs = []
            for i in range(BOOT):
                y = y0.copy(); free = np.where(y == 0)[0]
                w = np.ones(len(free)) if mech == "random" else np.exp(z[free])
                y[rng.choice(free, add, replace=False, p=w / w.sum())] = 1
                o = cv_predict(d, y, rows, c["SUBJECT_ID"].to_numpy(), ["M0", "M2"], i, False)
                from sklearn.metrics import roc_auc_score
                diffs.append(roc_auc_score(y, o["M2"]) - roc_auc_score(y, o["M0"]))
            out[f"s={s}, {mech}"] = {"added": add, "mean_diff": float(np.mean(diffs)), "range95": ci(diffs)}
            print(s, mech, out[f"s={s}, {mech}"], flush=True)
            (OUT / "bias.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    {"main": main, "bias": bias}[sys.argv[1]]()
