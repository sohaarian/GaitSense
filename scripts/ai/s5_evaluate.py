"""Stage 4-5 evaluation on test (and temporal) patients: metrics, H1/H2, breakdowns, uncertainty informativeness."""
import json, sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parents[2]; sys.path.insert(0, str(ROOT / "src"))
from gaitsense_ai import evaluation as E

A = ROOT / "results" / "ai"
sel = json.loads((A / "model_selection.json").read_text())
D = pd.read_csv(A / "predictions_test.csv.gz")
MODELS = ["M0a", "M0b", "M1", "M2", "M3", "M4"]
B = 2000
GAP_BINS = [-1, 7, 14, 28, 10_000]; GAP_LABELS = ["<=7d", "8-14d", "15-28d", ">28d"]
D["gap_bin"] = pd.cut(D.gap_days.round(), GAP_BINS, labels=GAP_LABELS).astype(str)
S = pd.concat([D, E.pointwise(D)], axis=1)
res = {"B": B, "targets": {}}
metric_cols = {"MAE": "abs_err", "CRPS": "crps", "IS95": "is95", "NLPD": "nlpd", "cov95": "cov95",
               "cov50": "cov50", "cov80": "cov80", "cov90": "cov90", "width95": "width95"}


def table(sub, W):
    out = {}
    for mdl in MODELS:
        x = sub[sub.model == mdl]
        summ = E.summarise(x)
        summ["ci95"] = {k: E.ci(E.cluster_boot_mean(x[c].to_numpy(), x.SUBJECT_ID.to_numpy(), weights=W))
                        for k, c in metric_cols.items()}
        out[mdl] = summ
    return out


def paired(sub, a, b, col, W, transform=None):
    xa = sub[sub.model == a].sort_values(["SUBJECT_ID", "k"]); xb = sub[sub.model == b].sort_values(["SUBJECT_ID", "k"])
    assert (xa.SESSION_ID.to_numpy() == xb.SESSION_ID.to_numpy()).all()
    pid = xa.SUBJECT_ID.to_numpy()
    da = E.cluster_boot_mean(xa[col].to_numpy(), pid, weights=W); db = E.cluster_boot_mean(xb[col].to_numpy(), pid, weights=W)
    if transform:
        da, db = transform(da), transform(db)
    est_a = xa[col].mean(); est_b = xb[col].mean()
    est = (transform(est_a) - transform(est_b)) if transform else est_a - est_b
    return {"estimate": float(est), "ci95": E.ci(da - db)}


for tgt in sel["targets"]:
    R = {}
    for split in ("test", "temporal"):
        sub = S[(S.target == tgt) & (S.split == split)]
        W = E.bootstrap_weights(sub.SUBJECT_ID.to_numpy(), B, seed=1)
        r = {"n_targets": int(sub[sub.model == "M3"].shape[0]), "n_patients": int(sub.SUBJECT_ID.nunique()),
             "models": table(sub, W)}
        comp = sel["targets"][tgt]["H1_comparator"]; comp_label = "M2" if comp.startswith("M2") else comp
        r["H1_comparator"] = comp_label
        r["H1_crps_M3_minus_comparator"] = paired(sub, "M3", comp_label, "crps", W)
        r["H1_success"] = bool(r["H1_crps_M3_minus_comparator"]["ci95"][1] < 0)
        r["MAE_M3_minus_M0a"] = paired(sub, "M3", "M0a", "abs_err", W)
        r["MAE_best_minus_M0a"] = {m: paired(sub, m, "M0a", "abs_err", W) for m in ("M1", "M2", "M4")}
        r["CRPS_vs_M3"] = {m: paired(sub, m, "M3", "crps", W) for m in MODELS if m != "M3"}
        ace = lambda c: np.abs(c - 0.95)
        r["H2_abs_cov_error_M3_minus_M0a"] = paired(sub, "M3", "M0a", "cov95", W, transform=ace)
        c3 = r["models"]["M3"]["cov95"]
        r["H2_success"] = bool(0.93 <= c3 <= 0.97 and r["H2_abs_cov_error_M3_minus_M0a"]["ci95"][1] < 0)
        r["H2_abs_cov_error_M2_minus_M0a"] = paired(sub, "M2", "M0a", "cov95", W, transform=ace)
        # secondary: variance-recalibrated M2/M3/M0 (factor from calibration patients)
        rec = {}
        for mdl in ("M0a", "M0b", "M2", "M3"):
            x = sub[sub.model == mdl]; f = x.recal.iat[0]
            rec[mdl] = {"factor": float(f), **E.summarise(E.pointwise(x.assign(sd=x.sd * f)[["y", "mu", "sd"]]))}
        r["recalibrated_secondary"] = rec
        if split == "test":
            br = {}
            for col in ("CLASS_LABEL", "n_prior_grp", "gap_bin"):
                br[col] = {}
                for lv, g in sub.groupby(col):
                    Wg = E.bootstrap_weights(g.SUBJECT_ID.to_numpy(), 1000, seed=2)
                    br[col][str(lv)] = {mdl: {**E.summarise(g[g.model == mdl]),
                                              "crps_ci95": E.ci(E.cluster_boot_mean(g[g.model == mdl].crps.to_numpy(),
                                                                                    g[g.model == mdl].SUBJECT_ID.to_numpy(), weights=Wg)),
                                              "cov95_ci95": E.ci(E.cluster_boot_mean(g[g.model == mdl].cov95.to_numpy(),
                                                                                     g[g.model == mdl].SUBJECT_ID.to_numpy(), weights=Wg))}
                                        for mdl in MODELS}
            r["breakdowns"] = br
            x3 = sub[sub.model == "M3"]
            r["E1_M3_sd_informativeness"] = E.sd_informativeness(x3.reset_index(drop=True), x3.sd.to_numpy())
            r["E1_M0b_sd_informativeness"] = E.sd_informativeness(sub[sub.model == "M0b"].reset_index(drop=True),
                                                                  sub[sub.model == "M0b"].sd.to_numpy())
            r["M3_variance_decomposition_mean"] = {
                "latent_trajectory_var": float(x3.latent_var.mean()), "extra_visit_var_sigma_w2": float(x3.sigma_w2.mean()),
                "within_visit_measurement_var": float(x3.sbar2.mean()),
                "note": "sigma_w^2 is not identifiable as measurement error; it mixes sensor/setup variation with "
                        "genuine change faster than the kernel represents"}
            r["pit_hist"] = {mdl: np.histogram(sub[sub.model == mdl].pit, bins=10, range=(0, 1))[0].tolist() for mdl in MODELS}
        R[split] = r
    res["targets"][tgt] = R
    t = R["test"]
    print(f"{tgt}: H1 {t['H1_success']} dCRPS {t['H1_crps_M3_minus_comparator']} | H2 {t['H2_success']} "
          f"cov95 M3 {t['models']['M3']['cov95']:.3f} M0a {t['models']['M0a']['cov95']:.3f} | "
          f"MAE M3-M0a {t['MAE_M3_minus_M0a']}")
(A / "prediction_results.json").write_text(json.dumps(res, indent=2, default=float))
rows = []
for tgt, R in res["targets"].items():
    for split, r in R.items():
        for mdl, s in r["models"].items():
            rows.append({"target": tgt, "split": split, "model": mdl, **{k: v for k, v in s.items() if k != "ci95"}})
pd.DataFrame(rows).to_csv(A / "prediction_metrics.csv", index=False)
