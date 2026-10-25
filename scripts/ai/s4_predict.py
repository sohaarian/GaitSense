"""Stages 4-5: future-visit prediction models.

  python scripts/ai/s4_predict.py select   fit on training, choose on calibration -> results/ai/model_selection.json
  python scripts/ai/s4_predict.py test     refuses to run without model_selection.json; evaluates test + temporal once

The select phase never builds test or temporal predictions (asserted).
"""
import json, pickle, platform, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn, scipy

ROOT = Path(__file__).parents[2]; sys.path.insert(0, str(ROOT / "src"))
from gaitsense_ai import SEED, __version__, splits
from gaitsense_ai import evaluation as E
from gaitsense_ai import prediction as P
from gaitsense_ai.longitudinal import KERNELS, LongitudinalGP

PHASE = sys.argv[1]
A = ROOT / "results" / "ai"
V = pd.read_csv(A / "visit_level_features.csv.gz", parse_dates=["SESSION_DATE"])
part = pd.read_csv(A / "partition.csv")
V = V[V.role.isin(["cohort", "temporal"])]
train_ids = part[(part.split == "train") & (part.role == "cohort")].SUBJECT_ID.to_numpy()
std = P.Standardiser().fit(V[V.SUBJECT_ID.isin(train_ids)])
GP_KERNELS = [k for k in KERNELS if not k.startswith("lmm")]


def predict_all(models, arrays, T, tgt):
    """Predictive distributions of every model for target rows T."""
    out = {}
    sbar2 = T.sbar2.to_numpy(); last_se2 = T.last_se.to_numpy() ** 2
    out["M0a"] = (T["last"].to_numpy(), np.sqrt(last_se2 + sbar2))
    out["M0b"] = (T["last"].to_numpy(), np.sqrt(last_se2 + sbar2 + models["tau"] ** 2))
    X = T[P.HIST].to_numpy()
    for name in ("M1", "M4"):
        mu = models[name].predict(X)
        sd = T.n_prior_grp.map(models[name + "_sd"]).to_numpy()
        out[name] = (mu, sd)
    for name in ["M2_" + k for k in ("lmm_int", "lmm_slope")] + ["M3_" + k for k in GP_KERNELS]:
        g = models[name]; mu, vl = P.gp_predict_rows(g, arrays, T)
        out[name] = (mu, np.sqrt(vl + g.sigma_w ** 2 + sbar2))
        out[name + "_latentvar"] = vl
    return out


if PHASE == "select":
    sel = {"version": __version__, "seed": SEED, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                        "scikit-learn": sklearn.__version__, "scipy": scipy.__version__}, "targets": {}}
    fitted = {}
    for tgt, se in P.TARGETS.items():
        t0 = time.time()
        arrays = P.patient_arrays(V, tgt, se, std)
        T = P.build_targets(arrays, std)
        Ttr, Tca = T[T.split == "train"], T[T.split == "calib"]
        splits.assert_only(Ttr, part, {"train"}, "training targets")
        splits.assert_only(Tca, part, {"calib"}, "calibration targets")
        m = {"tau": P.tau_moments(arrays, train_ids)}
        Xtr, ytr = Ttr[P.HIST].to_numpy(), Ttr.y.to_numpy()
        for name, make, grid in (("M1", P.make_ridge, P.RIDGE_GRID), ("M4", P.make_hgb, P.HGB_GRID)):
            params, cv_mae = P.cv_select(make, grid, Xtr, ytr, Ttr.SUBJECT_ID.to_numpy(), SEED)
            m[name] = make(**params).fit(Xtr, ytr); m[name + "_params"] = params; m[name + "_cv_mae"] = cv_mae
            m[name + "_sd"] = P.resid_sd_by_group(Tca, m[name].predict(Tca[P.HIST].to_numpy()))
        data = P.training_data(arrays, train_ids)
        ysd = float(np.std(np.concatenate([arrays[p][1] for p in train_ids])))
        for k in KERNELS:
            key = ("M2_" if k.startswith("lmm") else "M3_") + k
            m[key] = LongitudinalGP(k).fit(data, ysd, restarts=2, seed=SEED)
        preds = predict_all(m, arrays, Tca, tgt)
        calib = {name: E.summarise(E.pointwise(pd.DataFrame({"y": Tca.y, "mu": v[0], "sd": v[1]})))
                 for name, v in preds.items() if not name.endswith("_latentvar")}
        m2 = min(("M2_lmm_int", "M2_lmm_slope"), key=lambda k: m[k].aic)          # M2 structure by training AIC
        m3 = min(["M3_" + k for k in GP_KERNELS], key=lambda k: calib[k]["NLPD"])  # M3 kernel by calibration NLPD
        comp = min(["M0b", "M1", m2], key=lambda k: calib[k]["CRPS"])               # H1 comparator by calibration CRPS
        # secondary (not primary) variance recalibration factor from calibration standardised residuals
        rec = {k: float(np.sqrt(np.mean(((Tca.y - preds[k][0]) / preds[k][1]) ** 2))) for k in (m2, m3, "M0a", "M0b")}
        sel["targets"][tgt] = {
            "n_targets": {"train": len(Ttr), "calib": len(Tca)}, "tau_train": m["tau"],
            "M1_params": m["M1_params"], "M1_cv_mae": m["M1_cv_mae"], "M1_sd_by_prior": m["M1_sd"],
            "M4_params": m["M4_params"], "M4_cv_mae": m["M4_cv_mae"], "M4_sd_by_prior": m["M4_sd"],
            "M2_chosen": m2, "M2_aic": {k: m[k].aic for k in ("M2_lmm_int", "M2_lmm_slope")},
            "M3_chosen": m3, "M3_calib_nlpd": {k: calib["M3_" + k]["NLPD"] for k in GP_KERNELS},
            "H1_comparator": comp, "recalibration_factor": rec,
            "hyperparameters": {k: m[k].hyperparameters() | {"beta": m[k].beta.tolist(), "aic": m[k].aic,
                                                              "converged": m[k].converged}
                                for k in m if k.startswith(("M2_", "M3_"))},
            "calibration_metrics": calib, "seconds": round(time.time() - t0, 1)}
        fitted[tgt] = m
        print(tgt, "M2", m2, "M3", m3, "comparator", comp, f"{time.time() - t0:.0f}s", flush=True)
    sel["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    (A / "model_selection.json").write_text(json.dumps(sel, indent=2, default=float))
    with open(A / "models.pkl", "wb") as f:
        pickle.dump({"fitted": fitted, "std": std}, f)

elif PHASE == "test":
    selp = A / "model_selection.json"
    if not selp.exists():
        sys.exit("run the select phase first: model choices must be frozen before the test partition is opened")
    sel = json.loads(selp.read_text())
    with open(A / "models.pkl", "rb") as f:
        fitted = pickle.load(f)["fitted"]
    rows, res = [], {"opened": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "targets": {}}
    for tgt, se in P.TARGETS.items():
        s = sel["targets"][tgt]; m = fitted[tgt]
        arrays = P.patient_arrays(V, tgt, se, std); T = P.build_targets(arrays, std)
        Tev = T[T.split.isin(["test", "temporal"])].reset_index(drop=True)
        preds = predict_all(m, arrays, Tev, tgt)
        names = ["M0a", "M0b", "M1", s["M2_chosen"], s["M3_chosen"], "M4"]
        label = {s["M2_chosen"]: "M2", s["M3_chosen"]: "M3"}
        for nm in names:
            mu, sd = preds[nm]
            d = Tev[["SUBJECT_ID", "SESSION_ID", "split", "CLASS_LABEL", "k", "n_prior", "n_prior_grp", "gap_days",
                     "y", "last", "sbar2", "se2_target"]].copy()
            d["target"] = tgt; d["model"] = label.get(nm, nm); d["mu"] = mu; d["sd"] = sd
            d["latent_var"] = preds.get(nm + "_latentvar", np.full(len(d), np.nan))
            d["sigma_w2"] = m[nm].sigma_w ** 2 if nm.startswith(("M2", "M3")) else np.nan
            d["recal"] = s["recalibration_factor"].get(nm, np.nan)
            rows.append(d)
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(A / "predictions_test.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    (A / "test_opened.json").write_text(json.dumps(res, indent=2))
    print("test predictions written:", len(out))
