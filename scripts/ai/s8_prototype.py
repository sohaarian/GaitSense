"""Stage 8: GaitSense AI research prototype (static, self-contained HTML) built from validated test-partition outputs.

Writes prototype/gaitsense_ai_review.html (standalone document) and prototype/gaitsense_ai_review.fragment.html
(the same page without the document wrapper, for hosting as an artifact). Only test-partition patients are shown,
because only their predictions are out-of-sample.
"""
import json, sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parents[2]; A = ROOT / "results" / "ai"
V = pd.read_csv(A / "visit_level_features.csv.gz")
D = pd.read_csv(A / "predictions_test.csv.gz")
S = pd.read_csv(A / "monitoring_pairs.csv.gz")
pr = json.loads((A / "prediction_results.json").read_text())
mon = json.loads((A / "monitoring_results.json").read_text())
sel = json.loads((A / "model_selection.json").read_text())

tr = V[(V.role == "cohort") & (V.split == "train")]
se_hi = float(tr.d_peak_v_se.quantile(0.90))
test = V[(V.role == "cohort") & (V.split == "test")].sort_values(["SUBJECT_ID", "visit_no"])
ids = {sid: f"R-{i + 1:03d}" for i, sid in enumerate(sorted(test.SUBJECT_ID.unique()))}
CLS = {"A": "Ankle", "K": "Knee", "H": "Hip", "C": "Heel"}
r4 = lambda x: None if pd.isna(x) else round(float(x), 4)
Dt = D[D.split == "test"].set_index(["SESSION_ID", "target", "model"])
Sm = S.set_index("SESSION_ID_cur")

patients = []
for sid, p in test.groupby("SUBJECT_ID"):
    visits = []
    for _, v in p.iterrows():
        vv = {"no": int(v.visit_no), "day": int(v.day), "n_a": int(v.n_a), "n_u": int(v.n_u)}
        for f in ("peak_v", "braking", "propulsive", "stance_s"):
            vv[f] = [r4(v[f"{f}_a"]), r4(v[f"{f}_u"]), r4(v[f"d_{f}"]), r4(v[f"d_{f}_se"])]
        if v.visit_no >= 2:
            pred = {}
            for tgt in ("d_peak_v", "d_braking", "d_propulsive", "d_stance_s"):
                for m in ("M0b", "M1", "M2", "M3"):
                    if (v.SESSION_ID, tgt, m) in Dt.index:
                        row = Dt.loc[(v.SESSION_ID, tgt, m)]
                        pred.setdefault(tgt, {})[m] = [r4(row.mu), r4(row.sd)]
            vv["pred"] = pred
            if v.SESSION_ID in Sm.index:
                s = Sm.loc[v.SESSION_ID]
                vv["mon"] = {"A": bool(s.alert_A), "B": bool(s.alert_B), "C": bool(s.alert_C), "C0": bool(s.alert_C0),
                             "D": bool(s.alert_D), "dLSI": r4(s.dLSI), "zC": r4(s.score_C), "rD": r4(s.score_D),
                             "zBp": r4(s.zB_prev), "zBc": r4(s.zB_cur), "dA": r4(s.d_affected), "dU": r4(s.d_unaffected),
                             "muD": r4(s.mu_D), "sdD": r4(np.sqrt(s.var_D + s.se_cur ** 2))}
        visits.append(vv)
    patients.append({"id": ids[sid], "cls": CLS[p.CLASS_LABEL.iat[0]], "visits": visits})

T = pr["targets"]["d_peak_v"]["test"]
calib = {"overall": {m: [round(T["models"][m]["cov95"], 3), round(T["models"][m]["MAE"], 4), round(T["models"][m]["CRPS"], 4),
                         round(T["models"][m]["width95"], 4)] for m in ("M0a", "M0b", "M1", "M2", "M3", "M4")},
         "by_class": {CLS[c]: {m: round(T["breakdowns"]["CLASS_LABEL"][c][m]["cov95"], 3) for m in ("M0b", "M1", "M2", "M3")}
                      for c in "AKHC"},
         "n_targets": T["n_targets"], "n_patients": T["n_patients"],
         "mae_m3_minus_locf": T["MAE_M3_minus_M0a"], "crps_m3_minus_m2": T["H1_crps_M3_minus_comparator"]}
meta = {"se_hi": round(se_hi, 4), "tau": round(mon["tau"], 4), "mu_h": round(mon["healthy_reference"]["mu"], 4),
        "sd_h": round(mon["healthy_reference"]["sd"], 4), "kernel": sel["targets"]["d_peak_v"]["M3_chosen"].replace("M3_", ""),
        "kappa_matched": {k: round(v["kappa"]["estimate"], 2) for k, v in mon["matched"]["pairwise"].items()},
        "native_rates": {k: round(v["estimate"], 3) for k, v in mon["native"]["alert_rate"].items()}}
DATA = json.dumps({"patients": patients, "calib": calib, "meta": meta}, separators=(",", ":"))

PAGE = (ROOT / "prototype" / "ai_template.html").read_text().replace("/*__DATA__*/null", DATA)
(ROOT / "prototype" / "gaitsense_ai_review.fragment.html").write_text(PAGE)
(ROOT / "prototype" / "gaitsense_ai_review.html").write_text(
    "<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1,viewport-fit=cover\">"
    "</head><body>\n" + PAGE + "\n</body></html>\n")
print("patients", len(patients), "bytes", len(PAGE))
