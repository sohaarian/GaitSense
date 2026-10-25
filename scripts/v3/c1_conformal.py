"""Capability 1, v3.0 post-hoc improvement: conformal recalibration of the frozen v2.0 GP intervals.

Motivation (from v2.0 results): GP errors were heavier-tailed than Gaussian (50% intervals covered 56%, 95% covered
94.6%). Quantiles of |z| are fitted on CALIBRATION patients only and applied unchanged to test and temporal patients.
The v2.0 models, test predictions and primary results are not modified. Because the test set had already been
opened in v2.0, this comparison is post hoc. Writes results/v3/c1_conformal.json.
"""
import json, sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).parents[2]; sys.path.insert(0, str(ROOT / "src"))
from gaitsense_ai import prediction as P
from gaitsense_ai import evaluation as E
from gaitsense_platform.capability1 import ConformalCalibrator, load_v2

A = ROOT / "results" / "ai"
v2 = load_v2(A)
V = pd.read_csv(A / "visit_level_features.csv.gz", parse_dates=["SESSION_DATE"])
V = V[V.role.isin(["cohort", "temporal"])]
D = pd.read_csv(A / "predictions_test.csv.gz")
LEVELS = [0.5, 0.8, 0.9, 0.95]
out = {"post_hoc": True, "targets": {}}
for tgt, se in P.TARGETS.items():
    s = v2["selection"]["targets"][tgt]; g = v2["fitted"][tgt][s["M3_chosen"]]
    arrays = P.patient_arrays(V, tgt, se, v2["standardiser"]); T = P.build_targets(arrays, v2["standardiser"])
    Tc = T[T.split == "calib"]
    mu, vl = P.gp_predict_rows(g, arrays, Tc); sd = np.sqrt(vl + g.sigma_w ** 2 + Tc.sbar2.to_numpy())
    cal = ConformalCalibrator().fit(Tc.y.to_numpy(), mu, sd)
    r = {"calibration_n": int(len(Tc)), "conformal_quantiles": {str(l): cal.quantile(l) for l in LEVELS},
         "gaussian_quantiles": {str(l): float(norm.ppf(0.5 + l / 2)) for l in LEVELS}}
    for split in ("test", "temporal"):
        x = D[(D.target == tgt) & (D.model == "M3") & (D.split == split)]
        z = np.abs(x.y - x.mu) / x.sd
        rows = {}
        for l in LEVELS:
            qg, qc = norm.ppf(0.5 + l / 2), cal.quantile(l)
            rows[str(l)] = {"gaussian_coverage": float((z <= qg).mean()), "conformal_coverage": float((z <= qc).mean()),
                            "gaussian_width": float((2 * qg * x.sd).mean()), "conformal_width": float((2 * qc * x.sd).mean())}
        a = 0.05; qc = cal.quantile(0.95)
        lo, hi = x.mu - qc * x.sd, x.mu + qc * x.sd
        isc = (hi - lo) + (2 / a) * (lo - x.y) * (x.y < lo) + (2 / a) * (x.y - hi) * (x.y > hi)
        rows["interval_score_95"] = {"gaussian": float(E.interval_score(x.y, x.mu, x.sd).mean()), "conformal": float(isc.mean())}
        r[split] = {"n": int(len(x)), **rows}
    out["targets"][tgt] = r
    t = r["test"]
    print(tgt, {l: (round(t[l]["gaussian_coverage"], 3), round(t[l]["conformal_coverage"], 3)) for l in map(str, LEVELS)},
          "IS95", round(t["interval_score_95"]["gaussian"], 4), round(t["interval_score_95"]["conformal"], 4))
d = ROOT / "results" / "v3"; d.mkdir(parents=True, exist_ok=True)
(d / "c1_conformal.json").write_text(json.dumps(out, indent=2))
