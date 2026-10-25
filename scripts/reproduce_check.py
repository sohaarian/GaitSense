"""Independent re-run of key results, compared with the stored results/*.json files.

Re-runs the cohort, the primary point estimate (M0, M2 over 10 repeats) and the monitoring
variability figures from the source data, and reports any mismatch.
"""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).parents[1]; sys.path.insert(0, str(ROOT / "src"))
from gaitrec_rehab.analysis import cv_predict, load_data, metrics
from gaitrec_rehab.monitoring import between_session_sd
from gaitrec_rehab.sessions import cohort_session_table

main = json.loads((ROOT / "results" / "main.json").read_text())
var = json.loads((ROOT / "results" / "variability.json").read_text())
d = load_data(ROOT / "data"); c = d.cohort; y = c.event.to_numpy(); rows = np.arange(len(c))
checks = [("patients", len(c), main["n"]), ("events", int(y.sum()), main["events"])]
aur = {"M0": [], "M2": []}
for s in range(10):
    o = cv_predict(d, y, rows, c.SUBJECT_ID.to_numpy(), ["M0", "M2"], s, False)
    for k in aur: aur[k].append(metrics(y, o[k])["auroc"])
checks += [("M0 AUROC", np.mean(aur["M0"]), main["models"]["M0"]["auroc"]),
           ("M2 AUROC", np.mean(aur["M2"]), main["models"]["M2"]["auroc"]),
           ("M2-M0", np.mean(aur["M2"]) - np.mean(aur["M0"]), main["primary_auroc_diff_M2_M0"])]
st, _ = cohort_session_table(ROOT / "data")
checks.append(("extra SD loading", between_session_sd(st)["peak_loading"], var["extra_between_session_sd_full_cohort"]["peak_loading"]))
bad = 0
for name, got, want in checks:
    ok = abs(float(got) - float(want)) < 1e-9; bad += not ok
    print(("OK  " if ok else "DIFF"), name, round(float(got), 6), "stored", round(float(want), 6))
(ROOT / "results" / "reproduce_check.json").write_text(json.dumps({n: [float(g), float(w)] for n, g, w in checks}, indent=2))
sys.exit(1 if bad else 0)
