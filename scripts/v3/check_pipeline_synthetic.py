"""Pipeline check of the integrated platform on SYNTHETIC data (positive and negative controls). Not a clinical result.

Negative control: the outcome does not depend on the latent recovery rate (bio_effect = 0), so Model C should not
beat Model A. Positive control: it does (bio_effect = 1.5), so Model C should beat Model A.
Writes results/v3/synthetic_pipeline_check.json; every record carries the synthetic banner.
"""
import json, sys
from pathlib import Path

ROOT = Path(__file__).parents[2]; sys.path.insert(0, str(ROOT / "src"))
from gaitsense_platform import pipeline, synthetic
from gaitsense_platform.gate import SYNTHETIC_BANNER

cfg = pipeline.Config(outcome="meaningful_improvement", landmark_day=42, horizon_day=90, min_gap_days=14,
                      window_end_day=120, binary=True, clinical_cols=("age", "severity"))
res = {"banner": SYNTHETIC_BANNER, "controls": {}}
for name, eff in (("negative_control", 0.0), ("positive_control", 1.5)):
    v, c, o, m = synthetic.make(n_patients=800, bio_effect=eff, seed=11)
    r = pipeline.run(m, v, c, o, cfg, allow_synthetic=True)
    res["controls"][name] = {"bio_effect": eff, "linked_patients": r["linkage"]["linked_patients"],
                             "auroc": {k: round(x["auroc"], 3) for k, x in r["results"]["models"].items()},
                             "C_minus_A_auroc": r["results"]["C_minus_A"]["auroc"],
                             "C_minus_A_brier": r["results"]["C_minus_A"]["brier"], "gate": r["gate"]["status"]}
    print(name, res["controls"][name]["auroc"], res["controls"][name]["C_minus_A_auroc"])
d = ROOT / "results" / "v3"; d.mkdir(parents=True, exist_ok=True)
(d / "synthetic_pipeline_check.json").write_text(json.dumps(res, indent=2))
