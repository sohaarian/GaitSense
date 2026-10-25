"""Check numbers quoted in the v3.0 documents against results/v3 outputs. Exit 1 on any MISS."""
import json, sys
from pathlib import Path

ROOT = Path(__file__).parents[2]; R = ROOT / "results" / "v3"
syn = json.loads((R / "synthetic_pipeline_check.json").read_text())["controls"]
con = json.loads((R / "c1_conformal.json").read_text())["targets"]["d_peak_v"]
docs = {n: (ROOT / p).read_text() for n, p in (("changelog", "CHANGELOG.md"), ("protocol", "docs/v3_capability2_protocol.md"),
                                                ("c1", "docs/v3_capability1_update.md"))}
miss = 0


def need(doc, text):
    global miss
    ok = text in docs[doc]; miss += not ok
    print(f"{'OK  ' if ok else 'MISS'} [{doc}] {text}")


n, p = syn["negative_control"]["C_minus_A_auroc"], syn["positive_control"]["C_minus_A_auroc"]
need("changelog", f"(AUROC difference −{abs(n['estimate']):.3f}, 95% CI −{abs(n['ci95'][0]):.3f} to {n['ci95'][1]:.3f})")
need("changelog", f"(+{p['estimate']:.3f}, {p['ci95'][0]:.3f} to {p['ci95'][1]:.3f})")
need("c1", f"{con['calibration_n']} calibration predictions")
need("c1", f"test, {con['test']['n']} predictions")
for lv, lab in (("0.5", "50%"), ("0.8", "80%"), ("0.9", "90%"), ("0.95", "95%")):
    t = con["test"][lv]
    need("c1", f"| {lab} | {t['gaussian_coverage'] * 100:.1f}% | {t['conformal_coverage'] * 100:.1f}% | {t['gaussian_width']:.3f} | {t['conformal_width']:.3f} |")
i = con["test"]["interval_score_95"]; it = con["temporal"]["interval_score_95"]
need("c1", f"Gaussian {i['gaussian']:.3f}, conformal {i['conformal']:.3f}")
tp = con["temporal"]
need("c1", f"(50%: {tp['0.5']['gaussian_coverage'] * 100:.1f}% → {tp['0.5']['conformal_coverage'] * 100:.1f}%; 95%: "
           f"{tp['0.95']['gaussian_coverage'] * 100:.1f}% → {tp['0.95']['conformal_coverage'] * 100:.1f}%; interval score "
           f"{it['gaussian']:.3f} → {it['conformal']:.3f})")
print(f"\n{miss} misses" if miss else "\nall v3 document numbers match the results")
sys.exit(1 if miss else 0)
