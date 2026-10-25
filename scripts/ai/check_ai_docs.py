"""Check that numbers quoted in the v2.0 documents match results/ai outputs (like scripts/check_report.py).

Each check formats a value computed from the results files and requires that exact text in the named document(s).
Exit code 1 on any MISS.
"""
import json, sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parents[2]; A = ROOT / "results" / "ai"; DOCS = ROOT / "docs"
pr = json.loads((A / "prediction_results.json").read_text())
mon = json.loads((A / "monitoring_results.json").read_text())
pert = json.loads((A / "perturbation_results.json").read_text())
inj = json.loads((A / "injection_results.json").read_text())
tg = json.loads((A / "tau_by_gap.json").read_text())
s3 = json.loads((A / "stage3_summary.json").read_text())
sel = json.loads((A / "model_selection.json").read_text())
T = pr["targets"]["d_peak_v"]["test"]; Tt = pr["targets"]["d_peak_v"]["temporal"]
ms = (DOCS / "gaitsense_ai_manuscript.md").read_text()
pf = (DOCS / "yale_technical_portfolio.md").read_text()
nv = (DOCS / "novelty_assessment.md").read_text()
rd = (ROOT / "README.md").read_text()
miss = 0


def need(label, text, *docs):
    global miss
    names = {id(ms): "manuscript", id(pf): "portfolio", id(nv): "novelty", id(rd): "README"}
    for d in docs or (ms,):
        ok = text in d
        print(f"{'OK  ' if ok else 'MISS'} [{names[id(d)]:10s}] {label}: {text}")
        miss += not ok


pct = lambda x, d=1: f"{100 * x:.{d}f}%"
f4 = lambda x: f"{x:.4f}"
M = T["models"]
# cohort and design
V = pd.read_csv(A / "visit_level_features.csv.gz")
need("visits", f"{(V.role == 'cohort').sum():,} visits")
need("pairs", f"{sum(v for k, v in s3['pairs_by_split'].items() if k != 'temporal'):,} consecutive visit pairs")
need("partition", f"training (1,001), calibration (333) and test (333)")
need("test predictions", f"{T['n_targets']} test predictions ({T['n_patients']} patients)")
need("temporal predictions", f"{Tt['n_targets']} predictions")
need("healthy reference sessions", f"{mon['healthy_reference']['n_sessions']} training-control sessions")
# prediction table
for m, lab in (("M0a", "M0a last value, same-day noise"), ("M0b", "M0b last value + τ"), ("M1", "M1 ridge"),
               ("M2", "M2 mixed model (random slope)"), ("M3", "M3 GP (Matérn-3/2 + linear)"), ("M4", "M4 boosted trees")):
    need(f"{m} row", f"| {lab} | {f4(M[m]['MAE'])} | {f4(M[m]['CRPS'])} | {pct(M[m]['cov95'])} | {M[m]['width95']:.3f} |")
h1 = T["H1_crps_M3_minus_comparator"]
need("H1 estimate", f"−{abs(h1['estimate']):.5f} (95% CI −{abs(h1['ci95'][0]):.5f} to −{abs(h1['ci95'][1]):.5f})")
need("H1 abstract", f"by {abs(h1['estimate']):.5f} BW (95% CI {abs(h1['ci95'][1]):.5f} to {abs(h1['ci95'][0]):.5f}")
need("H1 relative", f"{abs(h1['estimate']) / M['M2']['CRPS'] * 100:.1f}%", ms, pf)
r1 = T["CRPS_vs_M3"]["M1"]
need("GP vs ridge", f"−{r1['estimate']:.5f} (−{r1['ci95'][1]:.5f} to {-r1['ci95'][0]:.5f})")
mae = T["MAE_M3_minus_M0a"]
need("MAE vs last", f"−{abs(mae['estimate']):.4f} BW, 95% CI −{abs(mae['ci95'][0]):.4f} to −{abs(mae['ci95'][1]):.4f}")
need("MAE vs last (results)", f"−{abs(mae['estimate']):.4f} (−{abs(mae['ci95'][0]):.4f} to −{abs(mae['ci95'][1]):.4f})")
tt = Tt["H1_crps_M3_minus_comparator"]
need("temporal H1", f"−{abs(tt['estimate']):.5f}, CI −{abs(tt['ci95'][0]):.5f} to {tt['ci95'][1]:.5f}")
need("secondary H1 all false", "not met for any secondary target" if not any(pr["targets"][t]["test"]["H1_success"] for t in pr["targets"] if t != "d_peak_v") else "XX")
need("M0a coverage CI", f"{pct(M['M0a']['cov95'])} (95% CI {pct(M['M0a']['ci95']['cov95'][0])[:-1]} to {pct(M['M0a']['ci95']['cov95'][1])})")
need("M3 coverage CI", f"{pct(M['M3']['cov95'])} ({pct(M['M3']['ci95']['cov95'][0])[:-1]} to {pct(M['M3']['ci95']['cov95'][1])})")
h2 = T["H2_abs_cov_error_M3_minus_M0a"]
need("H2 difference", f"−{abs(h2['estimate']):.3f} (−{abs(h2['ci95'][0]):.3f} to −{abs(h2['ci95'][1]):.3f})")
need("narrower", f"{round((1 - M['M3']['width95'] / M['M0b']['width95']) * 100)}% narrower", ms, pf, rd)
need("temporal coverage", f"temporal set ({pct(Tt['models']['M3']['cov95'])})")
need("50% coverage GP", f"GP {pct(M['M3']['cov50'], 0)}")
e1 = T["E1_M3_sd_informativeness"]
need("E1 spearman", f"Spearman {e1['spearman_sd_abs_error']:.2f}; M0b {T['E1_M0b_sd_informativeness']['spearman_sd_abs_error']:.2f}")
q = e1["by_sd_quintile"]
need("E1 quintile MAE", f"from {q[0]['mae']:.3f} BW in the lowest SD quintile to {q[-1]['mae']:.3f} BW")
vd = T["M3_variance_decomposition_mean"]
need("variance decomposition", f"({vd['latent_trajectory_var']:.5f} BW²), extra visit-level variation σ²_w ({vd['extra_visit_var_sigma_w2']:.5f}) and within-visit measurement variance ({vd['within_visit_measurement_var']:.5f})")
br = T["breakdowns"]["CLASS_LABEL"]
cov_g = [br[c]["M3"]["cov95"] for c in "AKHC"]
need("group coverage range", f"GP {round(min(cov_g) * 100)} to {round(max(cov_g) * 100)}%")
# monitoring
nr = mon["native"]["alert_rate"]
for a, lab in (("A", "A raw asymmetry (ΔLSI ≥ 10)"), ("B", "B healthy reference (band crossing)"), ("C", "C change limit (with τ)"), ("D", "D GP prediction deviation")):
    need(f"native rate {a}", f"| {lab} | {pct(nr[a]['estimate'])} |")
need("C0 rate", f"| Original limit, same-day noise only | {pct(nr['C0_original']['estimate'])} |")
kn = {k: v["kappa"]["estimate"] for k, v in mon["native"]["pairwise"].items()}
km = {k: v["kappa"]["estimate"] for k, v in mon["matched"]["pairwise"].items()}
need("native kappa range", f"κ {min(kn.values()):.2f} to {max(kn.values()):.2f}")
need("matched kappa range", f"κ {min(km.values()):.2f} to {max(km.values()):.2f}")
cd = mon["native"]["pairwise"]["C-D"]["kappa"]
need("C-D native kappa", f"κ {cd['estimate']:.2f}, 95% CI {cd['ci95'][0]:.2f} to {cd['ci95'][1]:.2f}")
ad = mon["matched"]["pairwise"]["A-D"]["kappa"]
need("A-D matched kappa", f"A with D, {ad['estimate']:.2f}, CI {ad['ci95'][0]:.2f} to {ad['ci95'][1]:.2f}")
prox = [v["share_with_a_score_within_10pct_of_threshold"] for v in mon["matched"]["threshold_proximity"].values()]
need("threshold proximity", f"Between {round(min(prox) * 100)}% and {round(max(prox) * 100)}%")
wl = mon["A_which_limb"]
need("which limb", f"{round(wl['share_unaffected_change_at_least_half_of_dY'] * wl['A_alerts'])} of {wl['A_alerts']} raw-asymmetry alerts")
# perturbations
E = pert["experiments"]
need("s_w", f"({pert['scales_from_training']['s_w_peak_v_BW']:.3f} BW)")
need("cv_w", f"CV ({pert['scales_from_training']['cv_w_peak_v'] * 100:.1f}%)")
d1 = [E[k]["D"]["flip_rate"]["estimate"] for k in ("offset_+1sw", "offset_-1sw", "scale_+1cv", "scale_-1cv")]
need("D ±1 flip", f"{min(d1) * 100:.1f} to {max(d1) * 100:.1f}% for D")
mp = pert["missing_previous_visit"]
need("missing visit C", f"{mp['C']['flip_rate'] * 100:.1f}% of the {mp['n_pairs']} eligible pairs")
need("missing visit D", f"in only {mp['D']['flip_rate'] * 100:.1f}%")
need("missing visit coverage", f"from {mp['D_cov95_full'] * 100:.1f}% to {mp['D_cov95_missing'] * 100:.1f}%")
k3 = E["trials_keep3"]
need("keep3", f"(C {k3['C']['flip_rate']['estimate'] * 100:.1f}%, D {k3['D']['flip_rate']['estimate'] * 100:.1f}%)")
# injections
X = inj["experiments"]
need("5% vertical", f"{round(X['vertical_-0.05']['Y']['C'] * 100)}% vs {round(X['vertical_-0.05']['Y']['C'] * 100)}%".replace(
    f"{round(X['vertical_-0.05']['Y']['C'] * 100)}% vs", f"{round(X['vertical_-0.05']['Y']['D'] * 100)}% vs"))
need("10% vertical", f"10% reduction in {round(X['vertical_-0.10']['Y']['C'] * 100)}% and {round(X['vertical_-0.10']['Y']['D'] * 100)}%")
need("baselines", f"{inj['baseline']['Y_C_down'] * 100:.1f}% and {inj['baseline']['Y_D_down'] * 100:.1f}%")
need("bilateral", f"(C {X['bilateral_vertical_-0.10']['Y']['C'] * 100:.1f}%, D {X['bilateral_vertical_-0.10']['Y']['D'] * 100:.1f}% for a 10% bilateral reduction")
need("bilateral abs", f"({round(X['bilateral_vertical_-0.10']['affected_abs_peak_v']['C'] * 100)}%, {round(X['bilateral_vertical_-0.10']['affected_abs_peak_v']['D'] * 100)}%)")
need("propulsive 20%", f"{round(X['propulsive_+0.20']['d_propulsive']['C'] * 100)}% (C) and {round(X['propulsive_+0.20']['d_propulsive']['D'] * 100)}% (D)")
need("LSI impulse", f"({round(X['propulsive_+0.20']['d_propulsive']['A_LSI10'] * 100)}%) only because it alerted on {round(inj['baseline']['d_propulsive_A_LSI10_any'] * 100)}%")
need("peak1", f"D {round(X['peak1_local_+0.15']['Y']['D'] * 100)}% for a 15% first-peak reduction")
need("B range", f"from {round(X['vertical_-0.02']['Y']['B'] * 100)}% to {round(X['vertical_-0.15']['Y']['B'] * 100)}%")
# tau by gap
g = tg["features"]["d_peak_v"]
need("same-day pairs", f"{g['same day']['pairs']} consecutive pairs")
need("same-day tau", f"{g['same day']['tau']:.3f} BW (95% CI {g['same day']['tau_ci95'][0]:.3f} to {g['same day']['tau_ci95'][1]:.3f})")
need("tau ratio", f"about {round((g['same day']['tau'] / g['1-7 d']['tau']) ** 2 * 100)}% of the weekly value")
need("tau ratio portfolio", f"about **{round((g['same day']['tau'] / g['1-7 d']['tau']) ** 2 * 100)}%**", pf)
need("kernel", sel["targets"]["d_peak_v"]["M3_chosen"].replace("M3_matern32_lin", "Matérn-3/2 + linear"))
print(f"\n{miss} misses" if miss else "\nall document numbers match the results")
sys.exit(1 if miss else 0)
