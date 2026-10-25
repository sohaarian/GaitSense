"""Check that numbers quoted in docs/technical_report.md match the result files."""
import json, sys
from pathlib import Path
R = Path(__file__).parents[1]
rep = (R / "docs" / "technical_report.md").read_text()
m = json.loads((R / "results" / "main.json").read_text()); b = json.loads((R / "results" / "bias.json").read_text())
mon = json.loads((R / "results" / "monitoring.json").read_text()); pc = json.loads((R / "results" / "positive_control.json").read_text())
f3 = lambda x: f"{x:.3f}"; s3 = lambda x: f"{x:+.3f}".replace("+", "+") ; ci = lambda a: f"{a[0]:.3f} to {a[1]:.3f}"
checks = {
  "M0 AUROC": f3(m["models"]["M0"]["auroc"]), "M1 AUROC": f3(m["models"]["M1"]["auroc"]),
  "M2 AUROC": f3(m["models"]["M2"]["auroc"]), "M2+ AUROC": f3(m["models"]["M2+"]["auroc"]),
  "primary diff": f"{m['primary_auroc_diff_M2_M0']:.3f}".replace("-", "−"),
  "primary CI": ci(m["primary_boot"]["auroc_diff_ci"]).replace("-", "−"),
  "S1": ci(m["S1_ci_oof"]).replace("-", "−"), "S2": ci(m["S2"]["ci_oof"]).replace("-", "−"),
  "S4": ci(m["S4"]["boot"]["auroc_diff_ci"]).replace("-", "−"), "E1": ci(m["E1"]["ci"]).replace("-", "−"),
  "E2": ci(m["E2_ci_oof"]).replace("-", "−"), "E3 M0": f"{m['E3']['M0']:.2f}", "E3 M2": f"{m['E3']['M2']:.2f}",
  "E3 n": f"{m['E3']['n']:,}", "E3 events": str(m["E3"]["events"]),
  "S2 n": f"{m['S2']['n']:,}", "S4 n": f"{m['S4']['n']:,}", "S4 events": str(m["S4"]["events"]),
  "E1 test": f"{m['E1']['test_n']} test ({m['E1']['test_events']})",
  "PC true": f3(pc["planted_true_auroc"]), "PC M2": f3(pc["recovered_M2"]), "PC M0": f3(pc["recovered_M0"]),
  "in-sample M0": f"{pc['M0_in_sample_unpenalised_auroc']:.2f}", "in-sample M2": f"{pc['M2_in_sample_unpenalised_auroc']:.2f}",
  "mon sessions": f"{mon['sessions']:,}", "mon 2plus": f"{mon['patients_2plus_sessions']:,}", "mon 3plus": f"{mon['patients_3plus_sessions']:,}",
  "declined %": f"{mon['overall']['loading_declined']*100:.1f}%", "persistent %": f"{mon['overall']['persistent_loading_deficit']*100:.1f}%",
  "above %": f"{mon['overall']['loading_above_typical_range']*100:.1f}%",
  "bias 0.6 gait": f"+{b['s=0.6, gait']['mean_diff']:.3f}", "bias 0.6 gait CI": f"{b['s=0.6, gait']['range95'][0]:.3f} to {b['s=0.6, gait']['range95'][1]:.3f}",
  "bias 0.8 gait": f"+{b['s=0.8, gait']['mean_diff']:.3f}",
}
v = json.loads((R / "results" / "variability.json").read_text())
w = v["within_session_only"]["peak_loading"]; sb = v["split_half"]["B_with_tau_from_A"]["peak_loading"]; sa = v["split_half"]["A_with_tau_from_B"]["peak_loading"]
ar = v["alert_rates"]; pct = lambda x: f"{100*x:.1f}%"
checks.update({
  "std SD within": f"{w['sd_of_standardised_change']:.2f}", "beyond within": pct(w["share_beyond_up"] + w["share_beyond_down"]),
  "up": pct(w["share_beyond_up"]), "down": pct(w["share_beyond_down"]),
  "session SE": f"{w['mean_within_session_se']:.3f}", "tau": f"{v['extra_between_session_sd_full_cohort']['peak_loading']:.3f}",
  "tau A": f"{v['split_half']['tau_from_A']['peak_loading']:.3f}", "tau B": f"{v['split_half']['tau_from_B']['peak_loading']:.3f}",
  "SD corr B": f"{sb['sd_of_standardised_change']:.2f}", "SD corr A": f"{sa['sd_of_standardised_change']:.2f}",
  "beyond corr B": pct(sb["share_beyond_up"] + sb["share_beyond_down"]), "beyond corr A": pct(sa["share_beyond_up"] + sa["share_beyond_down"]),
  "declined orig": pct(ar["original_within_session"]["loading_declined"]), "declined corr": pct(ar["corrected_between_session"]["loading_declined"]),
  "any orig": pct(ar["original_within_session"]["any"]), "any corr": pct(ar["corrected_between_session"]["any"]),
  "repeat orig n": str(v["repeat"]["original"]["declines"]), "repeat orig %": f"{100*v['repeat']['original']['followed_by_another_decline']:.0f}%",
  "repeat corr n": str(v["repeat"]["corrected"]["A"]["declines"] + v["repeat"]["corrected"]["B"]["declines"]),
  "z1.645": pct(v["declined_rate_by_z_corrected"]["1.645"]), "z2.576": pct(v["declined_rate_by_z_corrected"]["2.576"]),
})
gr = json.loads((R / "results" / "group_reliability.json").read_text())
for name in ("Knee", "Ankle", "Hip", "Heel"):
    t = gr["tau_by_group"][name]["peak_loading"]; cb = gr["calibration_by_group"][name]["peak_loading"]
    checks[f"tau {name}"] = f"{t['tau']:.3f} ({t['ci95'][0]:.3f} to {t['ci95'][1]:.3f})"
    checks[f"pooled {name}"] = pct(cb["pooled_threshold_beyond"]); checks[f"group {name}"] = pct(cb["group_threshold_beyond"])
    checks[f"pairs {name}"] = f"{gr['pairs_by_group'][name]:,}"
checks["het p loading"] = f"p = {gr['heterogeneity_permutation_p']['peak_loading']['p']:.2f}"
checks["het p braking"] = f"p = {gr['heterogeneity_permutation_p']['braking_impulse']['p']:.2f}"
checks["het p pushoff"] = f"p = {gr['heterogeneity_permutation_p']['propulsive_impulse']['p']:.3f}"
checks["hip pushoff tau"] = f"{gr['tau_by_group']['Hip']['propulsive_impulse']['tau']:.4f}"
checks["hip pushoff pooled"] = pct(gr["calibration_by_group"]["Hip"]["propulsive_impulse"]["pooled_threshold_beyond"])
checks["knee pushoff pooled"] = pct(gr["calibration_by_group"]["Knee"]["propulsive_impulse"]["pooled_threshold_beyond"])
md = json.loads((R / "results" / "missed_decline.json").read_text())
for d in ("0.05", "0.1", "0.15"):
    checks[f"detect orig {d}"] = pct(md["flagged_at"][d]["original"]); checks[f"detect corr {d}"] = pct(md["flagged_at"][d]["corrected"])
checks["median lim orig"] = f"{md['median_limit_bw']['original']:.3f}"; checks["median lim corr"] = f"{md['median_limit_bw']['corrected']:.3f}"
checks["50% corr"] = f"{md['delta_for_50pct']['corrected']}"; checks["80% orig"] = f"{md['delta_for_80pct']['original']}"
checks["80% corr"] = f"{md['delta_for_80pct']['corrected']}"
checks["between-patient SD"] = f"{md['scale_reference_bw']['sd_first_visit_loading_difference_between_patients']:.3f}"
gmin = min(g["corrected"] for g in md["by_group_at_0.10"].values()); gmax = max(g["corrected"] for g in md["by_group_at_0.10"].values())
checks["group detect range"] = f"{100*gmin:.0f}% to {100*gmax:.0f}%"
ind = json.loads((R / "results" / "independent_reliability.json").read_text())
assert ind["comparison_ok"], "independent re-implementation mismatch"
h = ind["heldout_beyond_loading"]
checks["indep odd"] = pct(h["odd_with_tau_from_even"]); checks["indep even"] = pct(h["even_with_tau_from_odd"])
checks["indep pooled"] = pct(h["pooled"])
rc = json.loads((R / "results" / "reproduce_check.json").read_text())
assert all(abs(a - b) < 1e-9 for a, b in rc.values()), "reproduction mismatch"
bad = {k: v for k, v in checks.items() if v not in rep}
for k, v in checks.items(): print(("OK   " if k not in bad else "MISS ") + k, v)
sys.exit(1 if bad else 0)
