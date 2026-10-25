"""Figures 1-8 for the GaitSense AI manuscript, generated only from results/ai outputs. Writes docs/figures/ai_fig*.png."""
import json, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).parents[2]; A = ROOT / "results" / "ai"; OUT = ROOT / "docs" / "figures"; OUT.mkdir(exist_ok=True)
# reference categorical palette (fixed order) and ink tokens
SER = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
MODELS = ["M0a", "M0b", "M1", "M2", "M3", "M4"]
MLAB = {"M0a": "M0a last value\n(same-day noise)", "M0b": "M0b last value\n(+τ)", "M1": "M1 ridge",
        "M2": "M2 mixed model", "M3": "M3 Gaussian\nprocess", "M4": "M4 boosted trees"}
MCOL = dict(zip(MODELS, SER))
MSHORT = {"M0a": "M0a\nlast value", "M0b": "M0b\nlast value+τ", "M1": "M1\nridge", "M2": "M2\nmixed", "M3": "M3\nGP", "M4": "M4\ntrees"}
ARCH = ["A", "B", "C", "D"]; ACOL = dict(zip(ARCH, SER[:4]))
ALAB = {"A": "A raw asymmetry", "B": "B healthy reference", "C": "C change limit", "D": "D predictive (GP)"}
CLS = {"A": "Ankle", "K": "Knee", "H": "Hip", "C": "Heel"}
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "figure.facecolor": SURF,
                     "axes.facecolor": SURF, "savefig.facecolor": SURF, "legend.frameon": False, "lines.linewidth": 2})
pr = json.loads((A / "prediction_results.json").read_text())
mon = json.loads((A / "monitoring_results.json").read_text())
pert = json.loads((A / "perturbation_results.json").read_text())
inj = json.loads((A / "injection_results.json").read_text())
s3 = json.loads((A / "stage3_summary.json").read_text())
V = pd.read_csv(A / "visit_level_features.csv.gz")
D = pd.read_csv(A / "predictions_test.csv.gz")
T = pr["targets"]["d_peak_v"]["test"]


def save(fig, name):
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight"); plt.close(fig); print("wrote", name)


# ---- Figure 1: cohort and longitudinal data availability
fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
flow = [("GaitRec patients", 2084), ("First-stay, unilateral,\nnormal shoes", 1759), ("Cohort (365-day window)", 1667),
        ("≥ 2 usable visits", s3["cohort_patients_with_2plus_visits"])]
y = np.arange(len(flow))[::-1]
ax[0].barh(y, [f[1] for f in flow], color=SER[0], height=0.55)
for yi, (lab, n) in zip(y, flow):
    ax[0].text(n + 30, yi, f"{n:,}", va="center", color=INK, fontsize=8)
ax[0].set_yticks(y, [f[0] for f in flow]); ax[0].set_xlim(0, 2500); ax[0].set_xlabel("Patients"); ax[0].grid(axis="y", visible=False)
ax[0].set_title("a  Patient selection", loc="left", color=INK)
pc = V[V.role == "cohort"].groupby("SUBJECT_ID").size()
cnt = pc.value_counts().sort_index()
ax[1].bar(cnt.index, cnt.values, color=SER[0], width=0.8)
ax[1].set_xticks(range(1, 13)); ax[1].set_xlabel("Usable first-stay visits per patient"); ax[1].set_ylabel("Patients"); ax[1].grid(axis="x", visible=False)
ax[1].set_title("b  Visits per patient", loc="left", color=INK)
P = pd.read_csv(A / "eligible_visit_pairs.csv.gz"); g = P[P.role == "cohort"].gap_days.clip(upper=42)
ax[2].hist(g, bins=np.arange(-0.5, 43.5, 1), color=SER[0])
ax[2].set_xlabel("Days between consecutive visits (≥42 pooled)"); ax[2].set_ylabel("Visit pairs"); ax[2].grid(axis="x", visible=False)
ax[2].set_title(f"c  Irregular spacing ({len(P[P.role=='cohort']):,} pairs)", loc="left", color=INK)
fig.tight_layout(); save(fig, "ai_fig1_cohort.png")

# ---- Figure 2: representative test-patient trajectories with M3 one-step predictions
d3 = D[(D.model == "M3") & (D.target == "d_peak_v") & (D.split == "test")]
fig, ax = plt.subplots(1, 4, figsize=(12, 3.0), sharey=True)
for i, c in enumerate("AKHC"):
    vt = V[(V.role == "cohort") & (V.split == "test") & (V.CLASS_LABEL == c)]
    pid = vt.groupby("SUBJECT_ID").size().sort_values(ascending=False).index[min(2, len(vt) - 1)]
    pv = vt[vt.SUBJECT_ID == pid].sort_values("visit_no"); pp = d3[d3.SUBJECT_ID == pid].sort_values("k")
    days = pv.set_index("visit_no").day
    ax[i].errorbar(pv.day, pv.d_peak_v, yerr=1.96 * pv.d_peak_v_se, fmt="o", ms=4, color=INK, elinewidth=1, capsize=0,
                   label="Observed (95% within-visit)")
    xd = pp.k.map(days)
    ax[i].fill_between(xd, pp.mu - 1.96 * pp.sd, pp.mu + 1.96 * pp.sd, color=SER[4], alpha=0.25, lw=0, label="M3 95% predictive interval")
    ax[i].plot(xd, pp.mu, "s", ms=4, color=SER[4], label="M3 prediction from earlier visits")
    ax[i].axhline(0, color=INK2, lw=0.8)
    ax[i].set_title(f"{CLS[c]} (test patient)", loc="left", color=INK); ax[i].set_xlabel("Days since first visit")
ax[0].set_ylabel("Affected − unaffected peak force (BW)")
ax[0].legend(loc="lower right", fontsize=7)
fig.tight_layout(); save(fig, "ai_fig2_trajectories.png")

# ---- Figure 3: prediction performance (test, primary target)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
for j, (met, lab) in enumerate((("MAE", "Mean absolute error (BW)"), ("CRPS", "CRPS (BW, lower is better)"))):
    est = [T["models"][m][met] for m in MODELS]; lo = [T["models"][m]["ci95"][met][0] for m in MODELS]
    hi = [T["models"][m]["ci95"][met][1] for m in MODELS]
    x = np.arange(len(MODELS))
    ax[j].bar(x, est, color=[MCOL[m] for m in MODELS], width=0.6)
    ax[j].errorbar(x, est, yerr=[np.subtract(est, lo), np.subtract(hi, est)], fmt="none", ecolor=INK, elinewidth=1)
    ax[j].set_xticks(x, [MSHORT[m] for m in MODELS], fontsize=7.5); ax[j].set_ylabel(lab); ax[j].grid(axis="x", visible=False)
    ax[j].set_ylim(min(lo) * 0.9, max(hi) * 1.05)
ax[0].set_title(f"a  Point accuracy ({T['n_targets']} test predictions, {T['n_patients']} patients)", loc="left", color=INK)
ax[1].set_title("b  Probabilistic accuracy", loc="left", color=INK)
fig.text(0.01, -0.03, "Bars: mean; whiskers: 95% patient-clustered bootstrap interval. Y axes do not start at zero.", color=INK2, fontsize=7)
fig.tight_layout(); save(fig, "ai_fig3_prediction.png")

# ---- Figure 4: calibration (empirical vs nominal coverage)
fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
nom = [0.5, 0.8, 0.9, 0.95]
for m in MODELS:
    emp = [T["models"][m][f"cov{int(n*100)}"] for n in nom]
    ax[0].plot(nom, emp, "-o", ms=4, color=MCOL[m], label=MLAB[m].replace("\n", " "))
ax[0].plot([0.45, 1], [0.45, 1], color=INK2, lw=1, ls="--"); ax[0].set_xlabel("Nominal coverage"); ax[0].set_ylabel("Empirical coverage (test)")
ax[0].set_title("a  Interval calibration, primary target", loc="left", color=INK); ax[0].legend(fontsize=6.5, loc="upper left")
tg = ["d_peak_v", "d_braking", "d_propulsive", "d_stance_s", "peak_v_a"]
x = np.arange(len(tg)); w = 0.13
for i, m in enumerate(MODELS):
    ax[1].bar(x + (i - 2.5) * w, [pr["targets"][t]["test"]["models"][m]["cov95"] for t in tg], width=w * 0.9, color=MCOL[m])
ax[1].axhline(0.95, color=INK2, ls="--", lw=1); ax[1].set_ylim(0.6, 1.0)
ax[1].set_xticks(x, ["Δ peak", "Δ braking", "Δ propulsive", "Δ stance", "affected peak"], fontsize=7)
ax[1].set_ylabel("95% interval coverage (test)"); ax[1].set_title("b  95% coverage by target", loc="left", color=INK); ax[1].grid(axis="x", visible=False)
ax[1].legend([plt.Rectangle((0, 0), 1, 1, color=MCOL[m]) for m in MODELS], [m for m in MODELS], fontsize=6.5, ncol=3, loc="lower right")
fig.tight_layout(); save(fig, "ai_fig4_calibration.png")

# ---- Figure 5: monitoring disagreement
fig, ax = plt.subplots(1, 3, figsize=(12, 3.4), gridspec_kw={"width_ratios": [1.1, 1, 1]})
x = np.arange(4)
nat = [mon["native"]["alert_rate"][a]["estimate"] for a in ARCH]; mat = [mon["matched"]["alert_rate"][a]["estimate"] for a in ARCH]
ax[0].bar(x - 0.18, nat, 0.34, color=[ACOL[a] for a in ARCH]); ax[0].bar(x + 0.18, mat, 0.34, color=[ACOL[a] for a in ARCH], hatch="///", edgecolor=SURF)
ax[0].set_xticks(x, [ALAB[a].replace(" ", "\n", 1) for a in ARCH], fontsize=7); ax[0].set_ylabel("Share of test visit pairs alerting")
ax[0].set_title("a  Alert rates: native rule (solid)\n    vs matched 10% (hatched)", loc="left", color=INK); ax[0].grid(axis="x", visible=False)
for j, mode in enumerate(("native", "matched")):
    K = np.full((4, 4), np.nan)
    for k, v in mon[mode]["pairwise"].items():
        a, b = k.split("-"); i1, i2 = ARCH.index(a), ARCH.index(b); K[i1, i2] = K[i2, i1] = v["kappa"]["estimate"]
    np.fill_diagonal(K, 1)
    im = ax[j + 1].imshow(K, vmin=-0.1, vmax=1, cmap="Blues")
    for i1 in range(4):
        for i2 in range(4):
            ax[j + 1].text(i2, i1, f"{K[i1, i2]:.2f}", ha="center", va="center", fontsize=8, color=INK if K[i1, i2] < 0.6 else "white")
    ax[j + 1].set_xticks(range(4), ARCH); ax[j + 1].set_yticks(range(4), ARCH); ax[j + 1].grid(False)
    ax[j + 1].set_title(f"{'bc'[j]}  Cohen's κ, {mode} {'rules' if mode=='native' else 'alert rate'}", loc="left", color=INK)
fig.colorbar(im, ax=ax[2], fraction=0.046)
fig.tight_layout(); save(fig, "ai_fig5_disagreement.png")

# ---- Figure 6: perturbation robustness
E = pert["experiments"]
fig, ax = plt.subplots(1, 3, figsize=(12, 3.3), sharey=True)
for j, (fam, keys, xl) in enumerate((("Force offset", ["offset_-2sw", "offset_-1sw", "offset_-0.5sw", "offset_+0.5sw", "offset_+1sw", "offset_+2sw"], ["−2", "−1", "−½", "+½", "+1", "+2"]),
                                     ("Amplitude scaling", ["scale_-2cv", "scale_-1cv", "scale_-0.5cv", "scale_+0.5cv", "scale_+1cv", "scale_+2cv"], ["−2", "−1", "−½", "+½", "+1", "+2"]),
                                     ("Processing / trials", ["trim_1samples", "trim_2samples", "trim_4samples", "timenorm_linear101", "timenorm_cubic101", "timenorm_linear51", "trials_drop1", "trials_drop2", "trials_keep3"],
                                      ["trim 1", "trim 2", "trim 4", "lin 101", "cub 101", "lin 51", "−1 trial", "−2 trials", "keep 3"]))):
    xx = np.arange(len(keys))
    for a in ARCH:
        ax[j].plot(xx, [E[k][a]["flip_rate"]["estimate"] for k in keys], "-o", ms=4, color=ACOL[a], label=ALAB[a])
    ax[j].set_xticks(xx, xl, fontsize=7, rotation=0 if j < 2 else 45); ax[j].set_title(f"{'abc'[j]}  {fam}", loc="left", color=INK)
ax[0].set_xlabel("Offset (× median within-visit SD)"); ax[1].set_xlabel("Scale change (× median within-visit CV)")
ax[0].set_ylabel("Alert flip rate (test pairs)"); ax[0].legend(fontsize=7)
fig.tight_layout(); save(fig, "ai_fig6_perturbation.png")

# ---- Figure 7: injected-change detection and the detection-stability trade-off
X = inj["experiments"]; dl = [-0.15, -0.10, -0.05, -0.02]
fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
for a in ARCH:
    ax[0].plot([-d * 100 for d in dl], [X[f"vertical_{d:+.2f}"]["Y"][a] for d in dl], "-o", ms=4, color=ACOL[a], label=f"{ALAB[a]} (baseline {inj['baseline']['Y_' + a + ('_any' if a == 'B' else '_down')]:.0%})")
ax[0].set_xlabel("Injected affected-leg vertical reduction (%)"); ax[0].set_ylabel("Share of test pairs detected")
ax[0].set_title("a  Vertical reduction (later visit)", loc="left", color=INK); ax[0].legend(fontsize=7)
for k, (t, lab) in enumerate((("propulsive", "Propulsive reduction"), ("braking", "Braking increase"))):
    tt = "d_propulsive" if t == "propulsive" else "d_braking"; dd = [0.05, 0.10, 0.20, 0.30]
    for a, key in (("A", "A_LSI10"), ("C", "C"), ("D", "D")):
        ax[1].plot([d * 100 for d in dd], [X[f"{t}_{d:+.2f}"][tt][key] for d in dd], "-o" if k == 0 else "--s", ms=4, color=ACOL[a],
                   label=f"{ALAB[a].split(' ')[0]} {lab.lower()} (baseline {inj['baseline'][tt + '_' + ('A_LSI10' if a == 'A' else a) + ('_down' if t == 'propulsive' else '_up')]:.0%})")
ax[1].set_xlabel("Injected AP change (%)"); ax[1].set_title("b  Impulse changes", loc="left", color=INK); ax[1].legend(fontsize=6)
for a, mk in (("C", "o"), ("D", "s")):
    zs = sorted(inj["tradeoff"], key=float)
    bx = [inj["tradeoff"][z][a]["baseline_any"] for z in zs]
    for dkey, ls in (("detect_-0.05", "-"), ("detect_-0.10", ":")):
        ax[2].plot(bx, [inj["tradeoff"][z][a][dkey] for z in zs], ls, marker=mk, ms=4, color=ACOL[a], label=f"{a}, {dkey[-5:]} injected")
ax[2].set_xlabel("Alert rate on unmodified pairs (not false positives)"); ax[2].set_ylabel("Detection of injected reduction")
ax[2].set_title("c  Trade-off as z varies (1.645–3.0)", loc="left", color=INK); ax[2].legend(fontsize=6)
fig.tight_layout(); save(fig, "ai_fig7_detection.png")

# ---- Figure 8: injury-group comparison
br = T["breakdowns"]["CLASS_LABEL"]
fig, ax = plt.subplots(1, 3, figsize=(12, 3.3))
x = np.arange(4); w = 0.13
for j, met in enumerate(("MAE", "CRPS", "cov95")):
    for i, m in enumerate(MODELS):
        ax[j].bar(x + (i - 2.5) * w, [br[c][m][met] for c in "AKHC"], width=w * 0.9, color=MCOL[m], label=MLAB[m].replace("\n", " "))
    ax[j].set_xticks(x, [f"{CLS[c]}\n(n={int(br[c]['M3']['n'])})" for c in "AKHC"], fontsize=7); ax[j].grid(axis="x", visible=False)
ax[0].set_ylabel("MAE (BW)"); ax[1].set_ylabel("CRPS (BW)"); ax[2].set_ylabel("95% coverage"); ax[2].axhline(0.95, color=INK2, ls="--", lw=1)
ax[2].set_ylim(0.6, 1.0)
fig.legend(*ax[0].get_legend_handles_labels(), loc="lower center", ncol=6, fontsize=7, bbox_to_anchor=(0.5, -0.08))
for j, t in enumerate(("a  Point error", "b  Probabilistic error", "c  95% interval coverage")):
    ax[j].set_title(t, loc="left", color=INK)
fig.tight_layout(); save(fig, "ai_fig8_injury_groups.png")
