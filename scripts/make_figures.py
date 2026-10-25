"""Figures for the technical report (docs/figures/)."""
import json, sys
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = Path(__file__).parents[1]; sys.path.insert(0, str(ROOT / "src"))
from gaitrec_rehab.monitoring import between_session_sd
from gaitrec_rehab.sessions import cohort_session_table

OUT = ROOT / "docs" / "figures"; OUT.mkdir(parents=True, exist_ok=True)
INK, MUTED, ACC, BAND, AMB = "#17211e", "#5a6661", "#1d5c74", "#d6e4ea", "#b06a00"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
st, _ = cohort_session_table(ROOT / "data"); st = st.sort_values(["SUBJECT_ID", "SESSION_DATE"])
main = json.loads((ROOT / "results" / "main.json").read_text())

# Figure 1: one patient's visits, same-day noise vs corrected threshold
pid = 1079; p = st[st.SUBJECT_ID == pid]
others = st[st.SUBJECT_ID != pid]; tau = between_session_sd(others)["peak_loading"]
x, y, se = p.day.to_numpy(), p.peak_loading.to_numpy(), p.peak_loading_se.to_numpy()
fig, ax = plt.subplots(figsize=(6.4, 2.9))
lo = [np.percentile(others[(others.CLASS_LABEL == "K") & (others.week == w)].peak_loading, 10) for w in p.week]
hi = [np.percentile(others[(others.CLASS_LABEL == "K") & (others.week == w)].peak_loading, 90) for w in p.week]
ax.fill_between(x, lo, hi, color=BAND, label="Knee patients, same week (10th–90th percentile)")
corr = 1.96 * np.sqrt(se[0] ** 2 + se ** 2 + tau ** 2); corr[0] = 0
ax.fill_between(x, y[0] - corr, y[0] + corr, color=AMB, alpha=.12, label="Corrected change threshold around first visit")
ax.axhline(0, color=MUTED, lw=.8, ls=(0, (3, 3)))
ax.errorbar(x, y, yerr=1.96 * se, color=ACC, marker="o", ms=4, lw=1.6, capsize=0, label="Patient: visit value ± same-day noise")
ax.set_xlabel("Days since first assessment"); ax.set_ylabel("Weight accepted, injured − other leg\n(body weights)")
ax.legend(frameon=False, fontsize=7.5, loc="upper left"); ax.set_title(f"Figure 1. Patient #{pid}, knee injury: five visits over 36 days", loc="left", fontsize=9.5, color=INK)
fig.tight_layout(); fig.savefig(OUT / "fig1_patient.png", dpi=200); plt.close(fig)

# Figure 2: standardised visit-to-visit changes, same-day noise only vs corrected (held-out half)
rng = np.random.default_rng(0); ids = st.SUBJECT_ID.unique(); half = set(rng.permutation(ids)[: len(ids) // 2])
A, B = st[st.SUBJECT_ID.isin(half)], st[~st.SUBJECT_ID.isin(half)]
def z(t, tau=0.0):
    g = t.groupby("SUBJECT_ID"); d = g.peak_loading.diff()
    se2 = t.peak_loading_se ** 2 + g.peak_loading_se.shift() ** 2 + tau ** 2
    return (d / np.sqrt(se2)).dropna().to_numpy()
z0 = z(st); zc = z(B, between_session_sd(A)["peak_loading"])
fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.6), sharey=True)
grid = np.linspace(-6, 6, 300); dens = np.exp(-grid ** 2 / 2) / np.sqrt(2 * np.pi)
for ax, zz, ttl in ((axs[0], z0, "Same-day noise only"), (axs[1], zc, "Corrected (estimated on other half)")):
    ax.hist(np.clip(zz, -6, 6), bins=np.linspace(-6, 6, 49), density=True, color=ACC, alpha=.35)
    ax.plot(grid, dens, color=INK, lw=1.2, label="Expected if limit is right")
    for v in (-1.96, 1.96): ax.axvline(v, color=AMB, lw=.9, ls="--")
    out = np.mean(np.abs(zz) > 1.96) * 100
    ax.set_title(f"{ttl}\nSD {zz.std():.2f}; {out:.1f}% beyond ±1.96 (5% expected)", fontsize=8.5, color=INK, loc="left")
    ax.set_xlabel("Change ÷ its estimated uncertainty")
axs[0].set_ylabel("Density"); axs[0].legend(frameon=False, fontsize=7.5)
fig.suptitle(f"Figure 2. Visit-to-visit changes in weight accepted by the injured leg\n(left: all {len(z0):,} pairs; right: the {len(zc):,} pairs of the half not used to estimate the correction)", x=0.01, ha="left", fontsize=9.5, color=INK)
fig.tight_layout(); fig.savefig(OUT / "fig2_variability.png", dpi=200); plt.close(fig)

# Figure 3: prediction results
rows = [("Primary: gait vs characteristics", main["primary_auroc_diff_M2_M0"], main["primary_boot"]["auroc_diff_ci"], True),
        ("Gait vs published score", main["S1_auroc_diff_M2_M1"], main["S1_ci_oof"], False),
        ("Later returns (day-90 landmark)", main["S2"]["auroc_diff"], main["S2"]["ci_oof"], False),
        ("No orthopaedic insole", main["S4"]["auroc_diff"], main["S4"]["boot"]["auroc_diff_ci"], True),
        ("Temporal hold-out, 2014–2017", main["E1"]["auroc_diff"], main["E1"]["ci"], False),
        ("Combined vs published score", main["E2_auroc_diff_M2plus_M1"], main["E2_ci_oof"], False)]
fig, ax = plt.subplots(figsize=(6.4, 2.9))
for i, (lab, est, ci, full) in enumerate(rows[::-1]):
    ax.plot(ci, [i, i], color=ACC if full else MUTED, lw=2 if full else 1.3)
    ax.plot(est, i, "o", color=ACC if full else MUTED, ms=5)
ax.axvline(0, color=INK, lw=.8); ax.axvline(0.05, color=AMB, lw=.9, ls="--")
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows[::-1]])
ax.set_xlabel("Difference in AUROC (95% interval)")
fig.suptitle("Figure 3. Gait features did not improve prediction of a later stay", x=0.01, ha="left", fontsize=9.5, color=INK)
fig.text(0.01, 0.01, "Dashed line: pre-specified meaningful gain (0.05). Dark: full-pipeline bootstrap.\nGrey: resampled fixed predictions (narrower than true uncertainty).", fontsize=7, color=MUTED)
fig.tight_layout(rect=(0, 0.09, 1, 1)); fig.savefig(OUT / "fig3_prediction.png", dpi=200); plt.close(fig)
print("figures written")
