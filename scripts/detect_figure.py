"""Figure 6: direct detection of push-off change (development CV ROC; sensitivity at 5% FPR, CV and hold-out)."""
import json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve
R = Path(__file__).parents[1]
d = pd.read_csv(R / "results" / "detect_cv_pairs.csv")
cv = json.loads((R / "results" / "detect_cv.json").read_text()); ho = json.loads((R / "results" / "detect_holdout.json").read_text())
M = ["V-thr", "R-gb", "R-cnn", "P-lin", "P-gb", "P-cnn"]
col = {"V-thr": "#9aa7b1", "R-gb": "#2f6f8f", "R-cnn": "#6fa3c0", "P-lin": "#d9a46b", "P-gb": "#b5523b", "P-cnn": "#7a2f1f"}
fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.9), dpi=200, gridspec_kw={"width_ratios": [1, 1.2]})
a = ax[0]
for m in ["V-thr", "R-gb", "P-gb", "P-cnn"]:
    f, t, _ = roc_curve(d.label, d["score_" + m]); a.plot(f, t, color=col[m], lw=1.6, label=f"{m} (AUC {cv['cv'][m]['roc_auc']:.2f})")
a.plot([0, 1], [0, 1], color="#ccc", lw=0.8, ls="--"); a.axvline(0.05, color="#888", lw=0.7, ls=":")
a.set_xlabel("False-positive rate"); a.set_ylabel("Sensitivity (true push-off changes)")
a.set_title("A. Development set, 5-fold CV (243 changes)", loc="left", fontsize=9); a.legend(frameon=False, fontsize=7, loc="lower right")
b = ax[1]; x = np.arange(len(M)); w = 0.38
s_cv = [cv["cv"][m]["sensitivity"] for m in M]; ci_cv = np.array([cv["cv_bootstrap"]["sensitivity_ci95"][m] for m in M])
s_ho = [ho["holdout"][m]["sensitivity"] for m in M]; ci_ho = np.array([ho["holdout_bootstrap"]["sensitivity_ci95"][m] for m in M])
b.bar(x - w / 2, s_cv, w, color=[col[m] for m in M], yerr=[np.array(s_cv) - ci_cv[:, 0], ci_cv[:, 1] - np.array(s_cv)], capsize=2, label="Development CV")
b.bar(x + w / 2, s_ho, w, color=[col[m] for m in M], alpha=0.35, yerr=[np.array(s_ho) - ci_ho[:, 0], ci_ho[:, 1] - np.array(s_ho)], capsize=2,
      label="Temporal hold-out (12 changes)")
b.set_xticks(x); b.set_xticklabels(M, fontsize=7.5); b.set_ylabel("Sensitivity at 5% false-positive rate"); b.set_ylim(0, 0.6)
from matplotlib.patches import Patch
b.set_title("B. Sensitivity with 95% intervals", loc="left", fontsize=9)
b.legend(handles=[Patch(color="#555"), Patch(color="#555", alpha=0.35)], labels=["Development CV (solid)", "Temporal hold-out, 12 changes (pale)"],
         frameon=False, fontsize=7, loc="upper left")
for s in ("top", "right"):
    a.spines[s].set_visible(False); b.spines[s].set_visible(False)
fig.suptitle("Figure 6. Detecting force-plate-defined push-off change from vertical force alone", x=0.01, ha="left", fontsize=10)
fig.tight_layout(); fig.savefig(R / "docs" / "figures" / "fig6_detection.png")
