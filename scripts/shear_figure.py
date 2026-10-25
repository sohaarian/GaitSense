"""Figure 5: AI-predicted vs force-plate change in push-off (test, 18-65), and agreement under degradation."""
import json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
R = Path(__file__).parents[1]
p = pd.read_csv(R / "results" / "shear_pairs_test.csv"); p = p[p.age_first.between(18, 65)]
s3 = json.loads((R / "results" / "shear_study3.json").read_text())["conditions"]
fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.9), dpi=200, gridspec_kw={"width_ratios": [1, 1.25]})
a = ax[0]
a.scatter(p.d_push * 1000, p.d_gb_push * 1000, s=6, c="#9aa7b1", lw=0, label="All visit pairs")
h = p[p.hard]
a.scatter(h.d_push * 1000, h.d_gb_push * 1000, s=16, c="#b5523b", lw=0, label="Hard subset (n = %d)" % len(h))
lim = 16; a.plot([-lim, lim], [-lim, lim], color="#555", lw=0.8, ls="--")
a.axhline(0, color="#ccc", lw=0.6); a.axvline(0, color="#ccc", lw=0.6)
a.set_xlim(-lim, lim); a.set_ylim(-lim, lim)
a.set_xlabel("Force-plate change in push-off (×10⁻³ BW·s)"); a.set_ylabel("AI-predicted change (×10⁻³ BW·s)")
a.set_title("A. Real visit-to-visit change", loc="left", fontsize=9); a.legend(frameon=False, fontsize=7, loc="upper left")
b = ax[1]
labels, vals, cols = [], [], []
for c in s3:
    if c["steps_per_leg"] in ("all", 3):
        labels.append(f"{c['fs_hz']} Hz, {'insole err.' if c['insole_error'] == 'insole' else 'clean'}, {'all' if c['steps_per_leg'] == 'all' else '3'} steps")
        vals.append(c["ccc_push_change"]); cols.append("#2f6f8f" if c["insole_error"] == "none" else "#8fb3c7")
y = np.arange(len(vals))[::-1]
b.barh(y, vals, color=cols); b.set_yticks(y); b.set_yticklabels(labels, fontsize=6.5)
b.set_xlabel("Agreement with force-plate change (CCC)"); b.set_xlim(0, 0.5)
b.set_title("B. Simulated wearable input", loc="left", fontsize=9)
for s in ("top", "right"):
    a.spines[s].set_visible(False); b.spines[s].set_visible(False)
fig.suptitle("Figure 5. Reconstructing push-off change from vertical force alone", x=0.01, ha="left", fontsize=10)
fig.tight_layout(); fig.savefig(R / "docs" / "figures" / "fig5_shear.png")
