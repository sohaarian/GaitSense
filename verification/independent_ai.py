"""Clean-room check of the GaitSense AI results (Stage 9).

Imports nothing from src/. Recomputes key numbers from the saved trial/visit tables and prediction files with
independently written code, and compares them with results/ai/*.json. Prints one line per check; exits 1 on any FAIL.
"""
import json, math, sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parents[1]; A = ROOT / "results" / "ai"
fails = []


def check(name, got, want, tol=1e-9):
    ok = (abs(got - want) <= tol * max(1.0, abs(want))) if isinstance(want, (int, float)) else got == want
    print(f"{'OK  ' if ok else 'FAIL'} {name:58s} {got!s:>14.14} {want!s:>14.14}")
    if not ok:
        fails.append(name)


V = pd.read_csv(A / "visit_level_features.csv.gz")
Tr = pd.read_csv(A / "trial_level_features.csv.gz")
part = pd.read_csv(A / "partition.csv")
pr = json.loads((A / "prediction_results.json").read_text())
mon = json.loads((A / "monitoring_results.json").read_text())
sel = json.loads((A / "model_selection.json").read_text())

# 1. partition and counts
check("each patient in exactly one partition", int(part.SUBJECT_ID.duplicated().sum()), 0)
coh = V[V.role == "cohort"]
check("cohort patients", int(coh.SUBJECT_ID.nunique()), 1667)
check("cohort patients with >=2 visits", int((coh.groupby("SUBJECT_ID").size() >= 2).sum()), 1441)
check("consecutive cohort pairs", int((coh.groupby("SUBJECT_ID").size() - 1).sum()), 4203)
for s in ("train", "calib", "test"):
    sh = (part[(part.role == "cohort")].split == s).mean()
    check(f"partition share {s}", round(sh, 2), {"train": 0.6, "calib": 0.2, "test": 0.2}[s], tol=0.011)

# 2. visit summaries recomputed from trials for 200 random sessions
rng = np.random.default_rng(7)
sess = rng.choice(coh.SESSION_ID.to_numpy(), 200, replace=False)
side = coh.set_index("SESSION_ID").affected_side
worst = 0.0
for sid in sess:
    t = Tr[(Tr.SESSION_ID == sid) & Tr.valid]
    a = t[t.side == side[sid]].peak_v.to_numpy(); u = t[t.side != side[sid]].peak_v.to_numpy()
    d = a.mean() - u.mean(); se = math.sqrt(a.var(ddof=1) / len(a) + u.var(ddof=1) / len(u))
    row = coh[coh.SESSION_ID == sid].iloc[0]
    worst = max(worst, abs(d - row.d_peak_v), abs(se - row.d_peak_v_se))
check("visit difference and SE recomputed (200 sessions, max abs diff)", worst, 0.0, tol=1e-12)

# 3. last-value prediction and test metrics recomputed from predictions file
D = pd.read_csv(A / "predictions_test.csv.gz")
Dp = D[(D.target == "d_peak_v") & (D.split == "test")]
prev = coh.sort_values(["SUBJECT_ID", "visit_no"]).groupby("SUBJECT_ID").d_peak_v.shift()
lastval = pd.Series(prev.to_numpy(), index=coh.sort_values(["SUBJECT_ID", "visit_no"]).SESSION_ID.to_numpy())
m0 = Dp[Dp.model == "M0a"]
check("M0a mean equals previous visit value (max abs diff)", float(np.max(np.abs(m0.mu.to_numpy() - lastval.loc[m0.SESSION_ID].to_numpy()))), 0.0, tol=1e-12)
erf = np.vectorize(math.erf)
Phi = lambda z: 0.5 * (1 + erf(z / math.sqrt(2))); phi = lambda z: np.exp(-z * z / 2) / math.sqrt(2 * math.pi)
for m in ("M0a", "M0b", "M1", "M2", "M3", "M4"):
    x = Dp[Dp.model == m]; z = (x.y - x.mu) / x.sd
    crps = float(np.mean(x.sd * (z * (2 * Phi(z) - 1) + 2 * phi(z) - 1 / math.sqrt(math.pi))))
    mae = float(np.mean(np.abs(x.y - x.mu))); cov = float(np.mean(np.abs(z) <= 1.959963984540054))
    R = pr["targets"]["d_peak_v"]["test"]["models"][m]
    check(f"{m} test MAE", mae, R["MAE"], 1e-9); check(f"{m} test CRPS", crps, R["CRPS"], 1e-7); check(f"{m} test cov95", cov, R["cov95"], 1e-9)
check("test targets (primary)", int(len(m0)), pr["targets"]["d_peak_v"]["test"]["n_targets"])
check("no calibration/training rows in test file", sorted(D.split.unique()), ["temporal", "test"])

# 4. tau from training consecutive pairs (method of moments), independently
tr = coh[coh.split == "train"].sort_values(["SUBJECT_ID", "visit_no"])
g = tr.groupby("SUBJECT_ID")
dd = g.d_peak_v.diff(); ss = tr.d_peak_v_se ** 2 + g.d_peak_v_se.shift() ** 2; ok = dd.notna()
tau = math.sqrt(max(dd[ok].var(ddof=1) - ss[ok].mean(), 0))
check("tau (training, primary target)", tau, sel["targets"]["d_peak_v"]["tau_train"], 1e-9)

# 5. monitoring: architecture C alert rate and C-D kappa on test pairs, recomputed
S = pd.read_csv(A / "monitoring_pairs.csv.gz")
zc = (S.Y_cur - S.Y_prev) / np.sqrt(S.se_cur ** 2 + S.se_prev ** 2 + tau ** 2)
check("architecture C native alert rate (test)", float((np.abs(zc) > 1.959963984540054).mean()), mon["native"]["alert_rate"]["C"]["estimate"], 1e-9)
zc0 = (S.Y_cur - S.Y_prev) / np.sqrt(S.se_cur ** 2 + S.se_prev ** 2)
check("original same-day limit crossing rate (test)", float((np.abs(zc0) > 1.959963984540054).mean()), mon["native"]["alert_rate"]["C0_original"]["estimate"], 1e-9)
a, b = (np.abs(zc) > 1.959963984540054).to_numpy(), S.alert_D.to_numpy(bool)
po = np.mean(a == b); pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
check("kappa C-D native (test)", float((po - pe) / (1 - pe)), mon["native"]["pairwise"]["C-D"]["kappa"]["estimate"], 1e-9)
hc = V[(V.role == "hc") & (V.split == "train") & (V.SPEED == 2)]
check("healthy reference mean (train controls, speed 2)", float(hc.d_peak_v.mean()), mon["healthy_reference"]["mu"], 1e-9)

# 6. order of work: model choices written before the test partition was opened
opened = json.loads((A / "test_opened.json").read_text())["opened"]
check("model selection finished before test opened", sel["finished"] < opened, True)
print(f"\n{len(fails)} failures" if fails else "\nall checks passed")
sys.exit(1 if fails else 0)
