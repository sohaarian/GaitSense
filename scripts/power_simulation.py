"""Preliminary precision/power simulation for the paired M2 vs M0 AUROC comparison.

Synthetic cohorts match the audited primary cohort: n = 1,667, ~84 events (5.0%).
M0: 6 patient-characteristic predictors. M2: M0 + 3 gait features.
Each cohort is analysed with the planned procedure: 5-fold patient-level CV,
ridge logistic regression with the penalty tuned by inner 5-fold CV, then a paired
DeLong test on the pooled out-of-fold predictions.
Scenarios set the TRUE AUROC of M0 and the TRUE gain from the gait features.
A second block injects non-differential and differential missed events.
"""
import sys
import numpy as np
from scipy.stats import norm
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

N, PREV = 1667, 84 / 1667

def delong_paired(y, a, b):
    pos, neg = a[y == 1], a[y == 0]
    pos2, neg2 = b[y == 1], b[y == 0]
    def comps(p, q):
        v10 = (p[:, None] > q[None, :]).mean(1) + 0.5 * (p[:, None] == q[None, :]).mean(1)
        v01 = (p[:, None] > q[None, :]).mean(0) + 0.5 * (p[:, None] == q[None, :]).mean(0)
        return v10, v01
    a10, a01 = comps(pos, neg); b10, b01 = comps(pos2, neg2)
    s10 = np.cov(np.vstack([a10, b10])); s01 = np.cov(np.vstack([a01, b01]))
    S = s10 / len(pos) + s01 / len(neg)
    d = a10.mean() - b10.mean()
    se = np.sqrt(S[0, 0] + S[1, 1] - 2 * S[0, 1])
    return d, se

def make(rng, b0, bg, n=None):
    n = n or N
    X0 = rng.standard_normal((n, 6)); G = rng.standard_normal((n, 3))
    lp = X0 @ np.full(6, b0) + G @ np.full(3, bg)
    lo, hi = -10, 5
    for _ in range(50):  # intercept giving ~5% prevalence
        c = (lo + hi) / 2
        if (1 / (1 + np.exp(-(c + lp)))).mean() > PREV: hi = c
        else: lo = c
    y = rng.random(n) < 1 / (1 + np.exp(-(c + lp)))
    return X0, G, y.astype(int), lp, X0 @ np.full(6, b0)

def oof(X, y, seed):
    out = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        m = LogisticRegressionCV(Cs=8, cv=5, penalty="l2", scoring="neg_log_loss", max_iter=2000)
        m.fit(X[tr], y[tr]); out[te] = m.predict_proba(X[te])[:, 1]
    return out

def calibrate(target0, gain, rng, big=200_000):
    """Coefficient sizes giving the target TRUE AUROCs, set on one large sample."""
    X0, G, _, _, _ = make(rng, 0.1, 0.0, big)
    def aucs(b0, bg):
        r = np.random.default_rng(7)
        X0, G, y, lp, lp0 = make(r, b0, bg, big)
        return roc_auc_score(y, lp0), roc_auc_score(y, lp)
    lo, hi = 0.01, 1.0
    for _ in range(25):
        b0 = (lo + hi) / 2
        if aucs(b0, 0.0)[0] < target0: lo = b0
        else: hi = b0
    if gain == 0: return b0, 0.0
    lo, hi = 0.0, 1.0
    for _ in range(25):
        bg = (lo + hi) / 2
        a0, a2 = aucs(b0, bg)
        if a2 - a0 < gain: lo = bg
        else: hi = bg
    return b0, bg

def run(target0, gain, sims, rng, miss=None):
    hits, diffs = 0, []
    b0, bg = calibrate(target0, gain, rng)
    for s in range(sims):
        X0, G, y, lp, lp0 = make(rng, b0, bg)
        if miss is not None:  # drop events from observation
            rate, differential = miss
            p = np.full(N, rate)
            if differential:  # higher-gait-risk patients more likely to be missed
                z = G.sum(1); p = np.clip(rate * 2 * (z > np.median(z)), 0, 1)
            y = np.where((y == 1) & (rng.random(N) < p), 0, y)
        X2 = np.hstack([X0, G])
        p0, p2 = oof(X0, y, s), oof(X2, y, s)
        d, se = delong_paired(y, p2, p0)
        diffs.append(d); hits += (d / se) > norm.ppf(0.975)
    diffs = np.array(diffs)
    return hits / sims, diffs.mean(), np.percentile(diffs, [2.5, 97.5])

SCENARIOS = [(a0, g, None) for a0 in (0.65, 0.70) for g in (0.0, 0.03, 0.05, 0.08)] + \
            [(0.65, 0.05, m) for m in ((0.2, False), (0.4, False), (0.2, True), (0.4, True))]

if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore")
    sims = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    part, parts = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (0, 1)
    for k, (a0, gain, miss) in enumerate(SCENARIOS):
        if k % parts != part: continue
        rng = np.random.default_rng(2026 + k)
        pw, md, ci = run(a0, gain, sims, rng, miss)
        tag = "none" if miss is None else f"{'differential' if miss[1] else 'random'} {miss[0]:.0%}"
        print(f"{k}\tM0 {a0:.2f}\tgain {gain:.2f}\tmissed {tag}\tpower {pw:.2f}\t"
              f"mean est {md:+.3f}\t95% range [{ci[0]:+.3f}, {ci[1]:+.3f}]", flush=True)
