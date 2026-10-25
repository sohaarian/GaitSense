# Capability 1 in v3.0 — status and evidence-based update

**Unchanged and fully supported:** the v2.0 forecasters (last value, ridge, mixed model, GP, boosted trees),
monitoring architectures A–D, perturbation and injection experiments, prototype, and every v2.0 result in
`results/ai/` (experiment `gaitsense-ai-2.0.0-065bf3843746`). `gaitsense_platform.capability1.load_v2()` exposes the
frozen models without refitting; `BiomechForecaster` applies the same model family to any dataset that meets the
biomechanical data contract.

## Post-hoc update: conformal recalibration of GP intervals

**Why.** v2.0 found that GP errors were heavier-tailed than a Gaussian assumes: on test patients, 50% intervals
covered 56.4% and 95% intervals covered 94.6% (primary target).

**What.** Split-conformal quantiles of |z| fitted on the 858 calibration predictions, applied unchanged to test and
temporal patients (`scripts/v3/c1_conformal.py`, `results/v3/c1_conformal.json`). Post hoc, because the test set was
opened in v2.0.

**Result (primary target, test, 816 predictions).**

| Nominal | Gaussian coverage | Conformal coverage | Gaussian mean width (BW) | Conformal mean width (BW) |
|---|---|---|---|---|
| 50% | 56.4% | 52.7% | 0.045 | 0.041 |
| 80% | 82.7% | 81.7% | 0.086 | 0.084 |
| 90% | 90.7% | 92.3% | 0.110 | 0.115 |
| 95% | 94.6% | 96.3% | 0.131 | 0.152 |

95% interval score: Gaussian 0.192, conformal 0.194 (lower is better). The temporal set showed the same pattern
(50%: 57.9% → 53.5%; 95%: 94.5% → 97.0%; interval score 0.187 → 0.192).

**Decision.** Conformal quantiles improve calibration of the narrower ranges (50–80%) but over-cover and widen the
95% range, with a slightly worse interval score. The evidence does not justify replacing the Gaussian 95% interval.
`ConformalCalibrator` is available as an option for narrower ranges; the default 95% interval stays Gaussian.
