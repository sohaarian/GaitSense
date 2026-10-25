"""Riley et al. (BMJ 2020;368:m441) criteria for a binary-outcome prediction model.

Inputs: n patients, observed event proportion, assumed Cox-Snell R^2 as a fraction
of its maximum. Output: largest number of predictor parameters meeting each criterion.
"""
import math

def max_r2cs(phi):
    ln_l0 = phi * math.log(phi) + (1 - phi) * math.log(1 - phi)   # null log-lik per patient
    return 1 - math.exp(2 * ln_l0)

def max_params(n, phi, frac, shrink=0.9, delta=0.05):
    mr = max_r2cs(phi)
    r2 = frac * mr
    # Criterion 1: expected uniform shrinkage >= 0.9
    p1 = n * (shrink - 1) * math.log(1 - r2 / shrink)
    # Criterion 2: optimism in Nagelkerke R^2 <= delta
    s2 = r2 / (r2 + delta * mr)
    p2 = n * (s2 - 1) * math.log(1 - r2 / s2)
    # Criterion 3: overall risk estimated within +/- 0.05
    n3 = (1.96 / 0.05) ** 2 * phi * (1 - phi)
    return mr, r2, p1, p2, n3

if __name__ == "__main__":
    for label, n, events in [("primary", 1667, 84), ("late-return", 1643, 60), ("landmark", 1073, 53)]:
        phi = events / n
        print(f"\n{label}: n={n}, events={events}, phi={phi:.4f}")
        for frac in (0.10, 0.15, 0.20):
            mr, r2, p1, p2, n3 = max_params(n, phi, frac)
            print(f"  R2cs={r2:.4f} ({frac:.0%} of max {mr:.3f}): "
                  f"crit1 max params={p1:.1f}, crit2 max params={p2:.1f}, crit3 needs n>={n3:.0f}")
