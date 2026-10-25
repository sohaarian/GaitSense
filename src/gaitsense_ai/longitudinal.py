"""Longitudinal Gaussian models for irregularly timed visits: linear mixed model (M2) and Gaussian process (M3).

Both are the same probabilistic model with different patient-level covariance functions:

    y_ij = x_ij' beta + f_i(t_ij) + e_ij,     f_i ~ GP(0, k_theta),     e_ij ~ N(0, sigma_w^2 + SE_ij^2)

* SE_ij is the *known* within-visit standard error of the visit mean (trial-to-trial variation).
* sigma_w^2 is extra visit-level variation that is not explained by the kernel. It is not identifiable as pure
  measurement error: it absorbs any genuine visit-to-visit change faster than the kernel can represent.
* M2 (LMM): k = s0^2 + rho*s0*s1*(t+t') + s1^2*t*t'   (random intercept and slope), or k = s0^2 (intercept only).
* M3 (GP): k = s_c^2 + s_m^2 * Matern_nu(|t-t'|/l)  [+ s_l^2 t t'], nu in {1/2, 3/2, 5/2, inf (RBF)}.

Hyperparameters theta, sigma_w are fitted by maximum marginal likelihood pooled over training patients, with beta
profiled out by generalised least squares. Prediction for visit k uses only visits 1..k-1 of the same patient.
Time unit: months (days / 30.44) since the patient's first included visit.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

MONTH = 30.44
KERNELS = ["lmm_int", "lmm_slope", "matern12", "matern32", "matern52", "rbf", "matern32_lin"]


def _matern(r, nu):
    if nu == 0.5:
        return np.exp(-r)
    if nu == 1.5:
        a = np.sqrt(3.0) * r; return (1 + a) * np.exp(-a)
    if nu == 2.5:
        a = np.sqrt(5.0) * r; return (1 + a + a * a / 3.0) * np.exp(-a)
    return np.exp(-0.5 * r * r)          # nu = inf: RBF


NU = {"matern12": 0.5, "matern32": 1.5, "matern52": 2.5, "rbf": np.inf, "matern32_lin": 1.5}


def n_params(kernel):
    return {"lmm_int": 1, "lmm_slope": 3, "matern32_lin": 4}.get(kernel, 3) + 1   # + log sigma_w


def kernel_matrix(kernel, p, t1, t2):
    """Covariance between time vectors t1 [..., n] and t2 [..., m] (broadcast over leading batch dims).
    p: unconstrained parameter vector (log scales, atanh correlation)."""
    T1, T2 = t1[..., :, None], t2[..., None, :]
    if kernel == "lmm_int":
        return np.exp(2 * p[0]) * np.ones(np.broadcast_shapes(T1.shape, T2.shape))
    if kernel == "lmm_slope":
        s0, s1, rho = np.exp(p[0]), np.exp(p[1]), np.tanh(p[2])
        return s0 ** 2 + rho * s0 * s1 * (T1 + T2) + s1 ** 2 * T1 * T2
    sc2, sm2, ell = np.exp(2 * p[0]), np.exp(2 * p[1]), np.exp(p[2])
    K = sc2 + sm2 * _matern(np.abs(T1 - T2) / ell, NU[kernel])
    if kernel == "matern32_lin":
        K = K + np.exp(2 * p[3]) * T1 * T2
    return K


def init_params(kernel, y_sd):
    s = np.log(max(y_sd, 1e-6))
    base = {"lmm_int": [s], "lmm_slope": [s, s - 1.0, 0.0], "matern32_lin": [s, s - 0.5, np.log(1.0), s - 1.5]}
    return np.array(base.get(kernel, [s, s - 0.5, np.log(1.0)]) + [s - 1.0])


@dataclass
class PatientData:
    """Ragged per-patient arrays grouped by number of visits for batched linear algebra."""
    groups: dict = field(default_factory=dict)      # n -> dict(t[B,n], y[B,n], se2[B,n], X[B,n,p])

    @classmethod
    def from_lists(cls, ts, ys, se2s, Xs):
        g = {}
        for t, y, s, X in zip(ts, ys, se2s, Xs):
            g.setdefault(len(t), []).append((t, y, s, X))
        return cls({n: {"t": np.array([a[0] for a in v]), "y": np.array([a[1] for a in v]),
                        "se2": np.array([a[2] for a in v]), "X": np.array([a[3] for a in v])} for n, v in g.items()})


class LongitudinalGP:
    def __init__(self, kernel: str):
        if kernel not in KERNELS:
            raise ValueError(kernel)
        self.kernel = kernel

    # ---- likelihood -------------------------------------------------------------------------------------------
    def _chol_terms(self, p, data):
        sw2 = np.exp(2 * p[-1]); out = []
        for n, d in data.groups.items():
            K = kernel_matrix(self.kernel, p[:-1], d["t"], d["t"])
            K = K + np.einsum("bi,ij->bij", sw2 + d["se2"], np.eye(n)) + 1e-10 * np.eye(n)
            L = np.linalg.cholesky(K)
            out.append((d, L))
        return out

    def _profile(self, p, data):
        """Return (negative log-likelihood, beta) with beta profiled by GLS."""
        terms = self._chol_terms(p, data)
        XtVX = 0; XtVy = 0; cache = []
        for d, L in terms:
            Xs = np.linalg.solve(L, d["X"]); ys = np.linalg.solve(L, d["y"][..., None])[..., 0]
            XtVX = XtVX + np.einsum("bnp,bnq->pq", Xs, Xs); XtVy = XtVy + np.einsum("bnp,bn->p", Xs, ys)
            cache.append((L, Xs, ys))
        beta = np.linalg.solve(XtVX + 1e-9 * np.eye(len(XtVy)), XtVy)
        nll = 0.0; N = 0
        for (L, Xs, ys) in cache:
            r = ys - Xs @ beta
            nll += 0.5 * np.sum(r * r) + np.sum(np.log(np.diagonal(L, axis1=1, axis2=2))); N += r.size
        return nll + 0.5 * N * np.log(2 * np.pi), beta

    def fit(self, data: PatientData, y_sd: float, restarts: int = 2, seed: int = 0):
        rng = np.random.default_rng(seed); best = None
        for k in range(restarts):
            p0 = init_params(self.kernel, y_sd) + (0 if k == 0 else rng.normal(0, 0.5, n_params(self.kernel)))
            f = lambda p: self._profile(p, data)[0] if np.all(np.abs(p) < 30) else 1e12
            r = minimize(f, p0, method="L-BFGS-B")
            if best is None or r.fun < best.fun:
                best = r
        self.p = best.x; self.nll, self.beta = self._profile(self.p, data)
        self.n_obs = sum(d["y"].size for d in data.groups.values())
        self.aic = 2 * self.nll + 2 * (n_params(self.kernel) + len(self.beta))
        self.converged = bool(best.success)
        return self

    @property
    def sigma_w(self):
        return float(np.exp(self.p[-1]))

    def hyperparameters(self) -> dict:
        p = self.p[:-1]
        if self.kernel == "lmm_int":
            h = {"s_intercept": float(np.exp(p[0]))}
        elif self.kernel == "lmm_slope":
            h = {"s_intercept": float(np.exp(p[0])), "s_slope_per_month": float(np.exp(p[1])), "rho": float(np.tanh(p[2]))}
        else:
            h = {"s_constant": float(np.exp(p[0])), "s_matern": float(np.exp(p[1])), "lengthscale_months": float(np.exp(p[2]))}
            if self.kernel == "matern32_lin":
                h["s_linear_per_month"] = float(np.exp(p[3]))
        h["sigma_w"] = self.sigma_w
        return h

    # ---- prediction -------------------------------------------------------------------------------------------
    def predict_next(self, t_hist, y_hist, se2_hist, X_hist, t_new, x_new):
        """Predictive mean and latent variance of f + x'beta at t_new given the history (no observation noise)."""
        p = self.p[:-1]; sw2 = self.sigma_w ** 2
        t_hist = np.asarray(t_hist, float); n = t_hist.size
        K = kernel_matrix(self.kernel, p, t_hist, t_hist) + np.diag(sw2 + np.asarray(se2_hist)) + 1e-10 * np.eye(n)
        ks = kernel_matrix(self.kernel, p, t_hist, np.array([t_new]))[:, 0]
        kss = kernel_matrix(self.kernel, p, np.array([t_new]), np.array([t_new]))[0, 0]
        L = np.linalg.cholesky(K)
        a = np.linalg.solve(L, ks); r = np.linalg.solve(L, np.asarray(y_hist) - np.asarray(X_hist) @ self.beta)
        mean = float(np.asarray(x_new) @ self.beta + a @ r)
        var = float(max(kss - a @ a, 0.0))
        return mean, var
