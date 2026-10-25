"""A small 1D convolutional network in NumPy (no deep-learning library needed).

Architecture (layout is samples x length x channels throughout):
    curve (101 x in_ch; e.g. 1 for one step, 4 for two legs at two visits) -> Conv1d(1->16, k=7, pad 3) -> ReLU
                    -> Conv1d(16->32, k=5, stride 2, pad 2) -> ReLU
                    -> mean pool within 6 time segments of stance (32 x 6 = 192; keeps timing, which
                       global pooling would discard)
    concat covariates -> Dense(64) -> ReLU -> Dense(n_out)
Trained with mean squared error and Adam. Gradients are checked numerically in tests/test_cnn1d.py.
"""
from __future__ import annotations

import numpy as np


def _cols(x, k, stride, pad):
    """im2col: (N, L, C) -> (N, Lout, k*C), ordered by kernel offset then channel."""
    xp = np.pad(x, ((0, 0), (pad, pad), (0, 0)))
    lout = (xp.shape[1] - k) // stride + 1
    return np.concatenate([xp[:, j: j + stride * lout: stride, :] for j in range(k)], axis=2), xp.shape


def _cols_back(dcols, xshape_p, k, stride, pad, c):
    dxp = np.zeros(xshape_p)
    lout = dcols.shape[1]
    for j in range(k):
        dxp[:, j: j + stride * lout: stride, :] += dcols[:, :, j * c:(j + 1) * c]
    return dxp[:, pad: xshape_p[1] - pad, :]


class CNN1D:
    def __init__(self, n_cov: int, n_out: int, seed: int = 0, c1: int = 16, c2: int = 32, hidden: int = 64, segments: int = 6,
                 in_ch: int = 1):
        r = np.random.default_rng(seed)
        he = lambda fan_in, shape: r.normal(0, np.sqrt(2.0 / fan_in), shape)
        self.cfg = dict(k1=7, s1=1, p1=3, k2=5, s2=2, p2=2, c1=c1, c2=c2, seg=segments)
        self.bins = np.array_split(np.arange(51), segments)
        self.p = {"W1": he(7 * in_ch, (7 * in_ch, c1)), "b1": np.zeros(c1),
                  "W2": he(5 * c1, (5 * c1, c2)), "b2": np.zeros(c2),
                  "W3": he(segments * c2 + n_cov, (segments * c2 + n_cov, hidden)), "b3": np.zeros(hidden),
                  "W4": he(hidden, (hidden, n_out)) * 0.5, "b4": np.zeros(n_out)}

    def forward(self, curve, cov, keep=False):
        c = self.cfg; p = self.p
        x = curve if curve.ndim == 3 else curve[:, :, None]
        col1, s1 = _cols(x, c["k1"], c["s1"], c["p1"]); z1 = col1 @ p["W1"] + p["b1"]; h1 = np.maximum(z1, 0)
        col2, s2 = _cols(h1, c["k2"], c["s2"], c["p2"]); z2 = col2 @ p["W2"] + p["b2"]; h2 = np.maximum(z2, 0)
        am = None
        g = np.concatenate([h2[:, b, :].mean(1) for b in self.bins] + [cov], axis=1)
        z3 = g @ p["W3"] + p["b3"]; h3 = np.maximum(z3, 0)
        out = h3 @ p["W4"] + p["b4"]
        if keep:
            self._cache = (col1, s1, z1, h1, col2, s2, z2, h2, am, g, z3, h3)
        return out

    def backward(self, dout):
        c = self.cfg; p = self.p
        col1, s1, z1, h1, col2, s2, z2, h2, am, g, z3, h3 = self._cache
        gr = {"W4": h3.T @ dout, "b4": dout.sum(0)}
        dh3 = dout @ p["W4"].T; dz3 = dh3 * (z3 > 0)
        gr["W3"] = g.T @ dz3; gr["b3"] = dz3.sum(0)
        dg = dz3 @ p["W3"].T
        c2 = c["c2"]
        dh2 = np.zeros_like(h2)
        for i, b in enumerate(self.bins):
            dh2[:, b, :] = dg[:, None, i * c2:(i + 1) * c2] / len(b)
        dz2 = dh2 * (z2 > 0)
        gr["W2"] = np.einsum("nlk,nlc->kc", col2, dz2); gr["b2"] = dz2.sum((0, 1))
        dh1 = _cols_back(dz2 @ p["W2"].T, s2, c["k2"], c["s2"], c["p2"], c["c1"])
        dz1 = dh1 * (z1 > 0)
        gr["W1"] = np.einsum("nlk,nlc->kc", col1, dz1); gr["b1"] = dz1.sum((0, 1))
        return gr

    def loss_grad(self, curve, cov, y):
        out = self.forward(curve, cov, keep=True)
        d = out - y
        return float((d ** 2).mean()), self.backward(2 * d / d.size)

    def fit(self, curve, cov, y, val=None, epochs=40, batch=256, lr=1e-3, patience=5, seed=0, verbose=False):
        r = np.random.default_rng(seed)
        m = {k: np.zeros_like(v) for k, v in self.p.items()}; v2 = {k: np.zeros_like(v) for k, v in self.p.items()}
        t = 0; best = (np.inf, None); wait = 0; self.history = []
        for ep in range(epochs):
            order = r.permutation(len(y))
            for i in range(0, len(y), batch):
                b = order[i:i + batch]
                _, gr = self.loss_grad(curve[b], cov[b], y[b]); t += 1
                for k in self.p:
                    m[k] = 0.9 * m[k] + 0.1 * gr[k]; v2[k] = 0.999 * v2[k] + 0.001 * gr[k] ** 2
                    self.p[k] -= lr * (m[k] / (1 - 0.9 ** t)) / (np.sqrt(v2[k] / (1 - 0.999 ** t)) + 1e-8)
            if val is not None:
                vl = float(((self.predict(*val[:2]) - val[2]) ** 2).mean()); self.history.append(vl)
                if verbose:
                    print(f"epoch {ep + 1} val mse {vl:.4f}", flush=True)
                if vl < best[0] - 1e-5:
                    best = (vl, {k: v.copy() for k, v in self.p.items()}); wait = 0
                else:
                    wait += 1
                    if wait >= patience:
                        break
        if best[1] is not None:
            self.p = best[1]
        return self

    def predict(self, curve, cov, batch=4096):
        return np.vstack([self.forward(curve[i:i + batch], cov[i:i + batch]) for i in range(0, len(curve), batch)])
