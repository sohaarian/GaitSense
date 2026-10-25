import numpy as np

from gaitrec_rehab.cnn1d import CNN1D


def test_cnn_gradients_match_numerical_gradients():
    r = np.random.default_rng(3)
    net = CNN1D(n_cov=3, n_out=2, seed=1, c1=4, c2=5, hidden=6)
    curve, cov, y = r.normal(size=(4, 101)), r.normal(size=(4, 3)), r.normal(size=(4, 2))
    _, gr = net.loss_grad(curve, cov, y)
    for k in ("W1", "b1", "W2", "b2", "W3", "W4"):
        idx = tuple(r.integers(0, s) for s in net.p[k].shape)
        old = net.p[k][idx]; eps = 1e-6
        net.p[k][idx] = old + eps; up, _ = net.loss_grad(curve, cov, y)
        net.p[k][idx] = old - eps; dn, _ = net.loss_grad(curve, cov, y)
        net.p[k][idx] = old
        num = (up - dn) / (2 * eps)
        assert abs(num - gr[k][idx]) <= 1e-5 + 1e-4 * abs(num), k


def test_cnn_learns_a_shape_feature():
    """A target that depends on the position of a bump can be learned; a constant model cannot."""
    r = np.random.default_rng(0)
    n = 1500; pos = r.integers(20, 80, n)
    t = np.arange(101)
    curve = np.exp(-0.5 * ((t[None, :] - pos[:, None]) / 4.0) ** 2) + 0.05 * r.normal(size=(n, 101))
    y = ((pos - 50) / 30.0)[:, None]
    cov = np.zeros((n, 1))
    net = CNN1D(n_cov=1, n_out=1, seed=0).fit(curve[:1200], cov[:1200], y[:1200],
                                               val=(curve[1200:], cov[1200:], y[1200:]), epochs=15)
    mse = float(((net.predict(curve[1200:], cov[1200:]) - y[1200:]) ** 2).mean())
    assert mse < 0.5 * float(y[1200:].var())


def test_cnn_multichannel_gradients_match_numerical_gradients():
    r = np.random.default_rng(5)
    net = CNN1D(n_cov=2, n_out=2, seed=2, c1=3, c2=4, hidden=5, in_ch=4)
    curve, cov, y = r.normal(size=(3, 101, 4)), r.normal(size=(3, 2)), r.normal(size=(3, 2))
    _, gr = net.loss_grad(curve, cov, y)
    for k in ("W1", "W2", "W3"):
        idx = tuple(r.integers(0, s) for s in net.p[k].shape)
        old = net.p[k][idx]; eps = 1e-6
        net.p[k][idx] = old + eps; up, _ = net.loss_grad(curve, cov, y)
        net.p[k][idx] = old - eps; dn, _ = net.loss_grad(curve, cov, y)
        net.p[k][idx] = old
        num = (up - dn) / (2 * eps)
        assert abs(num - gr[k][idx]) <= 1e-5 + 1e-4 * abs(num), k
