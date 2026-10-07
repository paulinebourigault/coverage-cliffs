"""Run: python -m pytest tests  (or python tests/test_tailcert.py)."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", "common"))
import tailcert as tc   # noqa: E402
import certs            # noqa: E402  (the paper's experiment code)


def _log(n, rng, M=4, eps=0.5):
    """Uniform-ish logger, target = action 0; Y ~ Exp with action-dependent scale."""
    pi0 = np.full(M, 1.0 / M)
    A = rng.integers(0, M, n)
    y = rng.exponential(1.0 + A, n)
    w_all = np.stack([(A == j) / pi0[j] for j in range(M)], 1)
    return w_all, y, float(1 / pi0.min())


def test_matches_paper_code():
    rng = np.random.default_rng(0)
    w, y, W = _log(5000, rng)
    tau = 3.0; K = w.shape[1]; C = np.log(K / 0.05)
    e = (y > tau).astype(float)
    for meth, f in (("betting", certs.exc_bet), ("bernstein", certs.exc_eb), ("pacbayes", certs.exc_pb)):
        ours = tc.exceedance_bound(w, y, tau, W=W, delta=0.05, method=meth)
        ref = np.array([f(w[:, j], e, W, C) for j in range(K)])
        assert np.allclose(ours, ref, rtol=1e-10, atol=1e-12), meth


def test_coverage():
    rng = np.random.default_rng(1)
    tau = 2.0; true_G = np.exp(-tau / 1.0)      # target = action 0, Exp(1)
    miss = {m: 0 for m in ("betting", "bernstein", "pacbayes")}
    R = 400
    for _ in range(R):
        w, y, W = _log(2000, rng)
        for m in miss:
            miss[m] += tc.exceedance_bound(w[:, 0], y, tau, W=W, method=m)[0] < true_G
    for m, k in miss.items():
        assert k / R <= 0.05 + 3 * np.sqrt(0.05 * 0.95 / R), (m, k / R)


def test_select_and_planning():
    rng = np.random.default_rng(2)
    w, y, W = _log(20000, rng)
    r = tc.select_and_certify(w, y, tau=6.0, p=0.9, utility=[0, 1, 2, 3], W=W)
    assert r["index"] in (0, 1)                 # actions 2,3 have P(Y>6) = e^-2, e^-1.5 > 0.1
    n_typ = tc.required_n(V_tau=0.01, gamma=0.02, W=4, K=4)
    n_guar = tc.required_n(V_tau=0.01, gamma=0.02, W=4, K=4, mode="guarantee")
    assert 0 < n_typ < n_guar
    pi = np.tile([0.9, 0.1, 0.0], (5, 1)); g = np.tile([0.01, 0.8, 0.5], (5, 1))
    pi0 = tc.tail_aware_logging(pi, g, floor=0.1)
    assert np.allclose(pi0.sum(1), 1) and pi0.min() >= 0.1 / 3 - 1e-12
    V = lambda q: (pi ** 2 / q * g).sum(1).mean()
    assert V(pi0) < V(0.9 * pi + 0.1 / 3)


def test_requires_known_finite_positive_weight_bound():
    w = np.ones(100); y = np.zeros(100)
    for method in ("betting", "bernstein", "pacbayes"):
        for invalid_W in (None, 0, -1, np.inf, np.nan, 0.99):
            with np.testing.assert_raises(ValueError):
                tc.exceedance_bound(w, y, tau=1.0, W=invalid_W, method=method)
        # Zero observed weights are possible under a valid positive global bound.
        bound = tc.exceedance_bound(np.zeros(100), y, tau=1.0, W=4.0, method=method)
        assert np.isfinite(bound).all()
        assert ((bound >= 0) & (bound <= 1)).all()
    for bad in (np.nan, np.inf, -1.0):
        bad_w = w.copy(); bad_w[0] = bad
        with np.testing.assert_raises(ValueError):
            tc.exceedance_bound(bad_w, y, tau=1.0, W=4.0)


def test_rejects_inputs_outside_certificate_domain():
    w = np.ones(10); y = np.zeros(10)
    for method in ("betting", "bernstein", "pacbayes"):
        for bad_y in (np.nan, np.inf, -np.inf):
            losses = y.copy(); losses[0] = bad_y
            with np.testing.assert_raises(ValueError):
                tc.exceedance_bound(w, losses, tau=1.0, W=1.0, method=method)
        for name, bad_values in (("delta", (0.0, 1.0, -0.1, np.nan, np.inf)),
                                 ("tau", (np.nan, np.inf, -np.inf)),
                                 ("K", (0, -1, 1.5, np.nan, True)),
                                 ("n_thresholds", (0, -1, 1.5, np.inf, True))):
            for bad in bad_values:
                args = dict(tau=1.0, W=1.0, method=method)
                args[name] = bad
                with np.testing.assert_raises(ValueError):
                    tc.exceedance_bound(w, y, **args)
    with np.testing.assert_raises(ValueError):
        tc.exceedance_bound(w[:1], y[:1], tau=1.0, W=1.0, method="bernstein")
    for bad_p in (0.0, 1.0, -0.1, 1.1, np.nan, np.inf):
        for fn in (tc.certify, tc.select_and_certify):
            with np.testing.assert_raises(ValueError):
                fn(w, y, tau=1.0, p=bad_p, W=1.0)


def test_pacbayes_scalar_and_per_column_kl():
    rng = np.random.default_rng(451)
    w, y, W = _log(1000, rng)
    # Each column is a different exact mixture of independently fixed action policies.
    rho = np.array([[0.7, 0.1, 0.1, 0.1], [0.1, 0.2, 0.3, 0.4]])
    mixed_w = w @ rho.T
    kl = np.sum(rho * np.log(4.0 * rho), axis=1)
    e = (y > 3.0).astype(float)
    vector = tc.exceedance_bound(mixed_w, y, tau=3.0, W=W, method="pacbayes", kl=kl)
    reference = np.array([certs.exc_pb(mixed_w[:, j], e, W, kl[j] + np.log(1 / 0.05))
                          for j in range(2)])
    np.testing.assert_allclose(vector, reference, rtol=1e-12, atol=1e-14)
    scalar = tc.exceedance_bound(mixed_w, y, tau=3.0, W=W, method="pacbayes", kl=1.0)
    repeated = tc.exceedance_bound(mixed_w, y, tau=3.0, W=W, method="pacbayes", kl=[1.0, 1.0])
    np.testing.assert_array_equal(scalar, repeated)
    for bad in (-1.0, np.nan, [-1.0, 0.0], [0.0, np.nan], [0.0], [[0.0, 1.0]]):
        with np.testing.assert_raises(ValueError):
            tc.exceedance_bound(mixed_w, y, tau=3.0, W=W, method="pacbayes", kl=bad)
    vacuous = tc.exceedance_bound(mixed_w, y, tau=3.0, W=W, method="pacbayes", kl=np.inf)
    np.testing.assert_array_equal(vacuous, np.ones(2))


if __name__ == "__main__":
    test_matches_paper_code(); test_coverage(); test_select_and_planning()
    test_requires_known_finite_positive_weight_bound()
    test_rejects_inputs_outside_certificate_domain(); test_pacbayes_scalar_and_per_column_kl()
    print("all tests passed")
