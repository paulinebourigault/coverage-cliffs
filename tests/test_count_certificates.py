"""Checks that the count-based certificates in sims/count_certificates.py equal the raw-data
implementations in common/certs.py, and that norm_bet is valid on contextual logs.
Run: python tests/test_count_certificates.py"""
import os
import sys

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
sys.path.insert(0, os.path.join(HERE, "..", "sims"))
import certs                      # noqa: E402
import count_certificates as cc   # noqa: E402


def _raw_log(rng, n, W, K, g):
    """Rare-action family: K target actions w.p. 1/W each, dummy otherwise; candidate j plays action j."""
    probs = np.r_[np.full(K, 1.0 / W), 1.0 - K / W]
    A = rng.choice(K + 1, size=n, p=probs)
    Z = (rng.random(n) < np.r_[g, 0.5][A]).astype(float)
    w = np.stack([W * (A == j) for j in range(K)]).astype(float)       # (K, n)
    return w, Z


def _block_counts(w, Z, W, nbs):
    m = (w > 0)
    Nb, Eb, s = [], [], 0
    for nb in nbs:
        Nb.append(m[:, s:s + nb].sum(1)); Eb.append((m[:, s:s + nb] * Z[s:s + nb]).sum(1)); s += nb
    return Nb, Eb


def test_equal_to_raw():
    rng = np.random.default_rng(0)
    for (n, W, K) in [(500, 2, 2), (3000, 8, 4), (20000, 32, 8), (777, 4, 1)]:
        g = rng.uniform(0.01, 0.3, K)
        w, Z = _raw_log(rng, n, W, K, g)
        C = np.log(K / 0.05)
        nbs = cc.blocks(n)
        Nb, Eb = _block_counts(w, Z, W, nbs)
        N, E = sum(Nb), sum(Eb); S = N - E
        Sb = [a - b for a, b in zip(Nb, Eb)]
        pairs = [(cc.exc_bet(Eb, nbs, n, W, C), certs.exc_bet(w, Z, W, C)),
                 (cc.exc_eb(E, n, W, C), certs.exc_eb(w, Z, W, C)),
                 (cc.cdf_eb(S, n, W, C), certs.cdf_eb(w, Z, W, C)),
                 (cc.norm_bet(Nb, Eb, nbs, n, W, C), certs.norm_bet(w, Z, W, C))]
        # cdf_bet: raw version = exc_bet machinery on 1 - X with X = w 1{Y<=tau}/W
        Xc = w * (1 - Z) / W
        A2, B2, D2 = certs.bet_quadratic(W * (1 - Xc), np.ones_like(Xc), W, C)  # test on 1 - Xc in [0,1]
        ucb = certs._smallest_crossing(A2, B2, D2, C)                            # unclipped UCB on E[1 - Xc]
        pairs.append((cc.cdf_bet(Sb, nbs, n, W, C), np.clip(1 - W * (1 - ucb), 0, 1)))
        for k, (a, b) in enumerate(pairs):
            assert np.allclose(a, b, rtol=1e-9, atol=1e-12), (k, n, W, K, a, b)


def _coverage(fn, R=3000, alpha=0.05, seed=1):
    rng = np.random.default_rng(seed)
    miss = 0
    for _ in range(R):
        miss += fn(rng)
    lo = stats.beta.ppf(0.025, miss, R - miss + 1) if miss else 0.0
    return miss / R, lo


def test_validity_contextual():
    """norm_bet on contextual stochastic logs with unequal weights: miss rate <= delta (up to MC error)."""
    M, n, delta = 4, 1500, 0.05
    pi0 = np.array([0.4, 0.3, 0.2, 0.1])

    def one(rng):
        x = rng.random(n)
        pi = np.stack([0.7 * x, 0.3 * (1 - x), 0.2 * np.ones(n), 0.8 - 0.4 * x], 1)
        pi /= pi.sum(1, keepdims=True)
        a = rng.choice(M, size=n, p=pi0)
        g = np.array([0.05, 0.4, 0.1, 0.02])[a] * (0.5 + x)
        z = (rng.random(n) < g).astype(float)
        w = pi[np.arange(n), a] / pi0[a]
        # true G = E_x sum_a pi(a|x) g(x,a), computed on a large independent sample
        xs = np.linspace(0, 1, 20001)
        pis = np.stack([0.7 * xs, 0.3 * (1 - xs), 0.2 * np.ones_like(xs), 0.8 - 0.4 * xs], 1)
        pis /= pis.sum(1, keepdims=True)
        G = (pis * np.array([0.05, 0.4, 0.1, 0.02]) * (0.5 + xs)[:, None]).sum(1).mean()
        W = 1.0 / pi0.min()
        return certs.norm_bet(w, z, W, np.log(1 / delta)) < G
    rate, lo = _coverage(one)
    assert lo <= 0.05, rate
    print(f"norm_bet contextual miss rate {rate:.4f} (delta 0.05)")


def test_counts_exact_and_cdf_exact_validity():
    """Exact rules at a boundary model: miss rates <= delta."""
    W, n, delta = 4, 2000, 0.05
    G = 0.08
    C = np.log(1 / delta)

    def one_counts(rng):
        N = rng.binomial(n, 1 / W); E = rng.binomial(N, G)
        return cc.counts_exact(N, E, C) < G

    def one_cdf(rng):
        S = rng.binomial(n, (1 - G) / W)
        return cc.cdf_exact(S, n, W, C) < G
    for f in (one_counts, one_cdf):
        rate, lo = _coverage(f, R=20000)
        assert lo <= 0.05, rate


if __name__ == "__main__":
    test_equal_to_raw(); test_counts_exact_and_cdf_exact_validity(); test_validity_contextual()
    print("all tests passed")
