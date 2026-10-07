"""E0: exact recovery of the exceedance empirical-Bernstein certificate from CDF moment summaries.

For each candidate j and threshold t, a CDF computation keeps S_F = sum_i w_ij 1{Y_i <= t} and
Q_F = sum_i w_ij^2 1{Y_i <= t}; two totals per candidate, S_W = sum_i w_ij and Q_W = sum_i w_ij^2, give
    S_E = S_W - S_F,   Q_E = Q_W - Q_F,
the first two sample moments of w_ij 1{Y_i > t}, hence the exceedance empirical-Bernstein bound
(certs.exc_eb) at every threshold.  This script checks the identity, the bounds and the certification
decisions against direct accumulation on
  * synthetic contextual logs with stochastic policies, zero and unequal weights, ties, no exceedances;
  * the reasoning-study outcome table (results/reasoning/table) under uniform logging with random
    stochastic target policies;
  * the RouterBench outcome table (downloaded on first use) under the same construction.
Agreement is an implementation check of the identity, not a performance claim.

usage: python check_reconstruction.py [--data-dir DIR]
"""
import argparse
import math
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
sys.path.insert(0, os.path.join(HERE, "..", "routerbench"))
import certs                      # noqa: E402

DELTA = 0.05


def fsum_rows(a):
    """Row sums with compensated (exact-rounding) summation."""
    return np.array([math.fsum(r) for r in a])


def check(w, y, taus, W, label, p=0.9):
    """w: (K, n) weights; y: (n,) losses; taus: thresholds.  Returns max abs differences."""
    K, n = w.shape
    C = certs.complexity(K, DELTA, n_thresholds=len(taus))
    L = C + np.log(2.0)
    SW, QW = fsum_rows(w), fsum_rows(w ** 2)
    out = dict(case=label, K=K, n=n, thresholds=len(taus), max_moment_diff=0.0, max_bound_diff=0.0,
               decision_mismatch=0, moment_violations=0)
    for t in taus:
        z = (y > t).astype(float)
        SF, QF = fsum_rows(w * (1 - z)), fsum_rows(w ** 2 * (1 - z))
        SE, QE = SW - SF, QW - QF                                  # reconstruction
        SEd, QEd = fsum_rows(w * z), fsum_rows(w ** 2 * z)          # direct accumulation
        out["max_moment_diff"] = max(out["max_moment_diff"], np.abs(SE - SEd).max(), np.abs(QE - QEd).max())
        # moment sanity: 0 <= Q_E <= W S_E and Q_E >= S_E^2 / n (up to rounding), never clipped silently
        tol = 1e-9 * max(1.0, QW.max())
        bad = (QE < -tol) | (QE > W * SE + tol) | (QE < SE ** 2 / n - tol)
        out["moment_violations"] += int(bad.sum())
        G = SE / n
        var = np.maximum(QE - SE ** 2 / n, 0.0) / (n - 1)
        U = np.clip(G + np.sqrt(2 * var * L / n) + 7 * W * L / (3 * (n - 1)), 0.0, 1.0)
        Ud = certs.exc_eb(w, z, W, C)                               # direct implementation
        out["max_bound_diff"] = max(out["max_bound_diff"], np.abs(U - Ud).max())
        out["decision_mismatch"] += int(((U <= 1 - p) != (Ud <= 1 - p)).sum())
    return out


def synthetic(rng):
    rows = []
    M, n = 6, 20000
    pi0 = np.array([0.3, 0.25, 0.2, 0.15, 0.07, 0.03])
    W = 1 / pi0.min()
    x = rng.random(n)
    a = rng.choice(M, size=n, p=pi0)
    y = np.round(rng.exponential(1 + a + x), 1)                     # rounding creates threshold ties
    K = 20
    logits = rng.normal(size=(K, M)); slope = rng.normal(size=(K, M))
    pi = np.exp(logits[:, None, :] + slope[:, None, :] * x[None, :, None])
    pi[:3, :, 5] = 0.0                                              # some candidates never use action 5
    pi /= pi.sum(-1, keepdims=True)
    w = pi[:, np.arange(n), a] / pi0[a][None]
    taus = list(np.unique(np.quantile(y, [0.5, 0.9, 0.99]))) + [float(y.max())]   # incl. no exceedances
    rows.append(check(w, y, taus, W, "synthetic contextual"))
    rows.append(check(np.zeros((2, n)), y, taus[:1], W, "synthetic all-zero weights"))
    return rows


def table_case(rng, Y, label, n=20000, K=16):
    N, M = Y.shape
    idx = rng.integers(0, N, n)
    a = rng.integers(0, M, n)
    y = Y[idx, a]
    pi = rng.dirichlet(np.ones(M) * 0.5, size=(K, N))               # random stochastic target policies
    w = pi[:, idx, a] * M
    taus = list(np.quantile(Y, [0.5, 0.9, 0.95, 0.99]))
    return check(w, y, taus, float(M), label)


def main():
    ap = argparse.ArgumentParser(description="E0: exact reconstruction check.")
    ap.add_argument("--data-dir", default=None, help="RouterBench data directory (see routerbench_replay)")
    ap.add_argument("--out", default=os.path.join(HERE, "..", "results", "reconstruction_check.csv"))
    a = ap.parse_args()
    rng = np.random.default_rng(20261004)
    rows = synthetic(rng)
    t = pd.read_parquet(os.path.join(HERE, "..", "results", "reasoning", "table", "table.parquet"))
    acts = [c[len("tokens__"):] for c in t.columns if c.startswith("tokens__")]
    rows.append(table_case(rng, t[["tokens__" + x for x in acts]].values.astype(float), "reasoning table (tokens)"))
    import routerbench_replay as rb
    _, _, Ccost, _ = rb.load_data(a.data_dir) if a.data_dir else rb.load_data()
    rows.append(table_case(rng, np.asarray(Ccost, float), "RouterBench table (cost)"))
    R = pd.DataFrame(rows)
    R.to_csv(a.out, index=False)
    pd.set_option("display.width", 200)
    print(R.to_string(index=False))
    assert (R.decision_mismatch == 0).all() and (R.moment_violations == 0).all()
    assert (R.max_bound_diff < 1e-9).all()
    print("reconstruction identical to direct accumulation in every case")


if __name__ == "__main__":
    main()
