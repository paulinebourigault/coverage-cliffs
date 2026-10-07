"""Appendix "Tighter inequalities on the same statistic": does a PAC-Bayes-kl exceedance certificate
(Seeger 2002; Maurer 2004) reduce the constant gap to the exact binomial reference?
Same rare-action design as sim_tail_boundary.py (W=64, gamma=(1-p)/2).

Outputs (default directory ../results/sims): kl_variant_results.csv (power per cell) and
kl_variant_summary.csv (n/n* at 90% power per method, and its ratio to the binomial reference).

usage: python sim_kl_variant.py [--reps 3000] [--seed 20271002] [--out-dir DIR]
"""
import argparse
import os

import numpy as np
import pandas as pd

import sim_tail_boundary as s

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results", "sims")


def kl(q, p):
    q = np.clip(q, 1e-15, 1 - 1e-15); p = np.clip(p, 1e-15, 1 - 1e-15)
    return q * np.log(q / p) + (1 - q) * np.log((1 - q) / (1 - p))


def kl_ucb(qhat, c):
    lo = np.array(qhat, float); hi = np.ones_like(lo)
    for _ in range(60):
        mid = (lo + hi) / 2
        ok = kl(qhat, mid) <= c
        lo = np.where(ok, mid, lo); hi = np.where(ok, hi, mid)
    return hi


def main(reps=3000, seed=20271002, out_dir=RESULTS):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed); d = 0.05; W = 64; rows = []
    for K in (2, 8, 32):
        for p, g in ((0.90, 0.05), (0.95, 0.025), (0.99, 0.005)):
            nstar = W * (1 - p) * np.log(K / d) / g ** 2
            for mult in np.geomspace(0.5, 30, 40):
                n = int(mult * nstar)
                j = rng.integers(K, size=reps); qs = np.full((reps, K), p - g); qs[np.arange(reps), j] = p + g
                probs = np.r_[np.full(K, 1 / W), 1 - K / W]
                N = rng.multinomial(n, probs, size=reps)[:, :K]; S = rng.binomial(N, qs); E = N - S
                Ghat = W * E / n; C = np.log(K / d); L = C + np.log(2.0); Ef = E.astype(float)
                ub = {"exc_pb": s.exc_ucb(Ghat, n, W, C),
                      "exc_eb": Ghat + np.sqrt(2 * W ** 2 * Ef * (n - Ef) / (n * (n - 1.0)) * L / n) + 7 * W * L / (3 * (n - 1.0)),
                      "exc_kl": W * kl_ucb(Ghat / W, (C + np.log(2 * np.sqrt(n))) / n),
                      "binom": 1 - s.cp_lower(S, N, d / K)}
                for m, u in ub.items():
                    sel = s.pick(-u, u <= 1 - p, rng)
                    rows.append(dict(K=K, p=p, mult=mult, method=m, power=np.mean(sel == j)))
            print("done", K, p, flush=True)
    df = pd.DataFrame(rows); df.to_csv(os.path.join(out_dir, "kl_variant_results.csv"), index=False)
    out = []
    for (K, p), dd in df.groupby(["K", "p"]):
        r = {"K": K, "p": p}
        for m, d2 in dd.groupby("method"):
            d2 = d2.sort_values("mult"); x, y = d2.mult.values, d2.power.values
            i = int(np.argmax(y >= 0.9)); t = (0.9 - y[i - 1]) / (y[i] - y[i - 1]) if i > 0 else 0
            r[m] = float(np.exp(np.log(x[i - 1]) + t * (np.log(x[i]) - np.log(x[i - 1])))) if i > 0 else x[0]
        out.append(r)
    o = pd.DataFrame(out)
    for m in ("exc_pb", "exc_eb", "exc_kl"):
        o[m + "/binom"] = o[m] / o.binom
    print(o.round(2).to_string(index=False)); o.to_csv(os.path.join(out_dir, "kl_variant_summary.csv"), index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="PAC-Bayes-kl exceedance variant on the rare-action class.")
    ap.add_argument("--reps", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=20271002)
    ap.add_argument("--out-dir", default=RESULTS)
    a = ap.parse_args()
    main(a.reps, a.seed, a.out_dir)
