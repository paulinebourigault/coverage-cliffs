"""E1: information removal and recovery on the rare-action family (Theorem 1 of the paper).

One context; K target actions, each logged with probability 1/W, and a dummy action with the remaining
mass; candidate j always plays action j, so w_j = W 1{A = j}.  Binary outcomes with tail probabilities
    P_0 (null)      : every target action u + gamma            (every candidate unsafe)
    P_j (alternative): action j u - gamma, the others u + gamma (j uniquely safe)
with u = 1 - p and margin gamma = c u.  The hidden safe index j is drawn uniformly per repetition.

Every rule outputs the FIRST certified candidate in the fixed order 1..K (a selection rule measurable in
the joint CDF indicators), at delta = 0.05 with per-candidate level delta / K:
  joint CDF indicators only : cdf_eb (UnO / Thomas interval), cdf_bet, cdf_exact (Clopper-Pearson on S_j)
  exceedance indicators     : exc_eb (Huang et al. 2022), exc_bet
  action counts added       : counts_exact (conditional binomial), norm_bet (normalization-aware betting)
The data are generated as exact per-block counts (sims/count_certificates.py).

Settings
  two_action  : K = W = 2, p in {0.80, 0.90, 0.95, 0.98, 0.99, 0.995}
  vary_K      : W = 32, K in {2, 4, 8, 16, 32}, p in {0.90, 0.99}
  vary_p      : W = 32, K = 8, p in {0.80, 0.95, 0.98}  (with vary_K: the u-scaling at fixed W, K)
  single      : K = 1, W in {1, 2, 5, 10, 20, 32} (logging probability 1/W of the target action),
                p = 0.95; the CDF penalty should vanish on-policy (W = 1)
For each setting, log sizes n = x * n_ref, n_ref = W u log(max(K, 2)... ) -- see N_REF below --, with
x on a geometric grid; power P(output = j) under P_j and false certification under P_j (output an unsafe
candidate), under P_0, and under the near-boundary null where every action has tail u (1 + 0.05).

Outputs (default ../results/information):
  power.csv     one row per (setting, W, K, p, n, method): power, false_alt, abstain
  validity.csv  one row per (setting, W, K, p, n, method, null): false-certification rate
  summary.csv   n at 90% power (log-interpolated; NaN if not reached) with the two lower bounds of
                Theorem 1 at beta = 0.1:  full logs  W u H / (16 gamma^2),  joint CDF  W p H / (8 gamma^2)
                (the joint-CDF lower bound is omitted for K = 1; see lower_bounds).

usage: python sim_information.py [--reps 2000] [--workers 32] [--out-dir DIR] [--quick]
"""
import argparse
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

import count_certificates as cc

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results", "information")
DELTA = 0.05
C_REL = 0.5                      # relative margin gamma = C_REL * u
BETA = 0.1                       # power target 1 - BETA for n90 and the lower bounds
METHODS = ["cdf_eb", "cdf_bet", "cdf_exact", "exc_eb", "exc_bet", "counts_exact", "norm_bet"]
INFO = {"cdf_eb": "cdf", "cdf_bet": "cdf", "cdf_exact": "cdf", "exc_eb": "exceedance",
        "exc_bet": "exceedance", "counts_exact": "counts", "norm_bet": "counts"}


def settings():
    out = [("two_action", 2, 2, p) for p in (0.80, 0.90, 0.95, 0.98, 0.99, 0.995)]
    out += [("vary_K", 32, K, p) for p in (0.90, 0.99) for K in (2, 4, 8, 16, 32)]
    out += [("vary_p", 32, 8, p) for p in (0.80, 0.95, 0.98)]
    out += [("single", W, 1, 0.95) for W in (1, 2, 5, 10, 20, 32)]
    return out


def n_ref(W, K, p):
    """Full-data scale W u log(K / delta) / gamma^2 of Theorem 1(a)."""
    u = 1 - p
    return W * u * np.log(K / DELTA) / (C_REL * u) ** 2


def draw(rng, n, W, K, g, reps):
    """Per-block counts (N_b, E_b) for every candidate; g has shape (reps, K)."""
    probs = np.r_[np.full(K, 1.0 / W), max(0.0, 1.0 - K / W)]
    nbs = cc.blocks(n)
    Nb, Eb = [], []
    for nb in nbs:
        N = rng.multinomial(nb, probs, size=reps)[:, :K]
        Nb.append(N); Eb.append(rng.binomial(N, g))
    return nbs, Nb, Eb


def bounds(nbs, Nb, Eb, n, W, K):
    C = np.log(K / DELTA)
    N, E = sum(Nb), sum(Eb)
    S = N - E
    Sb = [a - b for a, b in zip(Nb, Eb)]
    return {"cdf_eb": cc.cdf_eb(S, n, W, C), "cdf_bet": cc.cdf_bet(Sb, nbs, n, W, C),
            "cdf_exact": cc.cdf_exact(S, n, W, C), "exc_eb": cc.exc_eb(E, n, W, C),
            "exc_bet": cc.exc_bet(Eb, nbs, n, W, C), "counts_exact": cc.counts_exact(N, E, C),
            "norm_bet": cc.norm_bet(Nb, Eb, nbs, n, W, C)}


def first_certified(b, u):
    ok = b <= u
    return np.where(ok.any(1), ok.argmax(1), -1)


def run_setting(job):
    name, W, K, p, reps, grid, seed = job
    rng = np.random.default_rng([seed, int(1000 * p), W, K, ["two_action", "vary_K", "vary_p", "single"].index(name)])
    u = 1 - p; g = C_REL * u
    rows, vrows = [], []
    for x in grid:
        n = int(max(2 * W, round(x * n_ref(W, K, p))))
        # alternatives: hidden safe index j
        j = rng.integers(K, size=reps)
        gs = np.full((reps, K), u + g); gs[np.arange(reps), j] = u - g
        B = bounds(*draw(rng, n, W, K, gs, reps), n, W, K)
        for m in METHODS:
            out = first_certified(B[m], u)
            rows.append(dict(setting=name, W=W, K=K, p=p, u=u, gamma=g, n=n, x=n / n_ref(W, K, p),
                             method=m, info=INFO[m], power=np.mean(out == j),
                             false_alt=np.mean((out >= 0) & (out != j)), abstain=np.mean(out < 0)))
        # nulls: all unsafe at u + gamma, and near the boundary at u (1 + 0.05)
        for null, gv in (("family", u + g), ("near_boundary", u * 1.05)):
            B0 = bounds(*draw(rng, n, W, K, np.full((reps, K), gv), reps), n, W, K)
            for m in METHODS:
                vrows.append(dict(setting=name, W=W, K=K, p=p, n=n, x=n / n_ref(W, K, p), method=m,
                                  null=null, false_cert=np.mean(first_certified(B0[m], u) >= 0)))
    print(f"done {name} W={W} K={K} p={p}", flush=True)
    return rows, vrows


def n90(d):
    d = d.sort_values("n"); x, y = d.n.values.astype(float), d.power.values
    if not (y >= 1 - BETA).any():
        return np.nan
    i = int(np.argmax(y >= 1 - BETA))
    if i == 0:
        return x[0]
    t = (1 - BETA - y[i - 1]) / (y[i] - y[i - 1])
    return float(np.exp(np.log(x[i - 1]) + t * (np.log(x[i]) - np.log(x[i - 1]))))


def lower_bounds(W, K, p):
    """Theorem 1 lower bounds at power 1 - BETA and level DELTA.

    For K = 1, retain the full-log confidence bound but leave the CDF bound undefined: Theorem 1(b)
    assumes K >= 2.  The separate single-candidate proposition requires W >= 2 and validity over a
    larger model class; in particular its off-policy CDF penalty must not be applied at W = 1.
    """
    u = 1 - p; g = C_REL * u
    kl = (1 - BETA) * np.log((1 - BETA) / DELTA) + BETA * np.log(BETA / (1 - DELTA))
    H = max(kl, (1 - BETA) * np.log(K) - np.log(2)) if K >= 2 else kl
    lb_cdf = W * p * H / (8 * g ** 2) if K >= 2 else np.nan
    return W * u * H / (16 * g ** 2), lb_cdf


def main():
    ap = argparse.ArgumentParser(description="E1: information removal and recovery (rare-action family).")
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261004)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--points", type=int, default=44, help="log sizes per setting")
    ap.add_argument("--out-dir", default=RESULTS)
    ap.add_argument("--quick", action="store_true", help="smoke test: few reps and points")
    a = ap.parse_args()
    if a.quick:
        a.reps, a.points = 100, 10
    os.makedirs(a.out_dir, exist_ok=True)
    grid = np.geomspace(0.02, 5000, a.points)
    jobs = [(name, W, K, p, a.reps, grid, a.seed) for name, W, K, p in settings()]
    jobs.sort(key=lambda j: -j[3])                     # slowest (largest n) first
    with Pool(a.workers) as pool:
        res = pool.map(run_setting, jobs, chunksize=1)
    P = pd.DataFrame([r for rr, _ in res for r in rr])
    V = pd.DataFrame([r for _, vv in res for r in vv])
    P.to_csv(os.path.join(a.out_dir, "power.csv"), index=False)
    V.to_csv(os.path.join(a.out_dir, "validity.csv"), index=False)
    rows = []
    for (name, W, K, p), d in P.groupby(["setting", "W", "K", "p"]):
        lb_full, lb_cdf = lower_bounds(W, K, p)
        r = dict(setting=name, W=W, K=K, p=p, n_ref=n_ref(W, K, p), lb_full=lb_full, lb_cdf=lb_cdf)
        for m in METHODS:
            r[m] = n90(d[d.method == m])
        rows.append(r)
    S = pd.DataFrame(rows)
    S["cdf_exact/counts_exact"] = S.cdf_exact / S.counts_exact
    S["cdf_eb/exc_eb"] = S.cdf_eb / S.exc_eb
    S["p/u"] = S.p / (1 - S.p)
    S.to_csv(os.path.join(a.out_dir, "summary.csv"), index=False)
    pd.set_option("display.width", 250)
    print(S.round(3).to_string(index=False))
    print("\nmax false certification over n, by null and method:")
    print(V.groupby(["null", "method"]).false_cert.max().unstack("method").round(4).to_string())
    print("max false certification under alternatives:", P.groupby("method").false_alt.max().round(4).to_dict())


if __name__ == "__main__":
    main()
