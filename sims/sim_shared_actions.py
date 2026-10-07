"""E3: search cost with shared actions -- the affine-router family of the shared-action theorem.

Contexts x = (x1, x2) uniform on F_5^2 (25 contexts); five actions logged uniformly (W = 5).  The 125
deterministic routers f_{a,b,c}(x) = a x1 + b x2 + c (mod 5) all share the same five actions: two routers
with different slopes agree on exactly 1/5 of the contexts, routers with equal slopes never agree.
Base model P_0: every (context, action) has tail probability u + gamma.  Alternative P_j: the cells used by
router j have tail probability u - gamma (the perturbation g0 (1 - 2 gamma w_j / V_j) with V_j = 5 (u + gamma)),
so j has exceedance u - gamma and every other router u + gamma - 2 gamma * agreement(k, j) > u.

Candidate lists: the first K routers of a fixed random ordering of all 125 (nested, predeclared), for
K in {2, 5, 10, 25, 50, 125}; the hidden safe router is uniform within the list.  Duplicate control: a list
of K = 125 entries made of 5 distinct routers repeated 25 times (union-bound procedures pay log 125 although
only 5 distinct hypotheses exist).  Every rule outputs the first certified candidate (fixed order), at
delta = 0.05 with per-candidate level delta / K; weights are W 1{A = f(X)}, so certificates are exact
functions of per-block match counts (sims/count_certificates.py).

Outputs (default ../results/shared_actions): power.csv, validity.csv, summary.csv (n at 90% power with
the theorem's lower bound (1/16 gamma^2) max{V kl(1-beta||delta), V[(1-beta) log K - log 2]} and the
empirical-Bernstein sufficient size max{32 V L / gamma^2, 11 W L / gamma}, L = log(2 / min(delta/K, beta/2))).

usage: python sim_shared_actions.py [--reps 2000] [--workers 32] [--out-dir DIR] [--quick]
"""
import argparse
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

import count_certificates as cc

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results", "shared_actions")
DELTA = 0.05
BETA = 0.1
W = 5
METHODS = ["cdf_eb", "cdf_exact", "exc_eb", "exc_bet", "counts_exact", "norm_bet"]

# routers and cells
X1, X2 = np.meshgrid(np.arange(5), np.arange(5), indexing="ij")
X1, X2 = X1.ravel(), X2.ravel()                                    # 25 contexts
ROUTERS = [(a, b, c) for a in range(5) for b in range(5) for c in range(5)]
ACT = np.array([(a * X1 + b * X2 + c) % 5 for a, b, c in ROUTERS])  # (125 routers, 25 contexts)
CELL_X = np.repeat(np.arange(25), 5); CELL_A = np.tile(np.arange(5), 25)   # 125 (context, action) cells
MATCH = (ACT[:, CELL_X] == CELL_A[None]).astype(np.int64)                # (125 routers, 125 cells)
ORDER = np.random.default_rng(20261004).permutation(125)


def candidate_lists():
    out = [(f"K{K}", ORDER[:K]) for K in (2, 5, 10, 25, 50, 125)]
    out.append(("dup125", np.repeat(ORDER[:5], 25)))
    return out


def check_family(u, g):
    """Exact checks of the construction: agreement fractions, overlaps, margins and theorem conditions."""
    agree = (ACT[:, None, :] == ACT[None, :, :]).mean(-1)
    off = agree[~np.eye(125, dtype=bool)]
    assert set(np.round(np.unique(off), 12)) <= {0.0, 0.2}
    V = W * (u + g)
    assert V >= 3 * g * W and u + g <= 0.75
    Gk = u + g - 2 * g * off
    assert Gk.min() > u
    return dict(V=V, eta_max=float(off.max()), min_unsafe_margin=float(Gk.min() - u))


def draw(rng, n, gcell, idx, reps):
    """Per-block match counts (N_b, E_b) for the candidate list idx; gcell has shape (reps, 125)."""
    M = MATCH[idx].T                                                     # (125 cells, K)
    nbs = cc.blocks(n)
    Nb, Eb = [], []
    for nb in nbs:
        c = rng.multinomial(nb, np.full(125, 1 / 125), size=reps)        # (reps, 125)
        e = rng.binomial(c, gcell)
        Nb.append(c @ M); Eb.append(e @ M)
    return nbs, Nb, Eb


def bounds(nbs, Nb, Eb, n, K):
    C = np.log(K / DELTA)
    N, E = sum(Nb), sum(Eb)
    S = N - E
    return {"cdf_eb": cc.cdf_eb(S, n, W, C), "cdf_exact": cc.cdf_exact(S, n, W, C),
            "exc_eb": cc.exc_eb(E, n, W, C), "exc_bet": cc.exc_bet(Eb, nbs, n, W, C),
            "counts_exact": cc.counts_exact(N, E, C), "norm_bet": cc.norm_bet(Nb, Eb, nbs, n, W, C)}


def first_certified(b, u):
    ok = b <= u
    return np.where(ok.any(1), ok.argmax(1), -1)


def run(job):
    p, rel, lname, idx, reps, grid, seed = job
    u = 1 - p; g = rel * u; K = len(idx)
    rng = np.random.default_rng([seed, int(1000 * p), K, int(lname == "dup125")])
    nref = W * (u + g) * np.log(K / DELTA) / g ** 2
    rows, vrows = [], []
    for x in grid:
        n = int(round(x * nref))
        pos = rng.integers(K, size=reps)                                 # position of the safe router
        safe = idx[pos]
        gcell = np.where(MATCH[safe] == 1, u - g, u + g)                 # (reps, 125)
        B = bounds(*draw(rng, n, gcell, idx, reps), n, K)
        ok_out = (idx[None, :] == safe[:, None])                         # copies of the safe router
        for m in METHODS:
            o = first_certified(B[m], u)
            hit = np.where(o >= 0, ok_out[np.arange(reps), np.maximum(o, 0)], False)
            rows.append(dict(p=p, gamma=g, list=lname, K=K, n=n, x=n / nref, method=m,
                             power=hit.mean(), false_alt=np.mean((o >= 0) & ~hit), abstain=np.mean(o < 0)))
        for null, gv in (("family", u + g), ("near_boundary", u * 1.05)):
            B0 = bounds(*draw(rng, n, np.full((reps, 125), gv), idx, reps), n, K)
            for m in METHODS:
                vrows.append(dict(p=p, list=lname, K=K, n=n, method=m, null=null,
                                  false_cert=np.mean(first_certified(B0[m], u) >= 0)))
    print(f"done p={p} {lname}", flush=True)
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


def main():
    ap = argparse.ArgumentParser(description="E3: affine routers sharing five actions.")
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261005)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--points", type=int, default=36)
    ap.add_argument("--out-dir", default=RESULTS)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.reps, a.points = 100, 8
    os.makedirs(a.out_dir, exist_ok=True)
    grid = np.geomspace(0.05, 400, a.points)
    cfgs = [(0.95, 0.25), (0.90, 0.25)]          # (p, gamma / u): u = 0.05, gamma = 0.0125 as in the paper
    info = {f"p{p}": check_family(1 - p, r * (1 - p)) for p, r in cfgs}
    print("family checks:", info, flush=True)
    jobs = [(p, r, name, idx, a.reps, grid, a.seed) for p, r in cfgs for name, idx in candidate_lists()]
    with Pool(a.workers) as pool:
        res = pool.map(run, jobs, chunksize=1)
    P = pd.DataFrame([r for rr, _ in res for r in rr]); V = pd.DataFrame([r for _, vv in res for r in vv])
    P.to_csv(os.path.join(a.out_dir, "power.csv"), index=False)
    V.to_csv(os.path.join(a.out_dir, "validity.csv"), index=False)
    kl = (1 - BETA) * np.log((1 - BETA) / DELTA) + BETA * np.log(BETA / (1 - DELTA))
    rows = []
    for (p, lname, K), d in P.groupby(["p", "list", "K"]):
        u = 1 - p; g = d.gamma.iloc[0]; Vj = W * (u + g)
        Keff = 5 if lname == "dup125" else K
        L = np.log(2 / min(DELTA / K, BETA / 2))
        r = dict(p=p, list=lname, K=K, distinct=Keff,
                 lower_bound=max(Vj * kl, Vj * ((1 - BETA) * np.log(Keff) - np.log(2))) / (16 * g ** 2),
                 eb_sufficient=max(32 * Vj * L / g ** 2, 11 * W * L / g))
        for m in METHODS:
            r[m] = n90(d[d.method == m])
        rows.append(r)
    S = pd.DataFrame(rows).sort_values(["p", "K", "list"])
    S.to_csv(os.path.join(a.out_dir, "summary.csv"), index=False)
    with open(os.path.join(a.out_dir, "family_checks.txt"), "w") as fh:
        fh.write(repr(info) + "\n")
    pd.set_option("display.width", 250)
    print(S.round({"p": 3, "lower_bound": 1, "eb_sufficient": 1, **{m: 1 for m in METHODS}}).to_string(index=False))
    print("\nmax false certification:", V.groupby(["null", "method"]).false_cert.max().unstack().round(4).to_string())
    print("max false under alternatives:", P.groupby("method").false_alt.max().round(4).to_dict())


if __name__ == "__main__":
    main()
