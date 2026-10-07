"""Controlled study on the rare-action class of the information-separation theorem (tail-sharp boundary,
n* = W(1-p) log(K/delta) / gamma^2); produces the data behind fig_tail_boundary.pdf.

One context, K candidate actions each logged w.p. 1/W, a baseline action with the
remaining mass. Candidate j deterministically plays action j. Z = 1{Y <= tau}.
Null: every candidate has P(Z=1) = p - gamma (exceedance 1-p+gamma, unsafe).
Alternative j: candidate j has P(Z=1) = p + gamma (exceedance 1-p-gamma, safe).

Procedures (all at the same delta, uniform prior over K => complexity log K):
  exc     : exceedance PAC-Bayes envelope, min over a predeclared lambda grid
  exc_bet : exceedance betting certificate (appendix "The betting certificate")
  exc_eb  : exceedance, empirical Bernstein (same statistic as exc, variance-adaptive)
  cdf     : CDF-envelope PAC-Bayes Bernstein certificate, min over a predeclared lambda grid
  cdf_eb  : CDF, empirical Bernstein (= Thomas et al. 2015 / UnO bound without truncation)
  split   : select on first half (smallest empirical exceedance), certify on second half
            with the exceedance certificate at complexity log 1
  binom   : per-action exact Clopper-Pearson on (N_j, S_j) at delta/K (model-specific)
  naive   : exceedance rule omitting both search and lambda-grid penalties -- INVALID after selection
  ceiling : Neyman-Pearson average-power ceiling R_n(delta/K) (appendix "Oracle power ceiling";
            an upper bound on the power of any valid rule)
See appendix "Simulation details".

Outputs (default directory ../results/sims):
  tail_boundary_results.csv   power per (K, p, n, method); input of plot_tail_boundary.py
  tail_validity_results.csv   false-certification rate when every candidate is slightly unsafe

usage: python sim_tail_boundary.py [--reps 3000] [--seed 20270928] [--out-dir DIR] [--only-validity]
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results", "sims")
N_LAM = 60


def lam_grid(upper):
    return np.geomspace(1e-7 * upper, 0.999 * upper, N_LAM)


def exc_ucb(Ghat, n, W, C):
    """Upper confidence bound on exceedance via the posterior-uniform envelope and a paid lambda grid."""
    lams = lam_grid(2.0 / W)
    Cg = C + np.log(N_LAM)
    v = (Ghat[..., None] + Cg / (lams * n)) / (1.0 - lams * W / 2.0)
    return v.min(-1)


def cdf_radius(n, W, C):
    lams = lam_grid(3.0 / W)
    Cg = C + np.log(N_LAM)
    return (Cg / (lams * n) + lams * W / (2.0 * (1.0 - W * lams / 3.0))).min()


def cp_lower(S, N, a):
    out = np.zeros(S.shape)
    ok = S > 0
    out[ok] = stats.beta.ppf(a, S[ok], N[ok] - S[ok] + 1)
    return out


def pick(scores, feasible, rng, maximize=True):
    """Permutation-invariant choice among feasible candidates, random tie-breaking."""
    s = np.where(feasible, scores if maximize else -scores, -np.inf)
    s = s + 1e-12 * rng.random(s.shape)
    j = s.argmax(1)
    return np.where(feasible.any(1), j, -1)


_FAN = (np.log(0.5) + 0.5) / 0.25


def bet_ucb_counts(Eb, nbs, n, W, C):
    """certs.exc_bet specialised to X = w 1{Y>tau}/W in {0,1} (rare-action class), from per-block
    exceedance counts Eb (list of (reps, K) arrays) of sizes nbs: S1_b = S2_b = E_b."""
    A = 0.0; B = 0.0; C0 = 0.0; S1p = np.zeros_like(Eb[0], float); cp = 0
    for E, nb in zip(Eb, nbs):
        v = (0.25 + np.maximum(S1p - S1p ** 2 / max(cp, 1), 0.0)) / (cp + 1.0)
        lam = np.minimum(np.sqrt(2.0 * C / (v * n)), 0.5)
        A = A + _FAN * lam ** 2 * nb
        B = B + lam * nb - 2.0 * _FAN * lam ** 2 * E
        C0 = C0 + (-lam * E + _FAN * lam ** 2 * E)
        S1p = S1p + E; cp += nb
    disc = B ** 2 - 4.0 * A * (C0 - C)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (-B + np.sqrt(np.maximum(disc, 0.0))) / (2.0 * A)
    m = np.where((disc >= 0) & (C0 + B + A >= C), np.clip(r, 0.0, 1.0), 1.0)
    return np.minimum(W * m, 1.0)


def blocks(n, b0=64):
    out, b = [], b0
    while sum(out) < n:
        out.append(min(b, n - sum(out))); b *= 2
    return out


def run_cell(n, W, K, qs, p, delta, reps, rng):
    """qs: array (K,) of P(Z=1) per candidate for this model (safe index randomized by caller)."""
    C = np.log(K / delta)
    probs = np.r_[np.full(K, 1.0 / W), 1.0 - K / W]
    # data generated in doubling blocks (iid, so exact) so the betting certificate can bet predictably
    nbs = blocks(n); Nb, Eb = [], []
    for nb in nbs:
        Nk = rng.multinomial(nb, probs, size=reps)[:, :K]; Sk = rng.binomial(Nk, qs)
        Nb.append(Nk); Eb.append(Nk - Sk)
    N = sum(Nb); E = sum(Eb); S = N - E         # totals: target-action and exceedance counts
    out = {}
    ubet = bet_ucb_counts(Eb, nbs, n, W, C)
    out["exc_bet"] = pick(-ubet, ubet <= 1 - p, rng)
    Ghat = W * E / n
    ucb = exc_ucb(Ghat, n, W, C)
    out["exc"] = pick(-ucb, ucb <= 1 - p, rng)
    ucb0 = exc_ucb(Ghat, n, W, np.log(1 / delta) - np.log(N_LAM))   # no search cost, no grid cost
    out["naive"] = pick(-ucb0, ucb0 <= 1 - p, rng)
    lcb = W * S / n - cdf_radius(n, W, C)
    out["cdf"] = pick(lcb, lcb >= p, rng)
    # empirical-Bernstein CDF bound (Maurer-Pontil Thm 4, ln(2/delta)); X_i = w 1{Y<=tau} in {0,W}.
    # With truncation c=W this is identical to the Thomas et al. (2015) bound used by UnO.
    L = C + np.log(2.0)
    Sf = S.astype(float)
    var = W ** 2 * Sf * (n - Sf) / (n * (n - 1.0))   # float: int64 overflows at n ~ 5e8
    lcb_eb = W * S / n - np.sqrt(2 * var * L / n) - 7 * W * L / (3 * (n - 1.0))
    out["cdf_eb"] = pick(lcb_eb, lcb_eb >= p, rng)
    # exceedance with empirical Bernstein (variance-adaptive; same statistic as "exc")
    Ef = E.astype(float)
    var_e = W ** 2 * Ef * (n - Ef) / (n * (n - 1.0))
    ucb_eb = W * E / n + np.sqrt(2 * var_e * L / n) + 7 * W * L / (3 * (n - 1.0))
    out["exc_eb"] = pick(-ucb_eb, ucb_eb <= 1 - p, rng)
    lb = cp_lower(S, N, delta / K)
    out["binom"] = pick(lb, lb >= p, rng)
    # split: two independent halves of sizes n1, n2 (fresh multinomial draws; exact for iid data)
    n1 = n // 2; n2 = n - n1
    N1 = rng.multinomial(n1, probs, size=reps)[:, :K]; E1 = rng.binomial(N1, 1 - qs)
    N2 = rng.multinomial(n2, probs, size=reps)[:, :K]; E2all = rng.binomial(N2, 1 - qs)
    sel = pick(-(W * E1 / n1), np.ones_like(N1, bool), rng)
    E2 = E2all[np.arange(reps), sel]
    ucb2 = exc_ucb(W * E2 / n2, n2, W, np.log(1 / delta))
    out["split"] = np.where(ucb2 <= 1 - p, sel, -1)
    return out


def np_ceiling(n, W, q0, q1, alpha, sd=9.0):
    """Randomized Neyman-Pearson power at size alpha for P0^n vs Pj^n (differs only on action j).
    Sufficient statistic (N_j, S_j); N_j ~ Bin(n, 1/W), S_j | N_j ~ Bin(N_j, q)."""
    mu = n / W
    m = np.arange(max(0, int(mu - sd * np.sqrt(mu) - 5)), int(mu + sd * np.sqrt(mu) + 6))
    pm = stats.binom.pmf(m, n, 1.0 / W)
    lo = min(q0, q1); hi = max(q0, q1)
    smin = max(0, int(m.min() * lo - sd * np.sqrt(m.max() * 0.25) - 5))
    smax = int(m.max() * hi + sd * np.sqrt(m.max() * 0.25) + 6)
    s = np.arange(smin, smax + 1)
    M, Sg = np.meshgrid(m, s, indexing="ij")
    valid = Sg <= M
    P0 = np.where(valid, pm[:, None] * stats.binom.pmf(Sg, M, q0), 0.0).ravel()
    P1 = np.where(valid, pm[:, None] * stats.binom.pmf(Sg, M, q1), 0.0).ravel()
    llr = Sg * np.log(q1 / q0) + (M - Sg) * np.log((1 - q1) / (1 - q0))
    llr = llr.ravel()
    keep = (P0 > 0) | (P1 > 0)
    P0, P1, llr = P0[keep], P1[keep], llr[keep]
    omitted1 = max(0.0, 1.0 - P1.sum())          # conservatively credited to the ceiling
    order = np.argsort(-llr, kind="stable")
    c0 = np.cumsum(P0[order]); c1 = np.cumsum(P1[order])
    k = np.searchsorted(c0, alpha, side="right")
    if k >= len(order):
        return min(1.0, c1[-1] + omitted1)
    prev0 = c0[k - 1] if k > 0 else 0.0
    prev1 = c1[k - 1] if k > 0 else 0.0
    frac = (alpha - prev0) / P0[order[k]] if P0[order[k]] > 0 else 1.0
    return min(1.0, prev1 + frac * P1[order[k]] + omitted1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--reps", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=20270928)
    ap.add_argument("--out-dir", default=RESULTS)
    ap.add_argument("--only-validity", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    delta = 0.05
    rows = []
    # power study: margin gamma = (1-p)/2
    settings = [(0.90, 0.05), (0.95, 0.025), (0.99, 0.005)]
    for W, K in ([] if a.only_validity else [(64, 2), (64, 8), (64, 32)]):
        for p, g in settings:
            base = W * (1 - p) * np.log(K / delta) / g ** 2   # predicted scale n* (in samples)
            for mult in np.geomspace(0.05, 3000, 34):
                n = int(max(4 * W, mult * base))
                j = rng.integers(K, size=a.reps)
                qs = np.full((a.reps, K), p - g); qs[np.arange(a.reps), j] = p + g
                res = run_cell(n, W, K, qs, p, delta, a.reps, rng)
                # valid ceiling = min of (a) class null p-g at size delta/K (search-aware) and
                # (b) boundary null p at size delta (only action j differs from P_j in both)
                # ceiling only where informative (it is ~1 beyond 60 n*; enumeration cost grows with n)
                ceil = (min(np_ceiling(n, W, p - g, p + g, delta / K),
                            np_ceiling(n, W, p, p + g, delta)) if mult <= 60 else np.nan)
                for meth, sel in res.items():
                    rows.append(dict(W=W, K=K, p=p, gamma=g, n=n, n_over_W=n / W,
                                     x_norm=n / base, method=meth,
                                     power=np.mean(sel == j), abstain=np.mean(sel < 0)))
                rows.append(dict(W=W, K=K, p=p, gamma=g, n=n, n_over_W=n / W, x_norm=n / base,
                                 method="ceiling", power=ceil, abstain=np.nan))
            print("done", W, K, p, flush=True)
    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(a.out_dir, "tail_boundary_results.csv"), index=False)

    # validity: all candidates unsafe, exceedance just above 1-p
    vrows = []
    for W, K in [(64, 32)]:
        for p in [0.9, 0.95, 0.99]:
            for eps_rel in [0.02, 0.05, 0.2]:
                q = p - eps_rel * (1 - p)   # P(Z=1); exceedance = (1-p)(1+eps_rel) >= 1-p: unsafe
                for mult in [0.3, 1, 3, 10]:
                    n = int(mult * W * (1 - p) * np.log(K / delta) / (0.5 * (1 - p)) ** 2)
                    qs = np.full((a.reps * 3, K), q)
                    res = run_cell(n, W, K, qs, p, delta, a.reps * 3, rng)
                    for meth, sel in res.items():
                        fc = np.mean(sel >= 0)
                        vrows.append(dict(W=W, K=K, p=p, excess_rel=eps_rel, n=n, method=meth,
                                          false_cert=fc, reps=a.reps * 3))
    pd.DataFrame(vrows).to_csv(os.path.join(a.out_dir, "tail_validity_results.csv"), index=False)


if __name__ == "__main__":
    main()
