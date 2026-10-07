"""Certificates computed from sufficient statistics, for candidates whose weights take values in {0, W}.

In the rare-action family and the affine-router family, candidate j has weight w_j = W * 1{A = action of j}.
Every certificate in common/certs.py then depends on the log only through, per doubling block b,
    N_b = number of observations whose action matches the candidate,
    E_b = number of those with Y > tau,
so the experiments draw these counts directly (exact for i.i.d. logs) instead of individual observations.
tests/test_count_certificates.py checks each function against the raw-data implementation in certs.py.

Information classes (what each rule may use):
  joint CDF indicators only : S = N - E (per block and in total)       -> cdf_eb, cdf_bet, cdf_exact
  exceedance indicators     : E                                         -> exc_eb, exc_bet
  action counts added       : (N, E)                                    -> counts_exact, norm_bet
All functions return upper confidence bounds on the exceedance G (certify iff bound <= 1 - p), at
confidence exp(-C) per candidate; arrays broadcast over leading (reps, K) axes.
"""
import numpy as np
from scipy import stats

_FAN = (np.log(0.5) + 0.5) / 0.25   # log(1+y) >= y + _FAN y^2 for y >= -1/2


def blocks(n, b0=64):
    """Doubling block sizes used by the betting certificates (as in certs.exc_bet)."""
    out, b, tot = [], b0, 0
    while tot < n:
        out.append(min(b, n - tot)); tot += out[-1]; b *= 2
    return out


def _crossing(A, B, C0, C):
    """Smallest m in [0, 1] with A m^2 + B m + C0 >= C (quadratic increasing on [0, 1]); 1 if none."""
    reach = C0 + B + A >= C
    with np.errstate(invalid="ignore", divide="ignore"):
        disc = B ** 2 - 4.0 * A * (C0 - C)
        quad = (-B + np.sqrt(np.maximum(disc, 0.0))) / (2.0 * A)
        lin = (C - C0) / B
    r = np.where(A < 0, quad, lin)
    return np.where(reach, np.clip(np.nan_to_num(r, nan=0.0), 0.0, 1.0), 1.0)


def bet_mean_ucb(Xb, nbs, n, C):
    """Upper confidence bound on the mean of X in {0, 1} from per-block sums Xb (list of arrays), with the
    predictable plug-in bets of certs.exc_bet (S1 = S2 for binary X)."""
    A = 0.0; B = 0.0; C0 = 0.0; Sp = np.zeros_like(Xb[0], float); cp = 0
    for S, nb in zip(Xb, nbs):
        v = (0.25 + np.maximum(Sp - Sp ** 2 / max(cp, 1), 0.0)) / (cp + 1.0)
        lam = np.minimum(np.sqrt(2.0 * C / (v * n)), 0.5)
        A = A + _FAN * lam ** 2 * nb
        B = B + lam * nb - 2.0 * _FAN * lam ** 2 * S
        C0 = C0 + (-lam * S + _FAN * lam ** 2 * S)
        Sp = Sp + S; cp += nb
    return _crossing(A, B, C0, C)


# ---------------------------------------------------------------- exceedance indicators only
def exc_bet(Eb, nbs, n, W, C):
    """certs.exc_bet: X = w 1{Y > tau} / W = 1{match, Y > tau}."""
    return np.minimum(W * bet_mean_ucb(Eb, nbs, n, C), 1.0)


def exc_eb(E, n, W, C):
    """certs.exc_eb (Maurer-Pontil empirical Bernstein on w 1{Y > tau} in {0, W})."""
    L = C + np.log(2.0)
    Ef = np.asarray(E, float)
    var = W ** 2 * Ef * (n - Ef) / (n * (n - 1.0))
    return np.clip(W * Ef / n + np.sqrt(2 * var * L / n) + 7 * W * L / (3 * (n - 1.0)), 0.0, 1.0)


# ---------------------------------------------------------------- joint CDF indicators only
def cdf_eb(S, n, W, C):
    """certs.cdf_eb: 1 - empirical-Bernstein lower bound on F from w 1{Y <= tau} in {0, W}."""
    L = C + np.log(2.0)
    Sf = np.asarray(S, float)
    var = W ** 2 * Sf * (n - Sf) / (n * (n - 1.0))
    lcb = W * Sf / n - np.sqrt(2 * var * L / n) - 7 * W * L / (3 * (n - 1.0))
    return np.clip(1.0 - lcb, 0.0, 1.0)


def cdf_bet(Sb, nbs, n, W, C):
    """Betting lower bound on F = W E[X], X = w 1{Y <= tau} / W = 1{match, Y <= tau}: the same test as
    exc_bet applied to 1 - X, whose block sums are n_b - S_b.  Returns 1 - LCB_F."""
    ucb_comp = bet_mean_ucb([nb - S for S, nb in zip(Sb, nbs)], nbs, n, C)     # UCB on 1 - E[X]
    lcb_F = W * (1.0 - ucb_comp)
    return np.clip(1.0 - lcb_F, 0.0, 1.0)


def cdf_exact(S, n, W, C):
    """Exact one-sided Clopper-Pearson bound from the CDF success count alone: S ~ Bin(n, F / W), so
    LCB_F = W * beta.ppf(e^-C; S, n - S + 1).  The strongest simple rule using only the CDF indicators."""
    S = np.asarray(S)
    a = np.exp(-C)
    lo = np.where(S > 0, stats.beta.ppf(a, np.maximum(S, 1), n - S + 1), 0.0)
    return np.clip(1.0 - W * lo, 0.0, 1.0)


# ---------------------------------------------------------------- action counts added
def counts_exact(N, E, C):
    """Exact conditional bound: given N matched observations, E ~ Bin(N, G) (contexts and actions are
    independent of the outcome model's labels), so UCB_G = beta.ppf(1 - e^-C; E + 1, N - E); 1 if N = 0."""
    N = np.asarray(N); E = np.asarray(E)
    a = np.exp(-C)
    up = np.where(E < N, stats.beta.ppf(1.0 - a, E + 1, np.maximum(N - E, 1)), 1.0)
    return np.where(N > 0, up, 1.0)


def norm_bet(Nb, Eb, nbs, n, W, C):
    """certs.norm_bet with v = w / W = 1{match}: block sums s0 = t0 = N_b and s1 = t1 = E_b."""
    A = 0.0; B = 0.0; C0 = 0.0
    s0p = np.zeros_like(Nb[0], float); s1p = np.zeros_like(Nb[0], float); cp = 0
    for N, E, nb in zip(Nb, Eb, nbs):
        mhat = np.clip(W * s1p / max(cp, 1), 0.0, 1.0)
        sy = mhat * s0p - s1p
        sy2 = mhat ** 2 * s0p - 2.0 * mhat * s1p + s1p
        var = (0.25 + np.maximum(sy2 - sy ** 2 / max(cp, 1), 0.0)) / (cp + 1.0)
        lam = np.minimum(np.sqrt(2.0 * C / (var * n)), 0.5)
        A = A + _FAN * lam ** 2 * N
        B = B + lam * N - 2.0 * _FAN * lam ** 2 * E
        C0 = C0 + (-lam * E + _FAN * lam ** 2 * E)
        s0p = s0p + N; s1p = s1p + E; cp += nb
    return _crossing(A, B, C0, C)
