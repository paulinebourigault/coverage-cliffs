"""Implementation.  Bounds are upper confidence bounds on the exceedance G(tau) = P(Y > tau).

method="betting"   predictable plug-in betting test (Waudby-Smith & Ramdas 2024) on X = w 1{Y>tau}/W,
                   doubling blocks, closed-form inversion; finite candidate lists.
method="bernstein" empirical Bernstein (Maurer & Pontil 2009) on w 1{Y>tau}; finite candidate lists.
method="pacbayes"  PAC-Bayes exceedance envelope; also valid for randomized policies chosen from the
                   data (posterior rho with KL(rho || prior) passed as `kl`).
"""
import numpy as np

_FAN = (np.log(0.5) + 0.5) / 0.25   # log(1+y) >= y + _FAN y^2 for y >= -1/2
_N_LAM = 60


def _as2d(w):
    w = np.asarray(w, float)
    if w.ndim not in (1, 2):
        raise ValueError("w must have shape (n,) or (n, K)")
    return w[:, None] if w.ndim == 1 else w


def _probability(value, name):
    if np.ndim(value) != 0:
        raise ValueError(f"{name} must be a scalar in (0, 1)")
    value = float(value)
    if not np.isfinite(value) or not 0 < value < 1:
        raise ValueError(f"{name} must be in (0, 1)")
    return value


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or np.ndim(value) != 0:
        raise ValueError(f"{name} must be a positive integer")
    value = float(value)
    if not np.isfinite(value) or value < 1 or value != int(value):
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _bet(X, C, block0=64):
    """X in [0,1], shape (K, n); C = log(1/alpha) per row. Returns UCB on E[X]."""
    K, n = X.shape
    A = np.zeros(K); B = np.zeros(K); C0 = np.zeros(K)
    S1p = np.zeros(K); S2p = np.zeros(K); cp = 0
    start, b = 0, block0
    while start < n:
        stop = min(n, start + b)
        xb = X[:, start:stop]; nb = stop - start
        v = (0.25 + np.maximum(S2p - S1p ** 2 / max(cp, 1), 0.0)) / (cp + 1.0)
        lam = np.minimum(np.sqrt(2.0 * C / (v * n)), 0.5)
        S1 = xb.sum(1); S2 = (xb ** 2).sum(1)
        A += _FAN * lam ** 2 * nb
        B += lam * nb - 2.0 * _FAN * lam ** 2 * S1
        C0 += -lam * S1 + _FAN * lam ** 2 * S2
        S1p += S1; S2p += S2; cp += nb
        start, b = stop, 2 * b
    disc = B ** 2 - 4.0 * A * (C0 - C)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (-B + np.sqrt(np.maximum(disc, 0.0))) / (2.0 * A)
    return np.where((disc >= 0) & (C0 + B + A >= C), np.clip(r, 0.0, 1.0), 1.0)


def exceedance_bound(w, y, tau, W=None, delta=0.05, K=1, method="betting", kl=None, n_thresholds=1):
    """Upper confidence bound(s) on P_pi(Y > tau).

    w : (n,) or (n, K) importance weights pi_j(A_i|X_i) / pi0(A_i|X_i) of each candidate on the log
    y : (n,) observed losses;  tau : threshold;  W : required known almost-sure bound on the weights.
    W cannot be estimated by the largest observed weight: unseen actions may have larger weights.
    delta is split as delta / (K * n_thresholds) for finite lists (betting / bernstein);
    for method="pacbayes", each column must contain the exact posterior-averaged weights of its policy
    mixture; pass kl = KL(rho || prior), either one common scalar or one value per column. The prior and
    underlying policy family must be fixed independently of this log, and W must bound that family.
    Positive infinite KL yields the trivial bound one. For fixed candidates, omit kl to use log(K).
    Returns one bound per column (simultaneously valid with probability >= 1 - delta).
    """
    w = _as2d(w); y = np.asarray(y, float)
    n, K_w = w.shape
    delta = _probability(delta, "delta")
    K = _positive_integer(K, "K")
    n_thresholds = _positive_integer(n_thresholds, "n_thresholds")
    if np.ndim(tau) != 0 or not np.isfinite(tau):
        raise ValueError("tau must be a finite scalar")
    if y.shape != (n,):
        raise ValueError("y must have shape (n,) matching w")
    if not np.all(np.isfinite(y)):
        raise ValueError("observed losses y must be finite")
    if w.size == 0:
        raise ValueError("importance weights must contain at least one observation")
    if not np.all(np.isfinite(w)) or np.any(w < 0):
        raise ValueError("importance weights must be finite and nonnegative")
    if W is None:
        raise ValueError("W is required: supply a known almost-sure weight bound, not the observed maximum")
    W = float(W)
    if not np.isfinite(W) or W <= 0:
        raise ValueError("W must be finite and positive")
    if w.max() > W:
        raise ValueError("observed weight exceeds W")
    K = max(K, K_w)
    X = (w * (y > tau)[:, None]).T                      # (K_w, n)
    if method == "betting":
        C = np.full(K_w, np.log(K * n_thresholds / delta))
        return np.minimum(W * _bet(X / W, C), 1.0)
    if method == "bernstein":
        if n < 2:
            raise ValueError("the empirical-Bernstein certificate requires n >= 2")
        L = np.log(K * n_thresholds / delta) + np.log(2.0)
        m = X.mean(1); var = X.var(1, ddof=1)
        return np.clip(m + np.sqrt(2 * var * L / n) + 7 * W * L / (3 * (n - 1)), 0.0, 1.0)
    if method == "pacbayes":
        KL = np.asarray(np.log(K) if kl is None else kl, float)
        if KL.ndim == 0:
            KL = np.full(K_w, float(KL))
        elif KL.shape != (K_w,):
            raise ValueError("kl must be a scalar or have one value per column of w")
        if np.any(np.isnan(KL)) or np.any(KL < 0):
            raise ValueError("kl must be nonnegative (positive infinity gives a vacuous bound)")
        C = KL + np.log(n_thresholds / delta) + np.log(_N_LAM)
        lams = np.geomspace(1e-7 * 2.0 / W, 0.999 * 2.0 / W, _N_LAM)
        G = X.mean(1)[:, None]
        return np.minimum(((G + C[:, None] / (lams * n)) / (1.0 - lams * W / 2.0)).min(1), 1.0)
    raise ValueError("method must be 'betting', 'bernstein' or 'pacbayes'")


def certify(w, y, tau, p, W=None, delta=0.05, method="betting"):
    """Certify P_pi(Y > tau) <= 1 - p for ONE target policy fixed before the log.
    Returns dict(certified, bound, G_hat, V_tau_hat, n)."""
    p = _probability(p, "p")
    w = np.asarray(w, float).ravel()
    ub = float(exceedance_bound(w, y, tau, W=W, delta=delta, K=1, method=method)[0])
    return dict(certified=ub <= 1 - p, bound=ub, G_hat=float(np.mean(w * (np.asarray(y) > tau))),
                V_tau_hat=tail_overlap(w, y, tau), n=len(w))


def select_and_certify(w, y, tau, p, utility=None, W=None, delta=0.05, method="betting"):
    """Among K candidates (columns of w, fixed before the log), return the certified candidate with the
    highest utility (or the smallest bound if utility is None).  Valid after selection: all K bounds hold
    simultaneously with probability >= 1 - delta.  Returns dict(index (-1 if none), bounds, certified)."""
    p = _probability(p, "p")
    w = _as2d(w)
    ub = exceedance_bound(w, y, tau, W=W, delta=delta, K=w.shape[1], method=method)
    ok = ub <= 1 - p
    if not ok.any():
        return dict(index=-1, bounds=ub, certified=ok)
    score = -ub if utility is None else np.asarray(utility, float)
    return dict(index=int(np.argmax(np.where(ok, score, -np.inf))), bounds=ub, certified=ok)


def tail_overlap(w, y, tau):
    """Plug-in estimate of V_tau = E_pi0[w^2 1{Y > tau}] (per column if w is 2-D)."""
    w = np.asarray(w, float)
    e = (np.asarray(y) > tau)
    return np.mean(w ** 2 * (e if w.ndim == 1 else e[:, None]), axis=0)


def required_n(V_tau, gamma, W, K=1, delta=0.05, beta=0.1, mode="typical"):
    """Log size needed to certify a candidate whose exceedance is gamma below the budget 1 - p.

    mode="typical"   : n solving sqrt(2 V L / n) + 7 W L / (3 n) = gamma, L = log(2K/delta) -- the
                       empirical-Bernstein radius at the population variance (V ~ V_tau); this is the
                       retrospective planning proxy studied in the paper, not a guaranteed median.
    mode="guarantee" : max{32 V L_K / gamma^2, 11 W L_K / gamma}, L_K = log(2 / min(delta/K, beta/2))
                       from the empirical-Bernstein certification-and-power lemma: a specified safe
                       candidate is certified with probability >= 1 - beta. V_tau must be a justified
                       population-moment upper bound. This is not a guarantee that a selection rule
                       outputs that candidate; selected-output power needs separate error allocation.
    The betting certificate has no universal pointwise ordering relative to this proxy."""
    V, g = float(V_tau), float(gamma)
    if g <= 0:
        return float("inf")
    if mode == "guarantee":
        L = np.log(2.0 / min(delta / K, beta / 2.0))
        return float(max(32 * V * L / g ** 2, 11 * W * L / g))
    L = np.log(K / delta) + np.log(2.0)
    a, b = 7 * W * L / 3, np.sqrt(2 * V * L)
    x = (-b + np.sqrt(b ** 2 + 4 * a * g)) / (2 * a)       # x = 1/sqrt(n)
    return float(1.0 / x ** 2)


def tail_aware_logging(pi, g_hat, floor=0.1):
    """Logging policy pi0 proportional to pi * sqrt(g_hat), mixed with uniform exploration `floor`.
    pi, g_hat : (n_contexts, M) target probabilities and estimated P(Y > tau | x, a).
    With exact tail probabilities, the unmixed proportional rule minimizes the target's tail second
    moment over loggers when admissible (otherwise it gives its infimum). The uniform mixture ensures
    support and bounds weights by M / floor for floor > 0; it is not generally the constrained optimum.
    g_hat must be estimated on data separate from the certification log."""
    pi = np.asarray(pi, float); g = np.clip(np.asarray(g_hat, float), 0.0, 1.0)
    star = pi * np.sqrt(g)
    s = star.sum(1, keepdims=True)
    star = np.where(s > 0, star / np.where(s > 0, s, 1.0), pi)
    M = pi.shape[1]
    return (1 - floor) * star + floor / M
