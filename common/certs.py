"""Certificates shared by all experiments (simulations, RouterBench replay, reasoning-budget study).

Conventions
-----------
Data for one candidate policy at one threshold tau:
    w   : importance weights w_i = pi(A_i|X_i)/pi0(A_i|X_i), shape (..., n)
    exc : exceedance indicators 1{Y_i > tau},            shape (..., n)
W is a known bound on the weights; C is the complexity term
    C = KL(rho||pi) + log(|T|/delta)   (uniform prior over K candidates => C = log(K|T|/delta)).
Every function returns an upper bound on the EXCEEDANCE G(tau) = P(Y > tau), so that all methods are
compared on the same decision rule:  certify  <=>  bound <= 1 - p.
CDF-based methods bound F(tau) from below; we return 1 - LCB_F.

Methods
-------
exc_pb      : authors' specialization of a nonnegative-loss PAC-Bayes bound to weighted exceedances
exc_bet     : exceedance betting certificate (appendix "The betting certificate")
exc_eb      : exceedance, empirical Bernstein (Lemma 1; variance-adaptive; finite classes)
norm_bet    : normalization-aware betting certificate (uses the known weight mean E[w] = 1)
cdf_pb      : PAC-Bayes Bernstein lower bound on the IW-CDF (CDF-envelope route)
cdf_eb      : empirical-Bernstein (Maurer & Pontil 2009) lower bound on the IW-CDF
cdf_thomas  : Thomas et al. (2015) high-confidence lower bound on the IW-CDF indicator mean (the
              per-threshold interval used by UnO, Chandak et al. 2021); truncation c=W by default, in
              which case it equals cdf_eb, so it is not listed in METHODS.
With the fixed proxies used in the experiments, these CDF certificates use only w_i 1{Y_i <= tau};
norm_bet also uses the weights of non-exceedances and is outside that restriction.
"""
import numpy as np

N_LAM = 60


def lam_grid(upper, n_lam=N_LAM):
    return np.geomspace(1e-7 * upper, 0.999 * upper, n_lam)


def exc_pb(w, exc, W, C, n_lam=N_LAM):
    """Upper bound on G via the posterior-uniform exceedance envelope with a paid finite lambda grid."""
    n = w.shape[-1]
    Ghat = (w * exc).mean(-1)
    lams = lam_grid(2.0 / W, n_lam)
    Cg = C + np.log(n_lam)
    v = (Ghat[..., None] + Cg / (lams * n)) / (1.0 - lams * W / 2.0)
    return np.minimum(v.min(-1), 1.0)


def exc_pb_from_mean(Ghat, n, W, C, n_lam=N_LAM):
    lams = lam_grid(2.0 / W, n_lam)
    Cg = C + np.log(n_lam)
    v = (np.asarray(Ghat)[..., None] + Cg / (lams * n)) / (1.0 - lams * W / 2.0)
    return np.minimum(v.min(-1), 1.0)


def cdf_pb(w, exc, W, C, n_lam=N_LAM, V=None):
    """1 - LCB_F with the PAC-Bayes Bernstein CDF envelope. V (default W) must be an independently
    justified population second-moment upper bound, not an unprotected sample estimate."""
    n = w.shape[-1]
    Fhat = (w * (1 - exc)).mean(-1)
    V = W if V is None else V
    lams = lam_grid(3.0 / W, n_lam)
    Cg = C + np.log(n_lam)
    Cg = np.asarray(Cg, float)[..., None]
    r = (Cg / (lams * n) + lams * np.asarray(V, float)[..., None] / (2.0 * (1.0 - W * lams / 3.0))).min(-1)
    return np.clip(1.0 - (Fhat - r), 0.0, 1.0)


def cdf_eb(w, exc, W, C):
    """1 - LCB_F, empirical Bernstein (Maurer & Pontil 2009, Thm 4) on X = w 1{Y<=tau} in [0,W],
    at confidence exp(-C)."""
    n = w.shape[-1]
    X = w * (1 - exc)
    m = X.mean(-1)
    var = X.var(-1, ddof=1)
    L = C + np.log(2.0)  # Maurer-Pontil Thm 4 uses ln(2/delta) (variance is estimated)
    lcb = m - np.sqrt(2 * var * L / n) - 7 * W * L / (3 * (n - 1))
    return np.clip(1.0 - lcb, 0.0, 1.0)


def cdf_thomas(w, exc, W, C, c=None):
    """1 - LCB_F with the Thomas et al. (2015, Thm 1) bound for nonnegative X with truncation c,
    applied to X = w 1{Y<=tau}; confidence exp(-C)."""
    n = w.shape[-1]
    c = W if c is None else c
    Y = np.minimum(w * (1 - exc), c)
    L = C + np.log(2.0)  # Thomas et al. (2015) Thm 1 uses ln(2/delta)
    s1 = Y.sum(-1); s2 = (Y ** 2).sum(-1)
    spread = np.maximum(n * s2 - s1 ** 2, 0.0)
    lcb = s1 / n - 7 * c * L / (3 * (n - 1)) - (1.0 / n) * np.sqrt(2 * L * spread / (n - 1))
    return np.clip(1.0 - lcb, 0.0, 1.0)


def exc_eb(w, exc, W, C):
    """Upper bound on G, empirical Bernstein (Maurer & Pontil 2009, Thm 4) on X = w 1{Y>tau} in [0,W],
    at confidence exp(-C).  Same statistic as exc_pb (so the tail-mass advantage is kept) but the
    variance term adapts to the realised weight variance instead of the worst-case proxy W*G.
    Valid for a finite candidate class via C = log(K/delta); for general posteriors use exc_pb."""
    n = w.shape[-1]
    X = w * exc
    m = X.mean(-1)
    var = X.var(-1, ddof=1)
    L = C + np.log(2.0)  # Maurer-Pontil Thm 4 uses ln(2/delta)
    ucb = m + np.sqrt(2 * var * L / n) + 7 * W * L / (3 * (n - 1))
    return np.clip(ucb, 0.0, 1.0)


_FAN = (np.log(0.5) + 0.5) / 0.25   # log(1+y) >= y + _FAN*y^2 for y >= -1/2 (Fan, Grama & Liu 2015)


def bet_quadratic(w, exc, W, C, block0=64):
    """Coefficients (A, B, C0) of the concave quadratic lower bound Q(m) = A m^2 + B m + C0 on the log-wealth
    of the betting test used by exc_bet (m in units of G/W).  Q is increasing on [0, 1]."""
    n = w.shape[-1]
    X = (w * exc) / W
    C = np.asarray(C, float)
    shape = np.broadcast_shapes(C.shape, X.shape[:-1])
    A = np.zeros(shape); B = np.zeros(shape); C0 = np.zeros(shape)
    S1p = np.zeros(X.shape[:-1]); S2p = np.zeros(X.shape[:-1]); cp = 0
    start, b = 0, block0
    while start < n:
        stop = min(n, start + b)
        xb = X[..., start:stop]; nb = stop - start
        v = (0.25 + np.maximum(S2p - S1p ** 2 / max(cp, 1), 0.0)) / (cp + 1.0)
        lam = np.minimum(np.sqrt(2.0 * C / (v * n)), 0.5)
        S1 = xb.sum(-1); S2 = (xb ** 2).sum(-1)
        A = A + _FAN * lam ** 2 * nb
        B = B + lam * nb - 2.0 * _FAN * lam ** 2 * S1
        C0 = C0 + (-lam * S1 + _FAN * lam ** 2 * S2)
        S1p = S1p + S1; S2p = S2p + S2; cp += nb
        start, b = stop, 2 * b
    return A, B, C0


def exc_bet_stratified(strata, q, W, C, n_split=2001, n_bisect=50):
    """Upper bound on G = q1 G_1 + q2 G_2 for two independent strata (each i.i.d. within, fixed counts).
    strata: [(w1, exc1), (w2, exc2)], w_s of shape (K, n_s) (or (n_s,)); q = (q1, q2), q1 + q2 = 1.
    Under the truth, the product of the strata's betting wealths at their true means is an e-value
    (independent strata; each a nonnegative supermartingale), so with probability >= 1 - e^-C,
        min over splits {q1 W m1 + q2 W m2 = G}  [Q1(m1) + Q2(m2)]  <  C,
    where Q_s is the quadratic lower bound of exc_bet.  The returned bound is the smallest g at which this
    minimum reaches C (it is increasing in g); uses one confidence budget for both strata.
    The constrained quadratic is concave, so its exact minimum is attained at one of the two
    feasible endpoints. n_split is retained for backwards compatibility and is unused."""
    (A1, B1, D1), (A2, B2, D2) = [bet_quadratic(np.atleast_2d(w), np.atleast_2d(e), W, C) for w, e in strata]
    q1, q2 = q
    if not (0.0 < q1 < 1.0 and 0.0 < q2 < 1.0 and np.isclose(q1 + q2, 1.0)):
        raise ValueError("q must contain two positive mixture weights summing to one")

    def minQ(x):                                   # x = g / W, shape (K,)
        left = np.maximum(0.0, (x - q2) / q1)
        right = np.minimum(1.0, x / q1)

        def endpoint(m1):
            m2 = (x - q1 * m1) / q2
            return (A1 * m1 ** 2 + B1 * m1 + D1
                    + A2 * m2 ** 2 + B2 * m2 + D2)

        return np.minimum(endpoint(left), endpoint(right))
    lo = np.zeros_like(A1); hi = np.ones_like(A1)
    cert = minQ(hi) >= C                           # if even g = W is not rejected, bound is trivial
    for _ in range(n_bisect):
        mid = (lo + hi) / 2
        rej = minQ(mid) >= C
        hi = np.where(rej, mid, hi); lo = np.where(rej, lo, mid)
    return np.where(cert, np.minimum(W * hi, 1.0), 1.0)


def exc_bet(w, exc, W, C, block0=64):
    """Upper bound on G via a predictable-plug-in betting test (Waudby-Smith & Ramdas 2024) on
    X = w 1{Y>tau}/W in [0,1].  For a level alpha = exp(-C), wealth(m) = prod_i (1 + lam_i (m - X_i)) is a
    nonnegative supermartingale under E[X] >= m when lam_i is predictable and lam_i <= 1/2; Ville gives
    P(wealth(E X) >= 1/alpha) <= alpha.  lam is constant within doubling blocks, chosen from earlier blocks
    only, and does not depend on m, so the lower bound
        log wealth(m) >= sum_b [lam_b (n_b m - S1_b) + FAN lam_b^2 (n_b m^2 - 2 m S1_b + S2_b)]
    is a concave quadratic in m that is increasing on [0, 1] (lam <= 1/2).  The bound returned is
    W * (smallest m with that quadratic >= C), i.e. a valid (1-alpha) upper confidence bound on G.
    Valid for a finite candidate class via C = log(K/delta); for general posteriors use exc_pb."""
    n = w.shape[-1]
    X = (w * exc) / W
    C = np.asarray(C, float)
    shape = np.broadcast_shapes(C.shape, X.shape[:-1])
    A = np.zeros(shape); B = np.zeros(shape); C0 = np.zeros(shape)
    S1p = np.zeros(X.shape[:-1]); S2p = np.zeros(X.shape[:-1]); cp = 0
    start, b = 0, block0
    while start < n:
        stop = min(n, start + b)
        xb = X[..., start:stop]; nb = stop - start
        v = (0.25 + np.maximum(S2p - S1p ** 2 / max(cp, 1), 0.0)) / (cp + 1.0)   # predictable variance
        lam = np.minimum(np.sqrt(2.0 * C / (v * n)), 0.5)
        S1 = xb.sum(-1); S2 = (xb ** 2).sum(-1)
        A = A + _FAN * lam ** 2 * nb
        B = B + lam * nb - 2.0 * _FAN * lam ** 2 * S1
        C0 = C0 + (-lam * S1 + _FAN * lam ** 2 * S2)
        S1p = S1p + S1; S2p = S2p + S2; cp += nb
        start, b = stop, 2 * b
    # solve A m^2 + B m + (C0 - C) = 0 for the smallest root in [0, 1] (A < 0, increasing on [0, 1])
    disc = B ** 2 - 4.0 * A * (C0 - C)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (-B + np.sqrt(np.maximum(disc, 0.0))) / (2.0 * A)
    m = np.where((disc >= 0) & (C0 + B + A >= C), np.clip(r, 0.0, 1.0), 1.0)
    return np.minimum(W * m, 1.0)


def norm_bet(w, exc, W, C, block0=64):
    """Upper bound on G via a normalization-aware betting test (known-weight-mean control variate, in the
    spirit of Karampatziakis, Mineiro & Ramdas 2021).  With v_i = w_i / W in [0, 1] and
        wealth(m) = prod_i (1 + lam_i v_i (m - 1{Y_i > tau})),
    E[v (m - 1{Y > tau})] = (m E[w] - G) / W = (m - G) / W because E_pi0[w] = 1 (Assumption 1: the target's
    support is covered by the logger).  So wealth(G) is a nonnegative martingale for predictable
    lam_i <= 1/2, and Ville gives P(wealth(G) >= e^C) <= e^-C.  Unlike exc_bet, the test uses the weights of
    non-exceedances too (the action counts in the rare-action model), i.e. the information that
    unnormalized CDF indicators discard.  As in exc_bet, lam is constant within doubling blocks and set
    from earlier blocks only (plug-in variance of v (m_hat - Z) at the running estimate m_hat), and
        log wealth(m) >= sum_b [lam_b (m s0_b - s1_b) + FAN lam_b^2 (m^2 t0_b - 2 m t1_b + t1_b)]
    with block sums s0 = sum v, s1 = sum v Z, t0 = sum v^2, t1 = sum v^2 Z.  This quadratic is increasing on
    [0, 1] (each term has derivative lam v (1 + 2 FAN lam v (m - Z)) >= 0 since |2 FAN lam| < 1), so the
    bound returned is the smallest m in [0, 1] at which it reaches C (1 if none).  Finite candidate classes
    via C = log(K/delta)."""
    n = w.shape[-1]
    v = w / W
    Z = np.broadcast_to(exc, v.shape)
    C = np.asarray(C, float)
    shape = np.broadcast_shapes(C.shape, v.shape[:-1])
    A = np.zeros(shape); B = np.zeros(shape); C0 = np.zeros(shape)
    s0p = np.zeros(v.shape[:-1]); s1p = np.zeros(v.shape[:-1])
    t0p = np.zeros(v.shape[:-1]); t1p = np.zeros(v.shape[:-1]); cp = 0
    start, b = 0, block0
    while start < n:
        stop = min(n, start + b)
        vb = v[..., start:stop]; zb = Z[..., start:stop]; nb = stop - start
        mhat = np.clip(W * s1p / max(cp, 1), 0.0, 1.0)                 # predictable estimate of G
        sy = mhat * s0p - s1p                                           # sum of v (mhat - Z) so far
        sy2 = mhat ** 2 * t0p - 2.0 * mhat * t1p + t1p                  # sum of its squares (Z^2 = Z)
        var = (0.25 + np.maximum(sy2 - sy ** 2 / max(cp, 1), 0.0)) / (cp + 1.0)
        lam = np.minimum(np.sqrt(2.0 * C / (var * n)), 0.5)
        s0 = vb.sum(-1); s1 = (vb * zb).sum(-1); t0 = (vb ** 2).sum(-1); t1 = (vb ** 2 * zb).sum(-1)
        A = A + _FAN * lam ** 2 * t0
        B = B + lam * s0 - 2.0 * _FAN * lam ** 2 * t1
        C0 = C0 + (-lam * s1 + _FAN * lam ** 2 * t1)
        s0p = s0p + s0; s1p = s1p + s1; t0p = t0p + t0; t1p = t1p + t1; cp += nb
        start, b = stop, 2 * b
    return _smallest_crossing(A, B, C0, C)


def _smallest_crossing(A, B, C0, C):
    """Smallest m in [0, 1] with A m^2 + B m + C0 >= C for a quadratic (A <= 0) increasing on [0, 1];
    1 if the level is not reached on [0, 1]."""
    reach = C0 + B + A >= C
    with np.errstate(invalid="ignore", divide="ignore"):
        disc = B ** 2 - 4.0 * A * (C0 - C)
        quad = (-B + np.sqrt(np.maximum(disc, 0.0))) / (2.0 * A)
        lin = (C - C0) / B
    r = np.where(A < 0, quad, lin)
    return np.where(reach, np.clip(np.nan_to_num(r, nan=0.0), 0.0, 1.0), 1.0)


# cdf_thomas with c=W (no truncation) is algebraically identical to cdf_eb, so it is not listed
# separately: cdf_empbernstein is the UnO / Thomas et al. bound without truncation.
METHODS = {"exceedance": exc_pb, "exc_betting": exc_bet, "exc_empbernstein": exc_eb,
           "cdf_pacbayes": cdf_pb, "cdf_empbernstein": cdf_eb, "norm_betting": norm_bet}


def complexity(K, delta, n_thresholds=1, kl=None):
    """C = KL + log(|T|/delta); uniform prior over K candidates with point posteriors => KL = log K."""
    kl = np.log(K) if kl is None else kl
    return kl + np.log(n_thresholds / delta)


def select_and_certify(bounds, p, utility=None, rng=None):
    """bounds: (..., K) exceedance upper bounds. Returns index of certified candidate or -1.
    If utility (..., K) is given: maximize utility among certified; else pick smallest bound.
    Random tie-breaking."""
    rng = np.random.default_rng() if rng is None else rng
    feas = bounds <= 1 - p
    score = -bounds if utility is None else utility
    score = np.where(feas, score, -np.inf) + 1e-12 * rng.random(bounds.shape)
    j = score.argmax(-1)
    return np.where(feas.any(-1), j, -1)
