"""tailcert: certify a quantile (tail) constraint of a policy from logged contextual-bandit data.

Question answered: from a log collected by a known behavior policy pi0, can we certify, with confidence
1 - delta, that a target policy pi (or the best of K candidates, selected on the same log) satisfies
    P(Y > tau) <= 1 - p        (equivalently: its p-quantile of the loss Y is at most tau)?

Main entry points
    certify(w, y, tau, p, W=W)                 one target policy
    select_and_certify(w, y, tau, p, utility=utility, W=W)  best certified candidate among K
    tail_overlap(w, y, tau)                    estimated tail-effective overlap V_tau = E[w^2 1{Y > tau}]
    required_n(V_tau, gamma, ...)              log size needed (planning, before collecting data)
    tail_aware_logging(pi, g_hat)              tail-aware proposal mixed with uniform exploration

All certificates use the importance-weighted exceedance  G_hat = mean(w * 1{y > tau}).
Requirements: propensities pi0 known and > 0 wherever the target acts; an explicit finite positive
almost-sure weight bound W is required (the observed maximum is not substituted);
candidates, tau and p fixed before looking at the log.
"""
from .core import (certify, select_and_certify, exceedance_bound, tail_overlap, required_n,
                   tail_aware_logging)

__all__ = ["certify", "select_and_certify", "exceedance_bound", "tail_overlap", "required_n",
           "tail_aware_logging"]
__version__ = "0.1.0"
