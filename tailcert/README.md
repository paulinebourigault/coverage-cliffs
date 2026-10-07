# tailcert

Certify a **quantile (tail) constraint** of a policy from logged contextual-bandit data, e.g.
"with 95% confidence, at most 10% of requests exceed 2,000 tokens", before deploying it.

```python
import tailcert as tc
# w: importance weights pi(a|x)/pi0(a|x) on the log (n,) or (n, K); y: observed losses (n,)
tc.certify(w, y, tau=2000, p=0.90, W=20)                         # one fixed target policy
tc.select_and_certify(w_K, y, tau=2000, p=0.90, utility=acc_K, W=20)   # best certified of K candidates
```

Planning, before collecting data:

```python
V = tc.tail_overlap(w_pilot, y_pilot, tau)          # tail-effective overlap E[w^2 1{Y > tau}]
tc.required_n(V, gamma=0.02, W=20, K=32)              # log size needed (typical; mode="guarantee" requires a justified population-moment upper bound)
pi0 = tc.tail_aware_logging(pi, g_hat, floor=0.1)     # tail-aware proposal mixed with 10% uniform exploration
```

Methods: `"betting"` (default; no universal pointwise dominance), `"bernstein"`, `"pacbayes"` (also valid for randomized policies
selected on the same data; pass `kl`). Requirements: known logging propensities, an explicit known almost-sure weight bound `W` (the sample maximum is not valid), and
candidates / threshold / level fixed before the log. Only numpy is required. Tests: `python tests/test_tailcert.py`.

For PAC-Bayes use, the weighted statistic must be the exact posterior expectation and `W` must bound the underlying policy family. Passing an unprotected Monte Carlo approximation and the continuous generating posterior's KL does not establish validity. A fixed sampled mixture can instead be certified as a fixed policy (or as one member of a predeclared finite list).

For `exceedance_bound(..., method="pacbayes", kl=...)`, each column contains exact posterior-averaged weights, and `kl` is either a common scalar or one KL value per column, relative to an independently fixed prior. Infinite KL gives the trivial bound one. Empirical Bernstein requires at least two observations; all methods require finite observed losses and a finite threshold.

The `required_n` guarantee concerns certification of a specified safe candidate, not the probability that a selection rule outputs it. The typical mode is a planning proxy. The unmixed tail-aware proposal minimizes the fixed target's tail second moment when the supplied tail probabilities are exact and the proposal is admissible; adding uniform exploration enforces support but need not solve the corresponding constrained optimization problem.
