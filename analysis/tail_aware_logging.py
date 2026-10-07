"""Proposition 2 (tail-aware logging) on RouterBench (appendix "Tail-aware logging"): can logs designed for
the tail certify a policy's tail with fewer samples than logs collected by the policy itself (on-policy)?

RouterBench, fixed stochastic target routers pi_beta(a|x) = (1-beta) 1{a = Mistral-7B} + beta 1{a = GPT-4}
(a cheap default that escalates a fraction beta of requests to the strong model).  Cost threshold tau from
the main study; certify G(tau) = P(cost > tau) <= 1 - p for ONE fixed target (K = 1, no selection).

Logging policies (all fixed before any certification data; same uniform floor u):
  on-policy  : pi0 = (1-u) pi + u/M
  uniform    : pi0 = 1/M
  tail-aware : pi0 = (1-u) pi* + u/M,  pi*(a|x) ∝ pi(a|x) sqrt(ghat(x,a))   (Proposition 2),
               ghat = P(cost > tau | task, model) estimated on the DEV split only (smoothed).
For each logging policy and log size n we draw logs from the EVAL population, certify with the paper's
certificates (exc_bet, exc_pb, exc_eb; delta = 0.05) and record the certification rate; n90 = log size at
which the target is certified with probability 0.9.  Also reports the tail-effective overlap V_tau.

Outputs (default ../results/tail_aware/routerbench/): tail_aware_results.csv (certification rate per n),
tail_aware_results_meta.json (beta, G, W, V_tau per configuration), tail_aware_results_summary.csv (n90).
Requires the RouterBench table (downloaded by routerbench_replay.load_data into --data-dir).

usage: python tail_aware_logging.py [--reps 300] [--workers 8] [--data-dir DIR] [--out FILE.csv]
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
sys.path.insert(0, os.path.join(HERE, "..", "routerbench"))
import certs                      # noqa: E402
import routerbench_replay as rb   # noqa: E402

DELTA = 0.05
U = 0.1                           # uniform exploration floor shared by all logging policies


_CFG = {}


def run_config(job):
    """One (p, rel_margin, logging) configuration; data shared via fork-inherited globals."""
    p, r, name, reps, seed = job
    d = _CFG[(p, r)]
    P0 = d["logs"][name]; pie, Ee = d["pi_ev"], d["E_ev"]; Pe = P0[d["ev"]]
    W = float((pie / Pe).max()) * (1 + 1e-9)
    Vt = float((pie ** 2 / Pe * Ee).sum(1).mean())
    rng = np.random.default_rng([seed, int(p * 1000), int(r * 1000), ["on-policy", "uniform", "tail-aware"].index(name)])
    cum = np.cumsum(Pe, 1); M = Pe.shape[1]; rows = []
    C1 = np.log(1.0 / DELTA)
    for n in np.unique(np.geomspace(200, 400000, 32).astype(int)):
        cert = {}
        for c0 in range(0, reps, 50):                          # chunk reps to bound memory
            rc = min(50, reps - c0)
            idx = rng.integers(0, Pe.shape[0], size=(rc, n))
            A = np.minimum((rng.random((rc, n))[..., None] > cum[idx]).sum(-1), M - 1)
            w = pie[idx, A] / Pe[idx, A]
            exc = Ee[idx, A]
            for meth, f in (("exc_betting", certs.exc_bet), ("exceedance", certs.exc_pb),
                            ("exc_empbernstein", certs.exc_eb)):
                cert[meth] = cert.get(meth, 0) + int((f(w, exc, W, C1) <= 1 - p).sum())
            del idx, A, w, exc
        for meth, k in cert.items():
            rows.append(dict(p=p, rel_margin=r, beta=d["beta"], logging=name, method=meth, n=int(n),
                             cert_rate=k / reps))
    info = dict(beta=d["beta"], G=d["G"], W=W, V_tau=Vt, margin=(1 - p) - d["G"])
    print(f"p={p} r={r} beta={d['beta']:.4f} {name:10s} G={d['G']:.4f} W={W:.1f} V_tau={Vt:.4f}", flush=True)
    return rows, (f"p{p}_r{r}_{name}", info)


def main():
    from multiprocessing import Pool
    ap = argparse.ArgumentParser(description="Tail-aware logging on RouterBench.")
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20261003)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--data-dir", default=rb.DATA_DIR)
    ap.add_argument("--out", default=os.path.join(HERE, "..", "results", "tail_aware", "routerbench",
                                                  "tail_aware_results.csv"))
    a = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)

    df, Q, Ccost, audit = rb.load_data(a.data_dir)
    N, M = Ccost.shape
    split = np.random.default_rng(20261001).random(N)
    dev, ev = split < 0.3, split >= 0.3
    task = df.eval_name.values
    taus = {0.90: 0.000546386549881854, 0.95: 0.000630957344480193}   # main-study budgets (dev rule, meta.json)
    jobs = []
    for p, tau in taus.items():
        E = (Ccost > tau).astype(float)                      # exceedance indicator per (prompt, model)
        # ghat on DEV only: per (task, model) mean exceedance, shrunk to the per-model mean
        ghat = np.zeros((N, M))
        for m in range(M):
            gm = E[dev, m].mean()
            t_dev = pd.Series(E[dev, m]).groupby(task[dev]).agg(["sum", "count"])
            est = (t_dev["sum"] + 5 * gm) / (t_dev["count"] + 5)
            ghat[:, m] = pd.Series(task).map(est).fillna(gm).values
        ghat = np.clip(ghat, 1e-3, 1.0)
        g0, g4 = E[ev, 0].mean(), E[ev, 4].mean()
        for r in (0.5, 0.2, 0.1):                                            # relative margin below budget
            beta = float(((1 - r) * (1 - p) - g0) / (g4 - g0))               # target placed at G = (1-r)(1-p)
            pi = np.zeros((N, M)); pi[:, 0] = 1 - beta; pi[:, 4] = beta      # Mistral-7B / GPT-4
            G = float((pi[ev] * E[ev]).sum(1).mean())
            if G > 1 - p:
                continue                                                    # only certify safe targets
            star = pi * np.sqrt(ghat); star /= star.sum(1, keepdims=True)
            logs = {"on-policy": (1 - U) * pi + U / M,
                    "uniform": np.full((N, M), 1.0 / M),
                    "tail-aware": (1 - U) * star + U / M}
            _CFG[(p, r)] = dict(logs=logs, pi_ev=pi[ev], E_ev=E[ev], ev=ev, beta=beta, G=G)
            jobs += [(p, r, name, a.reps, a.seed) for name in logs]
    with Pool(a.workers) as pool:
        res = pool.map(run_config, jobs, chunksize=1)
    rows = [x for rr, _ in res for x in rr]
    meta = dict(m for _, m in res)
    out = pd.DataFrame(rows)
    out.to_csv(a.out, index=False)
    with open(a.out.replace(".csv", "_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1)

    def n90(d):
        d = d.sort_values("n"); x, y = d.n.values, d.cert_rate.values
        i = int(np.argmax(y >= 0.9))
        if y[i] < 0.9: return np.nan
        if i == 0: return float(x[0])
        t = (0.9 - y[i - 1]) / (y[i] - y[i - 1])
        return float(np.exp(np.log(x[i - 1]) + t * (np.log(x[i]) - np.log(x[i - 1]))))
    S = pd.DataFrame([dict(p=p, rel_margin=r, logging=l, method=m, n90=n90(d))
                      for (p, r, l, m), d in out.groupby(["p", "rel_margin", "logging", "method"])])
    T = S.pivot_table(index=["p", "rel_margin", "method"], columns="logging", values="n90")
    T["on/tail"] = T["on-policy"] / T["tail-aware"]
    pd.set_option("display.width", 200)
    print(T.round(1).to_string())
    T.to_csv(a.out.replace(".csv", "_summary.csv"))


if __name__ == "__main__":
    main()
