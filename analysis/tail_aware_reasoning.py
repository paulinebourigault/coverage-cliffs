"""Proposition 2 (tail-aware logging) on the reasoning study (appendix "Tail-aware logging"): same protocol as tail_aware_logging.py
(RouterBench), on the 7-configuration open-weight outcome table (outcome = generated tokens).

Fixed stochastic targets pi_beta(a|x) = (1-beta) 1{a = Qwen3-30B-A3B no-think} + beta 1{a = Qwen3-30B-A3B think}
(answer directly, escalate a fraction beta of requests to thinking), with beta set so that
G(tau) = P(tokens > tau) = (1-r)(1-p) on the EVAL population.  Budgets tau are the main study's dev-rule budgets.
Logging policies (fixed before any certification data):
  target     : (1-u) pi + u/M
  uniform    : 1/M
  tail-aware : (1-u) pi* + u/M,  pi* ∝ pi sqrt(ghat),  ghat = P(tokens > tau | dataset, level, action)
               estimated on the DEV split only (shrunk to the per-action mean).
Certify G(tau) <= 1-p for the fixed target (K = 1, delta = 0.05) with exc_bet / exc_pb / exc_eb.

Input: the reasoning-study outcome table (default ../results/reasoning/table/table.parquet).
Outputs (default ../results/tail_aware/reasoning/): tail_aware_reasoning_results.csv,
tail_aware_reasoning_results_meta.json, tail_aware_reasoning_results_summary.csv (n90).

usage: python tail_aware_reasoning.py [--reps 300] [--workers 8] [--table FILE.parquet] [--out FILE.csv]
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import certs                      # noqa: E402

DELTA = 0.05
U = 0.1
ACTS = ["q1.7b_nothink", "q1.7b_think", "q8b_nothink", "q8b_think", "r1d_qwen7b", "q30b_nothink", "q30b_think"]
A0, A1 = ACTS.index("q30b_nothink"), ACTS.index("q30b_think")
TAUS = {0.90: 2161.0, 0.95: 3509.0}   # main-study budgets (dev rule, certify/meta.json)
NAMES = ["target", "uniform", "tail-aware"]
_CFG = {}


def run_config(job):
    p, r, name, reps, seed, ns = job
    d = _CFG[(p, r)]
    Pe = d["logs"][name][d["ev"]]; pie, Ee = d["pi_ev"], d["E_ev"]
    W = float((pie / Pe).max()) * (1 + 1e-9)
    Vt = float((pie ** 2 / Pe * Ee).sum(1).mean())
    rng = np.random.default_rng([seed, int(p * 1000), int(r * 1000), NAMES.index(name)])
    cum = np.cumsum(Pe, 1); M = Pe.shape[1]; rows = []
    C1 = np.log(1.0 / DELTA)
    for n in ns:
        cert = {}
        for c0 in range(0, reps, 50):
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
            rows.append(dict(p=p, rel_margin=r, beta=d["beta"], logging=name, method=meth, n=int(n), cert_rate=k / reps))
    info = dict(beta=d["beta"], G=d["G"], W=W, V_tau=Vt, margin=(1 - p) - d["G"])
    print(f"p={p} r={r} beta={d['beta']:.4f} {name:10s} G={d['G']:.4f} W={W:.1f} V_tau={Vt:.4f}", flush=True)
    return rows, (f"p{p}_r{r}_{name}", info)


def n90(d):
    d = d.sort_values("n"); x, y = d.n.values, d.cert_rate.values
    i = int(np.argmax(y >= 0.9))
    if y[i] < 0.9: return np.nan
    if i == 0: return float(x[0])
    t = (0.9 - y[i - 1]) / (y[i] - y[i - 1])
    return float(np.exp(np.log(x[i - 1]) + t * (np.log(x[i]) - np.log(x[i - 1]))))


def main():
    from multiprocessing import Pool
    ap = argparse.ArgumentParser(description="Tail-aware logging on the reasoning study.")
    ap.add_argument("--table", default=os.path.join(HERE, "..", "results", "reasoning", "table", "table.parquet"))
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20261004)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--n-max", type=int, default=400000)
    ap.add_argument("--out", default=os.path.join(HERE, "..", "results", "tail_aware", "reasoning",
                                                  "tail_aware_reasoning_results.csv"))
    a = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)

    t = pd.read_parquet(a.table)
    dev, ev = (t.partition == "dev").values, (t.partition == "eval").values
    T = np.stack([t["tokens__" + x].values for x in ACTS], 1).astype(float)
    N, M = T.shape
    grp = (t.dataset + ":" + t.level.astype(str)).values
    ns = np.unique(np.geomspace(200, a.n_max, 32).astype(int))
    jobs = []
    for p, tau in TAUS.items():
        E = (T > tau).astype(float)
        ghat = np.zeros((N, M))
        for m in range(M):
            gm = E[dev, m].mean()
            s = pd.Series(E[dev, m]).groupby(grp[dev]).agg(["sum", "count"])
            est = (s["sum"] + 5 * gm) / (s["count"] + 5)
            ghat[:, m] = pd.Series(grp).map(est).fillna(gm).values
        ghat = np.clip(ghat, 1e-3, 1.0)
        g0, g1 = E[ev, A0].mean(), E[ev, A1].mean()
        for r in (0.5, 0.3, 0.2, 0.1):
            beta = float(((1 - r) * (1 - p) - g0) / (g1 - g0))
            if not 0 < beta < 1:
                print(f"skip p={p} r={r}: beta={beta:.4f} infeasible"); continue
            pi = np.zeros((N, M)); pi[:, A0] = 1 - beta; pi[:, A1] = beta
            G = float((pi[ev] * E[ev]).sum(1).mean())
            star = pi * np.sqrt(ghat); star /= star.sum(1, keepdims=True)
            logs = {"target": (1 - U) * pi + U / M, "uniform": np.full((N, M), 1.0 / M),
                    "tail-aware": (1 - U) * star + U / M}
            _CFG[(p, r)] = dict(logs=logs, pi_ev=pi[ev], E_ev=E[ev], ev=ev, beta=beta, G=G)
            jobs += [(p, r, name, a.reps, a.seed, ns) for name in NAMES]
    with Pool(a.workers) as pool:
        res = pool.map(run_config, jobs, chunksize=1)
    out = pd.DataFrame([x for rr, _ in res for x in rr])
    out.to_csv(a.out, index=False)
    with open(a.out.replace(".csv", "_meta.json"), "w") as fh:
        json.dump(dict(m for _, m in res), fh, indent=1)
    S = pd.DataFrame([dict(p=p, rel_margin=r, logging=l, method=m, n90=n90(d))
                      for (p, r, l, m), d in out.groupby(["p", "rel_margin", "logging", "method"])])
    Tb = S.pivot_table(index=["p", "rel_margin", "method"], columns="logging", values="n90")
    Tb["target/tail"] = Tb["target"] / Tb["tail-aware"]
    pd.set_option("display.width", 200)
    print(Tb.round(1).to_string())
    Tb.to_csv(a.out.replace(".csv", "_summary.csv"))


if __name__ == "__main__":
    main()
