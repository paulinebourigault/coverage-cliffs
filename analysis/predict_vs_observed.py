"""Appendix "Predicted versus observed log sizes": predict, from the measured overlaps, the log size at
which each certificate first certifies some router, and compare with the observed n50.

For candidate j with exceedance G_j < 1-p, an empirical-Bernstein certificate on the statistic X
(exceedance: X = w 1{Y>tau}; CDF: X = w 1{Y<=tau}) certifies once, ignoring sampling fluctuation,
    sqrt(2 Var(X) L / n) + 7 W L / (3 n) <= gamma_j,   gamma_j = (1-p) - G_j,   L = log(2K/delta).
Var(X) = V_tau - G^2 for the exceedance and V_F - F^2 for the CDF (V_tau, V_F: tail- and
bulk-effective overlaps, Theorem 3). The predicted n50 for "certify some router" is the minimum over
feasible candidates.

Inputs (in --res): ground_truth_routers.csv and either headline.csv (RouterBench replay) or
summary_main.csv + meta.json (reasoning-budget study).
Output: predict_vs_observed_K{K}.csv, the input of make_main_figures.py (fig_predict.pdf).

usage: python predict_vs_observed.py [--dataset routerbench|reasoning] [--res DIR] [--out FILE] [--K 32]
  defaults: routerbench -> reads ../results/routerbench/main, writes ../results/routerbench/
            reasoning   -> reads and writes ../results/reasoning/certify
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")
DEFAULTS = {"routerbench": (os.path.join(RESULTS, "routerbench", "main"), os.path.join(RESULTS, "routerbench")),
            "reasoning": (os.path.join(RESULTS, "reasoning", "certify"),) * 2}
DELTA = 0.05

ap = argparse.ArgumentParser(description="Predicted vs observed log sizes.")
ap.add_argument("--dataset", choices=sorted(DEFAULTS), default="routerbench")
ap.add_argument("--res", default=None, help="results directory (default: per --dataset)")
ap.add_argument("--out", default=None, help="output CSV (default: per --dataset)")
ap.add_argument("--K", type=int, default=32)
args = ap.parse_args()
res = args.res or DEFAULTS[args.dataset][0]
K = args.K
out_path = args.out or os.path.join(DEFAULTS[args.dataset][1], f"predict_vs_observed_K{K}.csv")

gt = pd.read_csv(os.path.join(res, "ground_truth_routers.csv")).iloc[:K]
L = np.log(K / DELTA) + np.log(2.0)
EPS_W = ((1.0, 5.0), (0.5, 10.0), (0.2, 25.0))
if os.path.exists(os.path.join(res, "headline.csv")):          # RouterBench replay output
    head = pd.read_csv(os.path.join(res, "headline.csv"))
    PS = (0.9, 0.95, 0.99)
else:                                                          # reasoning study: n50 from summary_main
    sm = pd.read_csv(os.path.join(res, "summary_main.csv"))
    sm = sm[(sm.K == K) & (sm.tag == "main")]

    def n50(d):
        d = d.sort_values("n"); x, y = d.n.values, d.cert_rate.values
        i = int(np.argmax(y >= 0.5))
        if y[i] < 0.5: return np.nan
        if i == 0: return float(x[0])
        t = (0.5 - y[i - 1]) / (y[i] - y[i - 1])
        return float(np.exp(np.log(x[i - 1]) + t * (np.log(x[i]) - np.log(x[i - 1]))))
    head = pd.DataFrame([dict(method=m, p=p, eps=e, K=K, n50=n50(d))
                         for (m, p, e), d in sm.groupby(["method", "p", "eps"])])
    Wm = json.load(open(os.path.join(res, "meta.json")))["W"]
    EPS_W = tuple((float(e), float(v[str(K)]) * (1 + 1e-6)) for e, v in Wm.items())
    PS = tuple(sorted(head.p.unique()))


@np.errstate(invalid="ignore")
def n_needed(var, W, gamma):
    a, b = 7 * W * L / 3, np.sqrt(2 * np.maximum(var, 0) * L)
    x = (-b + np.sqrt(b ** 2 + 4 * a * gamma)) / (2 * a)       # x = 1/sqrt(n)
    return 1.0 / x ** 2


rows = []
for p in PS:
    G = gt[f"G_true_p{p}"].values
    feas = G < 1 - p
    for eps, W in EPS_W:
        Vt, VF = gt[f"V_tau_p{p}_eps{eps}"].values, gt[f"V_F_p{p}_eps{eps}"].values
        gam = (1 - p) - G
        n_exc = n_needed(Vt - G ** 2, W, gam)[feas].min()
        n_cdf = n_needed(VF - (1 - G) ** 2, W, gam)[feas].min()
        obs = head[(head.K == K) & np.isclose(head.p, p) & np.isclose(head.eps, eps)].set_index("method").n50
        rows.append(dict(p=p, eps=eps, pred_exc=n_exc, obs_exc=obs.get("exc_empbernstein"),
                         pred_cdf=n_cdf, obs_cdf=obs.get("cdf_empbernstein"),
                         pred_ratio=n_cdf / n_exc,
                         obs_ratio=obs.get("cdf_empbernstein") / obs.get("exc_empbernstein")))
out = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(out.round(2).to_string(index=False))
os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
out.to_csv(out_path, index=False)
print("wrote", out_path)
