"""fig_tail_boundary.pdf and the n/n* summary table from the output of sim_tail_boundary.py.

Reads  <res>/tail_boundary_results.csv (and, if present, <res>/tail_validity_results.csv);
writes <out>/fig_tail_boundary.{pdf,png} and <out>/tail_boundary_summary.csv
(n/n* at which each method first reaches 90% power, log-interpolated).

usage: python plot_tail_boundary.py [--res ../results/sims] [--out ../figures]
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser(description="Plot fig_tail_boundary.pdf from simulation results.")
ap.add_argument("--res", default=os.path.join(HERE, "..", "results", "sims"))
ap.add_argument("--out", default=os.path.join(HERE, "..", "figures"))
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

df = pd.read_csv(os.path.join(args.res, "tail_boundary_results.csv"))
COL = {"exc": "#2a78d6", "exc_bet": "#4a3aa7", "exc_eb": "#e87ba4", "cdf": "#eb6834", "cdf_eb": "#eda100", "split": "#1baf7a", "binom": "#8a8984"}
LAB = {"exc": "Exceedance, PAC-Bayes", "exc_bet": "Exceedance, betting", "exc_eb": "Exceedance, emp. Bernstein", "cdf": "CDF, PAC-Bayes Bernstein", "cdf_eb": "CDF, emp. Bernstein / UnO",
       "split": "Split + exceedance",
       "binom": "Exact binomial (model-specific)"}
PSTY = {0.90: "-", 0.95: "--", 0.99: ":"}

plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#8a8984", "axes.labelcolor": "#0b0b0b",
                     "xtick.color": "#52514e", "ytick.color": "#52514e"})
Ks = sorted(df.K.unique())
fig, axes = plt.subplots(1, len(Ks), figsize=(6.8, 2.2), sharey=True)
for ax, K in zip(axes, Ks):
    d = df[df.K == K]
    for p in sorted(d.p.unique()):
        dp = d[d.p == p]
        c = dp[dp.method == "ceiling"].sort_values("x_norm").copy()
        c["power"] = c.power.fillna(1.0)   # ceiling not enumerated beyond 60 n* (it is 1 there)
        if p == 0.90:
            ax.fill_between(c.x_norm, 0, c.power, color="#d9d8d3", alpha=0.6, lw=0,
                            label="Oracle ceiling (any valid rule)")
        for m in ["exc", "exc_bet", "cdf", "cdf_eb", "split", "binom"]:
            s = dp[dp.method == m].sort_values("x_norm")
            ax.plot(s.x_norm, s.power, PSTY[p], color=COL[m], lw=1.6,
                    label=LAB[m] if p == 0.90 else None)
    ax.set_xscale("log"); ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"$K={K}$", fontsize=8)
    ax.set_xlabel(r"$n / n^\star$,  $n^\star = W(1-p)\log(K/\delta)/\gamma^2$")
    ax.grid(axis="y", color="#e8e7e2", lw=0.6)
axes[0].set_ylabel("P(select & certify safe)")
h, l = axes[0].get_legend_handles_labels()
h += [plt.Line2D([], [], color="#0b0b0b", lw=1.0, ls=PSTY[p]) for p in PSTY]
l += [f"line style: $p={p:.2f}$" for p in PSTY]
fig.legend(h, l, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.2), fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(args.out, "fig_tail_boundary.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(args.out, "fig_tail_boundary.png"), dpi=160, bbox_inches="tight")

# summary: n/n* at which each method first reaches 90% power (log-interpolated)
def cross(s, lvl=0.9):
    s = s.sort_values("x_norm"); x, y = s.x_norm.values, s.power.values
    i = np.argmax(y >= lvl)
    if y[i] < lvl: return np.nan
    if i == 0: return x[0]
    t = (lvl - y[i-1]) / (y[i] - y[i-1])
    return float(np.exp(np.log(x[i-1]) + t * (np.log(x[i]) - np.log(x[i-1]))))
rows = []
for (K, p), d in df.groupby(["K", "p"]):
    r = {"K": K, "p": p}
    for m in ["exc", "exc_bet", "exc_eb", "cdf", "cdf_eb", "split", "binom", "ceiling", "naive"]:
        r[m] = cross(d[d.method == m])
    rows.append(r)
S = pd.DataFrame(rows)
S["cdf/exc"] = S.cdf / S.exc; S["cdf_eb/exc_bet"] = S.cdf_eb / S.exc_bet; S["exc_bet/binom"] = S.exc_bet / S.binom; S["cdf_eb/exc"] = S.cdf_eb / S.exc; S["exc/binom"] = S.exc / S.binom; S["exc/ceiling"] = S.exc / S.ceiling
S["split/exc"] = S.split / S.exc
pd.set_option("display.width", 200)
print("n/n* at 90% power"); print(S.round(3).to_string(index=False))
S.to_csv(os.path.join(args.out, "tail_boundary_summary.csv"), index=False)

vpath = os.path.join(args.res, "tail_validity_results.csv")
if os.path.exists(vpath):
    v = pd.read_csv(vpath)
    print("\nfalse certification (max over n)")
    print(v.groupby(["p", "excess_rel", "method"]).false_cert.max().unstack("method").round(4))
