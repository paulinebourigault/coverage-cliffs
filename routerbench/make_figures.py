#!/usr/bin/env python3
"""fig_router.pdf (appendix "RouterBench certification curves") and summary tables from the
outputs of routerbench_replay.py.

  python make_figures.py [--res ../results/routerbench/main] [--stress ../results/routerbench/stress]
                         [--out ../figures] [--K 32] [--p 0.95] [--png]

fig_router.pdf      certification rate vs logged sample size n (Wilson 95% bands), one panel per
                    logging policy eps, one line per method; candidate family size K, level p
headline_table.csv  n needed for 50% / 90% certification (log-interpolated), all (method, p, eps, K)
stress_table.csv    per (tag, method, K, eps, p): max over n of the false-certification rate + CI
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results", "routerbench")

# colour = method identity; marker + linestyle = secondary encoding
STY = {
    "exceedance": dict(color="#2a78d6", marker="o", ls="-", label="Exceedance, PAC-Bayes"),
    "cdf_pacbayes": dict(color="#eb6834", marker="s", ls="-", label="CDF, PAC-Bayes Bernstein"),
    "cdf_empbernstein": dict(color="#eda100", marker="^", ls="-", label="CDF, emp. Bernstein (UnO)"),
    "exc_betting": dict(color="#4a3aa7", marker="D", ls="-", label="Exceedance, betting"),
    "exc_empbernstein": dict(color="#e87ba4", marker="v", ls=(0, (4, 2)), label="Exceedance, emp. Bernstein"),
    "split": dict(color="#1baf7a", marker="P", ls="-.", label="Split + exceedance"),
    "naive": dict(color="#e34948", marker="x", ls=":", label="Naive, no log K (invalid)"),
}
plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "lines.linewidth": 1.4,
    "lines.markersize": 3.5, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e3e3e3", "grid.linewidth": 0.5,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


PNG = False


def save(fig, fname):
    fig.savefig(fname, bbox_inches="tight")
    if PNG:
        fig.savefig(fname[:-4] + ".png", bbox_inches="tight", dpi=150)
    plt.close(fig)


HIDE = {"exc_empbernstein"}   # reported in the appendix tables only


def fig_certrate(s, meta, K, p, fname):
    epss = sorted(meta["eps"], reverse=True)
    fig, axes = plt.subplots(1, len(epss), figsize=(6.8, 2.2), sharey=True, squeeze=False)
    for ax, e in zip(axes[0], epss):
        for m, st in STY.items():
            if m in HIDE:
                continue
            sub = s[(s.method == m) & (s.eps == e) & (s.K == K) & (np.isclose(s.p, p))].sort_values("n")
            if sub.empty:
                continue
            ax.plot(sub.n, sub.cert_rate, color=st["color"], marker=st["marker"], ls=st["ls"],
                    label=st["label"], ms=3.5, mfc=st["color"])
            ax.fill_between(sub.n, sub.cert_lo, sub.cert_hi, color=st["color"], alpha=0.12, lw=0)
        W = meta["W"][str(e)][str(K)]
        ax.set_title(r"$\varepsilon=%g$  ($W=%g$)" % (e, W))
        ax.set_xscale("log")
        ax.set_xlabel("logged sample size $n$")
    axes[0][0].set_ylabel("certification rate")
    h, l = axes[0][0].get_legend_handles_labels()
    fig.tight_layout()
    fig.subplots_adjust(top=0.72)
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0))
    save(fig, fname)


def main():
    ap = argparse.ArgumentParser(description="RouterBench figure and tables.")
    ap.add_argument("--res", default=os.path.join(RESULTS, "main"), help="main-study results")
    ap.add_argument("--stress", default=os.path.join(RESULTS, "stress"), help="stress-test results")
    ap.add_argument("--out", default=os.path.join(HERE, "..", "figures"))
    ap.add_argument("--K", type=int, default=32)
    ap.add_argument("--p", type=float, default=0.95)
    ap.add_argument("--png", action="store_true", help="also write PNG copies")
    a = ap.parse_args()
    global PNG
    PNG = a.png
    os.makedirs(a.out, exist_ok=True)
    meta = json.load(open(os.path.join(a.res, "meta.json")))
    s = pd.read_csv(os.path.join(a.res, "summary_main.csv"))
    fig_certrate(s, meta, a.K, a.p, os.path.join(a.out, "fig_router.pdf"))
    hl = pd.read_csv(os.path.join(a.res, "headline.csv"))
    hl.to_csv(os.path.join(a.out, "headline_table.csv"), index=False)
    piv = hl.pivot_table(index=["p", "eps", "K"], columns="method", values="n90")
    print("n for 90% certification (nan = not reached):\n", piv.round(0).to_string())
    sp = os.path.join(a.stress, "summary_stress.csv")
    if os.path.exists(sp):
        ss = pd.read_csv(sp)
        idx = ss.groupby(["tag", "method", "K", "eps", "p"]).false_rate.idxmax()
        t = ss.loc[idx, ["tag", "method", "K", "eps", "p", "n", "reps", "false", "false_rate",
                         "false_lo", "false_hi", "cert_rate"]].rename(columns={"n": "n_at_max"})
        t.to_csv(os.path.join(a.out, "stress_table.csv"), index=False)
        print("stress: max false-cert rate over n\n",
              t[np.isclose(t.p, a.p)].pivot_table(index=["tag", "eps", "K"], columns="method",
                                                  values="false_rate").round(3).to_string())
    print("outputs in", a.out)


if __name__ == "__main__":
    main()
