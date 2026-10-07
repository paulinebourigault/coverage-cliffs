"""Main-text figures fig_teaser.pdf and fig_predict.pdf from the shipped results.

fig_teaser.pdf  : RouterBench, uniform logging (eps=1), K=32 -- logged requests needed to certify the selected
                  router's cost constraint with probability 0.9, per certificate and quantile level.
                  Certificates that never certify within the largest log size are drawn as hatched bars to
                  that size and marked with a cross.
fig_predict.pdf : predicted vs observed median log size (empirical-Bernstein certificates), from the measured
                  tail- and bulk-effective overlaps, RouterBench and reasoning study; censored observations
                  (never certified within the largest log) are drawn at that size with an upward arrow.
Inputs: ../results/routerbench/main/headline.csv, ../results/routerbench/predict_vs_observed_K32.csv,
        ../results/reasoning/certify/predict_vs_observed_K32.csv (written by predict_vs_observed.py).

usage: python make_main_figures.py [--out ../figures]
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")
RB = os.path.join(RESULTS, "routerbench", "main")
RB_PRED = os.path.join(RESULTS, "routerbench", "predict_vs_observed_K32.csv")
REASONING_PRED = os.path.join(RESULTS, "reasoning", "certify", "predict_vs_observed_K32.csv")
NMAX = {"rb": 300000, "reasoning": 98000}      # largest log size tested in each study

ap = argparse.ArgumentParser(description="Plot fig_teaser.pdf and fig_predict.pdf.")
ap.add_argument("--out", default=os.path.join(HERE, "..", "figures"))
OUT = ap.parse_args().out
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#8a8984", "axes.labelcolor": "#0b0b0b",
                     "xtick.color": "#52514e", "ytick.color": "#52514e", "legend.fontsize": 7})
INK2 = "#52514e"

# ------------------------------------------------------------------ teaser
h = pd.read_csv(os.path.join(RB, "headline.csv"))
h = h[(h.K == 32) & np.isclose(h.eps, 1.0)]
methods = [("exc_betting", "Exceedance, betting", "#4a3aa7"),
           ("exc_empbernstein", "Exceedance, emp. Bernstein [H22]", "#e87ba4"),
           ("cdf_empbernstein", "CDF, emp. Bernstein (UnO)", "#eda100"),
           ("cdf_pacbayes", "CDF, PAC-Bayes Bernstein", "#eb6834")]
ps = [0.90, 0.95, 0.99]
fig, ax = plt.subplots(figsize=(3.3, 2.35))
bw = 0.19
for k, (m, lab, col) in enumerate(methods):
    for i, p in enumerate(ps):
        v = h[(h.method == m) & np.isclose(h.p, p)].n90
        v = float(v.iloc[0]) if len(v) else np.nan
        x = i + (k - 1.5) * bw
        if np.isfinite(v):
            ax.bar(x, v, bw * 0.92, color=col, label=lab if i == 0 else None, zorder=2)
        else:
            ax.bar(x, NMAX["rb"], bw * 0.92, color="none", edgecolor=col, hatch="////", lw=0.8,
                   label=lab if i == 0 else None, zorder=2)
            ax.text(x, NMAX["rb"] * 1.12, "×", color=col, ha="center", va="bottom", fontsize=8, fontweight="bold")
ax.set_yscale("log"); ax.set_ylim(300, 4e6)
ax.set_xticks(range(len(ps))); ax.set_xticklabels([f"$p={p:.2f}$" for p in ps])
ax.set_ylabel("logged requests to certify\n(90% probability)")
ax.axhline(NMAX["rb"], color="#8a8984", lw=0.6, ls=":", zorder=1)
ax.text(1.0, NMAX["rb"] * 0.82, "largest log tested", fontsize=6, color=INK2, ha="center", va="top")
ax.grid(axis="y", color="#e8e7e2", lw=0.6, zorder=0)
ax.legend(loc="upper left", frameon=False, ncol=2, handlelength=1.0, columnspacing=0.8, fontsize=6)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_teaser.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT, "fig_teaser.png"), dpi=170, bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------------------ predicted vs observed
fig, ax = plt.subplots(figsize=(3.3, 2.6))
for src, path, mk in (("rb", RB_PRED, "o"), ("reasoning", REASONING_PRED, "s")):
    d = pd.read_csv(path)
    for stat, pc, oc, col in (("exceedance", "pred_exc", "obs_exc", "#4a3aa7"),
                              ("CDF", "pred_cdf", "obs_cdf", "#eda100")):
        pred, obs = d[pc].values, d[oc].values
        cens = ~np.isfinite(obs)
        lab = f"{stat}, {'RouterBench' if src == 'rb' else 'reasoning study'}"
        ax.scatter(pred[~cens], obs[~cens], s=22, marker=mk, color=col, edgecolor="white", lw=0.6,
                   zorder=3, label=lab)
        if cens.any():
            ax.scatter(pred[cens], np.full(cens.sum(), NMAX[src]), s=22, marker=mk, facecolor="none",
                       edgecolor=col, lw=0.9, zorder=3)
            for x in pred[cens]:
                ax.annotate("", xy=(x, NMAX[src] * 2.0), xytext=(x, NMAX[src] * 1.05),
                            arrowprops=dict(arrowstyle="->", color=col, lw=0.8))
lim = (300, 2e6)
ax.plot(lim, lim, color=INK2, lw=0.8, zorder=1)
ax.fill_between(lim, [l / 2 for l in lim], [l * 2 for l in lim], color="#e8e7e2", zorder=0, lw=0,
                label="within a factor of 2")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(*lim); ax.set_ylim(*lim)
ax.set_xlabel("predicted log size (from measured overlap)")
ax.set_ylabel("observed log size (median)")
ax.text(1.1e6, 6.5e5, "$y=x$", fontsize=6, color=INK2, ha="right")
ax.legend(loc="upper left", frameon=False, handletextpad=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_predict.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT, "fig_predict.png"), dpi=170, bbox_inches="tight")
print("wrote fig_teaser and fig_predict to", OUT)
