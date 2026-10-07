#!/usr/bin/env python3
"""Step 5 (CPU): diagnostic figures from certify/ and table/ outputs (PDF + PNG in figures/).

  fig_cert_rate.pdf     certification rate vs n, panels by eps (default p=0.95, K=32)
  fig_accuracy.pdf      true accuracy of the deployed router vs n (fallback = reference when no cert)
  fig_token_dist.pdf    survival curves of generated tokens per action, with tau and the cap
  fig_stress.pdf        stress test: false-certification rate per method (Wilson 95% CI)

  python make_figures.py --config configs/open.yaml [--table-dir ../results/reasoning/table]
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import grlib

STY = {
    "exceedance": dict(color="#2a78d6", marker="o", ls="-", label="Exceedance (ours)"),
    "exc_betting": dict(color="#4a3aa7", marker="D", ls="-", label="Exceedance, betting (ours)"),
    "exc_empbernstein": dict(color="#e87ba4", marker="v", ls=(0, (4, 2)), label="Exceedance, emp. Bernstein"),
    "cdf_pacbayes": dict(color="#eb6834", marker="s", ls="--", label="CDF PAC-Bayes"),
    "cdf_empbernstein": dict(color="#1baf7a", marker="^", ls="-.", label="CDF emp. Bernstein (= UnO/Thomas, c=W)"),
    "split": dict(color="#52514e", marker="P", ls=":", label="Sample split"),
    "naive": dict(color="#e34948", marker="x", ls=(0, (1, 1)), label="Naive (no log K; invalid)"),
}
ACT_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#52514e", "#e34948", "#7b5ea7", "#8c6d31"]
W1, W2 = 3.3, 6.8
plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 6.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "lines.linewidth": 1.3, "lines.markersize": 3.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": "#e3e3e3", "grid.linewidth": 0.5, "pdf.fonttype": 42, "ps.fonttype": 42,
})


def save(fig, figd, fname):
    fig.savefig(os.path.join(figd, fname), bbox_inches="tight")
    fig.savefig(os.path.join(figd, fname.replace(".pdf", ".png")), bbox_inches="tight", dpi=200)
    plt.close(fig)


def plot_dist(tab, meta, col, xlabel, fname, figd, taus=None, cap=None):
    fig, ax = plt.subplots(figsize=(W1, 2.4))
    for i, a in enumerate(meta["actions"]):
        t = np.sort(tab[col + a].values)
        surv = 1.0 - np.arange(1, len(t) + 1) / len(t)
        trunc = (tab["finish__" + a] == "length").mean()
        ax.step(t, np.maximum(surv, 0.5 / len(t)), where="post", color=ACT_COLORS[i % 8], lw=1.2,
                label=f"{a} (truncated {trunc:.1%})")
    for k, (p, tau) in enumerate(sorted((taus or {}).items())):
        ls = ":" if k == 0 else "--"
        ax.axvline(tau, color="#222222", lw=0.8, ls=ls)
        ax.axhline(1 - float(p), color="#999999", lw=0.6, ls=ls)
        ax.annotate(f"$\\tau_{{{float(p):g}}}$={tau:.3g}", (tau, 0.6 - 0.25 * k), xycoords=("data", "axes fraction"),
                    fontsize=6, ha="right", xytext=(-2, 0), textcoords="offset points")
    if cap:
        ax.axvline(cap, color="#e34948", lw=0.8)
        ax.annotate("cap", (cap, 0.95), xycoords=("data", "axes fraction"), fontsize=6, ha="right",
                    color="#e34948", xytext=(-2, 0), textcoords="offset points")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(xlabel); ax.set_ylabel("$P(Y > t)$")
    ax.legend(frameon=False, loc="lower left", fontsize=5.5)
    fig.tight_layout(); save(fig, figd, fname)


def by_eps_panels(s, ycol, ylabel, fname, meta, figd, K, ref=None, lo=None, hi=None):
    epss = sorted(s.eps.unique(), reverse=True)
    fig, axs = plt.subplots(1, len(epss), figsize=(W2, 2.1), sharey=True, squeeze=False)
    for ax, e in zip(axs[0], epss):
        for m, st in STY.items():
            d = s[(s.eps == e) & (s.method == m)].sort_values("n")
            if d.empty:
                continue
            ax.plot(d.n, d[ycol], color=st["color"], marker=st["marker"], ls=st["ls"], label=st["label"])
            if lo and m == "exceedance":  # Wilson band for Exc-PB only (all CIs are in the summary CSV)
                ax.fill_between(d.n, d[lo], d[hi], color=st["color"], alpha=0.15, lw=0)
        ax.axvline(meta["n_eval_pool"], color="#999999", lw=0.7, ls="--")
        if ref is not None:
            ax.axhline(ref, color="#999999", lw=0.8, ls=":")
        ax.set_xscale("log"); ax.set_xlabel("logged sample size n")
        W = meta["W"].get(str(e), {}).get(str(K))
        ax.set_title(f"$\\varepsilon$ = {e:g}" + (f"  (W = {W:.0f})" if W else ""))
    axs[0][0].set_ylabel(ylabel)
    h, l = axs[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    save(fig, figd, fname)


def main():
    ap = grlib.add_common_args(argparse.ArgumentParser())
    ap.add_argument("--table-dir", default=None, help="default: <run-dir>/table")
    ap.add_argument("--K", type=int, default=32)
    ap.add_argument("--p", type=float, default=0.95)
    args = ap.parse_args()
    cfg = grlib.load_config(args.config)
    rd = grlib.run_dir(cfg, args)
    cd = os.path.join(rd, "certify"); figd = os.path.join(rd, "figures"); os.makedirs(figd, exist_ok=True)
    meta = json.load(open(os.path.join(cd, "meta.json")))
    meta["W"] = {e: {str(k): v for k, v in d.items()} for e, d in meta["W"].items()}
    sm = pd.read_csv(os.path.join(cd, "summary_main.csv"))
    ss = pd.read_csv(os.path.join(cd, "summary_stress.csv"))
    s = sm[(sm.K == args.K) & np.isclose(sm.p, args.p)]

    # (1) certification rate
    by_eps_panels(s, "cert_rate", "certification rate", "fig_cert_rate.pdf",
                  meta, figd, args.K, lo="cert_lo", hi="cert_hi")
    # (2) accuracy of deployed router (fallback to the reference router when nothing certifies)
    by_eps_panels(s, "acc_deploy", "accuracy (deployed)", "fig_accuracy.pdf", meta, figd, args.K,
                  ref=meta["reference"]["acc_true"])

    # (3) per-action survival curves P(Y > t) with tau and the generation cap
    tab = pd.read_parquet(os.path.join(args.table_dir or os.path.join(rd, "table"), "table.parquet"))
    plot_dist(tab, meta, "tokens__", "generated tokens per request $t$", "fig_token_dist.pdf", figd,
              taus=meta["taus"], cap=max(meta["caps"].values()))

    # (4) stress test: false certification at the largest n, per eps, K
    x = ss[np.isclose(ss.p, args.p) & (ss.n == ss.n.max())]
    epss = sorted(x.eps.unique(), reverse=True); Ks = sorted(x.K.unique())
    fig, axs = plt.subplots(1, len(Ks), figsize=(W2, 2.0), sharey=True, squeeze=False)
    meths = [m for m in STY if m in set(x.method)]
    bw = 0.8 / len(meths)
    for ax, K in zip(axs[0], Ks):
        for mi, m in enumerate(meths):
            d = x[(x.K == K) & (x.method == m)].set_index("eps").reindex(epss)
            xs = np.arange(len(epss)) + (mi - (len(meths) - 1) / 2) * bw
            ax.bar(xs, d.false_rate, bw * 0.9, color=STY[m]["color"], label=STY[m]["label"])
            ax.plot(xs, d.false_rate, ls="none", marker=STY[m]["marker"], color=STY[m]["color"], ms=3)
            ax.errorbar(xs, d.false_rate, yerr=[d.false_rate - d.false_lo, d.false_hi - d.false_rate],
                        fmt="none", ecolor="#333333", lw=0.6, capsize=1)
        ax.axhline(0.05, color="#222222", lw=0.8, ls="--")
        ax.set_xticks(np.arange(len(epss))); ax.set_xticklabels([f"$\\varepsilon$={e:g}" for e in epss])
        ax.set_title(f"K = {K}, n = {int(x.n.max())}")
    axs[0][0].set_ylabel("false-cert. rate (stress)")
    h, l = axs[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    save(fig, figd, "fig_stress.pdf")
    print("figures in", figd, sorted(os.listdir(figd)))


if __name__ == "__main__":
    main()
