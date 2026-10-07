"""Figure fig_information.pdf from the E1 results (sim_information.py).

(a) power vs log size on the two-action family (K = W = 2, p = 0.99) for rules using the joint CDF
    indicators only, the exceedance indicators, and the action counts in addition;
(b) n at 90% power vs tail mass u = 1 - p at relative margin gamma = u/2 (K = W = 2), with 1/u and 1/u^2
    reference slopes;
(c) penalty n90(CDF-exact) / n90(counts-exact) for one candidate logged with probability 1/W (p = 0.95),
    against p/u = 19: the penalty vanishes on-policy (W = 1).

usage: python plot_information.py [--res DIR] [--out DIR]
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e8e7e2"
CDF, EXC, CNT = "#eb6834", "#4a3aa7", "#1baf7a"
STYLE = {  # method: (label, color, linestyle, marker)
    "cdf_eb": ("CDF, emp. Bernstein (UnO)", CDF, "--", None),
    "cdf_exact": ("CDF, exact", CDF, "-", "o"),
    "exc_eb": ("Exceedance, emp. Bernstein [H22]", EXC, "-", "s"),
    "norm_bet": ("Counts, normalized betting", CNT, "--", None),
    "counts_exact": ("Counts, exact conditional", CNT, "-", "^"),
}
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#8a8984", "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "legend.fontsize": 6.5})


def main():
    ap = argparse.ArgumentParser(description="Plot fig_information.pdf")
    ap.add_argument("--res", default=os.path.join(HERE, "..", "results", "information"))
    ap.add_argument("--out", default=os.path.join(HERE, "..", "figures"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    P = pd.read_csv(os.path.join(a.res, "power.csv"))
    S = pd.read_csv(os.path.join(a.res, "summary.csv"))
    fig, ax = plt.subplots(1, 3, figsize=(6.9, 2.25))

    # (a) power curves
    d = P[(P.setting == "two_action") & np.isclose(P.p, 0.99)]
    for m, (lab, col, ls, mk) in STYLE.items():
        s = d[d.method == m].sort_values("n")
        ax[0].plot(s.n, s.power, ls, color=col, lw=1.5, label=lab, marker=mk, ms=3, markevery=4)
        # Pointwise Wilson 95% intervals for 2000 independent simulation repetitions.
        z, reps = 1.96, 2000
        phat = s.power.to_numpy()
        den = 1 + z*z/reps
        center = (phat + z*z/(2*reps))/den
        half = z*np.sqrt(phat*(1-phat)/reps + z*z/(4*reps*reps))/den
        ax[0].fill_between(s.n.to_numpy(), center-half, center+half, color=col, alpha=0.12, lw=0)
    ax[0].set_xscale("log"); ax[0].set_ylim(-0.02, 1.02)
    ax[0].set_xlabel("log size $n$"); ax[0].set_ylabel("P(select & certify safe)")
    ax[0].set_title("(a) two actions, $p=0.99$", fontsize=8)
    ax[0].grid(axis="y", color=GRID, lw=0.6)
    ax[0].text(4e5, 0.12, "CDF only", color=INK2, fontsize=6.5)

    # (b) scaling in u at relative margin
    s = S[S.setting == "two_action"].sort_values("p")
    u = 1 - s.p.values
    for m in ("cdf_exact", "exc_eb", "counts_exact"):
        lab, col, ls, mk = STYLE[m]
        ax[1].plot(u, s[m].values, ls, color=col, lw=1.5, marker=mk, ms=4)
    for k, txt, base in ((1, r"$\propto 1/u$", s.counts_exact.values), (2, r"$\propto 1/u^2$", s.cdf_exact.values)):
        ref = base[0] * (u[0] / u) ** k
        ax[1].plot(u, ref * 0.55, ":", color="#8a8984", lw=0.9)
        ax[1].text(u[-1] * 1.05, ref[-1] * 0.45, txt, fontsize=6.5, color=INK2, va="top")
    ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].invert_xaxis()
    ax[1].set_xlabel("tail mass $u=1-p$  ($\\gamma=u/2$)"); ax[1].set_ylabel("log size for 90% power")
    ax[1].set_title("(b) scaling as $p\\to1$", fontsize=8)
    ax[1].grid(color=GRID, lw=0.6)

    # (c) single candidate: penalty vs logging probability
    s = S[S.setting == "single"].sort_values("W")
    ax[2].plot(s.W, s.cdf_exact / s.counts_exact, "-", color=CDF, lw=1.5, marker="o", ms=4,
               label="exact: CDF / counts")
    ax[2].plot(s.W, s.cdf_eb / s.exc_eb, "--", color=EXC, lw=1.5, marker="s", ms=3.5,
               label="EB: CDF / exceedance")
    ax[2].axhline(19, color="#8a8984", lw=0.9, ls=":")
    ax[2].text(1.05, 20.5, "$p/u = 19$", fontsize=6.5, color=INK2)
    ax[2].set_xscale("log"); ax[2].set_xticks([1, 2, 5, 10, 32]); ax[2].set_xticklabels(["1", "2", "5", "10", "32"])
    ax[2].set_ylim(0, 27)
    ax[2].set_xlabel("$W$ (target logged w.p. $1/W$)"); ax[2].set_ylabel("data ratio (CDF / full)")
    ax[2].set_title("(c) one candidate, $p=0.95$", fontsize=8)
    ax[2].grid(axis="y", color=GRID, lw=0.6)
    ax[2].legend(loc="lower right", frameon=False, borderaxespad=0.1, handlelength=1.8)

    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.08), handlelength=2.2)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "fig_information.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(a.out, "fig_information.png"), dpi=180, bbox_inches="tight")
    print("wrote", os.path.join(a.out, "fig_information.pdf"))


if __name__ == "__main__":
    main()
