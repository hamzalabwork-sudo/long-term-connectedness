"""Network figures of pairwise net connectedness (PNET) in the four cross-instrument exchange
networks, in the style of Shi & Lin (2026, Fig. 6), and a comparison with their published values.

PNET from l to j = CT[j, l] - CT[l, j] (time average of the dynamic measures, results/pairwise.csv);
arrows point from the net transmitter to the net receiver of each pair.

Outputs:
  paper/figs/fig_network.pdf/png     long band (cycles > 48 h): standard estimator (top row) and
                                     recommended estimator (bottom row), four exchange networks
  paper/figs/fig_network_time.pdf/png  time domain, standard estimator (replication of Shi & Lin)
  paper/tables/tab_pnet.tex          our time-domain PNET against the values in Shi & Lin's Fig. 6
  results/pnet.csv

Usage (from iref/):  python code/network_figs.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, FancyArrowPatch

from make_exhibits import BLUE, INK, INK2, ORANGE, save, sg, write  # noqa: F401  (rcParams on import)

EXCH = [("BN", "Binance"), ("OKX", "OKX"), ("BG", "Bitget"), ("BF", "Bitfinex")]
INS = {"BTCP": ("BTC-P", (0.0, 1.0)), "ETHS": ("ETH-S", (-1.0, 0.0)),
       "BTCS": ("BTC-S", (1.0, 0.0)), "ETHP": ("ETH-P", (0.0, -1.0))}
# Shi & Lin (2026), Fig. 6: (transmitter, receiver): value, exchange networks, time domain
PUBLISHED = {
    "BN": {("ETHS", "BTCP"): 1.024, ("ETHP", "BTCP"): 1.130, ("BTCS", "BTCP"): 0.040,
           ("ETHS", "BTCS"): 1.022, ("ETHP", "BTCS"): 1.133, ("ETHP", "ETHS"): 0.090},
    "OKX": {("ETHS", "BTCP"): 1.226, ("ETHP", "BTCP"): 1.065, ("BTCS", "BTCP"): 0.138,
            ("ETHS", "BTCS"): 1.141, ("ETHP", "BTCS"): 0.992, ("ETHS", "ETHP"): 0.195},
    "BG": {("ETHS", "BTCP"): 1.133, ("ETHP", "BTCP"): 0.823, ("BTCS", "BTCP"): 0.224,
           ("ETHS", "BTCS"): 0.941, ("ETHP", "BTCS"): 0.782, ("ETHS", "ETHP"): 0.243},
    "BF": {("ETHS", "BTCP"): 1.032, ("ETHP", "BTCP"): 0.926, ("BTCS", "BTCP"): 0.548,
           ("ETHS", "BTCS"): 0.645, ("ETHP", "BTCS"): 0.516, ("ETHS", "ETHP"): 0.251},
}
PAIRS = [("BTCP", "ETHS"), ("BTCP", "BTCS"), ("BTCP", "ETHP"), ("ETHS", "BTCS"), ("ETHS", "ETHP"), ("BTCS", "ETHP")]


def pnet(d, ex, est, dom):
    s = d[(d.network == f"exchange:{ex}") & (d.estimator == est) & (d.domain == dom)]
    M = {(r.receiver.split("_")[0], r.transmitter.split("_")[0]): r.value for r in s.itertuples()}
    out = {}
    for a, b in PAIRS:
        v = M[(b, a)] - M[(a, b)]                       # net from a to b
        out[(a, b) if v >= 0 else (b, a)] = abs(v)
    net = {i: sum(v for (t, r), v in out.items() if t == i) - sum(v for (t, r), v in out.items() if r == i) for i in INS}
    return out, net


def draw(ax, edges, net, title, scale):
    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(-1.45, 1.45)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=7.5, color=INK, loc="center", pad=2)
    curved = {frozenset(("BTCP", "ETHP")): 0.12, frozenset(("ETHS", "BTCS")): -0.12}
    for (t, r), v in edges.items():
        p, q = np.array(INS[t][1]), np.array(INS[r][1])
        rad = curved.get(frozenset((t, r)), 0.18 if (t, r) in [("ETHS", "BTCP"), ("BTCS", "ETHP"), ("BTCP", "BTCS"), ("ETHP", "ETHS")] else -0.18)
        w = 0.4 + 3.2 * v / scale
        ax.add_patch(FancyArrowPatch(p, q, connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>",
                                     mutation_scale=7 + 2 * w, lw=w, color="#9a9890", shrinkA=13, shrinkB=13, zorder=2))
        # label on the drawn curve (quadratic Bezier; control point on the side the arc bows to)
        ctrl = (p + q) / 2 + rad * np.array([-(q - p)[1], (q - p)[0]])
        pair = frozenset((t, r))
        s = 0.5
        if pair == frozenset(("ETHS", "BTCS")):        # crossing edges: label the left part of the horizontal one
            s = 0.3 if p[0] < q[0] else 0.7
        elif pair == frozenset(("BTCP", "ETHP")):      # and the upper part of the vertical one
            s = 0.72 if p[1] < q[1] else 0.28
        mid = (1 - s) ** 2 * p + 2 * s * (1 - s) * ctrl + s ** 2 * q
        ax.text(*mid, f"{v:.2f}", fontsize=6.2, color=INK, ha="center", va="center", zorder=4,
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))
    for i, (lab, pos) in INS.items():
        col = ORANGE if net[i] > 0 else BLUE
        ax.add_patch(Circle(pos, 0.27, color=col, zorder=3))
        ax.text(*pos, lab, fontsize=6.3, color="white", ha="center", va="center", fontweight="bold", zorder=5)


def main():
    d = pd.read_csv("results/pairwise.csv")
    rows = []
    for ex, _ in EXCH:
        for est in ("standard", "recommended"):
            for dom in ("time", "short", "medium", "long"):
                e, _ = pnet(d, ex, est, dom)
                for (t, r), v in e.items():
                    rows.append({"exchange": ex, "estimator": est, "domain": dom, "transmitter": t, "receiver": r, "pnet": v})
    p = pd.DataFrame(rows)
    p.to_csv("results/pnet.csv", index=False)

    # replication table
    lines = [r"\begin{table}[ht]\centering\footnotesize",
             r"\caption{Pairwise net connectedness in the exchange networks (time domain): \citet{shi2026}, Fig.~6, and our replication.}",
             r"\label{tab:pnet}", r"\begin{adjustbox}{max width=\linewidth}\begin{tabular}{llrrrrrrrr}\toprule",
             r" & & \multicolumn{2}{c}{Binance} & \multicolumn{2}{c}{OKX} & \multicolumn{2}{c}{Bitget} & \multicolumn{2}{c}{Bitfinex}\\",
             r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}",
             r"Pair (published direction) & & Published & Ours & Published & Ours & Published & Ours & Published & Ours\\\midrule"]
    agree = total = 0
    pub_all, our_all = [], []
    keys = [("ETHS", "BTCP"), ("ETHP", "BTCP"), ("BTCS", "BTCP"), ("ETHS", "BTCS"), ("ETHP", "BTCS"), ("ETHS/ETHP", "")]
    for t, r in keys:
        cells = []
        for ex, _ in EXCH:
            pub = PUBLISHED[ex]
            if t == "ETHS/ETHP":
                (pt, pr), pv = [(k, v) for k, v in pub.items() if set(k) == {"ETHS", "ETHP"}][0]
            else:
                pt, pr, pv = t, r, pub[(t, r)]
            e, _ = pnet(d, ex, "standard", "time")
            ov = e[(pt, pr)] if (pt, pr) in e else -e[(pr, pt)]
            agree += ov > 0
            total += 1
            pub_all.append(pv)
            our_all.append(ov)
            cells.append(f"{pv:.3f} & {sg(ov, 3)}")
        lab = r"ETH-S $\leftrightarrow$ ETH-P$^{a}$" if t == "ETHS/ETHP" else f"{INS[t][0]} $\\to$ {INS[r][0]}"
        lines.append(f"{lab} & & " + " & ".join(cells) + r"\\")
    corr = float(np.corrcoef(pub_all, our_all)[0, 1])
    lines += [r"\bottomrule\end{tabular}\end{adjustbox}",
              rf"\par\smallskip\parbox{{\linewidth}}{{\scriptsize Notes: time averages of dynamic pairwise net connectedness from the "
              rf"transmitter to the receiver (a negative value means the opposite direction). Ours: TVP-VAR(3), package centring, "
              rf"half-lives 15/7 days, $H=720$, 2021--2025. Published values read from Fig.~6 of \citet{{shi2026}}. "
              rf"$^{{a}}$Published direction ETH-P $\to$ ETH-S on Binance and ETH-S $\to$ ETH-P elsewhere. Directions agree in "
              rf"{agree} of {total} pairs; correlation of the values {corr:.2f}.}}", r"\end{table}"]
    write("tab_pnet", lines)
    print(f"directions agree {agree}/{total}, corr {corr:.2f}")

    # figures
    for name, specs in (("fig_network", [("standard", "long", "Standard estimator, long band"),
                                          ("recommended", "long", "Recommended estimator, long band")]),
                        ("fig_network_time", [("standard", "time", "Standard estimator, time domain")])):
        sub = p[p.domain.isin([s[1] for s in specs]) & p.estimator.isin([s[0] for s in specs])]
        scale = sub.pnet.max()
        fig, axes = plt.subplots(len(specs), 4, figsize=(6.3, 1.75 * len(specs) + 0.35))
        axes = np.atleast_2d(axes)
        for i, (est, dom, rowlab) in enumerate(specs):
            for jx, (ex, exlab) in enumerate(EXCH):
                e, net = pnet(d, ex, est, dom)
                draw(axes[i, jx], e, net, exlab if i == 0 else "", scale)
            axes[i, 0].text(-1.75, 0, rowlab, rotation=90, fontsize=7.2, color=INK2, ha="center", va="center")
        fig.legend(handles=[Circle((0, 0), 1, color=ORANGE, label="Net transmitter"),
                            Circle((0, 0), 1, color=BLUE, label="Net receiver")],
                   loc="lower center", ncol=2, fontsize=7, bbox_to_anchor=(0.5, -0.02))
        fig.subplots_adjust(left=0.06, right=0.99, top=0.93, bottom=0.08, wspace=0.05, hspace=0.12)
        save(fig, name)


if __name__ == "__main__":
    main()
