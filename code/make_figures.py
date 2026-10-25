"""Figures for the paper (vector PDF plus PNG preview).

Figure 1  Monthly long-band share of frequency connectedness (mean of the 8 networks) under
          package centring and causal EWMA centring (panel a), and the squared distance of
          the volatility level from the full-sample mean (panel b). Separate panels, one axis
          each.
Figure 2  Simulation: bias of the long-band TCI (estimate minus truth) for three estimators
          in four data-generating processes.

Palette: reference categorical slots 1-3 (validated all-pairs, light). Package = blue,
EWMA (fix) = orange, time-varying intercept = aqua; marker shapes differ as a second channel.

Usage (from the work directory):  python ../code/make_figures.py --out ../paper/figs
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 8,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.5, "grid.linestyle": "-", "axes.axisbelow": True,
    "legend.frameon": False, "pdf.fonttype": 42, "savefig.dpi": 300,
})


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def long_share(root, run):
    meta = json.load(open(os.path.join(root, run, "meta.json")))
    lb = meta["band_names"][-1]
    shares = []
    for spec in meta["networks"]:
        d = pd.read_csv(os.path.join(root, run, spec.replace(":", "_") + "_dynamic.csv.gz"),
                        index_col=0, parse_dates=True)
        shares.append((100 * d[f"TCI_{lb}"] / d["TCI_ftotal"]).resample("MS").mean())
    return pd.concat(shares, axis=1).mean(axis=1)


def figure1(args):
    pkg = long_share(args.conn, "shilin_package")
    ewm = long_share(args.conn, "shilin_ewma15d")
    p = pd.read_csv(args.panel, index_col=0, parse_dates=True)
    p = p[p.index <= pd.Timestamp("2025-12-31 23:00", tz="UTC")]
    lvl = p.mean(axis=1)
    d2 = ((lvl - lvl.mean()).resample("MS").mean()) ** 2

    fig, (a, b) = plt.subplots(2, 1, figsize=(6.3, 3.9), sharex=True,
                               gridspec_kw={"height_ratios": [1.6, 1], "hspace": 0.18})
    for s, col, mk, lab in ((pkg, BLUE, "o", "Package centring (full-sample mean)"),
                            (ewm, ORANGE, "s", "Causal EWMA centring (15-day half-life)")):
        a.plot(s.index, s.values, color=col, lw=1.4, marker=mk, ms=2.6, mew=0, label=lab)
    a.set_ylabel("Long band, % of total")
    a.set_ylim(0, 90)
    a.legend(loc="upper right", fontsize=7.5, handlelength=1.6)
    a.text(0.0, 1.02, "(a) Share of the long band (labelled > 24 h by the package) in total connectedness, mean of 8 networks",
           transform=a.transAxes, fontsize=8, color=INK, ha="left", va="bottom")
    b.plot(d2.index, d2.values, color=INK2, lw=1.2)
    b.fill_between(d2.index, 0, d2.values, color=INK2, alpha=0.08, lw=0)
    b.set_ylabel(r"$\delta_t^2$")
    b.set_ylim(bottom=0)
    b.text(0.0, 1.02, "(b) Squared distance of the monthly volatility level from the full-sample mean",
           transform=b.transAxes, fontsize=8, color=INK, ha="left", va="bottom")
    for ax in (a, b):
        clean(ax)
    fig.align_ylabels((a, b))
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(args.out, f"fig1_centring.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print("fig1: corr(package share, delta^2) =", round(float(pd.concat([pkg, d2], axis=1).corr().iloc[0, 1]), 2))


def figure2(args):
    cases = [("centering_level30d.csv", 1.0, "Observed level drift (30-day level)"),
             ("centering_level90d.csv", 1.0, "Observed level drift (90-day level)"),
             ("centering_btcs.csv", 1.0, "Observed level drift, BTC-spot network"),
             ("centering_randomlevel.csv", 1.0, "Random level drift (new path each run)"),
             ("centering_persistent.csv", 0.0, "Fast VAR, no drift"),
             ("centering_harvar.csv", 0.0, "Slow genuine spillovers (HAR-VAR), no drift")]
    cases = [c for c in cases if os.path.exists(os.path.join(args.sim, c[0]))]
    ests = [("package", BLUE, "o", "Package centring"),
            ("ewma15d", ORANGE, "s", "Causal EWMA centring"),
            ("tvintercept", AQUA, "^", "Time-varying intercept")]
    rows = []
    for f, scale, lab in cases:
        df = pd.read_csv(os.path.join(args.sim, f))
        truth = float(df.loc[df.estimator == "TRUTH", "long"].iloc[0])
        sim = df[(df.estimator != "TRUTH") & (np.isclose(df.scale, scale))]
        for e, *_ in ests:
            v = sim.loc[sim.estimator == e, "long"]
            rows.append({"case": lab, "est": e, "bias": v.mean() - truth,
                         "lo": v.min() - truth, "hi": v.max() - truth})
    r = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(6.3, 0.62 * len(cases) + 0.6))
    y0 = np.arange(len(cases))[::-1]
    offs = {"package": 0.22, "ewma15d": 0.0, "tvintercept": -0.22}
    ax.axvline(0, color=AXIS, lw=0.8, zorder=1)
    for e, col, mk, lab in ests:
        sub = r[r.est == e].set_index("case").loc[[c[2] for c in cases]]
        y = y0 + offs[e]
        ax.hlines(y, sub["lo"], sub["hi"], color=col, lw=1.2, zorder=2)
        ax.scatter(sub["bias"], y, color=col, marker=mk, s=26, zorder=3, label=lab,
                   edgecolors="white", linewidths=0.8)
    ax.set_yticks(y0)
    ax.set_yticklabels([c[2] for c in cases])
    ax.set_xlabel("Bias of the long-band TCI (estimate minus truth, percentage points)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", fontsize=7.5, handletextpad=0.3)
    clean(ax)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(args.out, f"fig2_simulation.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(r.round(2).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conn", default="results/conn")
    ap.add_argument("--sim", default="results/sim")
    ap.add_argument("--panel", default="data/processed/logrv_ds.csv.gz")
    ap.add_argument("--out", default="../paper/figs")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    figure1(args)
    figure2(args)


if __name__ == "__main__":
    main()
