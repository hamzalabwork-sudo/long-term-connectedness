"""Tables and figures of the IREF paper that are new relative to the letter.

Tables (paper/tables): tab_pseudo, tab_decay, tab_real, tab_indirect (if network results exist).
Figures (paper/figs, PDF + PNG): fig_hourlydaily, fig_tradeoff, fig_indirect.

Palette (letter's validated reference slots): TVP-VAR(3) blue, TVP-HAR-VAR orange; centring by line
style (solid EWMA, dashed package) and marker; data in ink. One y-axis per panel.

Usage (from iref/):  python code/make_exhibits.py
"""
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

R, T, F = "results", "paper/tables", "paper/figs"
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
EST = {("var3", "package"): ("TVP-VAR(3), package", BLUE, "--", "s"),
       ("var3", "ewma"): ("TVP-VAR(3), EWMA", BLUE, "-", "o"),
       ("harvar", "package"): ("TVP-HAR-VAR, package", ORANGE, "--", "s"),
       ("harvar", "ewma"): ("TVP-HAR-VAR, EWMA", ORANGE, "-", "o")}
NETLAB = {"instrument:BTCS": "BTC spot", "instrument:BTCP": "BTC perp.", "instrument:ETHS": "ETH spot",
          "instrument:ETHP": "ETH perp.", "exchange:BN": "Binance", "exchange:OKX": "OKX",
          "exchange:BG": "Bitget", "exchange:BF": "Bitfinex"}


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def save(fig, name):
    os.makedirs(F, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(F, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def sg(x, nd=1):
    """Signed number with a typographic minus."""
    return f"{x:+.{nd}f}".replace("-", "$-$")


def write(name, lines):
    os.makedirs(T, exist_ok=True)
    open(os.path.join(T, f"{name}.tex"), "w", encoding="utf-8").write("\n".join(lines))
    print("wrote", name)


# ------------------------------------------------------------------------------------- tables
def tab_pseudo():
    p = pd.read_csv(os.path.join(R, "pseudo_true.csv"))
    ls = pd.read_csv(os.path.join(R, "local_sampling.csv")).set_index("world")
    g = pd.read_csv(os.path.join(R, "predict_gap.csv"))
    lines = [r"\begin{table}[t]\centering\footnotesize",
             r"\caption{Pseudo-true values against simulation and data: bias of the long band (cycles $>$48 h).}",
             r"\label{tab:pseudo}", r"\begin{adjustbox}{max width=\linewidth}\begin{tabular}{llrrr}\toprule",
             r"\multicolumn{5}{l}{\emph{Panel A: simulated worlds (Binance network), estimate minus truth}}\\",
             r"World & Centring & Terms (A)+(B) & Incl.\ (C) & Monte Carlo\\\midrule"]
    for _, r in p.iterrows():
        if r.scale == 0 and r.world.startswith("drift"):
            continue
        w = (f"Level drift, $c={r.scale:g}$" if r.world.startswith("drift") else "Slow spillovers (HAR-VAR)")
        inc = "--"
        if r.world.startswith("slow") and r.estimator == "package":
            inc = sg(ls.loc['harvar', 'local_sample_bias_package'])
        lines.append(f"{w} & {'Package' if r.estimator == 'package' else 'EWMA'} & {sg(r.pt_long)} & {inc} & {sg(r.sim_long)}\\\\")
    lines += [r"\midrule", r"\multicolumn{5}{l}{\emph{Panel B: real data, package minus EWMA long band (mean of 8 networks)}}\\",
              r"Half-life & & Predicted & & Estimated\\\midrule"]
    for h, s in g.groupby("halflife_h"):
        lines.append(f"{h / 24:.1f} days & & {s.pred_gap.mean():.1f} [{s.pred_gap.min():.1f}, {s.pred_gap.max():.1f}] & & "
                     f"{s.obs_gap.mean():.1f} [{s.obs_gap.min():.1f}, {s.obs_gap.max():.1f}]\\\\")
    c = g[["pred_gap", "obs_gap"]].corr().iloc[0, 1]
    lines += [r"\bottomrule\end{tabular}\end{adjustbox}",
              r"\par\smallskip\parbox{\linewidth}{\scriptsize Notes: Panel A: worlds of Section~\ref{sec:simdesign}; $c$ scales the "
              r"observed 30-day level added to a VAR(3) fitted to de-levelled data. Terms (A)+(B): pseudo-true value of a TVP-VAR(3) "
              r"with 15-day forgetting (Proposition~\ref{prop:pseudo}) minus truth, computed from autocovariances. Incl.\ (C): "
              r"adds the local-sampling term, computed from forgetting-weighted sample moments of simulated paths (3 seeds). "
              r"Monte Carlo: TVP-VAR(3) estimates in the simulations of Table S5 of the Online Appendix (volatility half-life 7 "
              r"days; 10 replications in the drift world, 5 in the slow-spillover world); the runs behind Fig.~\ref{fig:tradeoff} "
              r"use a 2-day volatility half-life and 3 seeds and differ from these by at most 0.3 points. Panel B: drift reading of each network (VAR(3) on de-levelled data plus the "
              rf"observed level); coefficient and EWMA half-lives matched; brackets: range over networks. Correlation over the 24 "
              rf"network--half-life cells: {c:.2f}.}}", r"\end{table}"]
    write("tab_pseudo", lines)


def tab_decay():
    a = pd.concat([pd.read_csv(os.path.join(R, f)) for f in
                   ("select_halflife.csv", "select_halflife_daily_ext.csv", "select_halflife_hourly_ext.csv")])
    a = a.drop_duplicates(["freq", "network", "h_coef", "h_vol"], keep="last").reset_index(drop=True)
    best = a.loc[a.groupby(["freq", "network"]).score_sel.idxmax()].set_index(["freq", "network"])
    q = pd.read_csv(os.path.join(R, "select_q_scores.csv"))
    fg = q[q.q == "forgetting"].set_index("network")
    kf = q[q.q != "forgetting"].copy()
    kf = kf.loc[kf.groupby("network").score_sel.idxmax()].set_index("network")
    t = pd.read_csv(os.path.join(R, "score_tests.csv"))
    lines = [r"\begin{table}[t]\centering\footnotesize",
             r"\caption{Decay rates chosen by predictive log score.}", r"\label{tab:decay}",
             r"\begin{adjustbox}{max width=\linewidth}\begin{tabular}{lrrrrrr}\toprule",
             r"\multicolumn{7}{l}{\emph{Panel A: selected half-lives (days) and the standard Kalman filter}}\\",
             r" & \multicolumn{2}{c}{Hourly} & \multicolumn{2}{c}{Daily} & \multicolumn{2}{c}{Score, hourly (sel.\ period)}\\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
             r"Network & Coef. & Vol. & Coef. & Vol. & Forgetting & Kalman ($\hat q$)\\\midrule"]
    for n, lab in NETLAB.items():
        h, d = best.loc[("hourly", n)], best.loc[("daily", n)]
        lines.append(f"{lab} & {h.h_coef:g} & {h.h_vol:g} & {d.h_coef:g} & {d.h_vol:g} & {fg.loc[n, 'score_sel']:.4f} & "
                     f"{kf.loc[n, 'score_sel']:.4f}\\\\")
    lines += [r"\midrule", r"\multicolumn{7}{l}{\emph{Panel B: score difference against the modal selected coefficient half-life (nats per observation and series)}}\\",
              r" & \multicolumn{3}{c}{Selection period} & \multicolumn{3}{c}{Evaluation period}\\",
              r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
              r"Coefficient half-life & Difference & $t$ & & Difference & $t$ & \\\midrule"]
    for f, s in t.groupby("freq", sort=False):
        lines.append(rf"\multicolumn{{7}}{{l}}{{{f.capitalize()} data, against {s.vs.iloc[0]:g} days}}\\")
        for h, u in s.groupby("h_coef"):
            a1, b1 = u[u.period == "sel"].iloc[0], u[u.period == "eval"].iloc[0]
            lines.append(f"\\quad {h:g} days & {sg(a1.mean_diff, 4)} & {sg(a1.t)} & & {sg(b1.mean_diff, 4)} & {sg(b1.t)} & \\\\")
    lines += [r"\bottomrule\end{tabular}\end{adjustbox}",
              r"\par\smallskip\parbox{\linewidth}{\scriptsize Notes: one-step-ahead predictive log density of the TVP-VAR(3), "
              r"averaged over the four series, causal EWMA centring with the coefficient half-life; selection period 2022--2024, "
              r"evaluation period January 2025--September 2026. Hourly grids: coefficient 1--365 days, volatility 0.25--30 days; "
              r"daily grids: coefficient 1--730 days, volatility 1--365 days. Kalman: random-walk coefficients with state "
              r"covariance $q\,V_{\mathrm{OLS}}$, $q$ selected on the same criterion. Score columns: coefficient half-life 15 days, "
              r"volatility half-life fixed at 2 days for both filters. Panel B: network-average score difference against 15 days "
              r"(hourly) and 120 days (daily), Newey--West $t$ (lag one week); volatility half-life fixed at 2 days (hourly) and "
              r"60 days (daily).}",
              r"\end{table}"]
    write("tab_decay", lines)


def tab_real():
    d = pd.read_csv(os.path.join(R, "real_estimates.csv"))
    lines = [r"\begin{table}[t]\centering\footnotesize",
             r"\caption{Band connectedness of the eight crypto networks under eight estimators, 2021--2025.}",
             r"\label{tab:real}", r"\begin{adjustbox}{max width=\linewidth}\begin{tabular}{lllrrrrr}\toprule",
             r"Model & Centring & Window & Short & Medium & Long & Long$>$Short & Unstable\\\midrule"]
    for hl in (15, 60):
        for est in ("var3", "harvar"):
            for cent in ("package", "ewma"):
                s = d[(d.hl_coef == hl) & (d.estimator == est) & (d.centring == cent)]
                lab = "TVP-VAR(3)" if est == "var3" else "TVP-HAR-VAR"
                lines.append(f"{lab} & {'Package' if cent == 'package' else 'EWMA'} & {hl} d & {s.letter_s.mean():.1f} & "
                             f"{s.letter_m.mean():.1f} & {s.letter_l.mean():.1f} [{s.letter_l.min():.1f}, {s.letter_l.max():.1f}] & "
                             f"{int((s.letter_l > s.letter_s).sum())}/8 & "
                             + (f"{100 * s.n_unstable.mean() / 1767:.1f}\\%" if est == "harvar" else "--") + "\\\\")
        if hl == 15:
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule\end{tabular}\end{adjustbox}",
              r"\par\smallskip\parbox{\linewidth}{\scriptsize Notes: mean over the eight networks (range in brackets for the long "
              r"band); bands by cycle length: short $<$16 h, medium 16--48 h, long $>$48 h; $H=720$ h; connectedness every 24 h "
              r"from March 2021. Window: coefficient (and EWMA) half-life; volatility half-life 2 days. Long$>$Short: number of "
              r"networks. Unstable: share of dates at which the TVP-HAR-VAR's implied VAR(168) is explosive (replaced by the last "
              r"stable estimate). TCI is 72.2--72.4 for every estimator.}", r"\end{table}"]
    write("tab_real", lines)


def tab_indirect():
    files = sorted(glob.glob(os.path.join(R, "indirect_match_*.csv")))
    if not files:
        print("tab_indirect: no network results yet")
        return
    m = pd.concat([pd.read_csv(f) for f in files])
    real = pd.read_csv(os.path.join(R, "real_estimates.csv"))
    lines = [r"\begin{table}[t]\centering\footnotesize",
             r"\caption{Indirect inference: share of slow movement that is drift and the implied long band.}",
             r"\label{tab:indirect}", r"\begin{adjustbox}{max width=\linewidth}\begin{tabular}{lrrrrrrr}\toprule",
             r"Network & $\hat c$ & Implied long & Implied short & Standard estimator & TVP-HAR-VAR, EWMA, 60 d & "
             r"$Q_{\min}$ & Seeds\\\midrule"]
    for n, lab in NETLAB.items():
        r = m[m.network == n]
        if r.empty:
            continue
        r = r.iloc[0]
        rr = real[real.network == n].set_index(["estimator", "centring", "hl_coef"])
        lines.append(f"{lab} & {r.c_hat:.2f} [{r.c_lo_adj:.2f}, {r.c_hi_adj:.2f}] & {r.long:.1f} [{r.long_lo_adj:.1f}, "
                     f"{r.long_hi_adj:.1f}] & {r.short:.1f} & {rr.loc[('var3', 'package', 15), 'letter_l']:.1f} & "
                     f"{rr.loc[('harvar', 'ewma', 60), 'letter_l']:.1f} & {r.Q:.1f} & {int(r.seeds)}\\\\")
    lines += [r"\bottomrule\end{tabular}\end{adjustbox}",
              r"\par\smallskip\parbox{\linewidth}{\scriptsize Notes: worlds $y=\bar y+cL+x^{(c)}$ with $x^{(c)}$ a HAR-type VAR "
              r"fitted to $y-cL$; $\hat c$ minimises the simulation-variance-weighted distance $Q$ between simulated and real long "
              r"and short bands of six estimators (12 moments). Brackets: values of $c$ with $Q\le Q_{\min}+3.84\,\max(1,Q_{\min}/11)$, "
              r"i.e.\ a 95\% set whose threshold is scaled by the misfit of the one-parameter family, and the implied long bands. "
              r"Implied long and short: true bands of the selected world. Standard estimator: TVP-VAR(3), package centring, "
              r"15-day window, volatility half-life 2 days (31.7 on average with the 7-day volatility half-life of \citet{shi2026}, "
              r"Table~\ref{tab:replication}). Seeds: simulated paths per value of $c$.}", r"\end{table}"]
    write("tab_indirect", lines)


def tab_harvar_net():
    h = pd.read_csv(os.path.join(R, "harvar_summary.csv")).set_index("network")
    q = pd.read_csv(os.path.join(R, "select_q_scores.csv"))
    fg = q[q.q == "forgetting"].set_index("network")
    kf = q[q.q != "forgetting"].copy()
    kf["q"] = kf.q.astype(float)
    kf = kf.loc[kf.groupby("network").score_sel.idxmax()].set_index("network")
    lines = [r"\begin{table}[ht]\centering\footnotesize",
             r"\caption{Predictive log scores by network: TVP-VAR(3) with forgetting, standard Kalman filter and TVP-HAR-VAR.}",
             r"\label{tab:harvarnet}", r"\begin{adjustbox}{max width=\linewidth}\begin{tabular}{lrrrrrrr}\toprule",
             r" & \multicolumn{3}{c}{Selection period (2022--2024)} & \multicolumn{3}{c}{Evaluation period (2025--Sep 2026)} & \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
             r"Network & TVP-VAR(3) & Kalman & TVP-HAR-VAR & TVP-VAR(3) & Kalman & TVP-HAR-VAR & $\hat q$\\\midrule"]
    for n, lab in NETLAB.items():
        lines.append(f"{lab} & {sg(h.loc[n, 'var3_score_sel'], 4)} & {sg(kf.loc[n, 'score_sel'], 4)} & "
                     f"{sg(h.loc[n, 'har_score_sel'], 4)} & {sg(h.loc[n, 'var3_score_eval'], 4)} & "
                     f"{sg(kf.loc[n, 'score_eval'], 4)} & {sg(h.loc[n, 'har_score_eval'], 4)} & "
                     f"${kf.loc[n, 'q']:.0e}$\\\\".replace("e-", r"\times10^{-").replace("$\\\\", "}$\\\\"))
    lines += [r"\bottomrule\end{tabular}\end{adjustbox}",
              r"\par\smallskip\parbox{\linewidth}{\scriptsize Notes: mean one-step-ahead log predictive density per hour and "
              r"series. All models: coefficient half-life 15 days (Kalman: random-walk state covariance $\hat q\,V_{\mathrm{OLS}}$ "
              r"instead), volatility half-life 2 days, causal EWMA centring with a 15-day half-life, prior from the first 30 days. "
              r"The TVP-HAR-VAR adds the previous day's and week's average to three hourly lags.}", r"\end{table}"]
    write("tab_harvarnet", lines)


# ------------------------------------------------------------------------------------ figures
def fig_hourlydaily():
    data = pd.read_csv(os.path.join(R, "hourly_vs_daily.csv"))
    sim = pd.read_csv(os.path.join(R, "sim_aggregation.csv"))
    nets = sim.network.unique()
    sel = {"hourly": (15, 2), "daily": (120, 60)}
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.5), sharey=True)
    for ax, (freq, (h1, h2)) in zip(axes, sel.items()):
        bands = (["<2d"] if freq == "hourly" else []) + ["2-7d", "7-30d", ">30d"]
        x = np.arange(len(bands))
        d = data[(data.freq == freq) & (data.h_coef == h1) & (data.h_vol == h2) & (data.centring == "ewma") &
                 data.network.isin(nets)][bands].mean()
        ax.plot(x, d.values, color=INK, lw=2.4, alpha=0.35, marker="D", ms=6, mew=0, label="Data", zorder=2)
        for world, col, mk, lab in (("drift", BLUE, "o", "Simulated: level drift"),
                                    ("harvar", ORANGE, "s", "Simulated: slow spillovers")):
            s = sim[(sim.world == world) & (sim.freq == freq) & (sim.h_coef == h1) & (sim.h_vol == h2) &
                    (sim.centring == "ewma")][bands].mean()
            ax.plot(x, s.values, color=col, lw=1.2, ls="--", marker=mk, ms=4.5, mec="white", mew=0.8, label=lab, zorder=4)
        ax.set_xticks(x)
        ax.set_xticklabels([b.replace("d", " d").replace("<", "< ").replace(">", "> ") for b in bands])
        ax.set_title(f"({'a' if freq == 'hourly' else 'b'}) {freq.capitalize()} data, half-lives {h1}/{h2} d",
                     loc="left", fontsize=8, color=INK)
        ax.set_xlabel("Cycle length")
        clean(ax)
    axes[0].set_ylabel("Connectedness (points)")
    axes[1].legend(loc="upper left", fontsize=7)
    save(fig, "fig_hourlydaily")


def fig_tradeoff():
    s = pd.read_csv(os.path.join(R, "sim_harvar_hl.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.6), sharey=True)
    worlds = (("drift", "(a) Level drift (true long band 19.5)"), ("harvar", "(b) Slow spillovers (true long band 45.0)"))
    for ax, (w, title) in zip(axes, worlds):
        ax.axhline(0, color=AXIS, lw=0.8, zorder=1)
        for (est, cent), (lab, col, ls, mk) in EST.items():
            g = s[(s.world == w) & (s.estimator == est) & (s.centring == cent)].groupby("hl_coef").bias_long
            m, lo, hi = g.mean(), g.min(), g.max()
            ax.plot(m.index, m.values, color=col, ls=ls, lw=1.3, marker=mk, ms=4, mew=0, label=lab, zorder=3)
            ax.vlines(m.index, lo.values, hi.values, color=col, lw=0.8, zorder=2)
        ax.set_xscale("log")
        ax.set_xticks([15, 60, 120])
        ax.set_xticklabels(["15", "60", "120"])
        ax.minorticks_off()
        ax.set_xlabel("Coefficient and EWMA half-life (days)")
        ax.set_title(title, loc="left", fontsize=8, color=INK)
        clean(ax)
    axes[0].set_ylabel("Bias of the long band (points)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=7, bbox_to_anchor=(0.5, -0.06))
    fig.subplots_adjust(bottom=0.22)
    save(fig, "fig_tradeoff")


def fig_indirect():
    sim = pd.read_csv(os.path.join(R, "sim_indirect_exchange_BN", "estimates.csv"))
    tr = pd.read_csv(os.path.join(R, "sim_indirect_exchange_BN", "truth.csv"))
    real = pd.read_csv(os.path.join(R, "real_estimates.csv"))
    real = real[real.network == "exchange:BN"].set_index(["estimator", "centring", "hl_coef"])
    r = pd.read_csv(os.path.join(R, "indirect_match_exchange_BN.csv")).iloc[0]
    ch, clo, chi, implied = r.c_hat, r.c_lo_adj, r.c_hi_adj, r.long      # misfit-adjusted set, as in Table 7
    keys = [("var3", "package", 15), ("var3", "ewma", 15), ("harvar", "package", 15),
            ("harvar", "ewma", 15), ("harvar", "package", 60), ("harvar", "ewma", 60)]
    fig, axes = plt.subplots(2, 4, figsize=(6.3, 3.6), sharex=True)
    axes = axes.ravel()
    ax = axes[0]
    ax.axvspan(clo, chi, color=GRID, alpha=0.8, lw=0, zorder=0)
    ax.plot(tr.c, tr["long"], color=INK, lw=1.4, marker="D", ms=3.5)
    ax.plot([ch, ch], [tr["long"].min(), implied], color=INK2, lw=0.8, ls="--", zorder=1)
    ax.plot([0, ch], [implied, implied], color=INK2, lw=0.8, ls="--", zorder=1)
    ax.plot(ch, implied, marker="o", ms=6, color=AQUA, mec="white", mew=0.9, zorder=4)
    ax.annotate(f"implied {implied:.1f}\nat c = {ch:.2f}", (ch, implied), xytext=(6, 4), textcoords="offset points",
                fontsize=6.5, color=INK, va="bottom")
    ax.set_title("True long band", loc="left", fontsize=7.5, color=INK)
    clean(ax)
    for ax, (est, cent, hl) in zip(axes[1:], keys):
        lab, col, ls, mk = EST[(est, cent)]
        g = sim[(sim.estimator == est) & (sim.centring == cent) & (sim.hl_coef == hl)].groupby("c")["long"]
        ax.axvspan(clo, chi, color=GRID, alpha=0.8, lw=0, zorder=0)
        ax.fill_between(g.mean().index, g.min().values, g.max().values, color=col, alpha=0.15, lw=0)
        ax.plot(g.mean().index, g.mean().values, color=col, ls=ls, lw=1.3, marker=mk, ms=3.5, mew=0)
        ax.axhline(real.loc[(est, cent, hl), "letter_l"], color=INK, lw=0.9, ls=":")
        short = lab.replace("TVP-", "").replace(", package", ", pkg")
        ax.set_title(f"{short}, {hl} d", loc="left", fontsize=7.5, color=INK)
        clean(ax)
    axes[-1].axis("off")
    axes[-1].text(0.0, 0.95, "Lines: simulated long band\n(mean; band: seed range)\nDotted: real estimate\n"
                  f"Shaded: c in [{clo:.2f}, {chi:.2f}]\npkg: package centring\nEWMA: causal centring\n"
                  "15 d, 60 d: window half-life", transform=axes[-1].transAxes, fontsize=6.8, color=INK2, va="top")
    axes[3].xaxis.set_tick_params(labelbottom=True)   # the panel below it is blank
    for ax in axes[3:7]:
        ax.set_xlabel("Drift share c")
    for ax in (axes[0], axes[4]):
        ax.set_ylabel("Points")
    fig.tight_layout(w_pad=0.6, h_pad=0.8)
    save(fig, "fig_indirect")


if __name__ == "__main__":
    tab_pseudo()
    tab_decay()
    tab_real()
    tab_indirect()
    tab_harvar_net()
    fig_hourlydaily()
    fig_tradeoff()
    fig_indirect()
