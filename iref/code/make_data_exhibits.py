"""Preliminary analysis of the 16 hourly log realised-variance series, 2021-2025 (the series the
connectedness models use: log RV with hour-of-day means removed).

Table (paper/tables/tab_descriptive.tex): mean, standard deviation, skewness, excess kurtosis,
Jarque-Bera, ADF (constant, lag length by AIC up to 48), Ljung-Box Q(24) and Q(168), and the share of
variance in the 30-day centred moving average ("slow share").
Figure (paper/figs/fig_series.pdf/png): daily mean of the hourly series, one panel per instrument,
one line per exchange (colour and line style).

Usage (from iref/):  python code/make_data_exhibits.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller

from make_exhibits import AXIS, INK, INK2, clean, save, write  # noqa: F401  (rcParams set on import)

PANEL = "../work/data/processed/logrv_ds.csv.gz"
EX = [("BN", "Binance", "#2a78d6", "-"), ("OKX", "OKX", "#eb6834", "--"),
      ("BG", "Bitget", "#1baf7a", "-."), ("BF", "Bitfinex", "#eda100", ":")]
INS = [("BTCS", "BTC spot"), ("BTCP", "BTC perpetual"), ("ETHS", "ETH spot"), ("ETHP", "ETH perpetual")]


def star(p):
    return "^{***}" if p < 0.01 else "^{**}" if p < 0.05 else "^{*}" if p < 0.1 else ""


def main():
    p = pd.read_csv(PANEL, index_col=0, parse_dates=True)
    p = p[p.index <= pd.Timestamp("2025-12-31 23:00", tz="UTC")]
    rows = []
    for ins, ilab in INS:
        for ex, elab, _, _ in EX:
            c = f"{ins}_{ex}"
            x = p[c].interpolate().ffill().bfill()
            v = x.to_numpy()
            jb = stats.jarque_bera(v)
            adf = adfuller(v, maxlag=48, autolag="AIC", regression="c")
            lb = acorr_ljungbox(v, lags=[24, 168], return_df=True)
            slow = x.rolling(720, center=True, min_periods=1).mean().var() / x.var()
            rows.append({"series": f"{ilab}, {elab}", "mean": v.mean(), "sd": v.std(ddof=1),
                         "skew": stats.skew(v), "kurt": stats.kurtosis(v), "jb": jb.statistic, "jb_p": jb.pvalue,
                         "adf": adf[0], "adf_p": adf[1], "q24": lb.lb_stat.iloc[0], "q168": lb.lb_stat.iloc[1],
                         "slow": slow, "missing": int(p[c].isna().sum())})
            print(rows[-1]["series"], round(adf[0], 1), round(slow, 2), flush=True)
    d = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    d.to_csv("results/descriptive.csv", index=False)
    lines = [r"\begin{table}[t]\centering\footnotesize",
             r"\caption{Descriptive statistics of hourly log realised variance, January 2021--December 2025.}",
             r"\label{tab:descriptive}", r"\begin{adjustbox}{max width=\linewidth}\begin{tabular}{lrrrrrrrrr}\toprule",
             r"Series & Mean & SD & Skew. & Ex.\ kurt. & JB & ADF & $Q(24)$ & $Q(168)$ & Slow share\\\midrule"]
    def n(x, fmt):
        return f"${format(x, fmt)}$"
    for _, r in d.iterrows():
        lines.append(f"{r['series']} & {n(r['mean'], '.2f')} & {n(r['sd'], '.2f')} & {n(r['skew'], '.2f')} & "
                     f"{n(r['kurt'], '.2f')} & ${r['jb']:,.0f}{star(r['jb_p'])}$ & ${r['adf']:.2f}{star(r['adf_p'])}$ & "
                     f"{r['q24']:,.0f} & {r['q168']:,.0f} & {r['slow']:.2f}\\\\")
    lines += [r"\bottomrule\end{tabular}\end{adjustbox}",
              r"\par\smallskip\parbox{\linewidth}{\scriptsize Notes: log of hourly realised variance (sum of squared "
              r"five-minute log returns) after removing hour-of-day means estimated on 2021--2023; 43,824 hourly "
              r"observations per series. JB: Jarque--Bera statistic. ADF: augmented Dickey--Fuller $t$-statistic with a "
              r"constant, lag length chosen by AIC (at most 48). $Q(m)$: Ljung--Box statistic with $m$ lags (all $p<0.01$). "
              r"Slow share: variance of the 30-day centred moving average relative to the series' variance. "
              r"$^{***}$, $^{**}$, $^{*}$: significance at 1\%, 5\%, 10\%.}", r"\end{table}"]
    write("tab_descriptive", lines)

    # one line per instrument (mean over the four venues, which are visually indistinguishable), its
    # 30-day centred level, and the full-sample mean that package centring subtracts
    daily = p.resample("1D").mean()
    gap = max(float((daily[[f"{ins}_{e[0]}" for e in EX]].sub(
        daily[[f"{ins}_{e[0]}" for e in EX]].mean(1), axis=0)).abs().quantile(0.99).max()) for ins, _ in INS)
    print(f"99th pct abs deviation of a venue from the cross-venue daily mean: {gap:.2f}")
    fig, axes = plt.subplots(2, 2, figsize=(6.3, 3.8), sharex=True, sharey=True)
    for ax, (ins, ilab) in zip(axes.ravel(), INS):
        m = daily[[f"{ins}_{e[0]}" for e in EX]].mean(1)
        lvl = p[[f"{ins}_{e[0]}" for e in EX]].mean(1).rolling(720, center=True, min_periods=1).mean().resample("1D").mean()
        ax.plot(m.index, m.values, color="#86b6ef", lw=0.6, label="Daily mean")
        ax.plot(lvl.index, lvl.values, color="#2a78d6", lw=1.6, label="30-day centred level")
        ax.axhline(m.mean(), color=INK, lw=0.9, ls="--", label="Full-sample mean")
        ax.set_title(ilab, loc="left", fontsize=8, color=INK)
        clean(ax)
    for ax in axes[:, 0]:
        ax.set_ylabel("Log realised variance")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=7, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, "fig_series")
    pd.Series({"venue_dev_p99": gap}).to_csv("results/series_venue_gap.csv")


if __name__ == "__main__":
    main()
