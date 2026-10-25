"""LaTeX tables for the paper, generated from the result files (no hand-typed numbers).

  tab_replication.tex   published (Shi & Lin 2026) vs our package-centred vs EWMA-centred
                        TCI and band components, 8 networks, 2021-2025
  tab_forecast.tex      pre-registered forecast comparisons, validation 2024 and test 2025-26
  tab_robustness.tex    centring result across estimation settings (appendix)
  tab_simulation.tex    simulation, truth vs estimators (appendix)
  tab_cleaning.tex      data cleaning summary per series (appendix)
  tab_stage.tex         test-stage MCS, economic value, events (appendix)

Usage (from the work directory):  python ../code/make_tables.py --out ../paper/tables
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_shilin import PUBLISHED  # noqa: E402

NETS = list(PUBLISHED)
NET_LABEL = {"instrument:BTCS": "BTC spot", "instrument:BTCP": "BTC perpetual",
             "instrument:ETHS": "ETH spot", "instrument:ETHP": "ETH perpetual",
             "exchange:BN": "Binance", "exchange:OKX": "OKX", "exchange:BG": "Bitget",
             "exchange:BF": "Bitfinex"}


def run_means(root, run):
    meta = json.load(open(os.path.join(root, run, "meta.json")))
    out = {}
    for spec, info in meta["networks"].items():
        b = list(info["mean_TCI_bands"].values())
        out[spec] = [info["mean_TCI_time"]] + b
    return out


def net_signs(root, run):
    match = total = 0
    for spec, (*_, net) in PUBLISHED.items():
        d = pd.read_csv(os.path.join(root, run, spec.replace(":", "_") + "_dynamic.csv.gz"), index_col=0)
        for c, v in net.items():
            match += np.sign(d[f"NET_time_{c}"].mean()) == np.sign(v)
            total += 1
    return int(match), total


def f2(x):
    return f"{x:.2f}"


def tab_replication(args):
    pkg, ewm = run_means(args.conn, "shilin_package"), run_means(args.conn, "shilin_ewma15d")
    lines = [r"\begin{table}[t]\centering\footnotesize",
             r"\caption{Total connectedness (TCI) and its short, mid and long components, 2021--2025: "
             r"published values, our replication with package centring, and causal EWMA centring.}",
             r"\label{tab:replication}",
             r"\begin{tabular}{lrrrrrrrrrrrr}\toprule",
             r" & \multicolumn{4}{c}{Shi \& Lin (2026)} & \multicolumn{4}{c}{Package centring (ours)} "
             r"& \multicolumn{4}{c}{EWMA centring (ours)}\\",
             r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}\cmidrule(lr){10-13}",
             r"Network & TCI & Short & Mid & Long & TCI & Short & Mid & Long & TCI & Short & Mid & Long\\\midrule"]
    for i, spec in enumerate(NETS):
        if i == 4:
            lines.append(r"\addlinespace")
        pub = PUBLISHED[spec][:4]
        row = [NET_LABEL[spec]] + [f2(v) for v in pub] + [f2(v) for v in pkg[spec]] + [f2(v) for v in ewm[spec]]
        lines.append(" & ".join(row) + r"\\")
    m, t = net_signs(args.conn, "shilin_package")
    lines += [r"\bottomrule\end{tabular}",
              r"\par\smallskip\parbox{\linewidth}{\scriptsize Notes: averages of the dynamic measures, "
              r"TVP-VAR(3), $H=720$, coefficient and variance half-lives of 15 and 7 days, bands labelled "
              r"1--8, 8--24 and $>24$ h by ConnectednessApproach (true cycles of 2--16, 16--48 and $>48$ h). "
              r"The first four rows are cross-exchange instrument networks, the last four cross-instrument "
              rf"exchange networks. Signs of the time-domain NET of our package replication agree with the "
              rf"published signs for {m} of {t} variables.}}",
              r"\end{table}"]
    open(os.path.join(args.out, "tab_replication.tex"), "w").write("\n".join(lines))


def stars(p):
    return r"$^{**}$" if p < 0.01 else r"$^{*}$" if p < 0.05 else ""


def tab_forecast(args):
    rows = []
    for split in ("validation", "test"):
        d = os.path.join(args.fc, split)
        t = pd.read_csv(os.path.join(d, "tests.csv"))
        e = pd.read_csv(os.path.join(d, "tests_exploratory_vsM0.csv"))
        for m, lab in (("M3_0-8", "+ short band"), ("M3_8-24", "+ mid band"), ("M3_24-inf", "+ long band"),
                       ("M2", "+ time domain"), ("M4", "+ all bands")):
            sub = t[t.model == m].set_index("h")
            p = sub["Holm_p_H1"] if m.startswith("M3_") else sub["DM_p"]
            rows.append((split, "A", lab, {h: (sub.loc[h, "gain_pct_vs_M1"], p.loc[h]) for h in sub.index}))
            rows.append((split, "B", lab, {h: (sub.loc[h, "CW_t"], np.nan) for h in sub.index}))
        st = e[e.model == "S_time vs E0"].set_index("h")
        rows.append((split, "C", "Spillover vs equal weights", {h: (st.loc[h, "gain_pct_vs_M0"], st.loc[h, "Holm_p_H4"]) for h in st.index}))
        e0 = e[e.model == "E0"].set_index("h")
        rows.append((split, "C", "Equal-weight partners vs HAR", {h: (e0.loc[h, "gain_pct_vs_M0"], p_from_t(e0.loc[h, "DM_t"])) for h in e0.index}))
    hs = [1, 8, 24, 168]
    panels = {"A": r"A. QLIKE reduction (\%) relative to $\mathcal{M}_1$ (pre-specified, primary)",
              "B": r"B. Clark--West $t$-statistic, squared error, nested in $\mathcal{M}_1$",
              "C": r"C. Partner volatility, QLIKE reduction (\%)"}
    lines = [r"\begin{table}[t]\centering\footnotesize",
             r"\caption{Out-of-sample forecasts of average realised variance over the next $h$ hours.}",
             r"\label{tab:forecast}",
             r"\begin{tabular}{lrrrrrrrr}\toprule",
             r" & \multicolumn{4}{c}{Validation (2024)} & \multicolumn{4}{c}{Test (Jan 2025 -- Sep 2026)}\\",
             r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
             r"$h$ (hours) & " + " & ".join(str(h) for h in hs * 2) + r"\\\midrule"]
    for pan, title in panels.items():
        lines.append(rf"\multicolumn{{9}}{{l}}{{\emph{{{title}}}}}\\")
        names = [r[2] for r in rows if r[0] == "validation" and r[1] == pan]
        for name in names:
            cells = []
            for split in ("validation", "test"):
                vals = next(r[3] for r in rows if r[0] == split and r[1] == pan and r[2] == name)
                for h in hs:
                    v, p = vals[h]
                    cells.append(f"{v:.2f}" if pan == "B" else f"{v:+.2f}{stars(p)}")
            lines.append(f"\\quad {name} & " + " & ".join(cells) + r"\\")
    lines += [r"\bottomrule\end{tabular}",
              r"\par\smallskip\parbox{\linewidth}{\scriptsize Notes: losses pooled over 16 series; positive "
              r"reductions are improvements. $\mathcal{M}_1$: HAR plus lagged volatility of the six network partners. "
              r"Connectedness from the real-time TVP-VAR (causal EWMA centring; bands of 2--8, 8--24 and $>24$ hour "
              r"cycles). Panel A: one-sided Diebold--Mariano tests with Newey--West errors; $^{*}$/$^{**}$ significant "
              r"at 5\%/1\% after Holm correction across the 12 band tests (band rows), unadjusted for the time-domain "
              r"and all-band rows. Panel B (computed from the same losses after the test period was opened): "
              r"Clark--West statistics, one-sided critical value 1.645. Panel C "
              r"(added after the validation year, before the test): spillover-weighted versus equal-weighted partner "
              r"volatility, Holm-corrected across 16 tests; equal-weighted partners versus univariate HAR, unadjusted.}",
              r"\end{table}"]
    open(os.path.join(args.out, "tab_forecast.tex"), "w").write("\n".join(lines))


def tab_versions(args):
    """H3: band connectedness from different centrings, test period, QLIKE reduction vs M1."""
    t = pd.read_csv(os.path.join(args.fc, "test", "tests.csv"))
    names = {"": "Causal EWMA centring (primary)", f"@{args.window_run}": "First-30-day centring",
             f"@{args.tvint_run}": "Time-varying intercept", f"@{args.pkg_run}": "Package centring (look-ahead)"}
    hs = [1, 8, 24, 168]
    lines = [r"\begin{table}[ht]\centering\footnotesize",
             r"\caption{Test period: QLIKE reduction (\%) relative to $\mathcal{M}_1$ for band connectedness "
             r"built with different centrings.}", r"\label{tab:versions}",
             r"\begin{tabular}{llrrrr}\toprule", r"Centring & Band & " + " & ".join(map(str, hs)) + r"\\\midrule"]
    for suf, lab in names.items():
        for b, bl in (("0-8", "Short"), ("8-24", "Mid"), ("24-inf", "Long")):
            sub = t[t.model == f"M3_{b}{suf}"].set_index("h")
            if sub.empty:
                continue
            cells = [f"{sub.loc[h, 'gain_pct_vs_M1']:+.2f}{stars(sub.loc[h, 'DM_p'])}" for h in hs]
            lines.append(f"{lab if b == '0-8' else ''} & {bl} & " + " & ".join(cells) + r"\\")
        lines.append(r"\addlinespace")
    lines[-1] = r"\bottomrule\end{tabular}"
    lines += [r"\par\smallskip\parbox{\linewidth}{\scriptsize Notes: one-sided Diebold--Mariano tests, unadjusted; "
              r"$^{*}$/$^{**}$ 5\%/1\%. Package centring uses full-sample means and prior through September 2026 "
              r"and is therefore not available in real time.}", r"\end{table}"]
    open(os.path.join(args.out, "tab_versions.tex"), "w").write("\n".join(lines))


def tab_more(args):
    """Remaining pre-specified results: squared-error gains, per-series test counts, gamma = 1 fees and
    tracking error."""
    hs = [1, 8, 24, 168]
    rows = {("M3_0-8", "+ short band"), ("M3_8-24", "+ mid band"), ("M3_24-inf", "+ long band"),
            ("M2", "+ time domain"), ("M4", "+ all bands")}
    order = ["M3_0-8", "M3_8-24", "M3_24-inf", "M2", "M4"]
    lab = dict(rows)
    tv = pd.read_csv(os.path.join(args.fc, "validation", "tests.csv"))
    tt = pd.read_csv(os.path.join(args.fc, "test", "tests.csv"))
    per = pd.read_csv(os.path.join(args.fc, "test", "stage_per_series.csv"))
    econ = pd.read_csv(os.path.join(args.fc, "test", "stage_econ.csv"))
    lines = [r"\begin{table}[ht]\centering\footnotesize",
             r"\caption{Further pre-specified forecast results: squared-error gains, per-series tests, and economic value.}",
             r"\label{tab:more}", r"\begin{tabular}{lrrrrrrrr}\toprule",
             r" & \multicolumn{4}{c}{Validation (2024)} & \multicolumn{4}{c}{Test (Jan 2025 -- Sep 2026)}\\",
             r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
             r"$h$ (hours) & " + " & ".join(str(h) for h in hs * 2) + r"\\\midrule",
             r"\multicolumn{9}{l}{\emph{A. Squared-error reduction (\%) vs $\mathcal{M}_1$}}\\"]
    for m in order:
        cells = [f"{t[(t.model == m) & (t.h == h)].mse_gain_pct_vs_M1.iloc[0]:+.2f}" for t in (tv, tt) for h in hs]
        lines.append(f"\\quad {lab[m]} & " + " & ".join(cells) + r"\\")
    lines.append(r"\multicolumn{9}{l}{\emph{B. Series with a significant gain (of 16): DM / CW}}\\")
    for m in order[:3]:
        cells = ["", "", "", ""]
        for h in hs:
            r = per[(per.model == m) & (per.h == h)].iloc[0]
            cells.append(f"{int(r.series_DM_sig5)} / {int(r.series_CW_sig5)}")
        lines.append(f"\\quad {lab[m]} & " + " & ".join(cells) + r"\\")
    lines += [r"\bottomrule\end{tabular}"]
    lines += [r"\par\medskip", r"\begin{tabular}{lrrrr}\toprule",
              r" & \multicolumn{2}{c}{BTC perpetual} & \multicolumn{2}{c}{ETH perpetual}\\",
              r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
              r"C. Volatility targeting, test & Fee, $\gamma=1$ (bp) & Tracking error & Fee, $\gamma=1$ (bp) & Tracking error\\\midrule"]
    names = {"M0": "HAR", "M1": r"$\mathcal{M}_1$", "M2": "+ time domain", "M3_0-8": "+ short band",
             "M3_8-24": "+ mid band", "M3_24-inf": "+ long band", "M4": "+ all bands", "E0": "Equal-weight partners",
             "S_time": "Spillover-weighted partners"}
    for m, n in names.items():
        cells = [n]
        for a in ("BTC", "ETH"):
            r = econ[(econ.model == m) & (econ.asset == a)].iloc[0]
            cells += ["0" if m == "M1" else f"{r.fee_bp_g1:+.0f}", f"{r.tracking_err:.3f}"]
        lines.append(" & ".join(cells) + r"\\")
    lines += [r"\bottomrule\end{tabular}",
              r"\par\smallskip\parbox{\linewidth}{\scriptsize Notes: Panel A: pooled squared error of the log "
              r"target (the pre-specified secondary loss). Panel B: test period, number of series with a one-sided "
              r"Diebold--Mariano (QLIKE) / Clark--West gain against $\mathcal{M}_1$ significant at 5\%. "
              r"Panel C: Fleming--Kirby--Ostdiek fee versus $\mathcal{M}_1$ for relative risk aversion 1, and absolute "
              r"difference between the realised annualised volatility of the targeted position and the target.}",
              r"\end{table}"]
    open(os.path.join(args.out, "tab_more.tex"), "w").write("\n".join(lines))


PKG_DESC = {
    "dy2012": "Asset-class volatilities (Diebold--Yilmaz 2012)", "jcggh2022": "Metal and energy realised volatilities",
    "cgg2022": "Crude oil, absolute returns", "cgs2021": "Interest rate swaps", "acg2020": "Exchange rates",
    "bgu2021": "Oil futures and commodities", "cegg2022": "Oil shocks and exchange rates", "cg2021": "EMU bond yields",
    "cgp2024": "Clean energy stocks", "cgs2022": "European yield curves", "dy2009": "Equity returns (Diebold--Yilmaz 2009)",
    "g2020": "Exchange rates", "gcat2022": "Green bond and equity indices", "gg2018": "Economic uncertainty indices",
    "lw2021": "Sector returns", "ivol_crossasset": "Equity, oil, gold, euro", "ivol_usequity": "US equity indices",
    "ivol_global": "Equity, EM equity, oil, gold", "aaacgo2022": "Oil and Islamic sector stocks$^{a}$", "bcg2022": "Green and conventional bonds$^{a}$"}


def tab_pkgdata(args):
    blocks = [("Example datasets of ConnectednessApproach", pd.read_csv(os.path.join(args.pkg, "summary.csv")))]
    ivol_path = os.path.join(args.ivol, "summary.csv")
    if os.path.exists(ivol_path):
        blocks.append(("Implied-volatility networks (FRED, log levels)", pd.read_csv(ivol_path)))
    lines = [r"\begin{table}[t]\centering\footnotesize",
             r"\caption{Long-band connectedness under package and causal EWMA centring: the package's example "
             r"datasets and implied-volatility networks.}", r"\label{tab:pkgdata}",
             r"\begin{tabular}{llrrrrrrrrrr}\toprule",
             r" & & & & \multicolumn{3}{c}{Default set-up} & \multicolumn{5}{c}{$H=100$, split at $\pi/5$}\\",
             r"\cmidrule(lr){5-7}\cmidrule(lr){8-12}",
             r"Dataset & Content & $k$ & Slow & Pkg & EWMA & $\Delta$ & Pkg & EWMA & $\Delta$ & Long share & $\Delta$TCI\\\midrule"]
    for title, s in blocks:
        d, b = s[s.setup == "default"].set_index("dataset"), s[s.setup == "bk"].set_index("dataset")
        lines.append(rf"\multicolumn{{12}}{{l}}{{\emph{{{title}}}}}\\")
        for n in d.sort_values("slow_share", ascending=False).index:
            r1, r2 = d.loc[n], b.loc[n]
            lines.append(f"\\texttt{{{n.replace('_', chr(92) + '_')}}} & {PKG_DESC.get(n, '')} & {int(r1.k)} & {r1.slow_share:.2f} & "
                         f"{r1.long_pkg:.1f} & {r1.long_ewma:.1f} & {r1.d_long:+.1f} & {r2.long_pkg:.1f} & {r2.long_ewma:.1f} & "
                         f"{r2.d_long:+.1f} & {r2.long_share_ewma:.2f} & {r2.TCI_pkg - r2.TCI_ewma:+.1f}" + r"\\")
    lines += [r"\bottomrule\end{tabular}",
              r"\par\smallskip\parbox{\linewidth}{\scriptsize Notes: each package dataset accompanies a published study "
              r"documented in the package \citep{gabauer2025}. TVP-VAR(1) with the package defaults "
              r"($\kappa_1=\kappa_2=0.99$, Bayes prior); causal EWMA centring with the matching half-life (69 "
              r"observations) and a prior from the first 10\% of the sample. Default set-up: $H=10$, partition "
              r"$(\pi,\pi/2,0)$. Slow: share of variance in a centred moving average over 138 observations. Pkg and "
              r"EWMA: long component under each centring; $\Delta$: their difference; Long share: long component as a "
              r"share of total frequency connectedness under EWMA centring; $\Delta$TCI: change in total connectedness. "
              r"Implied-volatility networks: daily FRED closes of VIX, OVX, GVZ and EVZ (cross-asset, 2008--2025); VIX, "
              r"VXN, VXD and RVX (US equity, 2004--2026); VIX, VXEEM, OVX and GVZ (global, 2011--2026). "
              r"$^{a}$Prices converted to log returns, as in the source study. \texttt{gghm2022} (51 variables, "
              r"estimated with regularised VARs in its study) is omitted.}", r"\end{table}"]
    open(os.path.join(args.out, "tab_pkgdata.tex"), "w").write("\n".join(lines))


def tab_fiveband(args):
    runs = [("Package centring", "rob_five_pkg"), ("Causal EWMA centring", "rob_five_ewma")]
    if not all(os.path.exists(os.path.join(args.conn, r, "meta.json")) for _, r in runs):
        return
    lines = [r"\begin{table}[ht]\centering\footnotesize",
             r"\caption{Five-band decomposition (package labels), mean of the 8 networks, 2021--2025.}",
             r"\label{tab:fiveband}", r"\begin{tabular}{lrrrrrr}\toprule",
             r"Centring & TCI & 1--4 h & 4--8 h & 8--24 h & 24--168 h & $>$168 h\\\midrule"]
    for lab, run in runs:
        m = np.array(list(run_means(args.conn, run).values()))
        lines.append(lab + " & " + " & ".join(f"{v:.1f}" for v in m.mean(0)) + r"\\")
    a = np.array(list(run_means(args.conn, runs[0][1]).values()))
    b = np.array(list(run_means(args.conn, runs[1][1]).values()))
    lines.append("Difference & " + " & ".join(f"{v:+.1f}" for v in (a - b).mean(0)) + r"\\")
    lines += [r"\bottomrule\end{tabular}",
              r"\par\smallskip\parbox{\linewidth}{\scriptsize Notes: band edges as labelled by ConnectednessApproach "
              r"($\pi/\omega$); true cycle lengths are twice as long.}", r"\end{table}"]
    open(os.path.join(args.out, "tab_fiveband.tex"), "w").write("\n".join(lines))


def tab_staleness(args):
    t1 = pd.read_csv(os.path.join(args.stale, "staleness_sampling.csv"))
    t2 = pd.read_csv(os.path.join(args.stale, "staleness_monthly.csv"))
    bf = t1[t1.venue == "BF"].set_index("network")
    lab = {"BTCS": "BTC spot", "BTCP": "BTC perpetual", "ETHS": "ETH spot", "ETHP": "ETH perpetual"}
    lines = [r"\begin{table}[ht]\centering\footnotesize",
             r"\caption{Bitfinex in the instrument networks: net connectedness and stale prices, 2021--2025.}",
             r"\label{tab:staleness}", r"\begin{tabular}{lrrrrr}\toprule",
             r"Network & NET (5-min RV) & NET (15-min RV) & Relative staleness (pp) & Slope & Corr.\\\midrule"]
    for _, r in t2.iterrows():
        n = r["network"]
        lines.append(f"{lab[n]} & {bf.loc[n, 'NET_5m']:.2f} & {bf.loc[n, 'NET_15m']:.2f} & "
                     f"{r['mean_rel_stale_pp']:.1f} & {r['slope_per_pp']:.3f} & {r['corr']:.2f}" + r"\\")
    lines += [r"\bottomrule\end{tabular}",
              rf"\par\smallskip\parbox{{\linewidth}}{{\scriptsize Notes: time-domain NET, package settings. Relative "
              rf"staleness: share of minutes without a trade on Bitfinex minus the mean of the other venues. Slope: "
              rf"monthly NET on relative staleness (per percentage point). Network average: slope "
              rf"{t2['avg_slope'].iloc[0]:.3f}, Newey--West $t={t2['avg_t'].iloc[0]:.2f}$ (60 months).}}", r"\end{table}"]
    open(os.path.join(args.out, "tab_staleness.tex"), "w").write("\n".join(lines))


def p_from_t(t):
    import math
    return 0.5 * math.erfc(t / math.sqrt(2))


def tab_robustness(args):
    variants = [("Baseline", "shilin_package", "shilin_ewma15d"),
                ("2 lags", "rob_lag2_pkg", "rob_lag2_ewma"), ("4 lags", "rob_lag4_pkg", "rob_lag4_ewma"),
                ("$H=360$", "rob_H360_pkg", "rob_H360_ewma"), ("$H=1440$", "rob_H1440_pkg", "rob_H1440_ewma"),
                (r"$\kappa_1=\kappa_2=0.99$", "rob_kap099_pkg", "rob_kap099_ewmaM"),
                ("Half-lives 30/15 days", "rob_kap30_pkg", "rob_kap30_ewmaM"),
                ("True-period bands", "rob_true_pkg", "rob_true_ewma"),
                ("15-minute RV", "rob_rv15_pkg", "rob_rv15_ewma")]
    lines = [r"\begin{table}[ht]\centering\footnotesize",
             r"\caption{Effect of full-sample (package) versus causal EWMA centring across estimation settings, "
             r"2021--2025: mean of the 8 networks.}",
             r"\label{tab:robustness}", r"\begin{tabular}{lrrrrrrcc}\toprule",
             r" & \multicolumn{2}{c}{TCI} & \multicolumn{2}{c}{Long} & \multicolumn{2}{c}{$\Delta$Long} "
             r"& \multicolumn{2}{c}{Long$>$Short}\\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
             r"Setting & Pkg & EWMA & Pkg & EWMA & Mean & Range & Pkg & EWMA\\\midrule"]
    for lab, rp, re in variants:
        if not all(os.path.exists(os.path.join(args.conn, r, "meta.json")) for r in (rp, re)):
            lines.append(lab + " & " + " & ".join(["--"] * 8) + r"\\")
            continue
        a = np.array(list(run_means(args.conn, rp).values()))
        b = np.array(list(run_means(args.conn, re).values()))
        dl = a[:, 3] - b[:, 3]
        cells = [lab, f"{a[:, 0].mean():.1f}", f"{b[:, 0].mean():.1f}", f"{a[:, 3].mean():.1f}", f"{b[:, 3].mean():.1f}",
                 f"{dl.mean():.1f}", f"{dl.min():.1f}--{dl.max():.1f}",
                 f"{int((a[:, 3] > a[:, 1]).sum())}/8", f"{int((b[:, 3] > b[:, 1]).sum())}/8"]
        lines.append(" & ".join(cells) + r"\\")
    lines.append(r"\addlinespace")
    for lab, run in (("EWMA half-life 7 days", "rob_ewma7"), ("EWMA half-life 30 days", "rob_ewma30"),
                     ("Time-varying intercept", "rob_tvint")):
        if os.path.exists(os.path.join(args.conn, run, "meta.json")):
            m = np.array(list(run_means(args.conn, run).values()))
            lines.append(f"{lab} & & {m[:, 0].mean():.1f} & & {m[:, 3].mean():.1f} & & & & "
                         f"{int((m[:, 3] > m[:, 1]).sum())}/8" + r"\\")
    lines += [r"\bottomrule\end{tabular}",
              r"\par\smallskip\parbox{\linewidth}{\scriptsize Notes: one setting changed at a time from the "
              r"baseline (3 lags, $H=720$, half-lives 15/7 days, package band labels, 5-minute RV). $\Delta$Long is the "
              r"long component under package minus EWMA centring (mean and range over the 8 networks). "
              r"`Long$>$Short' counts networks in which the long component exceeds the short one. When the forgetting factors change, the EWMA half-life is matched to the coefficient half-life (2.9 and 30 days). The last rows "
              r"vary the real-time centring itself, in the EWMA column (the intercept model uses window centring).}",
              r"\end{table}"]
    open(os.path.join(args.out, "tab_robustness.tex"), "w").write("\n".join(lines))


def tab_simulation(args):
    cases = [("centering_level30d.csv", "Observed drift, 30-day level"),
             ("centering_level90d.csv", "Observed drift, 90-day level"),
             ("centering_btcs.csv", "Observed drift, BTC-spot network"),
             ("centering_randomlevel.csv", "Random drift (new path each run)"),
             ("centering_persistent.csv", "Fast VAR (raw series)"),
             ("centering_harvar.csv", "Slow spillovers (HAR-VAR, raw series)")]
    ests = [("package", "Package"), ("window30d", "First 30 d"), ("ewma15d", "EWMA 15 d"),
            ("ewma60d", "EWMA 60 d"), ("tvintercept", "TV intercept")]
    lines = [r"\begin{table}[ht]\centering\footnotesize",
             r"\caption{Simulation: long-band TCI (package labels), truth and mean estimates across replications.}",
             r"\label{tab:simulation}",
             r"\begin{tabular}{lrr" + "r" * len(ests) + r"}\toprule",
             r"Design & $c$ & Truth & " + " & ".join(e[1] for e in ests) + r"\\\midrule"]
    for f, lab in cases:
        path = os.path.join(args.sim, f)
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        truth = df.loc[df.estimator == "TRUTH", "long"].iloc[0]
        for scale in sorted(df.scale.dropna().unique()):
            sub = df[np.isclose(df.scale, scale)]
            cells = [lab, f"{scale:.1f}", f"{truth:.1f}"]
            for e, _ in ests:
                v = sub.loc[sub.estimator == e, "long"]
                cells.append(f"{v.mean():.1f}")
            lines.append(" & ".join(cells) + r"\\")
    lines += [r"\bottomrule\end{tabular}",
              r"\par\smallskip\parbox{\linewidth}{\scriptsize Notes: drift designs: $y_t=\bar y+c\,L_t+x_t$, with "
              r"$x_t$ a VAR(3) fitted to the de-levelled Binance (or BTC-spot) network and $L_t$ either its observed "
              r"slow level or, in the random design, a common AR(1) level with a 30-day half-life and the same "
              r"standard deviation, drawn afresh in each replication. The fast VAR is a VAR(3) and the slow-spillover "
              r"design a HAR-type VAR (daily and weekly cross-averages; largest root 0.9985) fitted to the raw series, so "
              r"that their low-frequency dynamics are genuine. 5--10 replications of 43,824 hours; standard deviations "
              r"across replications are at most 1.8 points except for first-30-day centring (up to 12.2).}", r"\end{table}"]
    open(os.path.join(args.out, "tab_simulation.tex"), "w").write("\n".join(lines))


def tab_cleaning(args):
    diag = json.load(open(args.diag))["series"]
    lines = [r"\begin{table}[ht]\centering\footnotesize",
             r"\caption{Data cleaning, January 2021 -- September 2026 (1-minute bars).}", r"\label{tab:cleaning}",
             r"\begin{tabular}{lrrrr}\toprule",
             r"Series & Minutes with trades (\%) & Bad prints removed & Dislocation minutes kept & "
             r"Missing hours\\\midrule"]
    for name, v in diag.items():
        lines.append(f"{name.replace('_', r'\_')} & {v['minutes_observed_pct']:.2f} & {v['bad_prints_removed']} & "
                     f"{v['dislocation_minutes_kept']} & {v['missing_hours_after_interp']}" + r"\\")
    lines += [r"\bottomrule\end{tabular}",
              r"\par\smallskip\parbox{\linewidth}{\scriptsize Notes: zero-volume bars carry no trade; bad prints "
              r"are deviations of more than 3\% from the cross-venue median lasting at most 3 minutes; longer "
              r"deviations are kept as genuine dislocations; missing hours remain after interpolating gaps of up "
              r"to 3 hours (50,376 hours in total). The forecasting exercise uses the same panel without any "
              r"interpolation, carrying the last value forward, so that it uses no future information.}", r"\end{table}"]
    open(os.path.join(args.out, "tab_cleaning.tex"), "w").write("\n".join(lines))


def tab_stage(args):
    d = os.path.join(args.fc, "test")
    mcs = pd.read_csv(os.path.join(d, "stage_mcs.csv"))
    econ = pd.read_csv(os.path.join(d, "stage_econ.csv"))
    lab = {"M0": "HAR", "M1": r"$\mathcal{M}_1$", "M2": "+ time domain", "M3_0-8": "+ short band",
           "M3_8-24": "+ mid band", "M3_24-inf": "+ long band", "M4": "+ all bands", "E0": "Equal-weight partners",
           "S_time": "Spillover-weighted partners"}
    events = pd.read_csv(os.path.join(d, "stage_events.csv"))
    per = pd.read_csv(os.path.join(d, "stage_per_series.csv"))
    lines = [r"\begin{table}[ht]\centering\scriptsize",
             r"\caption{Test period: model confidence set, economic value of volatility targeting, and event windows.}",
             r"\label{tab:stage}", r"\begin{tabular}{lrrrrrrrrrr}\toprule",
             r" & \multicolumn{4}{c}{MCS p-value ($h$)} & \multicolumn{2}{c}{BTC perpetual} & "
             r"\multicolumn{2}{c}{ETH perpetual} & \multicolumn{2}{c}{Events, $h=24$ (\%)}\\",
             r"\cmidrule(lr){2-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}\cmidrule(lr){10-11}",
             r"Model & 1 & 8 & 24 & 168 & Sharpe & Fee [95\% CI] & Sharpe & Fee [95\% CI] & Bybit & Cascade\\\midrule"]
    for m in ["M0", "M1", "M2", "M3_0-8", "M3_8-24", "M3_24-inf", "M4", "E0", "S_time"]:
        cells = [lab[m]]
        for h in (1, 8, 24, 168):
            r = mcs[(mcs.model == m) & (mcs.h == h)]
            cells.append(f"{r.mcs_p.iloc[0]:.2f}" if len(r) else "")
        for a in ("BTC", "ETH"):
            r = econ[(econ.model == m) & (econ.asset == a)]
            if not len(r):
                cells += ["", ""]
            elif m == "M1":
                cells += [f"{r.sharpe.iloc[0]:.2f}", "0"]
            else:
                r = r.iloc[0]
                cells += [f"{r.sharpe:.2f}", f"{r.fee_bp_g3:+.0f} [{r.fee_g3_ci_lo:+.0f}, {r.fee_g3_ci_hi:+.0f}]"]
        for ev in ("bybit_hack", "liquidation_cascade"):
            r = events[(events.model == m) & (events.event == ev) & (events.h == 24)]
            cells.append(f"{r.gain_pct_vs_M1.iloc[0]:+.1f}" if len(r) else ("0" if m == "M1" else ""))
        lines.append(" & ".join(cells) + r"\\")
    cw = per[(per.model.str.startswith("M3_")) & (per.h == 1)]
    lines += [r"\bottomrule\end{tabular}",
              r"\par\smallskip\parbox{\linewidth}{\scriptsize Notes: MCS of Hansen, Lunde and Nason (2011), "
              r"$T_{\max}$ statistic, circular block bootstrap (24 h, 5,000 draws) on pooled QLIKE; models with "
              r"p $\geq$ 0.10 are in the 90\% set. Economic value: daily volatility targeting with weight "
              r"$\min(\sigma^*/\hat\sigma,3)$, 5 bp per unit of turnover, funding not included; fee: Fleming, Kirby and "
              r"Ostdiek (2001) annual performance fee (bp) versus $\mathcal{M}_1$ for relative risk aversion 3, with a "
              r"20-day block-bootstrap interval. Events: QLIKE reduction versus $\mathcal{M}_1$ within $\pm$7 days of "
              r"the Bybit hack (21 February 2025) and the liquidation cascade (10 October 2025). Per-series tests "
              rf"at $h=1$: each band model beats $\mathcal{{M}}_1$ in {int(cw.series_DM_sig5.min())}--"
              rf"{int(cw.series_DM_sig5.max())} of 16 series (Diebold--Mariano, 5\%).}}", r"\end{table}"]
    open(os.path.join(args.out, "tab_stage.tex"), "w").write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conn", default="results/conn")
    ap.add_argument("--fc", default="results/forecast_causal")
    ap.add_argument("--sim", default="results/sim")
    ap.add_argument("--stale", default="results/staleness")
    ap.add_argument("--pkg", default="results/pkgdata")
    ap.add_argument("--ivol", default="results/ivol")
    ap.add_argument("--diag", default="data/processed/diagnostics.json")
    ap.add_argument("--window-run", default="rtc_window_true")
    ap.add_argument("--tvint-run", default="rtc_tvint_true")
    ap.add_argument("--pkg-run", default="pkgc_full_true")
    ap.add_argument("--out", default="../paper/tables")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for f in (tab_replication, tab_pkgdata, tab_forecast, tab_versions, tab_more, tab_robustness, tab_fiveband,
              tab_simulation, tab_cleaning, tab_stage, tab_staleness):
        try:
            f(args)
            print("wrote", f.__name__)
        except FileNotFoundError as e:
            print("skipped", f.__name__, "(missing input:", os.path.basename(str(e.filename)), ")")
    # shrink any table wider than the text block (never enlarge); needs \usepackage{adjustbox}
    for name in os.listdir(args.out):
        path = os.path.join(args.out, name)
        s = open(path).read()
        if r"\begin{tabular}" in s and "adjustbox" not in s:
            s = s.replace(r"\begin{tabular}", r"\begin{adjustbox}{max width=\linewidth}\begin{tabular}")
            s = s.replace(r"\end{tabular}", r"\end{tabular}\end{adjustbox}")
            open(path, "w").write(s)


if __name__ == "__main__":
    main()
