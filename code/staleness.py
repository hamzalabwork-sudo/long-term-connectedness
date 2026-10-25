"""Is Bitfinex's net-receiver role (Shi & Lin, 2026) a stale-price effect?

1. Sampling test: time-domain NET of each venue in the four instrument networks, with RV
   from 5-minute versus 15-minute returns (package settings, 2021-2025). Stale prices delay
   measured volatility at fine sampling, which shows up as "receiving"; that effect should
   fade at coarser sampling, while a genuine information-flow role should not.
2. Time-series test: monthly Bitfinex NET on its stale-minute share relative to the other
   venues (same instrument), slope t-statistic with Newey-West (3 lags).

Usage:
    python staleness.py --five shilin_package --fifteen shilin_package_rv15
"""
import argparse
import math
import os

import numpy as np
import pandas as pd

INSTRUMENTS = ["BTCS", "BTCP", "ETHS", "ETHP"]
EXCHANGES = ["BN", "OKX", "BG", "BF"]


def nw_ols(y, x, lags=3):
    X = np.column_stack([np.ones(len(x)), x])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    u = y - X @ beta
    S = (X * u[:, None]).T @ (X * u[:, None])
    for l in range(1, lags + 1):
        G = (X[l:] * u[l:, None]).T @ (X[:-l] * u[:-l, None])
        S += (1 - l / (lags + 1)) * (G + G.T)
    XtXi = np.linalg.inv(X.T @ X)
    V = XtXi @ S @ XtXi
    return beta[1], beta[1] / math.sqrt(V[1, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/conn")
    ap.add_argument("--five", default="shilin_package")
    ap.add_argument("--fifteen", default="shilin_package_rv15")
    ap.add_argument("--stale", default="data/processed/stale_share.csv.gz")
    ap.add_argument("--end", default="2025-12-31 23:00")
    ap.add_argument("--out", default="results/staleness")
    args = ap.parse_args()

    print("1. Mean time-domain NET in the instrument networks, 5-min vs 15-min RV")
    rows = []
    for inst in INSTRUMENTS:
        d5 = pd.read_csv(os.path.join(args.root, args.five, f"instrument_{inst}_dynamic.csv.gz"), index_col=0)
        d15 = pd.read_csv(os.path.join(args.root, args.fifteen, f"instrument_{inst}_dynamic.csv.gz"), index_col=0)
        for ex in EXCHANGES:
            c = f"{inst}_{ex}"
            rows.append({"network": inst, "venue": ex, "NET_5m": d5[f"NET_time_{c}"].mean(),
                         "NET_15m": d15[f"NET_time_{c}"].mean()})
    t1 = pd.DataFrame(rows)
    t1["change"] = t1["NET_15m"] - t1["NET_5m"]
    print(t1.pivot(index="venue", columns="network", values=["NET_5m", "NET_15m"]).round(2).to_string())

    print("\n2. Monthly Bitfinex NET on its relative stale share (5-min RV, package settings)")
    stale = pd.read_csv(args.stale, index_col=0, parse_dates=True)
    stale = stale[stale.index <= pd.Timestamp(args.end, tz="UTC")]
    out = []
    pooled_y, pooled_x = [], []
    for inst in INSTRUMENTS:
        d5 = pd.read_csv(os.path.join(args.root, args.five, f"instrument_{inst}_dynamic.csv.gz"),
                         index_col=0, parse_dates=True)
        net = d5[f"NET_time_{inst}_BF"].resample("MS").mean()
        others = [f"{inst}_{e}" for e in EXCHANGES if e != "BF"]
        rel = (stale[f"{inst}_BF"] - stale[others].mean(axis=1)).resample("MS").mean()
        df = pd.concat([net.rename("net"), rel.rename("rel_stale")], axis=1).dropna()
        b, t = nw_ols(df["net"].to_numpy(), 100 * df["rel_stale"].to_numpy())
        out.append({"network": inst, "months": len(df), "corr": df.corr().iloc[0, 1],
                    "slope_per_pp": b, "NW_t": t,
                    "mean_rel_stale_pp": 100 * df["rel_stale"].mean()})
        pooled_y.append(df["net"].to_numpy() - df["net"].mean())
        pooled_x.append(100 * (df["rel_stale"].to_numpy() - df["rel_stale"].mean()))
    t2 = pd.DataFrame(out)
    print(t2.round(3).to_string(index=False))
    b, t = nw_ols(np.concatenate(pooled_y), np.concatenate(pooled_x))
    print(f"pooled (within-network): slope {b:.3f} NET points per pp of relative staleness, NW t {t:.2f}")
    # network-averaged monthly series (the four networks share no Bitfinex series, but share the venue)
    ya = pd.concat([pd.Series(y) for y in pooled_y], axis=1).mean(axis=1).to_numpy()
    xa = pd.concat([pd.Series(x) for x in pooled_x], axis=1).mean(axis=1).to_numpy()
    ba, ta = nw_ols(ya, xa)
    print(f"network-averaged: slope {ba:.3f}, NW t {ta:.2f}")
    os.makedirs(args.out, exist_ok=True)
    t1.to_csv(os.path.join(args.out, "staleness_sampling.csv"), index=False)
    t2.assign(pooled_slope=b, pooled_t=t, avg_slope=ba, avg_t=ta).to_csv(
        os.path.join(args.out, "staleness_monthly.csv"), index=False)


if __name__ == "__main__":
    main()
