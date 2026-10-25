"""Does the centring effect appear in the example datasets distributed with ConnectednessApproach?

Each dataset (one per published study) is estimated with the package's default TVP-VAR settings
(1 lag, forgetting factors 0.99/0.99, Bayes prior) under package centring (full-sample mean and
prior) and causal EWMA centring (half-life matched to the coefficient forgetting, about 69
observations; prior from the first 10% of the sample). Two frequency set-ups:
  "default":  H = 10, partition (pi, pi/2, 0)          (ConnectednessApproach defaults)
  "bk":       H = 100, partition (pi + 1e-5, pi/5, 0)   (common Barunik-Krehlik-style split)
For each dataset, the slow-movement share is the variance of a centred moving average over twice
the half-life, relative to total variance, averaged over series. Price-level datasets
(aaacgo2022, bcg2022) are converted to log returns (x100), as in the studies they come from;
gghm2022 (51 variables, estimated with regularised VARs in its study) is excluded.

Usage (from the work directory):  python ../code/package_datasets.py --data pkgdata/csv --out results/pkgdata
"""
import argparse
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from connectedness import connectedness, tvp_var  # noqa: E402

LEVELS = {"aaacgo2022", "bcg2022"}
EXCLUDE = {"gghm2022"}
VOLATILITY = {"dy2012", "jcggh2022", "cgg2022", "ivol_crossasset", "ivol_usequity", "ivol_global"}
SETUPS = {"default": (10, [math.pi, math.pi / 2, 0.0]),
          "bk": (100, [math.pi + 1e-5, math.pi / 5, 0.0])}


def load(path, name):
    df = pd.read_csv(path)
    x = df.drop(columns=["date"]).apply(pd.to_numeric, errors="coerce")
    x = x.loc[:, x.notna().mean() > 0.9]                     # drop non-numeric columns (e.g. a date column)
    x = x.ffill().bfill()
    if name in LEVELS:
        x = 100 * np.log(x).diff().dropna()
    return x.to_numpy(float), list(x.columns)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="pkgdata/csv")
    ap.add_argument("--out", default="results/pkgdata")
    ap.add_argument("--kappa", type=float, default=0.99)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    hl = math.log(0.5) / math.log(args.kappa)                  # ~69 observations
    rows = []
    for f in sorted(os.listdir(args.data)):
        name = f[:-4]
        if name in EXCLUDE:
            continue
        x, cols = load(os.path.join(args.data, f), name)
        T, k = x.shape
        slow = pd.DataFrame(x).rolling(int(2 * hl), center=True, min_periods=1).mean()
        slow_share = float((slow.var() / pd.DataFrame(x).var()).mean())
        prior = max(50, T // 10)
        kind = "volatility" if name in VOLATILITY else ("prices->returns" if name in LEVELS else "returns/other")
        fits = {"package": tvp_var(x, 1, args.kappa, args.kappa, mode="package"),
                "ewma": tvp_var(x, 1, args.kappa, args.kappa, mode="realtime", prior_size=prior,
                                demean="ewma", demean_halflife=hl)}
        for setup, (H, part) in SETUPS.items():
            out = {}
            for c, fit in fits.items():
                r = connectedness(fit["B_t"], fit["Q_t"], H, partition=part, device="cpu")
                skip = prior                                    # same evaluation window for both
                out[c] = (float(r["time_TCI"][skip:].mean()), float(r["freq_TCI"][skip:, 0].mean()),
                          float(r["freq_TCI"][skip:, 1].mean()), float(r["freqtotal_TCI"][skip:].mean()))
            row = {"dataset": name, "type": kind, "k": k, "T": T, "setup": setup, "slow_share": slow_share,
                   "TCI_pkg": out["package"][0], "TCI_ewma": out["ewma"][0],
                   "short_pkg": out["package"][1], "short_ewma": out["ewma"][1],
                   "long_pkg": out["package"][2], "long_ewma": out["ewma"][2],
                   "long_share_pkg": out["package"][2] / out["package"][3],
                   "long_share_ewma": out["ewma"][2] / out["ewma"][3]}
            row["d_long"] = row["long_pkg"] - row["long_ewma"]
            row["rank_flip"] = (row["long_pkg"] > row["short_pkg"]) != (row["long_ewma"] > row["short_ewma"])
            rows.append(row)
            print(f"{name:<12} {kind:<16} k={k:2d} T={T:5d} {setup:<8} slow={slow_share:.2f}  "
                  f"TCI {row['TCI_pkg']:5.1f}/{row['TCI_ewma']:5.1f}  long {row['long_pkg']:5.1f}/{row['long_ewma']:5.1f}  "
                  f"dLong {row['d_long']:+5.1f}  flip={row['rank_flip']}", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(args.out, "summary.csv"), index=False)
    for setup in SETUPS:
        s = res[res.setup == setup]
        print(f"\n[{setup}] corr(slow share, dLong) = {s[['slow_share', 'd_long']].corr().iloc[0, 1]:.2f}; "
              f"datasets with |dLong| > 2: {int((s.d_long.abs() > 2).sum())}/{len(s)}; ranking flips: {int(s.rank_flip.sum())}")


if __name__ == "__main__":
    main()
