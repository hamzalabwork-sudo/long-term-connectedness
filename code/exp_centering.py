"""Does the band split of TVP-VAR frequency connectedness depend on how the data are centred?

Separates the two full-sample steps of ConnectednessApproach (centring and prior) on one
network, and tries causal EWMA centring as a real-time alternative.

Usage:
    python exp_centering.py --panel ../data/processed/logrv_ds_bn.csv.gz \
        --cols BTCS_BN BTCP_BN ETHS_BN ETHP_BN
"""
import argparse

import pandas as pd

from connectedness import bayes_prior, connectedness, partition_from_periods, tvp_var


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", required=True)
    ap.add_argument("--cols", nargs="+", required=True)
    ap.add_argument("--end", default="2025-12-31 23:00")
    ap.add_argument("--prior-hours", type=int, default=720)
    args = ap.parse_args()

    p = pd.read_csv(args.panel, index_col=0, parse_dates=True)
    p = p[p.index <= pd.Timestamp(args.end, tz="UTC")][args.cols].interpolate().ffill().bfill()
    x = p.to_numpy()
    k1, k2 = 0.5 ** (1 / 360), 0.5 ** (1 / 168)
    part = partition_from_periods([8, 24], package_convention=True)
    full_prior, win_prior = bayes_prior(x, 3), bayes_prior(x, 3, args.prior_hours)
    w = args.prior_hours
    cfgs = {
        "a full-mean, full-prior (package)": dict(prior=full_prior, demean="full"),
        "b full-mean, window-prior": dict(prior=win_prior, demean="full"),
        "c window-mean, full-prior": dict(prior=full_prior, demean="window", prior_size=w),
        "d window-mean, window-prior": dict(prior=win_prior, demean="window", prior_size=w),
        "e EWMA-mean hl=15d, window-prior": dict(prior=win_prior, demean="ewma", prior_size=w,
                                                 demean_halflife=360),
        "f EWMA-mean hl=7d, window-prior": dict(prior=win_prior, demean="ewma", prior_size=w,
                                                demean_halflife=168),
    }
    years = p.index.year
    print(f"{'config':<36}{'TCI':>7}{'0-8':>7}{'8-24':>7}{'24+':>7}   long-band share of total, by year")
    for name, c in cfgs.items():
        fit = tvp_var(x, 3, k1, k2, **c)
        r = connectedness(fit["B_t"], fit["Q_t"], 720, partition=part)
        b, tot = r["freq_TCI"], r["freqtotal_TCI"]
        share = pd.Series(b[:, 2] / tot).groupby(years).mean().round(2).tolist()
        print(f"{name:<36}{r['time_TCI'].mean():7.1f}{b[:, 0].mean():7.1f}{b[:, 1].mean():7.1f}"
              f"{b[:, 2].mean():7.1f}   {share}", flush=True)
    print("years:", sorted(set(years)))
    print("mean log RV by year:", {c: p[c].groupby(years).mean().round(2).tolist() for c in args.cols[:1]})


if __name__ == "__main__":
    main()
