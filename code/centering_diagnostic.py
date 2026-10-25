"""Mechanism check: does the long-band share track the local distance from the centring constant?

A zero-intercept (TVP-)VAR fitted to y_t = mu_t + x_t after subtracting a constant m sees a
local offset delta_t = mu_t - m. For an AR(1) the pseudo-true slope is
    phi* = (delta^2 + rho * s2) / (delta^2 + s2),
which rises towards one with delta^2 / s2, so low-frequency spectral mass, and with it the
long band, grows with the squared local offset. Prediction: under package centring the
monthly long-band share rises with delta_t^2; under causal EWMA centring (delta_t ~ 0) it
does not.

For each network and month: delta^2 = squared distance of the network's mean log RV in that
month from the full-sample mean (2021-2025), pooled regression of the long share on delta^2
with network fixed effects (demeaned within network).

Usage:
    python centering_diagnostic.py --runs shilin_package shilin_ewma15d
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from staleness import nw_ols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/conn")
    ap.add_argument("--runs", nargs="+", default=["shilin_package", "shilin_ewma15d"])
    ap.add_argument("--panel", default="data/processed/logrv_ds.csv.gz")
    ap.add_argument("--end", default="2025-12-31 23:00")
    args = ap.parse_args()

    panel = pd.read_csv(args.panel, index_col=0, parse_dates=True)
    panel = panel[panel.index <= pd.Timestamp(args.end, tz="UTC")]
    for run in args.runs:
        meta = json.load(open(os.path.join(args.root, run, "meta.json")))
        long_band = meta["band_names"][-1]
        ys, xs, per_net = [], [], []
        for spec, info in meta["networks"].items():
            dyn = pd.read_csv(os.path.join(args.root, run, spec.replace(":", "_") + "_dynamic.csv.gz"),
                              index_col=0, parse_dates=True)
            share = (dyn[f"TCI_{long_band}"] / dyn["TCI_ftotal"]).resample("MS").mean()
            lvl = panel[info["cols"]].mean(axis=1)
            delta2 = ((lvl - lvl.mean()).resample("MS").mean()) ** 2
            df = pd.concat([share.rename("share"), delta2.rename("d2")], axis=1).dropna()
            per_net.append({"network": spec, "corr": df.corr().iloc[0, 1]})
            ys.append(df["share"].to_numpy() - df["share"].mean())
            xs.append(df["d2"].to_numpy() - df["d2"].mean())
        y, x = np.concatenate(ys), np.concatenate(xs)
        b, t = nw_ols(y, x, lags=3)
        r2 = np.corrcoef(y, x)[0, 1] ** 2
        corrs = pd.DataFrame(per_net)
        print(f"{run:<18} long-share on delta^2 (within network): slope {b:.3f}, NW t {t:.2f}, "
              f"R2 {r2:.2f}; per-network corr {corrs['corr'].min():.2f}..{corrs['corr'].max():.2f}")


if __name__ == "__main__":
    main()
