"""Residualised panel in the spirit of Shi & Lin's (2026) pre-estimation robustness check:
each series is regressed (full sample, OLS with intercept) on the cross-sectional mean log
realised variance of its own asset (BTC or ETH, all venues and instruments), and the residual
is used in place of the series. This removes the common volatility level, including its drift.

Usage (from the work directory):
    python ../code/make_resid_panel.py --panel data/processed/logrv_ds.csv.gz \
        --out data/processed/logrv_ds_resid.csv.gz
"""
import argparse

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="data/processed/logrv_ds.csv.gz")
    ap.add_argument("--out", default="data/processed/logrv_ds_resid.csv.gz")
    args = ap.parse_args()
    p = pd.read_csv(args.panel, index_col=0, parse_dates=True)
    out = pd.DataFrame(index=p.index)
    for asset in ("BTC", "ETH"):
        cols = [c for c in p.columns if c.startswith(asset)]
        m = p[cols].mean(axis=1)
        for c in cols:
            ok = p[c].notna() & m.notna()
            X = np.column_stack([np.ones(ok.sum()), m[ok]])
            b = np.linalg.lstsq(X, p.loc[ok, c], rcond=None)[0]
            out[c] = p[c] - (b[0] + b[1] * m)
            print(f"{c}: slope on {asset} mean {b[1]:.3f}, residual sd {out[c].std():.3f} (raw {p[c].std():.3f})")
    out[p.columns].to_csv(args.out)


if __name__ == "__main__":
    main()
