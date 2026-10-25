"""Mechanism checks requested in the internal review.

1. Inference that does not overstate precision: the eight networks share series, so the
   long-band share and the squared level offset are averaged across networks first, giving one
   monthly series (60 months). The share is regressed on delta and delta^2 with Newey-West (3)
   errors. Both signs of the offset should raise the share (high-volatility 2021 and
   low-volatility 2023), i.e. delta^2 matters and delta does not, which a
   "turbulence raises long-run spillovers" story would not predict.
2. Multivariate pseudo-true calculation. A zero-intercept VAR(1) fitted to y_t = delta + x_t,
   where x_t is a VAR(1) with autocovariances Gamma_0, Gamma_1, converges to
       A* = M1 M0^{-1},  Sigma* = M0 - A* M1',  M0 = Gamma_0 + dd',  M1 = Gamma_1 + dd'.
   The connectedness of (A*, Sigma*) is computed with the same code for a common offset
   d = k * sd(x) * 1, k = 0, 0.5, 1, 2, from a VAR(1) fitted to the de-levelled Binance network.

Usage (from the work directory):  python ../code/mechanism_checks.py
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.linalg import solve_discrete_lyapunov

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from connectedness import connectedness, partition_from_periods  # noqa: E402
from staleness import nw_ols  # noqa: E402


def avg_long_share(root, run, end):
    meta = json.load(open(os.path.join(root, run, "meta.json")))
    lb = meta["band_names"][-1]
    s = []
    for spec in meta["networks"]:
        d = pd.read_csv(os.path.join(root, run, spec.replace(":", "_") + "_dynamic.csv.gz"),
                        index_col=0, parse_dates=True)
        s.append((d[f"TCI_{lb}"] / d["TCI_ftotal"]).resample("MS").mean())
    out = pd.concat(s, axis=1).mean(axis=1)
    return out[out.index <= pd.Timestamp(end, tz="UTC")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conn", default="results/conn")
    ap.add_argument("--panel", default="data/processed/logrv_ds.csv.gz")
    ap.add_argument("--end", default="2025-12-31 23:00")
    args = ap.parse_args()

    p = pd.read_csv(args.panel, index_col=0, parse_dates=True)
    p = p[p.index <= pd.Timestamp(args.end, tz="UTC")]
    lvl = p.mean(axis=1)
    delta = (lvl - lvl.mean()).resample("MS").mean()
    print("1. Network-averaged monthly long-band share on delta and delta^2 (NW 3 lags, 60 months)")
    for run in ("shilin_package", "shilin_ewma15d", "rob_true_pkg", "rob_true_ewma"):
        if not os.path.exists(os.path.join(args.conn, run, "meta.json")):
            continue
        y = avg_long_share(args.conn, run, args.end)
        df = pd.concat([y.rename("y"), delta.rename("d")], axis=1).dropna()
        X = np.column_stack([np.ones(len(df)), df["d"], df["d"] ** 2])
        beta = np.linalg.lstsq(X, df["y"].to_numpy(), rcond=None)[0]
        u = df["y"].to_numpy() - X @ beta
        S = (X * u[:, None]).T @ (X * u[:, None])
        for l in range(1, 4):
            G = (X[l:] * u[l:, None]).T @ (X[:-l] * u[:-l, None])
            S += (1 - l / 4) * (G + G.T)
        V = np.linalg.inv(X.T @ X) @ S @ np.linalg.inv(X.T @ X)
        t = beta / np.sqrt(np.diag(V))
        r2 = 1 - u.var() / df["y"].var()
        b2, t2 = nw_ols(df["y"].to_numpy(), (df["d"] ** 2).to_numpy(), lags=3)
        print(f"   {run:<16} delta: {beta[1]:+.3f} (t {t[1]:+.2f})  delta^2: {beta[2]:+.3f} (t {t[2]:+.2f})  "
              f"R2 {r2:.2f} | delta^2 alone: t {t2:+.2f}")

    print("\n2. Pseudo-true zero-intercept VAR(1) under a common offset (Binance network)")
    cols = ["BTCS_BN", "BTCP_BN", "ETHS_BN", "ETHP_BN"]
    y = p[cols].interpolate(limit_area="inside").ffill().bfill()
    x = (y - y.rolling(720, center=True, min_periods=1).mean()).to_numpy()
    X, Y = x[:-1], x[1:]
    A = np.linalg.lstsq(X, Y, rcond=None)[0].T
    U = Y - X @ A.T
    Sig = U.T @ U / len(U)
    G0 = solve_discrete_lyapunov(A, Sig)
    G1 = A @ G0
    sd = np.sqrt(np.diag(G0)).mean()
    for part_name, part in (("package labels", partition_from_periods([8, 24], True)),
                            ("true periods", partition_from_periods([8, 24], False))):
        print(f"   bands: {part_name}")
        for k in (0.0, 0.5, 1.0, 2.0):
            d = np.full(4, k * sd)
            M0, M1 = G0 + np.outer(d, d), G1 + np.outer(d, d)
            As = M1 @ np.linalg.inv(M0)
            Ss = M0 - As @ M1.T
            r = connectedness(As[None], Ss[None], 720, partition=part, device="cpu")
            print(f"      offset {k:3.1f} sd: max|eig| {np.abs(np.linalg.eigvals(As)).max():.3f}  "
                  f"TCI {r['time_TCI'][0]:5.1f}  short {r['freq_TCI'][0, 0]:5.1f}  mid {r['freq_TCI'][0, 1]:5.1f}  "
                  f"long {r['freq_TCI'][0, 2]:5.1f}")


if __name__ == "__main__":
    main()
