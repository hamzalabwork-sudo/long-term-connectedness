"""Can the data tell exogenous level drift from slow genuine spillovers?

Idea: with genuine (slow) spillovers, today's shocks feed into the slow component later on;
with exogenous level drift, the slow component moves independently of past shocks.

Procedure on hourly data y (T x k):
  1. Fit a short VAR(3) with intercept by OLS (full sample) and take its residuals e_t, the
     hourly shocks.
  2. Aggregate to non-overlapping weeks: level L_w = mean of y in week w, shock S_w = mean of e in
     week w (both averaged across series as well, giving one common series each).
  3. Regress the change in level from week w to weeks w+2..w+5 on S_w, controlling for L_w.
     Statistic = t-value of the coefficient on S_w (Newey-West). Week w+1 is skipped so that the
     fast reversal of the shock itself does not enter.
Under exogenous drift the coefficient should be about zero; under slow spillovers positive.

Runs the statistic on simulated drift worlds, slow-spillover worlds and fast-VAR worlds (known
truth), then on real data.

Usage (from work/):  python ../code/drift_diagnostic.py --panel data/processed/logrv_ds.csv.gz
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim_centering import fit_dgp, random_level, simulate  # noqa: E402
from staleness import nw_ols  # noqa: E402

WEEK = 168


def var_residuals(y, p=3):
    T, k = y.shape
    W = np.hstack([np.ones((T - p, 1))] + [y[p - j - 1:T - j - 1] for j in range(p)])
    C = np.linalg.lstsq(W, y[p:], rcond=None)[0]
    e = np.full_like(y, np.nan)
    e[p:] = y[p:] - W @ C
    return e


def statistic(y, skip=1, ahead=4):
    e = var_residuals(y)
    nw = len(y) // WEEK
    L = np.nanmean(y[:nw * WEEK].reshape(nw, WEEK, -1), axis=(1, 2))
    S = np.nanmean(e[:nw * WEEK].reshape(nw, WEEK, -1), axis=(1, 2))
    fut = np.array([L[w + 1 + skip:w + 1 + skip + ahead].mean() - L[w] if w + 1 + skip + ahead <= nw else np.nan
                    for w in range(nw)])
    ok = np.isfinite(fut) & np.isfinite(S)
    X = np.column_stack([S[ok], L[ok] - L[ok].mean()])
    yv = fut[ok]
    # Newey-West t on S, partialling out L
    resid_S = X[:, 0] - X[:, 1] * (X[:, 1] @ X[:, 0]) / (X[:, 1] @ X[:, 1])
    resid_y = yv - X[:, 1] * (X[:, 1] @ yv) / (X[:, 1] @ X[:, 1])
    b, t = nw_ols(resid_y, resid_S, lags=ahead + skip)
    return b, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="data/processed/logrv_ds.csv.gz")
    ap.add_argument("--cols", nargs="+", default=["BTCS_BN", "BTCP_BN", "ETHS_BN", "ETHP_BN"])
    ap.add_argument("--end", default="2025-12-31 23:00")
    ap.add_argument("--seeds", type=int, default=20)
    args = ap.parse_args()
    p = pd.read_csv(args.panel, index_col=0, parse_dates=True)
    p = p[p.index <= pd.Timestamp(args.end, tz="UTC")][args.cols].interpolate().ffill().bfill()
    y = p.to_numpy()
    T, k = y.shape

    L30, Phi_d, Sig_d = fit_dgp(y, 3, 720, "drift")
    _, Phi_h, Sig_h = fit_dgp(y, 3, 720, "harvar")
    _, Phi_f, Sig_f = fit_dgp(y, 3, 720, "persistent")
    sd = float(pd.DataFrame(y).rolling(720, center=True, min_periods=1).mean().std().mean())
    worlds = {
        "drift, observed level": lambda s: y.mean(0) + L30 + simulate(Phi_d, Sig_d, T, s),
        "drift, random level": lambda s: y.mean(0) + random_level(T, k, 720, sd, s) + simulate(Phi_d, Sig_d, T, s),
        "slow spillovers (HAR-VAR)": lambda s: y.mean(0) + simulate(Phi_h, Sig_h, T, s),
        "fast VAR, no drift": lambda s: y.mean(0) + simulate(Phi_f, Sig_f, T, s),
    }
    rows = []
    for name, gen in worlds.items():
        ts = [statistic(gen(s))[1] for s in range(args.seeds)]
        rows.append({"world": name, "mean_t": np.mean(ts), "sd_t": np.std(ts),
                     "share_t>1.96": np.mean(np.array(ts) > 1.96)})
        print(f"{name:<28} t: mean {np.mean(ts):+.2f}  sd {np.std(ts):.2f}  share>1.96 {np.mean(np.array(ts) > 1.96):.2f}", flush=True)
    b, t = statistic(y)
    print(f"{'REAL DATA (Binance network)':<28} b {b:+.3f}  t {t:+.2f}")


if __name__ == "__main__":
    main()
