"""How fast should the TVP-VAR forget? Choose coefficient and volatility half-lives by the
out-of-sample predictive log score, on hourly and on daily data.

Hourly: the causal panel (log RV, hour-of-day means from 2021-2023), 3 lags.
Daily:  log of the average hourly RV per UTC day (from the causal raw panel), 3 lags.
Centring: causal EWMA with half-life equal to the coefficient half-life (matched); prior from
the first 30 days. Half-lives are in days for both frequencies, so the two selections are
comparable in calendar time.
Selection period: 2022-01-01 to 2024-12-31; evaluation period: 2025-01-01 to 2026-09-30.
Scores are mean log predictive density per observation (and per series).

Usage (from iref/):  python code/select_halflife.py --workers 24
"""
import argparse
import itertools
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from predict_gap import NETWORKS  # noqa: E402

H_COEF = [1, 3, 7, 15, 30, 60, 120, 365]
H_VOL = [1, 3, 7, 15, 30]
SEL = ("2022-01-01", "2024-12-31 23:00")
EVAL = ("2025-01-01", "2026-09-30 23:00")


def load(freq, root):
    if freq == "hourly":
        p = pd.read_csv(os.path.join(root, "logrv_ds_causal.csv.gz"), index_col=0, parse_dates=True)
        return p.ffill().bfill(), 24
    r = pd.read_csv(os.path.join(root, "logrv_raw_causal.csv.gz"), index_col=0, parse_dates=True)
    d = np.log(np.exp(r).resample("1D").mean())
    return d.ffill().bfill(), 1


def job(args):
    freq, spec, cols, h1, h2, root = args
    from tvp_predictive import tvp_var_logscore
    p, per_day = load(freq, root)
    p = p[p.index <= pd.Timestamp(EVAL[1], tz="UTC")]
    x = p[cols].to_numpy()
    k1, k2 = 0.5 ** (1 / (h1 * per_day)), 0.5 ** (1 / (h2 * per_day))
    s = tvp_var_logscore(x, 3, k1, k2, prior_size=30 * per_day, demean="ewma", demean_halflife=h1 * per_day)
    s = pd.Series(s, index=p.index)
    out = {"freq": freq, "network": spec, "h_coef": h1, "h_vol": h2}
    for name, (a, b) in (("sel", SEL), ("eval", EVAL)):
        v = s[(s.index >= pd.Timestamp(a, tz="UTC")) & (s.index <= pd.Timestamp(b, tz="UTC"))]
        out[f"score_{name}"] = float(v.mean()) / len(cols)
        out[f"nan_{name}"] = int(v.isna().sum())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="../work/data/processed")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--freqs", nargs="+", default=["daily", "hourly"])
    ap.add_argument("--h-coef", nargs="+", type=float, default=H_COEF)
    ap.add_argument("--h-vol", nargs="+", type=float, default=H_VOL)
    ap.add_argument("--out", default="results/select_halflife.csv")
    args = ap.parse_args()
    root = os.path.abspath(args.root)
    jobs = [(f, spec, cols, h1, h2, root) for f in args.freqs for spec, cols in NETWORKS.items()
            for h1, h2 in itertools.product(args.h_coef, args.h_vol)]
    rows = []
    with ProcessPoolExecutor(args.workers) as ex:
        for i, r in enumerate(ex.map(job, jobs, chunksize=1)):
            rows.append(r)
            if (i + 1) % 20 == 0:
                print(f"  {i + 1}/{len(jobs)}", flush=True)
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    for f, g in df.groupby("freq"):
        best = g.loc[g.groupby("network").score_sel.idxmax()]
        print(f"\n[{f}] best half-lives (days) by selection-period log score:")
        print(best[["network", "h_coef", "h_vol", "score_sel", "score_eval"]].to_string(index=False))
        m = g.groupby(["h_coef", "h_vol"]).score_sel.mean().unstack()
        print(f"[{f}] mean selection score over networks (rows h_coef, cols h_vol):\n{m.round(3).to_string()}")


if __name__ == "__main__":
    main()
