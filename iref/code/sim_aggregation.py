"""Simulation: does the estimator itself, applied to hourly data and to their daily aggregate,
reproduce the hourly-versus-daily difference in frequency profiles seen in the data?

Worlds (from one real network, as in the letter's sim_centering.py):
  harvar: HAR-type VAR(168) fitted to the raw hourly network (slow dynamics are genuine);
  drift : VAR(3) fitted to de-levelled data plus the observed 30-day level.
Each seed simulates T hourly observations y (log RV scale). Daily series: log of the average of
exp(y) over each 24-hour block, as in the real daily panel. Both are estimated with the TVP-VAR(3)
at the rates used in hourly_vs_daily.py, package and causal EWMA centring, H = 30 days, bands
< 2 d, 2-7 d, 7-30 d, > 30 d.

Usage (from iref/):  python code/sim_aggregation.py --networks exchange:BN instrument:BTCS --seeds 3 --workers 12
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "code"))
from predict_gap import NETWORKS  # noqa: E402

SETTINGS = {"hourly": [(15, 2)], "daily": [(15, 7), (120, 60)]}


def job(a):
    from connectedness import connectedness, tvp_var
    from hourly_vs_daily import partition
    from sim_centering import fit_dgp, simulate
    spec, world, seed, root = a
    ds = pd.read_csv(os.path.join(root, "logrv_ds_causal.csv.gz"), index_col=0, parse_dates=True)
    ds = ds[ds.index <= pd.Timestamp("2025-12-31 23:00", tz="UTC")].ffill().bfill()
    y0 = ds[NETWORKS[spec]].to_numpy()
    L, Phi, Sigma = fit_dgp(y0, 3, 720, world)
    T, k = y0.shape
    y = y0.mean(0) + simulate(Phi, Sigma, T, seed) + (L if world == "drift" else 0)
    nd = T // 24
    yd = np.log(np.exp(y[:nd * 24]).reshape(nd, 24, k).mean(1))
    out = []
    for freq, data, per_day in (("hourly", y, 24), ("daily", yd, 1)):
        H = 30 * per_day
        part = partition(per_day, H)
        names = (["<2d"] if per_day > 1 else []) + ["2-7d", "7-30d", ">30d"]
        skip = 60 * per_day
        for h1, h2 in SETTINGS[freq]:
            k1, k2 = 0.5 ** (1 / (h1 * per_day)), 0.5 ** (1 / (h2 * per_day))
            for cent in ("package", "ewma"):
                if cent == "package":
                    fit = tvp_var(data, 3, k1, k2, mode="package")
                else:
                    fit = tvp_var(data, 3, k1, k2, mode="realtime", prior_size=30 * per_day,
                                  demean="ewma", demean_halflife=h1 * per_day)
                r = connectedness(fit["B_t"], fit["Q_t"], H, partition=part, device="cpu")
                row = {"network": spec, "world": world, "seed": seed, "freq": freq, "h_coef": h1, "h_vol": h2,
                       "centring": cent, "TCI": float(r["freqtotal_TCI"][skip:].mean())}
                for b, nm in enumerate(names):
                    row[nm] = float(r["freq_TCI"][skip:, b].mean())
                out.append(row)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="../work/data/processed")
    ap.add_argument("--networks", nargs="+", default=["exchange:BN", "instrument:BTCS"])
    ap.add_argument("--worlds", nargs="+", default=["harvar", "drift"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="results/sim_aggregation.csv")
    args = ap.parse_args()
    root = os.path.abspath(args.root)
    jobs = [(s, w, seed, root) for s in args.networks for w in args.worlds for seed in range(args.seeds)]
    rows = []
    with ProcessPoolExecutor(args.workers) as ex:
        for r in ex.map(job, jobs, chunksize=1):
            rows += r
            print(f"  done {len(rows) // 6}/{len(jobs)}", flush=True)
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    cols = ["TCI", "<2d", "2-7d", "7-30d", ">30d"]
    print(df.groupby(["world", "freq", "h_coef", "h_vol", "centring"])[cols].mean().round(1).to_string())


if __name__ == "__main__":
    main()
