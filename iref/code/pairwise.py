"""Average pairwise connectedness matrices of the crypto networks, for network figures.

Estimators:
  standard     TVP-VAR(3), package centring, half-lives 15/7 days (the settings of Shi & Lin, 2026)
  recommended  TVP-HAR-VAR, causal EWMA centring, coefficient (and EWMA) half-life 60 days, volatility
               half-life 2 days; explosive dates replaced by the last stable estimate
Connectedness every 24th hour from 2021-03-01 to 2025-12-31, H = 720, the letter's bands (true cycles
< 16 h, 16-48 h, > 48 h) and the time domain. Output (long format): network, estimator, domain,
receiver, transmitter, value = time average of CT[receiver, transmitter] (percent). Pairwise net
connectedness from l to j is CT[j, l] - CT[l, j].

Usage (from iref/):  python code/pairwise.py --workers 8 --devices cuda:0
"""
import argparse
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "code"))
from predict_gap import NETWORKS  # noqa: E402

BANDS = ["short", "medium", "long"]


def job(a):
    from connectedness import connectedness, partition_from_periods, tvp_var
    from run_harvar import stable
    from tvp_harvar import tvp_harvar
    spec, root, device = a
    ds = pd.read_csv(os.path.join(root, "logrv_ds.csv.gz"), index_col=0, parse_dates=True)
    ds = ds[ds.index <= pd.Timestamp("2025-12-31 23:00", tz="UTC")]
    cols = NETWORKS[spec]
    x = ds[cols].interpolate().ffill().bfill().to_numpy()
    T, k = x.shape
    start = int(np.searchsorted(ds.index, pd.Timestamp("2021-03-01", tz="UTC")))
    sel = np.arange(start, T, 24)
    part = partition_from_periods([8, 24], package_convention=True)
    f = tvp_var(x, 3, 0.5 ** (1 / 360), 0.5 ** (1 / 168), mode="package")
    fits = {"standard": (f["B_t"][sel], f["Q_t"][sel])}
    h = tvp_harvar(x, 0.5 ** (1 / 1440), 0.5 ** (1 / 48), 720, 1440, keep_every=24, keep_from=start, demean="ewma")
    ok = np.array([stable(b, k) for b in h["B"]])
    idx = np.maximum.accumulate(np.where(ok, np.arange(len(ok)), -1))
    keep = idx >= 0
    fits["recommended"] = (h["B"][idx[keep]], h["Q"][keep])
    rows = []
    for est, (B, Q) in fits.items():
        r = connectedness(B, Q, 720, partition=part, device=device, chunk=128)
        mats = {"time": r["time_CT"].mean(0)}
        for b, nm in enumerate(BANDS):
            mats[nm] = r["freq_CT"][:, b].mean(0)
        for dom, M in mats.items():
            for j in range(k):
                for l in range(k):
                    rows.append({"network": spec, "estimator": est, "domain": dom, "receiver": cols[j],
                                 "transmitter": cols[l], "value": float(M[j, l])})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="../work/data/processed")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--devices", nargs="+", default=["cpu"])
    ap.add_argument("--out", default="results/pairwise.csv")
    args = ap.parse_args()
    root = os.path.abspath(args.root)
    jobs = [(s, root, args.devices[i % len(args.devices)]) for i, s in enumerate(NETWORKS)]
    rows = []
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn")) as ex:
        for r in ex.map(job, jobs, chunksize=1):
            rows += r
            print(f"  done {r[0]['network']}", flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
