"""Long-band connectedness of the crypto networks under the estimators compared in the Monte Carlo:
TVP-VAR(3) and TVP-HAR-VAR, package (full-sample) and causal EWMA centring, coefficient (and EWMA)
half-lives of 15 and 60 days; volatility half-life 2 days; prior from the first 30 days (EWMA) or
the full sample (package TVP-VAR(3), as in ConnectednessApproach).

Hourly panel: data/processed/logrv_ds.csv.gz (the letter's connectedness panel), 2021-2025.
Connectedness every 24th hour from 2021-03-01, H = 720; explosive HAR dates are replaced by the last
stable estimate (and counted). Bands: the letter's (true cycles < 16 h, 16-48 h, > 48 h) and
calendar (< 2 d, 2-7 d, 7-30 d, > 30 d).

Usage (from iref/):  python code/real_estimates.py --workers 8 --devices cuda:0
"""
import argparse
import math
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

K2 = 0.5 ** (1 / 48)


def job(a):
    from connectedness import connectedness, partition_from_periods, tvp_var
    from run_harvar import stable
    from tvp_harvar import tvp_harvar
    spec, hl, root, device = a
    ds = pd.read_csv(os.path.join(root, "logrv_ds.csv.gz"), index_col=0, parse_dates=True)
    ds = ds[ds.index <= pd.Timestamp("2025-12-31 23:00", tz="UTC")]
    x = ds[NETWORKS[spec]].interpolate().ffill().bfill().to_numpy()
    T, k = x.shape
    start = int(np.searchsorted(ds.index, pd.Timestamp("2021-03-01", tz="UTC")))
    sel = np.arange(start, T, 24)
    K1 = 0.5 ** (1 / (hl * 24))
    parts = {"letter": (partition_from_periods([8, 24], package_convention=True), ["s", "m", "l"]),
             "calendar": ([math.pi + 1e-5] + [2 * math.pi / (d * 24) for d in (2, 7, 30)] + [0.0],
                          ["<2d", "2-7d", "7-30d", ">30d"])}
    out = []
    for cent in ("package", "ewma"):
        if cent == "package":
            f3 = tvp_var(x, 3, K1, K2, mode="package")
        else:
            f3 = tvp_var(x, 3, K1, K2, mode="realtime", prior_size=720, demean="ewma", demean_halflife=hl * 24)
        fh = tvp_harvar(x, K1, K2, 720, hl * 24, keep_every=24, keep_from=start,
                        demean="full" if cent == "package" else "ewma")
        ok = np.array([stable(b, k) for b in fh["B"]])
        idx = np.maximum.accumulate(np.where(ok, np.arange(len(ok)), -1))
        keep = idx >= 0
        models = {"var3": (f3["B_t"][sel], f3["Q_t"][sel], 0), "harvar": (fh["B"][idx[keep]], fh["Q"][keep], int((~ok).sum()))}
        for est, (B, Q, nu) in models.items():
            row = {"network": spec, "hl_coef": hl, "estimator": est, "centring": cent, "n_unstable": nu}
            for pname, (part, names) in parts.items():
                r = connectedness(B, Q, 720, partition=part, device=device, chunk=128)
                row[f"{pname}_TCI"] = float(r["freqtotal_TCI"].mean())
                for b, nm in enumerate(names):
                    row[f"{pname}_{nm}"] = float(r["freq_TCI"][:, b].mean())
            out.append(row)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="../work/data/processed")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--devices", nargs="+", default=["cpu"])
    ap.add_argument("--hl-coef", nargs="+", type=float, default=[15, 60])
    ap.add_argument("--out", default="results/real_estimates.csv")
    args = ap.parse_args()
    root = os.path.abspath(args.root)
    combos = [(s, h) for h in args.hl_coef for s in NETWORKS]
    jobs = [(s, h, root, args.devices[i % len(args.devices)]) for i, (s, h) in enumerate(combos)]
    rows = []
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn")) as ex:
        for r in ex.map(job, jobs, chunksize=1):
            rows += r
            print(f"  done {len(rows) // 4}/{len(jobs)}", flush=True)
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    cols = ["letter_l", "calendar_<2d", "calendar_2-7d", "calendar_7-30d", "calendar_>30d", "letter_TCI", "n_unstable"]
    print(df.groupby(["hl_coef", "estimator", "centring"])[cols].mean().round(1).to_string())


if __name__ == "__main__":
    main()
