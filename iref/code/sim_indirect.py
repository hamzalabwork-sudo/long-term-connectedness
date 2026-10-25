"""Indirect inference for the long band: which mix of level drift and genuine slow dynamics
reproduces what the different estimators report on the real data?

World family (Binance network), indexed by c in [0, 1]:
    y_t = ybar + c L_t + x_t^(c),   x^(c) = HAR-type VAR(168) fitted to y - c L,
where L is the observed 30-day level. c = 0: all slow movement is genuine dynamics (the HAR world);
c = 1: the observed level is an exogenous drift and the dynamics are fitted to de-levelled data.
The true connectedness of world c is that of x^(c).

For each world and seed, the estimators applied to the real data are run on the simulated data:
TVP-VAR(3) and TVP-HAR-VAR, package and causal EWMA centring, half-life 15 days; TVP-HAR-VAR at
60 days (both centrings). Volatility half-life 2 days; connectedness every 24th hour after 60 days,
H = 720, the letter's bands; explosive HAR dates carried forward.

Matching (match_indirect.py): choose c so the simulated estimator outputs are closest to the real
ones; the implied long band is the truth of that world.

Usage (from iref/):  python code/sim_indirect.py --workers 15 --devices cuda:0
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
K2 = 0.5 ** (1 / 48)
VARIANTS = [("var3", "package", 15), ("var3", "ewma", 15), ("harvar", "package", 15), ("harvar", "ewma", 15),
            ("harvar", "package", 60), ("harvar", "ewma", 60)]


def world(c, root, cols):
    from sim_centering import fit_dgp
    ds = pd.read_csv(os.path.join(root, "logrv_ds.csv.gz"), index_col=0, parse_dates=True)
    y0 = ds[ds.index <= pd.Timestamp("2025-12-31 23:00", tz="UTC")][cols].interpolate().ffill().bfill().to_numpy()
    L, _, _ = fit_dgp(y0, 3, 720, "drift")                    # observed 30-day level (demeaned)
    _, Phi, Sigma = fit_dgp(y0 - c * L, 3, 720, "harvar")
    return y0, L, Phi, Sigma


def estimate(y, est, cent, hl, start, device):
    from connectedness import connectedness, partition_from_periods, tvp_var
    from run_harvar import stable
    from tvp_harvar import tvp_harvar
    T, k = y.shape
    K1 = 0.5 ** (1 / (hl * 24))
    part = partition_from_periods([8, 24], package_convention=True)
    if est == "var3":
        f = tvp_var(y, 3, K1, K2, mode="package") if cent == "package" else \
            tvp_var(y, 3, K1, K2, mode="realtime", prior_size=720, demean="ewma", demean_halflife=hl * 24)
        sel = np.arange(start, T, 24)
        B, Q, nu = f["B_t"][sel], f["Q_t"][sel], 0
    else:
        f = tvp_harvar(y, K1, K2, 720, hl * 24, keep_every=24, keep_from=start,
                       demean="full" if cent == "package" else "ewma")
        ok = np.array([stable(b, k) for b in f["B"]])
        idx = np.maximum.accumulate(np.where(ok, np.arange(len(ok)), -1))
        keep = idx >= 0
        B, Q, nu = f["B"][idx[keep]], f["Q"][keep], int((~ok).sum())
    r = connectedness(B, Q, 720, partition=part, device=device, chunk=128)
    return {"TCI": float(r["freqtotal_TCI"].mean()), "short": float(r["freq_TCI"][:, 0].mean()),
            "mid": float(r["freq_TCI"][:, 1].mean()), "long": float(r["freq_TCI"][:, 2].mean()), "n_unstable": nu}


def job(a):
    from sim_centering import simulate
    c, seed, root, device, cols = a
    y0, L, Phi, Sigma = world(c, root, cols)
    T, _ = y0.shape
    y = y0.mean(0) + c * L + simulate(Phi, Sigma, T, seed)
    out = []
    for est, cent, hl in VARIANTS:
        out.append({"c": c, "seed": seed, "estimator": est, "centring": cent, "hl_coef": hl,
                    **estimate(y, est, cent, hl, 60 * 24, device)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="../work/data/processed")
    ap.add_argument("--c", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--workers", type=int, default=15)
    ap.add_argument("--devices", nargs="+", default=["cpu"])
    ap.add_argument("--network", default="exchange:BN")
    ap.add_argument("--out", default="results/sim_indirect_exchange_BN")
    args = ap.parse_args()
    root = os.path.abspath(args.root)
    os.makedirs(args.out, exist_ok=True)
    from connectedness import connectedness, partition_from_periods
    part = partition_from_periods([8, 24], package_convention=True)
    truth = []
    for c in args.c:
        _, _, Phi, Sigma = world(c, root, NETWORKS[args.network])
        r = connectedness(Phi[None], Sigma[None], 720, partition=part, device="cpu")
        truth.append({"c": c, "TCI": float(r["freqtotal_TCI"][0]), "short": float(r["freq_TCI"][0, 0]),
                      "mid": float(r["freq_TCI"][0, 1]), "long": float(r["freq_TCI"][0, 2])})
        print(f"truth c={c:.2f}: " + "  ".join(f"{k} {v:5.1f}" for k, v in truth[-1].items() if k != "c"), flush=True)
    pd.DataFrame(truth).to_csv(os.path.join(args.out, "truth.csv"), index=False)
    combos = [(c, s) for c in args.c for s in range(args.seeds)]
    jobs = [(c, s, root, args.devices[i % len(args.devices)], NETWORKS[args.network]) for i, (c, s) in enumerate(combos)]
    rows = []
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn")) as ex:
        for r in ex.map(job, jobs, chunksize=1):
            rows += r
            pd.DataFrame(rows).to_csv(os.path.join(args.out, "estimates.csv"), index=False)   # save as we go
            print(f"  done {len(rows) // len(VARIANTS)}/{len(jobs)}", flush=True)
    df = pd.DataFrame(rows)
    print(df.groupby(["c", "estimator", "centring", "hl_coef"])[["long", "short"]].mean().round(1).unstack(0).to_string())


if __name__ == "__main__":
    main()
