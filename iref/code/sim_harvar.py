"""Monte Carlo: does the TVP-HAR-VAR recover the long component where the TVP-VAR(3) cannot?

Worlds from the Binance network (as in the letter's sim_centering.py): drift (VAR(3) on de-levelled
data plus the observed 30-day level), harvar (HAR-type VAR(168), slow spillovers genuine), persistent
(VAR(3) on raw data, no level). Estimators: TVP-VAR(3) and TVP-HAR-VAR, each with package
(full-sample) and causal EWMA centring; half-lives 15 d (coefficients) and 2 d (volatility).
Connectedness every 24th hour after the first 60 days, H = 720, the letter's bands (true cycles
< 16 h, 16-48 h, > 48 h); explosive dates dropped and counted. Truth: the DGP's connectedness.

Usage (from iref/):  python code/sim_harvar.py --seeds 3 --workers 9
"""
import argparse
import os
import sys
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "code"))
COLS = ["BTCS_BN", "BTCP_BN", "ETHS_BN", "ETHP_BN"]
K2 = 0.5 ** (1 / 48)


def dgp(world, root):
    from sim_centering import fit_dgp
    ds = pd.read_csv(os.path.join(root, "logrv_ds.csv.gz"), index_col=0, parse_dates=True)
    y0 = ds[ds.index <= pd.Timestamp("2025-12-31 23:00", tz="UTC")][COLS].interpolate().ffill().bfill().to_numpy()
    L, Phi, Sigma = fit_dgp(y0, 3, 720, world)
    return y0, L, Phi, Sigma


def job(a):
    from connectedness import connectedness, partition_from_periods, tvp_var
    from run_harvar import stable
    from sim_centering import simulate
    from tvp_harvar import tvp_harvar
    world, seed, root, device, hl, unstable = a
    K1 = 0.5 ** (1 / (hl * 24))
    y0, L, Phi, Sigma = dgp(world, root)
    T, k = y0.shape
    y = y0.mean(0) + simulate(Phi, Sigma, T, seed) + (L if world == "drift" else 0)
    part = partition_from_periods([8, 24], package_convention=True)
    start = 60 * 24
    out = []
    for cent in ("package", "ewma"):
        if cent == "package":
            f3 = tvp_var(y, 3, K1, K2, mode="package")
        else:
            f3 = tvp_var(y, 3, K1, K2, mode="realtime", prior_size=720, demean="ewma", demean_halflife=hl * 24)
        sel = np.arange(start, T, 24)
        r3 = connectedness(f3["B_t"][sel], f3["Q_t"][sel], 720, partition=part, device=device)
        fh = tvp_harvar(y, K1, K2, 720, hl * 24, keep_every=24, keep_from=start, demean="full" if cent == "package" else "ewma")
        ok = np.array([stable(b, k) for b in fh["B"]])
        Bh, Qh = fh["B"], fh["Q"]
        if unstable == "carry":                     # replace explosive dates by the last stable estimate
            idx = np.maximum.accumulate(np.where(ok, np.arange(len(ok)), -1))
            keep = idx >= 0
            Bh, Qh = Bh[idx[keep]], Qh[keep]
        else:
            Bh, Qh = Bh[ok], Qh[ok]
        rh = connectedness(Bh, Qh, 720, partition=part, device=device, chunk=128)
        for est, r, nu in (("var3", r3, 0), ("harvar", rh, int((~ok).sum()))):
            out.append({"world": world, "seed": seed, "hl_coef": hl, "unstable": unstable, "estimator": est, "centring": cent, "n_unstable": nu,
                        "TCI": float(r["freqtotal_TCI"].mean()), "short": float(r["freq_TCI"][:, 0].mean()),
                        "mid": float(r["freq_TCI"][:, 1].mean()), "long": float(r["freq_TCI"][:, 2].mean())})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="../work/data/processed")
    ap.add_argument("--worlds", nargs="+", default=["drift", "harvar", "persistent"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--workers", type=int, default=9)
    ap.add_argument("--devices", nargs="+", default=["cpu"], help="assigned to jobs in turn")
    ap.add_argument("--hl-coef", nargs="+", type=float, default=[15], help="coefficient (and EWMA) half-lives, days")
    ap.add_argument("--unstable", choices=["drop", "carry"], default="drop")
    ap.add_argument("--out", default="results/sim_harvar.csv")
    args = ap.parse_args()
    root = os.path.abspath(args.root)
    from connectedness import connectedness, partition_from_periods
    part = partition_from_periods([8, 24], package_convention=True)
    truth = {}
    for w in args.worlds:
        _, _, Phi, Sigma = dgp(w, root)
        r = connectedness(Phi[None], Sigma[None], 720, partition=part, device="cpu")
        truth[w] = (float(r["freqtotal_TCI"][0]), *[float(v) for v in r["freq_TCI"][0]])
        print(f"truth {w:<11} TCI {truth[w][0]:5.1f}  short {truth[w][1]:5.1f} mid {truth[w][2]:5.1f} long {truth[w][3]:5.1f}", flush=True)
    combos = [(w, s, h) for w in args.worlds for h in args.hl_coef for s in range(args.seeds)]
    jobs = [(w, s, root, args.devices[i % len(args.devices)], h, args.unstable) for i, (w, s, h) in enumerate(combos)]
    rows = []
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn")) as ex:   # torch is loaded: no fork
        for r in ex.map(job, jobs, chunksize=1):
            rows += r
            print(f"  done {len(rows) // 4}/{len(jobs)}", flush=True)
    df = pd.DataFrame(rows)
    for w in args.worlds:
        m = df.world == w
        for i, c in enumerate(["TCI", "short", "mid", "long"]):
            df.loc[m, f"bias_{c}"] = df.loc[m, c] - truth[w][i]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(df.groupby(["world", "hl_coef", "estimator", "centring"])[["bias_TCI", "bias_short", "bias_mid", "bias_long", "n_unstable"]]
          .mean().round(1).to_string())


if __name__ == "__main__":
    main()
