"""Check connectedness.py against the CRAN ConnectednessApproach code on a real sub-panel.

Runs validate_r.R and our implementation (mode "package", i.e. with the package's
full-sample steps) on the same data, and reports the largest absolute differences.

Usage:
    python validate_r.py --panel data/processed/logrv_ds.csv.gz --cols BTCS_BN BTCP_BN ETHS_BN ETHP_BN \
        --hours 1000 --nlag 2 --nfore 100 --kappa1 0.99 --kappa2 0.99 --out results/validate/a
"""
import argparse
import math
import os
import subprocess
import time

import numpy as np
import pandas as pd

from connectedness import connectedness, partition_from_periods, tvp_var


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="data/processed/logrv_ds.csv.gz")
    ap.add_argument("--cols", nargs="+", required=True)
    ap.add_argument("--start", default=None)
    ap.add_argument("--hours", type=int, default=1000)
    ap.add_argument("--nlag", type=int, default=2)
    ap.add_argument("--nfore", type=int, default=100)
    ap.add_argument("--kappa1", type=float, default=0.99)
    ap.add_argument("--kappa2", type=float, default=0.99)
    ap.add_argument("--periods", nargs="+", type=float, default=[8, 24],
                    help="band edges as labelled by the package (omega = pi/P)")
    ap.add_argument("--rsrc", default="rsrc")
    ap.add_argument("--out", default="results/validate/a")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df = pd.read_csv(args.panel, index_col=0, parse_dates=True)[args.cols].dropna()
    if args.start:
        df = df[df.index >= pd.Timestamp(args.start, tz="UTC")]
    df = df.iloc[:args.hours]
    panel_csv = os.path.join(args.out, "panel.csv")
    df.index = df.index.strftime("%Y-%m-%d %H:%M:%S")
    df.to_csv(panel_csv)
    x = df.to_numpy()
    T, k = x.shape
    part = partition_from_periods(args.periods, package_convention=True)

    t0 = time.time()
    cmd = ["Rscript", os.path.join(os.path.dirname(os.path.abspath(__file__)), "validate_r.R"),
           panel_csv, args.out, args.rsrc, str(args.nlag), str(args.nfore),
           str(args.kappa1), str(args.kappa2)] + [repr(p) for p in part[1:-1]]
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout.strip())
    if r.returncode != 0:
        print(r.stderr[-3000:])
        raise SystemExit("R run failed")
    r_sec = time.time() - t0

    t0 = time.time()
    fit = tvp_var(x, args.nlag, args.kappa1, args.kappa2, mode="package")
    res = connectedness(fit["B_t"], fit["Q_t"], args.nfore, partition=part, device=args.device)
    py_sec = time.time() - t0

    def rd(name):
        return pd.read_csv(os.path.join(args.out, name)).to_numpy()

    B_r = rd("B_t.csv").reshape(T, k * args.nlag, k).transpose(0, 2, 1)   # undo column-major c()
    Q_r = rd("Q_t.csv").reshape(T, k, k).transpose(0, 2, 1)
    time_r = rd("time.csv")
    ftci_r = rd("freq_TCI.csv")                                        # Total, band1, band2, ...
    nb = len(part) - 1

    rows = [
        ("B_t", np.abs(fit["B_t"] - B_r).max(), np.abs(B_r).max()),
        ("Q_t", np.abs(fit["Q_t"] - Q_r).max(), np.abs(Q_r).max()),
        ("time TCI", np.abs(res["time_TCI"] - time_r[:, 0]).max(), np.abs(time_r[:, 0]).max()),
        ("time NET", np.abs(res["time_NET"] - time_r[:, 1:]).max(), np.abs(time_r[:, 1:]).max()),
        ("freq total TCI", np.abs(res["freqtotal_TCI"] - ftci_r[:, 0]).max(), np.abs(ftci_r[:, 0]).max()),
    ]
    for b in range(nb):
        rows.append((f"freq TCI band {b}", np.abs(res["freq_TCI"][:, b] - ftci_r[:, b + 1]).max(),
                     np.abs(ftci_r[:, b + 1]).max()))
        net_r = rd(f"freq_NET_{b + 1}.csv")
        rows.append((f"freq NET band {b}", np.abs(res["freq_NET"][:, b] - net_r).max(), np.abs(net_r).max()))
    print(f"\nT={T} k={k} nlag={args.nlag} nfore={args.nfore} kappa=({args.kappa1}, {args.kappa2}) "
          f"partition={[round(p, 5) for p in part]}")
    print(f"unstable steps (fallback used): {int(fit['unstable'].sum())}")
    print(f"{'quantity':<20}{'max |py - R|':>16}{'max |R|':>12}")
    for name, d, scale in rows:
        print(f"{name:<20}{d:>16.3e}{scale:>12.3f}")
    print(f"R seconds: {r_sec:.1f}   python seconds: {py_sec:.1f}")


if __name__ == "__main__":
    main()
