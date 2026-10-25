"""Dynamic time- and frequency-domain connectedness for the Shi & Lin (2026) networks.

Networks (each a separate TVP-VAR, as in Shi & Lin):
  instrument:BTCS  -> BTCS_BN, BTCS_OKX, BTCS_BG, BTCS_BF   (cross-exchange instrument network)
  exchange:BN      -> BTCS_BN, BTCP_BN, ETHS_BN, ETHP_BN    (cross-instrument exchange network)
  all16            -> all sixteen series in one system

Modes (see connectedness.py): "package" reproduces ConnectednessApproach (full-sample means
and prior); "realtime" uses only the first --prior-hours for means and prior.

Usage:
    python run_connectedness.py --panel data/processed/logrv_ds.csv.gz --start 2021-01-01 \
        --end 2025-12-31 --mode package --name shilin_package
"""
import argparse
import json
import math
import os
import subprocess
import time

import numpy as np
import pandas as pd
import torch

from connectedness import connectedness, partition_from_periods, tvp_var


def pick_device(spec, min_free_mb=3000):
    """'auto' -> the CUDA device with the most free memory (useful on shared machines), else CPU.

    Free memory is read from nvidia-smi: querying a full GPU through torch needs a CUDA
    context, which itself fails with 'out of memory'.
    """
    if spec != "auto":
        return spec
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True,
                             timeout=20).stdout
        free = {int(i): int(f) for i, f in (line.split(",") for line in out.strip().splitlines())}
        best = max(free, key=free.get)
        if free[best] >= min_free_mb and torch.cuda.is_available():
            return f"cuda:{best}"
    except Exception:
        pass
    return "cpu"

INSTRUMENTS = ["BTCS", "BTCP", "ETHS", "ETHP"]
EXCHANGES = ["BN", "OKX", "BG", "BF"]


def network_columns(spec):
    if spec == "all16":
        return [f"{i}_{e}" for e in EXCHANGES for i in INSTRUMENTS]
    kind, code = spec.split(":")
    if kind == "instrument":
        return [f"{code}_{e}" for e in EXCHANGES]
    if kind == "exchange":
        return [f"{i}_{code}" for i in INSTRUMENTS]
    raise ValueError(spec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="data/processed/logrv_ds.csv.gz")
    ap.add_argument("--start", default="2021-01-01")
    ap.add_argument("--end", default="2025-12-31 23:00")
    ap.add_argument("--networks", nargs="+",
                    default=[f"instrument:{i}" for i in INSTRUMENTS] + [f"exchange:{e}" for e in EXCHANGES])
    ap.add_argument("--mode", choices=["package", "realtime"], default="package")
    ap.add_argument("--prior-hours", type=int, default=None,
                    help="prior/means window; package mode default = whole sample")
    ap.add_argument("--demean", choices=["full", "window", "ewma"], default=None,
                    help="override the centering implied by --mode")
    ap.add_argument("--demean-halflife", type=float, default=None, help="hours, for --demean ewma")
    ap.add_argument("--intercept", action="store_true", help="time-varying intercept in the state")
    ap.add_argument("--nlag", type=int, default=3)
    ap.add_argument("--nfore", type=int, default=720)
    ap.add_argument("--half-lives", nargs=2, type=float, default=[15, 7],
                    help="coefficient and variance half-lives in days (Shi & Lin: 15, 7)")
    ap.add_argument("--periods", nargs="+", type=float, default=[8, 24])
    ap.add_argument("--true-periods", action="store_true",
                    help="band edges are true cycle lengths (omega = 2pi/P); default follows the "
                         "package labels (omega = pi/P) as in Shi & Lin")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="results/conn")
    ap.add_argument("--name", required=True)
    args = ap.parse_args()

    out_dir = os.path.join(args.out, args.name)
    os.makedirs(out_dir, exist_ok=True)
    args.device = pick_device(args.device)
    print("device:", args.device, flush=True)
    kappa1 = 0.5 ** (1 / (24 * args.half_lives[0]))
    kappa2 = 0.5 ** (1 / (24 * args.half_lives[1]))
    part = partition_from_periods(args.periods, package_convention=not args.true_periods)
    edges = [0] + sorted(args.periods) + [math.inf]
    band_names = [f"{a:g}-{b:g}" for a, b in zip(edges[:-1], edges[1:])]

    panel = pd.read_csv(args.panel, index_col=0, parse_dates=True)
    panel = panel[(panel.index >= pd.Timestamp(args.start, tz="UTC")) &
                  (panel.index <= pd.Timestamp(args.end, tz="UTC"))]
    meta = {"args": vars(args), "kappa1": kappa1, "kappa2": kappa2, "partition": part,
            "band_names": band_names, "hours": len(panel), "networks": {}}

    for spec in args.networks:
        cols = network_columns(spec)
        df = panel[cols]
        n_nan = int(df.isna().sum().sum())
        if args.mode == "realtime":       # carry the last value forward: no look-ahead
            df = df.ffill().bfill()
        else:                              # package replication (full-sample anyway)
            df = df.interpolate(limit_area="inside").ffill().bfill()
        x = df.to_numpy()
        t0 = time.time()
        fit = tvp_var(x, args.nlag, kappa1, kappa2, mode=args.mode, prior_size=args.prior_hours,
                      demean=args.demean, demean_halflife=args.demean_halflife,
                      intercept=args.intercept)
        t_fit = time.time() - t0
        res = connectedness(fit["B_t"], fit["Q_t"], args.nfore, partition=part, device=args.device)
        t_all = time.time() - t0

        tag = spec.replace(":", "_")
        np.savez_compressed(os.path.join(out_dir, f"{tag}.npz"), time_CT=res["time_CT"],
                            freq_CT=res["freq_CT"], unstable=fit["unstable"], mu=fit["mu"],
                            index=df.index.as_unit("ns").asi8, cols=np.array(cols))
        dyn = {"TCI_time": res["time_TCI"], "TCI_ftotal": res["freqtotal_TCI"]}
        for b, bn in enumerate(band_names):
            dyn[f"TCI_{bn}"] = res["freq_TCI"][:, b]
        for i, c in enumerate(cols):
            for key in ("NET", "TO", "FROM"):
                dyn[f"{key}_time_{c}"] = res[f"time_{key}"][:, i]
                for b, bn in enumerate(band_names):
                    dyn[f"{key}_{bn}_{c}"] = res[f"freq_{key}"][:, b, i]
        pd.DataFrame(dyn, index=df.index).to_csv(os.path.join(out_dir, f"{tag}_dynamic.csv.gz"),
                                                 float_format="%.5f")
        meta["networks"][spec] = {"cols": cols, "nan_filled": n_nan,
                                  "unstable_steps": int(fit["unstable"].sum()),
                                  "seconds_tvpvar": round(t_fit, 1), "seconds_total": round(t_all, 1),
                                  "mean_TCI_time": float(res["time_TCI"].mean()),
                                  "mean_TCI_bands": {bn: float(res["freq_TCI"][:, b].mean())
                                                     for b, bn in enumerate(band_names)}}
        print(spec, json.dumps(meta["networks"][spec]), flush=True)

    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1, default=str)


if __name__ == "__main__":
    main()
