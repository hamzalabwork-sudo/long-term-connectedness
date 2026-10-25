"""Frequency connectedness from hourly and from daily data, on common calendar-time bands.

Hourly: causal log-RV panel (hour-of-day adjusted), H = 720 h. Daily: log of average hourly RV per
UTC day, H = 30 d. Both horizons are 30 days, so both have the same lowest non-zero frequency
(30-day cycles). Bands by true cycle length: < 2 d (hourly only), 2-7 d, 7-30 d, > 30 d (the
zero-frequency bin and below; with H = 30 days, slower variation lands here).
TVP-VAR(3), causal EWMA centring matched to the coefficient half-life, prior from the first 30
days; also package centring. Half-life settings:
  hourly: (15, 7) Shi & Lin; (15, 2) chosen by log score
  daily:  (15, 7) the same calendar rates; (120, 60) chosen by log score
Evaluation from 2021-03-01 (after the prior window) to 2025-12-31.

Usage (from iref/):  python code/hourly_vs_daily.py
"""
import argparse
import math
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "code"))
from connectedness import connectedness, tvp_var  # noqa: E402
from predict_gap import NETWORKS  # noqa: E402
from select_halflife import load  # noqa: E402

BANDS_DAYS = [2, 7, 30]
SETTINGS = {"hourly": [(15, 7), (15, 2)], "daily": [(15, 7), (120, 60)]}


def partition(per_day, H):
    edges = [p * per_day for p in BANDS_DAYS]
    part = [math.pi + 1e-5] + [2 * math.pi / e for e in edges] + [0.0]
    if per_day == 1:                                   # no cycles shorter than 2 days at daily frequency
        part = [math.pi + 1e-5] + [2 * math.pi / e for e in edges[1:]] + [0.0]
    return part


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="../work/data/processed")
    ap.add_argument("--start-eval", default="2021-03-01")
    ap.add_argument("--end", default="2025-12-31 23:00")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="results/hourly_vs_daily.csv")
    args = ap.parse_args()
    rows = []
    for freq, settings in SETTINGS.items():
        p, per_day = load(freq, args.root)
        p = p[p.index <= pd.Timestamp(args.end, tz="UTC")]
        ev = p.index >= pd.Timestamp(args.start_eval, tz="UTC")
        H = 30 * per_day
        part = partition(per_day, H)
        names = (["<2d"] if per_day > 1 else []) + ["2-7d", "7-30d", ">30d"]
        for h1, h2 in settings:
            k1, k2 = 0.5 ** (1 / (h1 * per_day)), 0.5 ** (1 / (h2 * per_day))
            for spec, cols in NETWORKS.items():
                x = p[cols].to_numpy()
                for cent in ("package", "ewma"):
                    if cent == "package":
                        fit = tvp_var(x, 3, k1, k2, mode="package")
                    else:
                        fit = tvp_var(x, 3, k1, k2, mode="realtime", prior_size=30 * per_day,
                                      demean="ewma", demean_halflife=h1 * per_day)
                    r = connectedness(fit["B_t"], fit["Q_t"], H, partition=part, device=args.device)
                    row = {"freq": freq, "h_coef": h1, "h_vol": h2, "network": spec, "centring": cent,
                           "TCI": float(r["freqtotal_TCI"][ev].mean())}
                    for b, nm in enumerate(names):
                        row[nm] = float(r["freq_TCI"][ev, b].mean())
                    rows.append(row)
                    print(f"{freq:<6} ({h1:>3},{h2:>2}) {spec:<16} {cent:<7} TCI {row['TCI']:5.1f}  " +
                          "  ".join(f"{nm} {row[nm]:5.1f}" for nm in names), flush=True)
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    cols = ["TCI", "<2d", "2-7d", "7-30d", ">30d"]
    print("\nmean over networks:")
    print(df.groupby(["freq", "h_coef", "h_vol", "centring"])[cols].mean().round(1).to_string())


if __name__ == "__main__":
    main()
