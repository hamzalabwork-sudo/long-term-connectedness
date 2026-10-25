"""Implied-volatility networks from FRED daily closes (log levels), as used next to the
ConnectednessApproach example datasets.

Input: FRED CSV downloads in data/ivol/<SERIES>CLS.csv (columns observation_date, <SERIES>CLS), e.g.
  https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS
Output: ivoldata/csv/ivol_<name>.csv with a date column and log index levels, on the dates where all
series of the network are observed.
  crossasset: VIX, OVX, GVZ, EVZ
  usequity:   VIX, VXN, VXD, RVX
  global:     VIX, VXEEM, OVX, GVZ

Usage (from the work directory):  python ../code/build_ivol.py
"""
import argparse
import os

import numpy as np
import pandas as pd

NETWORKS = {"crossasset": ["VIX", "OVX", "GVZ", "EVZ"],
            "usequity": ["VIX", "VXN", "VXD", "RVX"],
            "global": ["VIX", "VXEEM", "OVX", "GVZ"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/ivol")
    ap.add_argument("--out", default="ivoldata/csv")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for name, series in NETWORKS.items():
        cols = []
        for s in series:
            d = pd.read_csv(os.path.join(args.src, f"{s}CLS.csv"), index_col=0, parse_dates=True)
            cols.append(pd.to_numeric(d.iloc[:, 0], errors="coerce").rename(s))
        x = np.log(pd.concat(cols, axis=1, sort=True).dropna())
        x.index.name = "date"
        x.to_csv(os.path.join(args.out, f"ivol_{name}.csv"))
        print(f"ivol_{name}: {len(x)} days, {x.index[0].date()} to {x.index[-1].date()}")


if __name__ == "__main__":
    main()
