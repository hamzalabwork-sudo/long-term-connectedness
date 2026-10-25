"""Compare our connectedness runs with Shi & Lin (2026, FRL 113, 110724), Tables 2 and 3.

Published values are averages over 2021-01-01..2025-12-31: TCI in the time domain and its
short / mid / long components (bands labelled 1-8, 8-24, 24-inf h by the package), and the
time-domain NET of each variable. Static tables are time averages of the dynamic tables,
so they are compared with the means of our dynamic series.

Usage:
    python compare_shilin.py --runs shilin_package shilin_window30d shilin_ewma15d
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

# network: (TCI, short, mid, long, {variable: NET})
PUBLISHED = {
    "instrument:BTCS": (74.34, 23.34, 16.81, 34.20, {"BTCS_BN": 0.04, "BTCS_OKX": 0.33, "BTCS_BG": 0.91, "BTCS_BF": -1.28}),
    "instrument:BTCP": (74.47, 23.79, 16.94, 33.75, {"BTCP_BN": 0.73, "BTCP_OKX": 0.47, "BTCP_BG": 1.04, "BTCP_BF": -2.23}),
    "instrument:ETHS": (74.24, 24.61, 16.90, 32.73, {"ETHS_BN": 0.45, "ETHS_OKX": 1.11, "ETHS_BG": 0.98, "ETHS_BF": -2.53}),
    "instrument:ETHP": (74.07, 25.25, 16.82, 32.00, {"ETHP_BN": 1.33, "ETHP_OKX": 0.62, "ETHP_BG": 0.69, "ETHP_BF": -2.64}),
    "exchange:BN": (70.35, 22.81, 15.30, 32.24, {"BTCS_BN": -2.12, "BTCP_BN": -2.19, "ETHS_BN": 1.96, "ETHP_BN": 2.35}),
    "exchange:OKX": (70.41, 22.65, 15.51, 32.25, {"BTCS_OKX": -1.99, "BTCP_OKX": -2.43, "ETHS_OKX": 2.56, "ETHP_OKX": 1.86}),
    "exchange:BG": (70.00, 22.53, 15.50, 31.97, {"BTCS_BG": -1.50, "BTCP_BG": -2.18, "ETHS_BG": 2.32, "ETHP_BG": 1.36}),
    "exchange:BF": (69.52, 21.70, 15.44, 32.38, {"BTCS_BF": -0.61, "BTCP_BF": -2.51, "ETHS_BF": 1.93, "ETHP_BF": 1.19}),
}


def load_run(root, name):
    meta = json.load(open(os.path.join(root, name, "meta.json")))
    bands = meta["band_names"]
    out = {}
    for spec, info in meta["networks"].items():
        dyn = pd.read_csv(os.path.join(root, name, spec.replace(":", "_") + "_dynamic.csv.gz"),
                          index_col=0, parse_dates=True)
        out[spec] = {"dyn": dyn, "cols": info["cols"], "bands": bands}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/conn")
    ap.add_argument("--runs", nargs="+", required=True)
    args = ap.parse_args()

    for name in args.runs:
        run = load_run(args.root, name)
        print(f"\n=== {name}")
        print(f"{'network':<17}{'TCI':>13}{'short':>13}{'mid':>13}{'long':>13}"
              f"{'NET sign match':>16}{'max |dNET|':>12}   long share by year 2021..2025")
        for spec, (tci, s, m, l, net) in PUBLISHED.items():
            if spec not in run:
                continue
            d, b = run[spec]["dyn"], run[spec]["bands"]
            ours = [d["TCI_time"].mean()] + [d[f"TCI_{bn}"].mean() for bn in b]
            pub = [tci, s, m, l]
            cells = "".join(f"{o:6.2f}/{p:<6.2f}" for o, p in zip(ours, pub))
            onet = {c: d[f"NET_time_{c}"].mean() for c in run[spec]["cols"]}
            match = sum(np.sign(onet[c]) == np.sign(v) for c, v in net.items())
            dmax = max(abs(onet[c] - v) for c, v in net.items())
            share = (d[f"TCI_{b[-1]}"] / d["TCI_ftotal"]).groupby(d.index.year).mean().round(2).tolist()
            print(f"{spec:<17}{cells}{match:>12}/4{dmax:>12.2f}   {share}")
        print("cells are ours/published")


if __name__ == "__main__":
    main()
