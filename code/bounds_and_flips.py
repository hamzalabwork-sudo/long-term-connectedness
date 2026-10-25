"""Two additions for the letter.

1. Equivalence-style bounds: one-sided 95% upper bound on the pooled QLIKE gain (% of the
   benchmark's mean loss) of each connectedness model over M1, from the Newey-West standard
   error of the pooled loss differential, i.e. the largest improvement the data allow.
2. Role flips: for package versus causal EWMA centring (2021-2025, same evaluation window),
   the number of variables whose net connectedness changes sign, and of directed pairs whose
   net pairwise connectedness (C_ij - C_ji) changes sign, by band and in the time domain.

Usage (from the work directory):
    python ../code/bounds_and_flips.py --fc results/forecast_causal --conn results/conn --out results/extra
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import nw_mean_se  # noqa: E402


def bounds(fc):
    rows = []
    for split in ("validation", "test"):
        L = np.load(os.path.join(fc, split, "losses.npz"))
        meta = json.load(open(os.path.join(fc, split, "meta.json")))
        models = [m for m in meta["models"] if m.startswith("M3_") and "@" not in m] + ["M2", "M4"]
        for h in meta["args"]["horizons"]:
            q1 = L[f"M1|h{h}|qlike"]
            ok = np.logical_and.reduce([np.isfinite(L[f"{m}|h{h}|qlike"]) for m in models + ["M1"]])
            base = np.nanmean(np.where(ok, q1, np.nan), 1)
            T = int(ok.any(1).sum())
            lags = max(h, int(4 * (T / 100) ** (2 / 9)))
            for m in models:
                lm = np.nanmean(np.where(ok, L[f"{m}|h{h}|qlike"], np.nan), 1)
                mean, se, _ = nw_mean_se(base - lm, lags)
                scale = 100 / np.nanmean(base)
                rows.append({"split": split, "h": h, "model": m, "gain_pct": mean * scale,
                             "upper95_pct": (mean + 1.645 * se) * scale, "lower95_pct": (mean - 1.645 * se) * scale})
    return pd.DataFrame(rows)


def flips(conn, run_a, run_b, skip=720):
    meta = json.load(open(os.path.join(conn, run_a, "meta.json")))
    bands = meta["band_names"]
    rows = []
    for spec in meta["networks"]:
        za = np.load(os.path.join(conn, run_a, spec.replace(":", "_") + ".npz"))
        zb = np.load(os.path.join(conn, run_b, spec.replace(":", "_") + ".npz"))
        doms = {"time": (za["time_CT"][skip:].mean(0), zb["time_CT"][skip:].mean(0))}
        for b, bn in enumerate(bands):
            doms[bn] = (za["freq_CT"][skip:, b].mean(0), zb["freq_CT"][skip:, b].mean(0))
        k = za["time_CT"].shape[1]
        iu = np.triu_indices(k, 1)
        for dom, (ca, cb) in doms.items():
            neta = ca.sum(0) - np.diag(ca) - (ca.sum(1) - np.diag(ca))     # TO - FROM
            netb = cb.sum(0) - np.diag(cb) - (cb.sum(1) - np.diag(cb))
            pa, pb = (ca - ca.T)[iu], (cb - cb.T)[iu]
            rows.append({"network": spec, "domain": dom,
                         "net_flips": int((np.sign(neta) != np.sign(netb)).sum()), "net_n": k,
                         "pair_flips": int((np.sign(pa) != np.sign(pb)).sum()), "pair_n": len(pa)})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fc", default="results/forecast_causal")
    ap.add_argument("--conn", default="results/conn")
    ap.add_argument("--out", default="results/extra")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    b = bounds(args.fc)
    b.to_csv(os.path.join(args.out, "bounds.csv"), index=False)
    for split in ("validation", "test"):
        s = b[(b.split == split) & b.model.str.startswith("M3_")]
        print(f"[{split}] band models, largest one-sided 95% upper bound on the QLIKE gain: "
              + ", ".join(f"h{h}: {s[s.h == h].upper95_pct.max():.2f}%" for h in (1, 8, 24, 168)))

    for a, bb, lab in (("shilin_package", "shilin_ewma15d", "package labels"),
                       ("rob_true_pkg", "rob_true_ewma", "true periods")):
        f = flips(args.conn, a, bb)
        f.to_csv(os.path.join(args.out, f"flips_{bb}.csv"), index=False)
        g = f.groupby("domain")[["net_flips", "net_n", "pair_flips", "pair_n"]].sum()
        print(f"\nRole flips, package vs EWMA centring ({lab}):")
        print(g.to_string())


if __name__ == "__main__":
    main()
