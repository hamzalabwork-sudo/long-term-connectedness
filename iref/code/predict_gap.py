"""Does the pseudo-true decomposition predict the observed package-minus-EWMA gap in the real
crypto networks, and its dependence on the half-life?

For each network and half-life h (coefficient forgetting and EWMA centring matched):
  - drift reading: x = VAR(3) fitted to the de-levelled data, level = observed 30-day level;
    predicted gap = pseudo-true(package) - pseudo-true(EWMA) (offset term minus filter-plus-
    residual-level term);
  - observed gap: long component under package minus EWMA centring from the letter's runs
    (shilin_package / shilin_ewma15d at 15 days; rob_kap099_* at 2.9 days; rob_kap30_* at 30).
Bands: true cycles <16 h, 16-48 h, >48 h (the letter's package-labelled 1-8, 8-24, >24 h).

Usage (from iref/):  python code/predict_gap.py
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "code"))
import spectral as sp  # noqa: E402
from pseudo_true import NLAG, pseudo_true_path, to_lags  # noqa: E402
from sim_centering import fit_dgp  # noqa: E402

EX = ["BN", "OKX", "BG", "BF"]
INS = ["BTCS", "BTCP", "ETHS", "ETHP"]
NETWORKS = {f"instrument:{i}": [f"{i}_{e}" for e in EX] for i in INS}
NETWORKS.update({f"exchange:{e}": [f"{i}_{e}" for i in INS] for e in EX})
HALF = {69: ("rob_kap099_pkg", "rob_kap099_ewmaM"), 360: ("shilin_package", "shilin_ewma15d"),
        720: ("rob_kap30_pkg", "rob_kap30_ewmaM")}


def observed_long(conn, run, spec):
    m = json.load(open(os.path.join(conn, run, "meta.json")))["networks"][spec]
    return list(m["mean_TCI_bands"].values())[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="../work/data/processed/logrv_ds.csv.gz")
    ap.add_argument("--conn", default="../work/results/conn")
    ap.add_argument("--end", default="2025-12-31 23:00")
    ap.add_argument("--step", type=int, default=72)
    ap.add_argument("--out", default="results/predict_gap.csv")
    args = ap.parse_args()
    p = pd.read_csv(args.panel, index_col=0, parse_dates=True)
    p = p[p.index <= pd.Timestamp(args.end, tz="UTC")].interpolate().ffill().bfill()
    n = 2 ** 17
    w = sp.grid(n)
    rows = []
    for spec, cols in NETWORKS.items():
        y = p[cols].to_numpy()
        L, Phi, Sigma = fit_dgp(y, NLAG, 720, "drift")
        A = to_lags(Phi, len(cols))
        fx = sp.var_spectrum(A, Sigma, w)
        Gx = sp.autocov_from_spectrum(fx, NLAG)
        truth = sp.freq_connectedness(A, Sigma, [16, 48], n=n)["TCI"][2]
        for h, (rp, re) in HALF.items():
            h2 = np.abs(sp.ewma_transfer(w, h)) ** 2
            Gfx = sp.autocov_from_spectrum(fx * h2[:, None, None], NLAG)
            pt = pseudo_true_path(Gx, Gfx, L, 1.0, h, h, args.step, 720)
            obs = observed_long(args.conn, rp, spec) - observed_long(args.conn, re, spec)
            rows.append({"network": spec, "halflife_h": h, "truth_long_drift": truth,
                         "pt_pkg": pt["package"][2], "pt_ewma": pt["ewma"][2],
                         "pred_gap": pt["package"][2] - pt["ewma"][2], "obs_gap": obs,
                         "obs_pkg": observed_long(args.conn, rp, spec), "obs_ewma": observed_long(args.conn, re, spec)})
            r = rows[-1]
            print(f"{spec:<16} h={h:4d}  predicted gap {r['pred_gap']:5.1f}  observed {obs:5.1f}  "
                  f"| pkg pt {r['pt_pkg']:5.1f} obs {r['obs_pkg']:5.1f}  ewma pt {r['pt_ewma']:5.1f} obs {r['obs_ewma']:5.1f}",
                  flush=True)
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print("\nmean over networks:")
    print(df.groupby("halflife_h")[["pred_gap", "obs_gap", "pt_pkg", "obs_pkg", "pt_ewma", "obs_ewma"]].mean().round(1).to_string())
    print(f"corr(pred, obs) across network x half-life: {df[['pred_gap', 'obs_gap']].corr().iloc[0, 1]:.2f}")


if __name__ == "__main__":
    main()
