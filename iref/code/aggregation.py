"""Does temporal aggregation explain why hourly and daily data give different frequency profiles?

Take a population hourly process x (the HAR-type VAR(168) fitted to the raw hourly network, in
which slow movements are genuine dynamics; alternatively the de-levelled VAR(3) plus the observed
level). Compute exactly:
  hourly: VAR(3) projection of the hourly autocovariances, connectedness with H = 720 h;
  daily:  autocovariances of daily averages  Gd(j) = (1/24^2) sum_{a,b=0}^{23} Gh(24 j + a - b),
          VAR(3) projection, connectedness with H = 30 d.
Bands by true cycle length: < 2 d, 2-7 d, 7-30 d, > 30 d. These are pseudo-true values of a
constant-coefficient VAR(3) at each frequency, compared with the full-sample-centred TVP-VAR
estimates of hourly_vs_daily.py.

Usage (from iref/):  python code/aggregation.py
"""
import math
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "code"))
import spectral as sp  # noqa: E402
from connectedness import connectedness  # noqa: E402
from predict_gap import NETWORKS  # noqa: E402
from pseudo_true import to_lags  # noqa: E402
from sim_centering import fit_dgp  # noqa: E402

BANDS_DAYS = [2, 7, 30]


def part(per_day):
    edges = [p * per_day for p in BANDS_DAYS] if per_day > 1 else [7, 30]
    return [math.pi + 1e-5] + [2 * math.pi / e for e in edges] + [0.0]


def conn_H(G, p, H, partition):
    A, S = sp.yule_walker(G, p)
    B = np.concatenate(list(A), axis=1)[None]
    r = connectedness(B, S[None], H, partition=partition, device="cpu")
    return r["freq_TCI"][0], r["freqtotal_TCI"][0]


def daily_autocov(Gh, maxlag_d):
    k = Gh.shape[1]
    out = np.zeros((maxlag_d + 1, k, k))

    def g(l):
        return Gh[l] if l >= 0 else Gh[-l].T
    for j in range(maxlag_d + 1):
        acc = np.zeros((k, k))
        for a in range(24):
            for b in range(24):
                acc += g(24 * j + a - b)
        out[j] = acc / 576
    return out


def main():
    p = pd.read_csv("../work/data/processed/logrv_raw_causal.csv.gz", index_col=0, parse_dates=True)
    p = p[p.index <= pd.Timestamp("2025-12-31 23:00", tz="UTC")].ffill().bfill()
    ds = pd.read_csv("../work/data/processed/logrv_ds_causal.csv.gz", index_col=0, parse_dates=True)
    ds = ds[ds.index <= pd.Timestamp("2025-12-31 23:00", tz="UTC")].ffill().bfill()
    n = 2 ** 17
    w = sp.grid(n)
    rows = []
    for spec, cols in NETWORKS.items():
        y = ds[cols].to_numpy()
        for reading in ("harvar", "drift"):
            L, Phi, Sigma = fit_dgp(y, 3, 720, reading)
            A = to_lags(Phi, len(cols))
            f = sp.var_spectrum(A, Sigma, w)
            Gh = sp.autocov_from_spectrum(f, 24 * 5 + 24)
            if reading == "drift":                       # add the observed level's sample second moments
                Lc = L - L.mean(0)
                T = len(Lc)
                for l in range(Gh.shape[0]):
                    Gh[l] += Lc[l:].T @ Lc[:T - l] / T
            hb, ht = conn_H(Gh[:4], 3, 720, part(24))
            Gd = daily_autocov(Gh, 4)
            db, dt = conn_H(Gd, 3, 30, part(1))
            rows.append({"network": spec, "reading": reading, "hourly_total": ht, "h_<2d": hb[0], "h_2-7d": hb[1],
                         "h_7-30d": hb[2], "h_>30d": hb[3], "daily_total": dt, "d_2-7d": db[0], "d_7-30d": db[1],
                         "d_>30d": db[2]})
            print(f"{spec:<16} {reading:<6} hourly {ht:5.1f} [{' '.join(f'{v:5.1f}' for v in hb)}]  "
                  f"daily {dt:5.1f} [{' '.join(f'{v:5.1f}' for v in db)}]", flush=True)
    df = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/aggregation.csv", index=False)
    print(df.groupby("reading").mean(numeric_only=True).round(1).T.to_string())


if __name__ == "__main__":
    main()
