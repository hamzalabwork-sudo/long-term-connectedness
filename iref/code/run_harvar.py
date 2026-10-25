"""TVP-HAR-VAR versus TVP-VAR(3) on the hourly networks: predictive log score and frequency
profile (calendar bands < 2 d, 2-7 d, 7-30 d, > 30 d and the letter's bands), H = 720 h.

Both: coefficient half-life 15 days, volatility half-life 2 days, causal EWMA centring (15 days),
prior from the first 30 days. Connectedness at every 24th hour from 2021-03-01 to 2025-12-31;
dates whose VAR(168) companion is explosive are dropped and counted.

Usage (from iref/):  python code/run_harvar.py --workers 8 --device cuda:0
"""
import argparse
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "code"))
from predict_gap import NETWORKS  # noqa: E402
from select_halflife import EVAL, SEL, load  # noqa: E402

K1, K2 = 0.5 ** (1 / 360), 0.5 ** (1 / 48)


def fit_job(a):
    from tvp_harvar import tvp_harvar
    spec, cols, root = a
    p, _ = load("hourly", root)
    p = p[p.index <= pd.Timestamp(EVAL[1], tz="UTC")]
    x = p[cols].to_numpy()
    t0 = int(np.searchsorted(p.index, pd.Timestamp("2021-03-01", tz="UTC")))
    t1 = int(np.searchsorted(p.index, pd.Timestamp("2025-12-31 23:00", tz="UTC"), side="right"))
    r = tvp_harvar(x, K1, K2, 720, 360, keep_every=24, keep_from=t0)
    keep = r["t"] < t1
    s = pd.Series(r["score"] / len(cols), index=p.index)
    sc = {nm: float(s[(s.index >= pd.Timestamp(a_, tz="UTC")) & (s.index <= pd.Timestamp(b_, tz="UTC"))].mean())
          for nm, (a_, b_) in (("sel", SEL), ("eval", EVAL))}
    return spec, sc, r["B"][keep], r["Q"][keep]


def stable(B, k):
    p = B.shape[1] // k
    F = np.zeros((k * p, k * p))
    F[:k] = B
    F[k:, :-k] = np.eye(k * (p - 1))
    # spectral radius by power iteration is unreliable; use eigvals on the companion
    return np.max(np.abs(np.linalg.eigvals(F))) < 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="../work/data/processed")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="results/harvar")
    args = ap.parse_args()
    root = os.path.abspath(args.root)
    os.makedirs(args.out, exist_ok=True)
    from connectedness import connectedness, partition_from_periods
    from tvp_predictive import tvp_var_logscore
    parts = {"letter": (partition_from_periods([8, 24], package_convention=True), ["s", "m", "l"]),
             "calendar": ([math.pi + 1e-5] + [2 * math.pi / (d * 24) for d in (2, 7, 30)] + [0.0],
                          ["<2d", "2-7d", "7-30d", ">30d"])}
    jobs = [(s, c, root) for s, c in NETWORKS.items()]
    rows = []
    with ProcessPoolExecutor(args.workers) as ex:
        for spec, sc, B, Q in ex.map(fit_job, jobs, chunksize=1):
            k = Q.shape[1]
            ok = np.array([stable(b, k) for b in B])
            p, _ = load("hourly", root)
            x = p[p.index <= pd.Timestamp(EVAL[1], tz="UTC")][NETWORKS[spec]].to_numpy()
            s3 = tvp_var_logscore(x, 3, K1, K2, 720, "ewma", 360) / k
            idx = p.index[p.index <= pd.Timestamp(EVAL[1], tz="UTC")]
            s3 = pd.Series(s3, index=idx)
            row = {"network": spec, "n_dates": len(B), "n_unstable": int((~ok).sum()),
                   "har_score_sel": sc["sel"], "har_score_eval": sc["eval"]}
            for nm, (a_, b_) in (("sel", SEL), ("eval", EVAL)):
                row[f"var3_score_{nm}"] = float(s3[(s3.index >= pd.Timestamp(a_, tz="UTC")) &
                                                   (s3.index <= pd.Timestamp(b_, tz="UTC"))].mean())
            for pname, (part, names) in parts.items():
                r = connectedness(B[ok], Q[ok], 720, partition=part, device=args.device, chunk=128)
                row[f"{pname}_TCI"] = float(r["freqtotal_TCI"].mean())
                for b, nm in enumerate(names):
                    row[f"{pname}_{nm}"] = float(r["freq_TCI"][:, b].mean())
            rows.append(row)
            print(f"{spec:<16} unstable {row['n_unstable']}/{row['n_dates']}  score sel HAR {sc['sel']:.4f} vs VAR3 "
                  f"{row['var3_score_sel']:.4f}  eval {sc['eval']:.4f} vs {row['var3_score_eval']:.4f} | calendar "
                  f"{row['calendar_<2d']:5.1f} {row['calendar_2-7d']:5.1f} {row['calendar_7-30d']:5.1f} "
                  f"{row['calendar_>30d']:5.1f} | letter l {row['letter_l']:5.1f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, "summary.csv"), index=False)
    print(df.mean(numeric_only=True).round(4).to_string())


if __name__ == "__main__":
    main()
