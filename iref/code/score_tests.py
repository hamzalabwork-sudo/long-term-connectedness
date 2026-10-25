"""Are decay rates distinguishable by forecast performance? Diebold-Mariano-type tests on
predictive log-score differences, hourly and daily, pooled over the 8 networks.

For each frequency, candidate coefficient half-lives (days) are compared with the selected one at
the selected volatility half-life. The loss differential at date t is the network-average log
score of the candidate minus that of the selected rate; its mean is tested with Newey-West errors
(lag = 1 week of observations). Selection period 2022-2024 and evaluation period 2025-01 to
2026-09, as in select_halflife.py.

Usage (from iref/):  python code/score_tests.py --workers 16
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from predict_gap import NETWORKS  # noqa: E402
from select_halflife import EVAL, SEL, load  # noqa: E402

CANDIDATES = {"hourly": ([3, 7, 15, 30, 60, 120], 15, 2), "daily": ([15, 30, 60, 120, 240, 365], 120, 60)}


def job(a):
    freq, spec, cols, h1, h2, root = a
    from tvp_predictive import tvp_var_logscore
    p, per_day = load(freq, root)
    p = p[p.index <= pd.Timestamp(EVAL[1], tz="UTC")]
    s = tvp_var_logscore(p[cols].to_numpy(), 3, 0.5 ** (1 / (h1 * per_day)), 0.5 ** (1 / (h2 * per_day)),
                         prior_size=30 * per_day, demean="ewma", demean_halflife=h1 * per_day)
    return freq, spec, h1, pd.Series(s / len(cols), index=p.index)


def nw_t(d, lag):
    d = d - d.mean()
    n = len(d)
    v = d @ d / n
    for l in range(1, lag + 1):
        v += 2 * (1 - l / (lag + 1)) * (d[l:] @ d[:-l]) / n
    return np.sqrt(v / n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="../work/data/processed")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="results/score_tests.csv")
    args = ap.parse_args()
    root = os.path.abspath(args.root)
    jobs = [(f, spec, cols, h1, h2, root) for f, (hs, _, h2) in CANDIDATES.items()
            for spec, cols in NETWORKS.items() for h1 in hs]
    S = {}
    with ProcessPoolExecutor(args.workers) as ex:
        for f, spec, h1, s in ex.map(job, jobs, chunksize=1):
            S[(f, spec, h1)] = s
    rows = []
    for f, (hs, best, _) in CANDIDATES.items():
        lag = 168 if f == "hourly" else 7
        avg = {h: pd.concat([S[(f, spec, h)] for spec in NETWORKS], axis=1).mean(1) for h in hs}
        for h in hs:
            if h == best:
                continue
            for per, (a, b) in (("sel", SEL), ("eval", EVAL)):
                d = (avg[h] - avg[best])
                d = d[(d.index >= pd.Timestamp(a, tz="UTC")) & (d.index <= pd.Timestamp(b, tz="UTC"))].dropna().to_numpy()
                se = nw_t(d, lag)
                rows.append({"freq": f, "h_coef": h, "vs": best, "period": per, "mean_diff": d.mean(),
                             "t": d.mean() / se, "n": len(d)})
                print(f"{f:<6} h_coef {h:>4} vs {best:>3} [{per}]  mean diff {d.mean():+.4f}  t {d.mean() / se:+6.2f}", flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
