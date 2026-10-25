"""Standard Kalman-filter TVP-VAR (estimated constant state covariance) versus the forgetting-factor
TVP-VAR: predictive log score, and the long component at the selected state-noise scale.

Hourly causal panel, 3 lags, volatility half-life 2 days, causal EWMA centring (15 days).
State-noise scale q is chosen on the selection period (2022-2024) by mean log score, the same
criterion that selects the forgetting rate (15 days). Connectedness (H = 720) over 2021-03 to
2025-12 with the letter's package-labelled bands (true cycles < 16 h, 16-48 h, > 48 h) and with
calendar bands (< 2 d, 2-7 d, 7-30 d, > 30 d).

Usage (from iref/):  python code/select_q.py --workers 16 --device cuda:0
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

Q_GRID = [0.0, 1e-10, 3e-10, 1e-9, 3e-9, 1e-8, 3e-8, 1e-7]
K2 = 0.5 ** (1 / 48)


def period_mean(s, idx, a, b):
    m = (idx >= pd.Timestamp(a, tz="UTC")) & (idx <= pd.Timestamp(b, tz="UTC"))
    return float(np.nanmean(s[m]))


def score_job(a):
    from tvp_predictive import tvp_var_kf, tvp_var_logscore
    spec, cols, q, root = a
    p, _ = load("hourly", root)
    p = p[p.index <= pd.Timestamp(EVAL[1], tz="UTC")]
    x = p[cols].to_numpy()
    if q == "forgetting":
        s = tvp_var_logscore(x, 3, 0.5 ** (1 / 360), K2, 720, "ewma", 360)
    else:
        s = tvp_var_kf(x, 3, q, K2, 720, "ewma", 360)["score"]
    s = s / len(cols)
    return {"network": spec, "q": q, "score_sel": period_mean(s, p.index, *SEL),
            "score_eval": period_mean(s, p.index, *EVAL)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="../work/data/processed")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="results/select_q")
    args = ap.parse_args()
    root = os.path.abspath(args.root)
    os.makedirs(args.out, exist_ok=True)
    jobs = [(s, c, q, root) for s, c in NETWORKS.items() for q in Q_GRID + ["forgetting"]]
    with ProcessPoolExecutor(args.workers) as ex:
        sc = pd.DataFrame(list(ex.map(score_job, jobs, chunksize=1)))
    sc.to_csv(os.path.join(args.out, "scores.csv"), index=False)
    kf = sc[sc.q != "forgetting"].copy()
    kf["q"] = kf.q.astype(float)
    best = kf.loc[kf.groupby("network").score_sel.idxmax()].set_index("network")
    fg = sc[sc.q == "forgetting"].set_index("network")
    print("selected q and scores (KF vs forgetting 15 d):")
    for n in NETWORKS:
        print(f"  {n:<16} q={best.loc[n, 'q']:.0e}  sel {best.loc[n, 'score_sel']:.4f} vs {fg.loc[n, 'score_sel']:.4f}  "
              f"eval {best.loc[n, 'score_eval']:.4f} vs {fg.loc[n, 'score_eval']:.4f}", flush=True)

    from connectedness import connectedness, partition_from_periods, tvp_var
    from tvp_predictive import tvp_var_kf
    p, _ = load("hourly", root)
    p = p[p.index <= pd.Timestamp("2025-12-31 23:00", tz="UTC")]
    ev = p.index >= pd.Timestamp("2021-03-01", tz="UTC")
    parts = {"letter": (partition_from_periods([8, 24], package_convention=True), ["s", "m", "l"]),
             "calendar": ([math.pi + 1e-5] + [2 * math.pi / (d * 24) for d in (2, 7, 30)] + [0.0],
                          ["<2d", "2-7d", "7-30d", ">30d"])}
    rows = []
    for n, cols in NETWORKS.items():
        x = p[cols].to_numpy()
        fits = {"kf": tvp_var_kf(x, 3, float(best.loc[n, "q"]), K2, 720, "ewma", 360),
                "forgetting": tvp_var(x, 3, 0.5 ** (1 / 360), K2, mode="realtime", prior_size=720,
                                      demean="ewma", demean_halflife=360)}
        for est, fit in fits.items():
            row = {"network": n, "estimator": est, "q": float(best.loc[n, "q"]) if est == "kf" else np.nan}
            for pname, (part, names) in parts.items():
                r = connectedness(fit["B_t"], fit["Q_t"], 720, partition=part, device=args.device)
                row[f"{pname}_TCI"] = float(r["freqtotal_TCI"][ev].mean())
                for b, nm in enumerate(names):
                    row[f"{pname}_{nm}"] = float(r["freq_TCI"][ev, b].mean())
            rows.append(row)
            print(f"  {n:<16} {est:<10} letter bands s/m/l {row['letter_s']:5.1f} {row['letter_m']:5.1f} {row['letter_l']:5.1f}"
                  f"  calendar {row['calendar_<2d']:5.1f} {row['calendar_2-7d']:5.1f} {row['calendar_7-30d']:5.1f} "
                  f"{row['calendar_>30d']:5.1f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, "connectedness.csv"), index=False)
    print(df.groupby("estimator").mean(numeric_only=True).round(1).T.to_string())


if __name__ == "__main__":
    main()
