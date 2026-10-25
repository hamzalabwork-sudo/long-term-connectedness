"""Simulation: which centring recovers the true frequency connectedness when the level drifts?

Data-generating process, built from one real network so that it looks like the data:
    y_t = ybar + c * L_t + x_t
  x_t  stationary VAR(3) with constant (Phi, Sigma), fitted by OLS to the real network after
       removing its slow level, so its connectedness is known exactly (same GFEVD code),
  L_t  the real network's slow level (30-day centred moving average, demeaned), i.e. the
       2021-2025 rise and fall of crypto volatility; c in {0, 0.5, 1} scales the drift.

Estimators (Shi & Lin settings: 3 lags, half-lives 15/7 days, H = 720, package bands):
  package      full-sample centring and prior (ConnectednessApproach)
  window30d    centring and prior from the first 30 days
  ewma15d      causal EWMA centring, half-life 15 days (prior from the first 30 days)
  ewma60d      causal EWMA centring, half-life 60 days
  tvintercept  time-varying intercept in the state, first-30-day centring and prior

Usage:
    python sim_centering.py --panel ../data/processed/logrv_ds.csv.gz --seeds 10 --workers 5
"""
import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

import numpy as np
import pandas as pd

ESTIMATORS = {
    "package": dict(mode="package"),
    "window30d": dict(mode="realtime", demean="window"),
    "ewma15d": dict(mode="realtime", demean="ewma", demean_halflife=360),
    "ewma60d": dict(mode="realtime", demean="ewma", demean_halflife=1440),
    "tvintercept": dict(mode="realtime", demean="window", intercept=True),
}
BANDS = ["short", "mid", "long"]


def fit_dgp(y, nlag, level_hours, dgp="drift"):
    """dgp="drift": slow level L (centred moving average) plus an OLS VAR on y - L, i.e. the
    low-frequency movement is a level drift outside the VAR.
    dgp="persistent": OLS VAR on y itself and no level, i.e. all low-frequency movement is
    genuine VAR dynamics (true long-run connectedness that a good estimator must keep).
    dgp="harvar": HAR-type VAR on y itself, y_t = sum_{j<=3} A_j y_{t-j} + B_d mean24_{t-1}
    + B_w mean168_{t-1} + e_t, i.e. slow (daily and weekly) genuine cross-spillovers; written
    as the equivalent VAR(168)."""
    yd = pd.DataFrame(y)
    if dgp == "drift":
        L = yd.rolling(level_hours, center=True, min_periods=1).mean().to_numpy()
    else:
        L = np.repeat(y.mean(0, keepdims=True), len(y), axis=0)
    d = y - L
    T, k = d.shape
    if dgp == "harvar":
        m24 = pd.DataFrame(d).rolling(24).mean().shift(1).to_numpy()
        m168 = pd.DataFrame(d).rolling(168).mean().shift(1).to_numpy()
        s = 168
        W = np.hstack([np.ones((T - s, 1))] + [d[s - j - 1:T - j - 1] for j in range(nlag)]
                      + [m24[s:], m168[s:]])
        C = np.linalg.lstsq(W, d[s:], rcond=None)[0]
        U = d[s:] - W @ C
        Bd = C[1 + nlag * k:1 + (nlag + 1) * k].T
        Bw = C[1 + (nlag + 1) * k:1 + (nlag + 2) * k].T
        Phi = np.zeros((k, 168 * k))
        for j in range(168):                          # coefficient on lag j + 1
            blk = Bw / 168 + (Bd / 24 if j < 24 else 0)
            if j < nlag:
                blk = blk + C[1 + j * k:1 + (j + 1) * k].T
            Phi[:, j * k:(j + 1) * k] = blk
    else:
        W = np.hstack([np.ones((T - nlag, 1))] + [d[nlag - j - 1:T - j - 1] for j in range(nlag)])
        C = np.linalg.lstsq(W, d[nlag:], rcond=None)[0]
        U = d[nlag:] - W @ C
        Phi = C[1:].T                                 # (k, nlag*k): B[e, j*k + v]
    Sigma = U.T @ U / len(U)
    return L - L.mean(0), Phi, Sigma


def companion_maxeig(Phi):
    k = Phi.shape[0]
    p = Phi.shape[1] // k
    comp = np.vstack([Phi, np.hstack([np.eye(k * (p - 1)), np.zeros((k * (p - 1), k))])])
    return float(np.abs(np.linalg.eigvals(comp)).max())


def random_level(T, k, halflife, sd, seed, burn=20000):
    """Common AR(1) level with the given half-life (hours), scaled to standard deviation sd."""
    rng = np.random.default_rng(10_000 + seed)
    phi = 0.5 ** (1 / halflife)
    e = rng.standard_normal(T + burn)
    lvl = np.empty(T + burn)
    lvl[0] = 0.0
    for t in range(1, T + burn):
        lvl[t] = phi * lvl[t - 1] + e[t]
    lvl = lvl[burn:]
    lvl = (lvl - lvl.mean()) / lvl.std() * sd
    return np.repeat(lvl[:, None], k, axis=1)


def simulate(Phi, Sigma, T, seed, burn=20000):
    k = Sigma.shape[0]
    nlag = Phi.shape[1] // k
    rng = np.random.default_rng(seed)
    eps = rng.multivariate_normal(np.zeros(k), Sigma, size=T + burn)
    x = np.zeros((T + burn, k))
    for t in range(nlag, T + burn):
        x[t] = Phi @ x[t - nlag:t][::-1].reshape(-1) + eps[t]
    return x[burn:]


def summarize(res, skip, suffix=""):
    out = {"TCI_time" + suffix: float(res["time_TCI"][skip:].mean()),
           "ftotal" + suffix: float(res["freqtotal_TCI"][skip:].mean())}
    for b, name in enumerate(BANDS):
        out[name + suffix] = float(res["freq_TCI"][skip:, b].mean())
    return out


def run_one(job):
    from connectedness import connectedness, partition_from_periods, tvp_var
    from run_connectedness import pick_device

    seed, scale, cfg = job
    device = pick_device("auto")
    T, k = cfg["L"].shape
    x = simulate(cfg["Phi"], cfg["Sigma"], T, seed)
    L = cfg["L"] if cfg["level_mode"] == "observed" else \
        random_level(T, k, cfg["level_halflife"], cfg["level_sd"], seed)
    y = cfg["ybar"] + scale * L + x
    parts = {"": partition_from_periods([8, 24], package_convention=True),
             "_true": partition_from_periods([8, 24], package_convention=False)}
    rows = []
    for name, kw in ESTIMATORS.items():
        prior_size = None if kw["mode"] == "package" else cfg["prior_hours"]
        t0 = time.time()
        fit = tvp_var(y, 3, cfg["kappa1"], cfg["kappa2"], prior_size=prior_size, **kw)
        row = {"seed": seed, "scale": scale, "estimator": name, "unstable": int(fit["unstable"].sum())}
        for suffix, part in parts.items():
            res = connectedness(fit["B_t"], fit["Q_t"], cfg["nfore"], partition=part, device=device)
            row.update(summarize(res, cfg["prior_hours"], suffix))
        row["seconds"] = round(time.time() - t0, 1)
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="../data/processed/logrv_ds.csv.gz")
    ap.add_argument("--cols", nargs="+", default=["BTCS_BN", "BTCP_BN", "ETHS_BN", "ETHP_BN"])
    ap.add_argument("--end", default="2025-12-31 23:00")
    ap.add_argument("--level-hours", type=int, default=720)
    ap.add_argument("--dgp", choices=["drift", "persistent", "harvar"], default="drift")
    ap.add_argument("--level-mode", choices=["observed", "random"], default="observed",
                    help="drift path: the observed slow level, or a fresh common AR(1) level per seed")
    ap.add_argument("--level-halflife", type=float, default=720, help="hours, for --level-mode random")
    ap.add_argument("--prior-hours", type=int, default=720)
    ap.add_argument("--nfore", type=int, default=720)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--scales", nargs="+", type=float, default=[0.0, 0.5, 1.0])
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--out", default="../results/sim/centering.csv")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    from connectedness import connectedness, partition_from_periods

    p = pd.read_csv(args.panel, index_col=0, parse_dates=True)
    p = p[p.index <= pd.Timestamp(args.end, tz="UTC")][args.cols].interpolate().ffill().bfill()
    y = p.to_numpy()
    L, Phi, Sigma = fit_dgp(y, 3, args.level_hours, args.dgp)
    k = len(args.cols)
    print(f"dgp={args.dgp} level_hours={args.level_hours} level_mode={args.level_mode} "
          f"cols={args.cols} scales={args.scales}")
    level_sd = float(pd.DataFrame(y).rolling(args.level_hours, center=True, min_periods=1).mean().std().mean())
    print(f"DGP: k={k}, T={len(y)}, lags={Phi.shape[1] // k}, max |eig| = {companion_maxeig(Phi):.4f}, "
          f"sd(level) = {L.std(0).round(3)}, sd(VAR innovations) = {np.sqrt(np.diag(Sigma)).round(3)}")

    truth = connectedness(Phi[None], Sigma[None], args.nfore,
                          partition=partition_from_periods([8, 24], package_convention=True), device="cpu")
    truth_t = connectedness(Phi[None], Sigma[None], args.nfore,
                            partition=partition_from_periods([8, 24], package_convention=False), device="cpu")
    truth_row = {"seed": -1, "scale": np.nan, "estimator": "TRUTH", **summarize(truth, 0),
                 **summarize(truth_t, 0, "_true")}
    print("truth:", {key: round(v, 2) for key, v in truth_row.items() if isinstance(v, float)})

    cfg = {"Phi": Phi, "Sigma": Sigma, "L": L, "ybar": y.mean(0), "prior_hours": args.prior_hours,
           "nfore": args.nfore, "kappa1": 0.5 ** (1 / 360), "kappa2": 0.5 ** (1 / 168),
           "level_mode": args.level_mode, "level_halflife": args.level_halflife, "level_sd": level_sd}
    jobs = [(seed, scale, cfg) for scale in args.scales for seed in range(args.seeds)]
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    rows = [truth_row]
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn")) as ex:
        for i, r in enumerate(ex.map(run_one, jobs)):
            rows += r
            print(f"  done {i + 1}/{len(jobs)} (seed {jobs[i][0]}, scale {jobs[i][1]})", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)

    sim = df[df.estimator != "TRUTH"]
    cols = ["TCI_time"] + BANDS + [b + "_true" for b in BANDS]
    mean = sim.groupby(["scale", "estimator"])[cols].mean()
    sd = sim.groupby(["scale", "estimator"])[cols].std()
    print("\nmean over seeds (sd); truth:", {c: round(truth_row[c], 2) for c in cols})
    for (scale, est), row in mean.iterrows():
        cells = "  ".join(f"{c} {row[c]:6.2f} ({sd.loc[(scale, est), c]:.2f})" for c in cols)
        print(f"scale {scale:3.1f}  {est:<12} {cells}")


if __name__ == "__main__":
    main()
