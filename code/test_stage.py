"""Test-stage analyses pre-registered in sections 8-10, run once on the forecast losses.

1. Model Confidence Set (Hansen, Lunde & Nason 2011), 90%, T_max statistic, circular block
   bootstrap (block 24 h, 5,000 draws) on pooled QLIKE, models
   {M0, M1, M2, M3_<short>, M3_<mid>, M3_<long>, M4}, for each horizon.
2. Per-series tests: Diebold-Mariano on QLIKE and Clark-West on z for M3_b vs M1 and S_d vs E0,
   number of series significant at 5% (one-sided).
3. Economic value: daily volatility targeting of the Binance BTC and ETH perpetuals. The
   weight for day d is w = min(sigma* / sigma_hat, 3), with sigma_hat from the h = 24 forecast
   made at 23:00 of day d-1 and sigma* the average daily volatility of the training period
   (2021-01-31..2023-12-31); turnover costs 5 bp. Reported: realised volatility, tracking
   error to sigma*, Sharpe ratio, and the Fleming, Kirby & Ostdiek (2001) performance fee
   versus M1 for relative risk aversion 1 and 3. Funding payments are not included.
4. Event windows: pooled QLIKE change versus M1 within +-7 days of the Bybit hack
   (2025-02-21) and the liquidation cascade (2025-10-10).

Usage:
    python test_stage.py --split test
"""
import argparse
import glob
import json
import math
import os

import numpy as np
import pandas as pd

from evaluate import nw_tstat, p_one_sided

EVENTS = {"bybit_hack": "2025-02-21", "liquidation_cascade": "2025-10-10"}


def mcs(L, names, alpha=0.10, B=5000, block=24, seed=0):
    """MCS p-values with the T_max statistic; L is (T, m) with no missing values."""
    rng = np.random.default_rng(seed)
    T, m = L.shape
    nb = int(math.ceil(T / block))
    Lb = np.empty((B, m))
    for b in range(B):
        starts = rng.integers(0, T, nb)
        idx = ((starts[:, None] + np.arange(block)[None, :]).ravel()[:T]) % T
        Lb[b] = L[idx].mean(0)
    Lbar = L.mean(0)
    live, p_run, pvals = list(range(m)), 0.0, {}
    while len(live) > 1:
        d = Lbar[live] - Lbar[live].mean()
        db = Lb[:, live] - Lb[:, live].mean(1, keepdims=True)
        se = np.sqrt(((db - d) ** 2).mean(0))
        t = d / se
        p = float((((db - d) / se).max(1) >= t.max()).mean())
        p_run = max(p_run, p)
        worst = live[int(np.argmax(t))]
        pvals[names[worst]] = p_run
        live.remove(worst)
    pvals[names[live[0]]] = 1.0
    return {n: (pvals[n], pvals[n] >= alpha) for n in names}


def fko_fee(r_new, r_base, gamma):
    """Annualised fee (bp) making a quadratic-utility investor indifferent (FKO 2001)."""
    def u(r):
        R = 1 + r
        return np.mean(R - gamma / (2 * (1 + gamma)) * R ** 2)
    target, lo, hi = u(r_base), -0.01, 0.01
    for _ in range(100):                               # bisection on the daily fee
        mid = (lo + hi) / 2
        if u(r_new - mid) > target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2 * 365 * 1e4


def fko_fee_ci(r_new, r_base, gamma, B=2000, block=20, seed=0):
    """95% circular block-bootstrap interval for the FKO fee (paired days, 20-day blocks)."""
    rng = np.random.default_rng(seed)
    T = len(r_new)
    nb = int(math.ceil(T / block))
    fees = np.empty(B)
    for b in range(B):
        idx = ((rng.integers(0, T, nb)[:, None] + np.arange(block)[None, :]).ravel()[:T]) % T
        fees[b] = fko_fee(r_new[idx], r_base[idx], gamma)
    return np.percentile(fees, [2.5, 97.5])


def daily_closes(raw, asset, start, end):
    files = sorted(glob.glob(os.path.join(raw, "binance", f"binance_{asset}_perp_*.csv.gz")))
    frames = []
    for f in files:
        month = os.path.basename(f).split("_")[-1][:7]
        if start[:7] <= month <= end[:7]:
            frames.append(pd.read_csv(f, usecols=["ts_ms", "close"]))
    d = pd.concat(frames)
    s = pd.Series(d["close"].to_numpy(float), index=pd.to_datetime(d["ts_ms"], unit="ms", utc=True))
    return s.sort_index().resample("1D").last()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/forecast")
    ap.add_argument("--split", default="test")
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--panel-raw", default="data/processed/logrv_raw.csv.gz")
    ap.add_argument("--panel-ds", default="data/processed/logrv_ds.csv.gz")
    ap.add_argument("--train", nargs=2, default=["2021-01-31", "2023-12-31"])
    ap.add_argument("--boot", type=int, default=5000)
    args = ap.parse_args()
    d = os.path.join(args.root, args.split)
    L = np.load(os.path.join(d, "losses.npz"))
    meta = json.load(open(os.path.join(d, "meta.json")))
    models, bands = list(meta["models"]), meta["bands_primary"]
    horizons = meta["args"]["horizons"]
    series = list(L["series"])
    idx = pd.to_datetime(L["index"], utc=True)
    out = {}

    # 1. MCS -----------------------------------------------------------------------------
    mcs_set = ["M0", "M1", "M2"] + [f"M3_{b}" for b in bands] + ["M4"]
    rows = []
    for h in horizons:
        q = np.stack([np.nanmean(L[f"{m}|h{h}|qlike"], 1) for m in mcs_set], 1)
        q = q[np.isfinite(q).all(1)]
        res = mcs(q, mcs_set, B=args.boot)
        for m, (p, inside) in res.items():
            rows.append({"h": h, "model": m, "mean_qlike": float(q[:, mcs_set.index(m)].mean()),
                         "mcs_p": p, "in_90pct_MCS": inside})
    out["mcs"] = pd.DataFrame(rows)

    # 2. per-series tests ------------------------------------------------------------------
    rows = []
    pairs = [(f"M3_{b}", "M1") for b in bands] + [(m, "E0") for m in models if m.startswith("S_")]
    for h in horizons:
        lags = max(h, int(4 * (len(idx) / 100) ** (2 / 9)))
        for m, base in pairs:
            n_dm = n_cw = 0
            for j in range(len(series)):
                qb, qm = L[f"{base}|h{h}|qlike"][:, j], L[f"{m}|h{h}|qlike"][:, j]
                ok = np.isfinite(qb) & np.isfinite(qm)
                n_dm += p_one_sided(nw_tstat(qb[ok] - qm[ok], lags)[0]) < 0.05
                eb, em = L[f"{base}|h{h}|se"][:, j], L[f"{m}|h{h}|se"][:, j]
                zb, zm = L[f"{base}|h{h}|zhat"][:, j], L[f"{m}|h{h}|zhat"][:, j]
                f = eb - (em - (zb - zm) ** 2)
                n_cw += p_one_sided(nw_tstat(f[ok], lags)[0]) < 0.05
            rows.append({"h": h, "model": m, "vs": base, "series_DM_sig5": int(n_dm),
                         "series_CW_sig5": int(n_cw), "of": len(series)})
    out["per_series"] = pd.DataFrame(rows)

    # 3. economic value ----------------------------------------------------------------------
    raw_p = pd.read_csv(args.panel_raw, index_col=0, parse_dates=True)
    ds_p = pd.read_csv(args.panel_ds, index_col=0, parse_dates=True)
    t0, t1 = (pd.Timestamp(x, tz="UTC") for x in args.train)
    econ_models = ["M0", "M1", "M2"] + [f"M3_{b}" for b in bands] + ["M4", "E0"] + \
        [m for m in models if m.startswith("S_")]
    rows = []
    for c, asset in (("BTCP_BN", "BTC"), ("ETHP_BN", "ETH")):
        j = series.index(c)
        season = (raw_p[c] - ds_p[c]).groupby(raw_p.index.hour).mean()     # log factor per hour
        s_bar = float(np.exp(season).mean())                              # ds -> raw, daily sum
        rv_day = np.exp(raw_p[c]).resample("1D").sum(min_count=20)
        tr = rv_day[(rv_day.index >= t0) & (rv_day.index <= t1)]
        sigma_star = float(np.sqrt(tr).mean())
        closes = daily_closes(args.raw, asset, str((idx[0] - pd.Timedelta(days=2)).date()), str(idx[-1].date()))
        ret = np.log(closes).diff()
        origins = idx[idx.hour == 23]
        for m in econ_models:
            F = pd.Series(L[f"{m}|h24|F"][:, j], index=idx).reindex(origins)
            sig_hat = np.sqrt(24 * F * s_bar)
            w = np.minimum(sigma_star / sig_hat, 3.0)
            w.index = w.index.normalize() + pd.Timedelta(days=1)            # applies to next day
            df = pd.concat([w.rename("w"), ret.rename("r")], axis=1, sort=True).dropna()
            turn = df["w"].diff().abs().fillna(df["w"].abs())
            net = df["w"] * df["r"] - 0.0005 * turn
            rows.append({"asset": asset, "model": m, "days": len(df), "net": net,
                         "ann_vol": float(net.std() * math.sqrt(365)),
                         "target_ann_vol": sigma_star * math.sqrt(365),
                         "sharpe": float(net.mean() / net.std() * math.sqrt(365)),
                         "mean_w": float(df["w"].mean())})
    econ = []
    for asset in ("BTC", "ETH"):
        base = next(r for r in rows if r["asset"] == asset and r["model"] == "M1")
        for r in [r for r in rows if r["asset"] == asset]:
            al = pd.concat([r["net"], base["net"]], axis=1, sort=True).dropna()
            rn, rb = al.iloc[:, 0].to_numpy(), al.iloc[:, 1].to_numpy()
            lo3, hi3 = fko_fee_ci(rn, rb, 3) if r["model"] != "M1" else (0.0, 0.0)
            econ.append({k: v for k, v in r.items() if k != "net"} | {
                "tracking_err": abs(r["ann_vol"] - r["target_ann_vol"]),
                "fee_bp_g1": fko_fee(rn, rb, 1), "fee_bp_g3": fko_fee(rn, rb, 3),
                "fee_g3_ci_lo": lo3, "fee_g3_ci_hi": hi3})
    out["econ"] = pd.DataFrame(econ)

    # 4. event windows -----------------------------------------------------------------------
    rows = []
    for ev, day in EVENTS.items():
        c0 = pd.Timestamp(day, tz="UTC")
        win = (idx >= c0 - pd.Timedelta(days=7)) & (idx < c0 + pd.Timedelta(days=8))
        if not win.any():
            continue
        for h in horizons:
            base = np.nanmean(L[f"M1|h{h}|qlike"][win], 1)
            for m in ["M0", "M2"] + [f"M3_{b}" for b in bands] + ["M4", "E0"] + \
                    [m for m in models if m.startswith("S_")]:
                lm = np.nanmean(L[f"{m}|h{h}|qlike"][win], 1)
                ok = np.isfinite(base) & np.isfinite(lm)
                rows.append({"event": ev, "h": h, "model": m, "hours": int(ok.sum()),
                             "gain_pct_vs_M1": 100 * (1 - lm[ok].mean() / base[ok].mean())})
    out["events"] = pd.DataFrame(rows)

    for k, v in out.items():
        v.to_csv(os.path.join(d, f"stage_{k}.csv"), index=False)
    with pd.option_context("display.width", 220, "display.max_rows", 300):
        print("MCS (90%):")
        print(out["mcs"].pivot(index="model", columns="h", values="mcs_p").round(3).to_string())
        print("\nPer-series tests (series significant at 5%, of 16):")
        print(out["per_series"].pivot(index="model", columns="h", values="series_DM_sig5").to_string())
        print("\nEconomic value (vs M1):")
        print(out["econ"].drop(columns=["days"]).round(4).to_string(index=False))
        if len(out["events"]):
            print("\nEvent windows, QLIKE gain % vs M1:")
            print(out["events"].pivot_table(index=["event", "model"], columns="h",
                                            values="gain_pct_vs_M1").round(2).to_string())


if __name__ == "__main__":
    main()
