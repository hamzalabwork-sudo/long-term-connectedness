"""Build the hourly log realized volatility panel from 1-minute klines.

Pipeline (each step is reported in the diagnostics file):
  1. Load 1-minute closes per series onto a complete UTC minute grid. Zero-volume bars
     count as minutes without a trade (exchanges repeat the last price in them, including
     during outages such as OKX perpetuals on 2021-01-10 19:27-21:14).
  2. Bad-print filter: a close is dropped when its log deviation from the cross-venue
     median of the same asset (spot and perpetual, all venues) exceeds --tick-threshold
     for at most --max-bad-run consecutive minutes. Longer excursions are kept: they are
     genuine dislocations on stress days, not data errors.
  3. Previous-tick fill of the 1-minute close, at most --ffill-cap minutes, so that a
     venue without trades keeps its last price (this is where staleness enters). Longer
     stretches stay missing, and the return spanning them is dropped.
  4. 5-minute log returns from the last close in each 5-minute bucket.
  5. Hourly RV = sum of squared 5-minute returns, rescaled by 12 / n_valid when a few
     returns are missing; hours with fewer than --min-returns valid returns are missing.
  6. log RV, with exact zeros floored at the 0.1% quantile of positive training-period RV.
  7. Short gaps (<= --max-gap hours) linearly interpolated in log RV; longer gaps stay NaN.
  8. Intraday seasonality: hour-of-day (or hour-of-week) means of log RV estimated on the
     training period only and subtracted from the whole sample.

Usage:
    python build_rv.py --raw data/raw --out data/processed --start 2021-01 --end 2026-09
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

INSTRUMENT_CODE = {("BTC", "spot"): "BTCS", ("BTC", "perp"): "BTCP",
                   ("ETH", "spot"): "ETHS", ("ETH", "perp"): "ETHP"}
EXCHANGE_CODE = {"binance": "BN", "okx": "OKX", "bitget": "BG", "bitfinex": "BF", "bybit": "BB"}


def months(start, end):
    return [p.strftime("%Y-%m") for p in pd.period_range(start, end, freq="M")]


def load_series(raw, ex, asset, inst, month_list, grid):
    frames, missing = [], []
    for m in month_list:
        path = os.path.join(raw, ex, f"{ex}_{asset}_{inst}_{m}.csv.gz")
        if not os.path.exists(path):
            missing.append(m)
            continue
        df = pd.read_csv(path, usecols=["ts_ms", "close", "volume"])
        if len(df):
            frames.append(df)
    if not frames:
        return pd.Series(np.nan, index=grid), missing
    df = pd.concat(frames).drop_duplicates("ts_ms")
    # a zero-volume bar carries no trade (exchanges repeat the last price, also during outages)
    df = df[df["volume"].astype(float) > 0]
    s = pd.Series(df["close"].to_numpy(float),
                  index=pd.to_datetime(df["ts_ms"], unit="ms", utc=True))
    return s.reindex(grid), missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--start", default="2021-01")
    ap.add_argument("--end", default="2026-09")
    ap.add_argument("--exchanges", nargs="+", default=["binance", "okx", "bitget", "bitfinex"])
    ap.add_argument("--train-end", default="2023-12-31 23:00", help="last hour of the training period")
    ap.add_argument("--tick-threshold", type=float, default=0.03)
    ap.add_argument("--max-bad-run", type=int, default=3,
                    help="longest excursion (minutes) still treated as a bad print")
    ap.add_argument("--ffill-cap", type=int, default=15, help="minutes")
    ap.add_argument("--sample-minutes", type=int, default=5, choices=[1, 5, 10, 15, 20, 30],
                    help="return sampling interval for RV (coarser sampling dampens staleness)")
    ap.add_argument("--min-returns", type=int, default=None,
                    help="valid returns needed per hour (default: 75%% of 60/sample-minutes)")
    ap.add_argument("--max-gap", type=int, default=3, help="hours")
    ap.add_argument("--seasonality", choices=["hod", "how", "none"], default="hod")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    per_hour = 60 // args.sample_minutes
    if args.min_returns is None:
        args.min_returns = int(np.ceil(0.75 * per_hour))

    month_list = months(args.start, args.end)
    t0 = pd.Timestamp(month_list[0] + "-01", tz="UTC")
    t1 = (pd.Period(month_list[-1], freq="M") + 1).to_timestamp().tz_localize("UTC")
    grid = pd.date_range(t0, t1, freq="1min", inclusive="left")
    train_end = pd.Timestamp(args.train_end, tz="UTC")

    diag = {"args": vars(args), "series": {}}
    logrv_cols, stale_cols = {}, {}
    for asset in ("BTC", "ETH"):
        closes, raw_missing = {}, {}
        for ex in args.exchanges:
            for inst in ("spot", "perp"):
                name = f"{INSTRUMENT_CODE[(asset, inst)]}_{EXCHANGE_CODE[ex]}"
                closes[name], raw_missing[name] = load_series(args.raw, ex, asset, inst, month_list, grid)
        px = pd.DataFrame(closes)
        lp = np.log(px)

        # 2. bad prints relative to the cross-venue median of the same asset
        med = lp.median(axis=1, skipna=True)
        enough = lp.notna().sum(axis=1) >= 3
        dev = lp.sub(med, axis=0)
        far = (dev.abs() > args.tick_threshold) & enough.to_numpy()[:, None]
        # only short excursions that revert count as bad prints; longer ones are genuine
        # dislocations (e.g. Binance ETH perpetual on 2021-05-19, 34 minutes up to 21% away)
        bad = far.copy()
        for name in far.columns:
            f = far[name]
            run_id = (f != f.shift()).cumsum()
            run_len = f.groupby(run_id).transform("size")
            bad[name] = f & (run_len <= args.max_bad_run)
        lp = lp.mask(bad)

        for name in lp.columns:
            s = lp[name]
            observed = s.notna()
            # 3. previous-tick fill, capped
            filled = s.ffill(limit=args.ffill_cap)
            # 4. returns from the last close in each sampling bucket (default 5 minutes)
            p5 = filled.resample(f"{args.sample_minutes}min").last()
            r5 = p5.diff()
            # 5. hourly RV with rescaling for a few missing returns
            n_valid = r5.resample("1h").count()
            rv = (r5 ** 2).resample("1h").sum() * float(per_hour) / n_valid.where(n_valid > 0)
            rv = rv.where(n_valid >= args.min_returns)
            # 6. log RV with a training-period floor for exact zeros
            pos_train = rv[(rv > 0) & (rv.index <= train_end)]
            floor = float(pos_train.quantile(0.001)) if len(pos_train) else np.nan
            n_zero = int((rv == 0).sum())
            lrv = np.log(rv.where(rv > 0, floor).where(rv.notna()))
            # 7. short gaps
            n_missing_before = int(lrv.isna().sum())
            if args.max_gap > 0:          # --max-gap 0 keeps the panel strictly causal
                lrv = lrv.interpolate(limit=args.max_gap, limit_area="inside")
            logrv_cols[name] = lrv
            # staleness: share of minutes without a new 1-minute bar
            stale_cols[name] = 1.0 - observed.resample("1h").mean()

            years = lrv.index.year
            diag["series"][name] = {
                "raw_months_missing": raw_missing[name],
                "minutes_observed_pct": round(100 * float(observed.mean()), 3),
                "bad_prints_removed": int(bad[name].sum()),
                "dislocation_minutes_kept": int((far[name] & ~bad[name]).sum()),
                "zero_rv_hours": n_zero,
                "rv_floor": floor,
                "missing_hours_before_interp": n_missing_before,
                "missing_hours_after_interp": int(lrv.isna().sum()),
                "stale_minutes_pct_by_year": {int(y): round(100 * float(v), 2) for y, v in
                                              stale_cols[name].groupby(stale_cols[name].index.year).mean().items()},
                "mean_logrv_by_year": {int(y): round(float(v), 3) for y, v in lrv.groupby(years).mean().items()},
            }

    order = [f"{INSTRUMENT_CODE[(a, i)]}_{EXCHANGE_CODE[ex]}"
             for ex in args.exchanges for a in ("BTC", "ETH") for i in ("spot", "perp")]
    logrv = pd.DataFrame(logrv_cols)[order]
    stale = pd.DataFrame(stale_cols)[order]

    # 8. intraday seasonality from the training period only
    train = logrv.index <= train_end
    if args.seasonality == "none":
        season = pd.DataFrame(0.0, index=logrv.index, columns=logrv.columns)
    else:
        key = logrv.index.hour if args.seasonality == "hod" else logrv.index.dayofweek * 24 + logrv.index.hour
        key = pd.Index(key, name="slot")
        means = logrv[train].groupby(key[train]).mean()
        means = means - means.mean()  # keep the level; remove only the intraday shape
        season = means.reindex(key).set_axis(logrv.index)
    logrv_ds = logrv - season

    suffix = f"_{args.tag}" if args.tag else ""
    logrv.to_csv(os.path.join(args.out, f"logrv_raw{suffix}.csv.gz"))
    logrv_ds.to_csv(os.path.join(args.out, f"logrv_ds{suffix}.csv.gz"))
    stale.to_csv(os.path.join(args.out, f"stale_share{suffix}.csv.gz"))
    diag["panel"] = {"hours": len(logrv), "complete_hours": int(logrv_ds.notna().all(axis=1).sum()),
                     "first": str(logrv.index[0]), "last": str(logrv.index[-1])}
    with open(os.path.join(args.out, f"diagnostics{suffix}.json"), "w") as f:
        json.dump(diag, f, indent=1, default=str)

    with pd.option_context("display.width", 200, "display.max_columns", 20):
        summary = pd.DataFrame({k: {c: v[c] for c in ("minutes_observed_pct", "bad_prints_removed",
                                                       "dislocation_minutes_kept", "zero_rv_hours",
                                                       "missing_hours_after_interp")}
                                for k, v in diag["series"].items()}).T
        print(summary.to_string())
        print(diag["panel"])


if __name__ == "__main__":
    main()
