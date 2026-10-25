"""Quality check of the pilot klines: coverage, gaps, and agreement with Binance spot.

For each series and tag it reports
  - coverage: share of expected 1-minute bars present, and the longest gap in minutes
  - zero-volume share (stale bars)
  - usable hours: hours with all twelve 5-minute returns available
  - agreement with Binance spot of the same asset: median |close deviation| in bp,
    correlation of 5-minute log returns, and correlation of hourly log realized volatility

Usage:
    python3 pilot_check.py --raw data/pilot/raw --out data/pilot/quality.csv
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

MIN = 60_000


def load(path):
    df = pd.read_csv(path)
    if df.empty:
        return df
    df["ts"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
    return df.set_index("ts").sort_index()


def five_min_returns(df, start, end):
    # last close in each 5-minute bucket on a complete grid; a missing bucket stays NaN
    grid = pd.date_range(start, end, freq="5min", inclusive="left", tz="UTC")
    close = df["close"].resample("5min").last().reindex(grid)
    return np.log(close).diff()


def hourly_log_rv(r5):
    sq = (r5 ** 2).resample("1h")
    rv = sq.sum(min_count=12)  # require all twelve 5-minute returns
    return np.log(rv.where(rv > 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/pilot/raw")
    ap.add_argument("--out", default="data/pilot/quality.csv")
    args = ap.parse_args()

    log = {r["series"]: r for r in json.load(open(os.path.join(args.raw, "download_log.json")))}
    rows = []
    for name, rec in sorted(log.items()):
        tag = rec["tag"]
        path = os.path.join(args.raw, name + ".csv")
        out = {"series": name, "exchange": rec["exchange"], "asset": rec["asset"],
               "inst": rec["inst"], "tag": tag, "expected": rec["expected"],
               "rows": rec.get("rows", 0), "error": rec.get("error", "")}
        if out["rows"] == 0 or not os.path.exists(path):
            rows.append(out)
            continue
        df = load(path)
        start = pd.Timestamp(tag + ("-01" if tag.count("-") == 1 else ""), tz="UTC")
        end = start + (pd.offsets.MonthBegin(1) if tag.count("-") == 1 else pd.Timedelta(days=1))
        minutes = pd.date_range(start, end, freq="1min", inclusive="left", tz="UTC")
        present = minutes.isin(df.index)
        gaps, run = [], 0
        for p in present:
            run = 0 if p else run + 1
            gaps.append(run)
        out["coverage_pct"] = round(100 * present.mean(), 2)
        out["max_gap_min"] = int(max(gaps)) if gaps else 0
        out["zero_vol_pct"] = round(100 * (df["volume"].astype(float) == 0).mean(), 2)

        r5 = five_min_returns(df, start, end)
        lrv = hourly_log_rv(r5)
        out["usable_hours_pct"] = round(100 * lrv.notna().mean(), 2)

        ref = os.path.join(args.raw, f"binance_{rec['asset']}_spot_{tag}.csv")
        if os.path.exists(ref) and not name.startswith(f"binance_{rec['asset']}_spot"):
            bdf = load(ref)
            joined = df[["close"]].join(bdf[["close"]], rsuffix="_ref", how="inner").astype(float)
            dev = (joined["close"] / joined["close_ref"] - 1).abs() * 1e4
            out["median_dev_bp"] = round(float(dev.median()), 2)
            out["p99_dev_bp"] = round(float(dev.quantile(0.99)), 2)
            br5 = five_min_returns(bdf, start, end)
            out["corr_r5"] = round(float(r5.corr(br5)), 3)
            out["corr_logrv"] = round(float(lrv.corr(hourly_log_rv(br5))), 3)
        rows.append(out)

    q = pd.DataFrame(rows)
    q.to_csv(args.out, index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.max_rows", 200):
        print(q.drop(columns=["expected"]).to_string(index=False))


if __name__ == "__main__":
    main()
