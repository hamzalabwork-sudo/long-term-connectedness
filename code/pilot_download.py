"""Pilot download of 1-minute klines for BTC/ETH spot and USDT perpetuals on five exchanges.

Purpose: confirm that free public sources cover the sample window (history depth at
Jan 2021, and the most recent data) before committing to the full 2021-2026 download.

Usage:
    python3 pilot_download.py --out data/pilot/raw --tags 2021-01 2026-09-29
    python3 pilot_download.py --only bitfinex bitget ...

A tag of the form YYYY-MM fetches a whole calendar month; YYYY-MM-DD fetches one day.
Only the standard library is used.
"""
import argparse
import calendar
import csv
import io
import json
import os
import time
import urllib.error
import urllib.request
import zipfile

UA = {"User-Agent": "Mozilla/5.0 (academic research data pilot)"}
ASSETS = ("BTC", "ETH")
INSTRUMENTS = ("spot", "perp")
MIN = 60_000


def get(url, tries=6, pause=1.0):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise
            if e.code in (418, 429) or e.code >= 500:
                time.sleep(pause * 2 ** k)
                continue
            raise
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            time.sleep(pause * 2 ** k)
    raise RuntimeError(f"failed after {tries} tries: {url}")


def getj(url, **kw):
    return json.loads(get(url, **kw))


def window(tag):
    """Return (start_ms, end_ms) in UTC for a YYYY-MM or YYYY-MM-DD tag; end is exclusive."""
    parts = [int(p) for p in tag.split("-")]
    if len(parts) == 2:
        y, m = parts
        start = calendar.timegm((y, m, 1, 0, 0, 0))
        ny, nm = (y + 1, 1) if m == 12 else (y, m + 1)
        end = calendar.timegm((ny, nm, 1, 0, 0, 0))
    else:
        y, m, d = parts
        start = calendar.timegm((y, m, d, 0, 0, 0))
        end = start + 86_400
    return start * 1000, end * 1000


# ---------------------------------------------------------------- exchanges
# Each fetcher returns a sorted list of (ts_ms, open, high, low, close, volume).

def binance(asset, inst, start, end, tag):
    sym = f"{asset}USDT"
    base = "spot" if inst == "spot" else "futures/um"
    freq = "monthly" if tag.count("-") == 1 else "daily"
    url = f"https://data.binance.vision/data/{base}/{freq}/klines/{sym}/1m/{sym}-1m-{tag}.zip"
    raw = get(url)
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        text = z.read(z.namelist()[0]).decode()
    out = {}
    for r in csv.reader(io.StringIO(text)):
        if not r or not r[0].isdigit():  # futures files carry a header row
            continue
        ts = int(r[0])
        if ts > 10 ** 14:  # spot files switched to microseconds in 2025
            ts //= 1000
        if start <= ts < end:
            out[ts] = (ts, r[1], r[2], r[3], r[4], r[5])
    return sorted(out.values())


def okx(asset, inst, start, end, tag):
    inst_id = f"{asset}-USDT" + ("-SWAP" if inst == "perp" else "")
    out, after = {}, end  # `after` returns candles strictly earlier than ts
    while True:
        d = getj(f"https://www.okx.com/api/v5/market/history-candles"
                 f"?instId={inst_id}&bar=1m&after={after}&limit=100")
        if d.get("code") != "0":
            raise RuntimeError(f"okx error: {d}")
        data = d["data"]
        if not data:
            break
        for r in data:
            ts = int(r[0])
            if start <= ts < end:
                out[ts] = (ts, r[1], r[2], r[3], r[4], r[5])
        oldest = min(int(r[0]) for r in data)
        if oldest <= start or oldest >= after:
            break
        after = oldest
        time.sleep(0.12)
    return sorted(out.values())


def bybit(asset, inst, start, end, tag):
    cat = "spot" if inst == "spot" else "linear"
    sym = f"{asset}USDT"
    out, t, step = {}, start, 1000 * MIN
    while t < end:
        e = min(t + step, end) - 1
        d = getj(f"https://api.bybit.com/v5/market/kline"
                 f"?category={cat}&symbol={sym}&interval=1&start={t}&end={e}&limit=1000")
        if d.get("retCode") != 0:
            raise RuntimeError(f"bybit error: {d}")
        for r in d["result"]["list"]:
            ts = int(r[0])
            if start <= ts < end:
                out[ts] = (ts, r[1], r[2], r[3], r[4], r[5])
        t += step
        time.sleep(0.1)
    return sorted(out.values())


BITFINEX_SYMBOLS = {
    ("BTC", "spot"): "tBTCUST", ("ETH", "spot"): "tETHUST",
    ("BTC", "perp"): "tBTCF0:USTF0", ("ETH", "perp"): "tETHF0:USTF0",
}


def bitfinex(asset, inst, start, end, tag):
    sym = BITFINEX_SYMBOLS[(asset, inst)]
    out, t = {}, start
    while t < end:
        d = getj(f"https://api-pub.bitfinex.com/v2/candles/trade:1m:{sym}/hist"
                 f"?start={t}&end={end - 1}&limit=10000&sort=1")
        if not d:
            break
        for r in d:  # MTS, OPEN, CLOSE, HIGH, LOW, VOLUME
            ts = int(r[0])
            if start <= ts < end:
                out[ts] = (ts, r[1], r[3], r[4], r[2], r[5])
        last = int(d[-1][0])
        if last + MIN <= t:
            break
        t = last + MIN
        time.sleep(3.0)  # public candles endpoint is tightly rate limited
    return sorted(out.values())


def bitget(asset, inst, start, end, tag):
    sym = f"{asset}USDT"
    if inst == "spot":
        base = (f"https://api.bitget.com/api/v2/spot/market/history-candles"
                f"?symbol={sym}&granularity=1min&limit=200")
    else:
        base = (f"https://api.bitget.com/api/v2/mix/market/history-candles"
                f"?symbol={sym}&productType=USDT-FUTURES&granularity=1m&limit=200")
    out, t_end = {}, end
    while t_end > start:
        d = getj(f"{base}&endTime={t_end}")
        if d.get("code") != "00000":
            raise RuntimeError(f"bitget error: {d}")
        data = d["data"]
        if not data:
            break
        for r in data:
            ts = int(r[0])
            if start <= ts < end:
                out[ts] = (ts, r[1], r[2], r[3], r[4], r[5])
        oldest = min(int(r[0]) for r in data)
        if oldest >= t_end:
            break
        t_end = oldest
        time.sleep(0.12)
    return sorted(out.values())


FETCHERS = {"binance": binance, "okx": okx, "bybit": bybit, "bitfinex": bitfinex, "bitget": bitget}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/pilot/raw")
    ap.add_argument("--tags", nargs="+", default=["2021-01", "2026-09-29"])
    ap.add_argument("--only", nargs="*", default=list(FETCHERS))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    log = []
    for tag in args.tags:
        start, end = window(tag)
        expected = (end - start) // MIN
        for ex in args.only:
            for asset in ASSETS:
                for inst in INSTRUMENTS:
                    name = f"{ex}_{asset}_{inst}_{tag}"
                    t0 = time.time()
                    rec = {"series": name, "exchange": ex, "asset": asset, "inst": inst,
                           "tag": tag, "expected": expected}
                    try:
                        rows = FETCHERS[ex](asset, inst, start, end, tag)
                        with open(os.path.join(args.out, name + ".csv"), "w", newline="") as f:
                            w = csv.writer(f)
                            w.writerow(["ts_ms", "open", "high", "low", "close", "volume"])
                            w.writerows(rows)
                        rec.update(rows=len(rows),
                                   first=rows[0][0] if rows else None,
                                   last=rows[-1][0] if rows else None)
                    except Exception as e:  # keep going; the log records what failed
                        rec.update(rows=0, error=f"{type(e).__name__}: {e}"[:300])
                    rec["seconds"] = round(time.time() - t0, 1)
                    log.append(rec)
                    print(json.dumps(rec), flush=True)
    with open(os.path.join(args.out, "download_log.json"), "w") as f:
        json.dump(log, f, indent=1)


if __name__ == "__main__":
    main()
