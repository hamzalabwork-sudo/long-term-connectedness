"""Full 1-minute kline download for one exchange, month by month (default Jan 2021 - Sep 2026).

Resumable: a series-month is skipped when its file already exists. Files are written to a
temporary name and renamed once complete, so an interrupted run never leaves a partial
file that looks finished. A series-month that fails after all retries gets no file and is
picked up again on the next run.

Usage (run one process per exchange):
    python3 full_download.py --exchange okx --out data/raw --start 2021-01 --end 2026-09
"""
import argparse
import csv
import gzip
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pilot_download import ASSETS, FETCHERS, INSTRUMENTS, MIN, window  # noqa: E402


def months(start, end):
    y, m = map(int, start.split("-"))
    ey, em = map(int, end.split("-"))
    while (y, m) <= (ey, em):
        yield f"{y:04d}-{m:02d}"
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exchange", required=True, choices=list(FETCHERS))
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--start", default="2021-01")
    ap.add_argument("--end", default="2026-09")
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--reverse", action="store_true",
                    help="newest month first; run alongside a forward process to halve the time")
    args = ap.parse_args()

    out_dir = os.path.join(args.out, args.exchange)
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "download_log.jsonl")

    tags = list(months(args.start, args.end))
    for tag in (reversed(tags) if args.reverse else tags):
        start, end = window(tag)
        for asset in ASSETS:
            for inst in INSTRUMENTS:
                name = f"{args.exchange}_{asset}_{inst}_{tag}"
                path = os.path.join(out_dir, name + ".csv.gz")
                if os.path.exists(path):
                    continue
                rec = {"series": name, "exchange": args.exchange, "asset": asset, "inst": inst,
                       "tag": tag, "expected": (end - start) // MIN}
                t0, rows = time.time(), None
                for attempt in range(args.retries):
                    try:
                        rows = FETCHERS[args.exchange](asset, inst, start, end, tag)
                        rec.pop("error", None)
                        break
                    except Exception as e:  # API-level errors are often transient
                        rec["error"] = f"{type(e).__name__}: {e}"[:300]
                        time.sleep(30 * (attempt + 1))
                if rows is not None:
                    tmp = path + ".tmp"
                    with gzip.open(tmp, "wt", newline="") as f:
                        w = csv.writer(f)
                        w.writerow(["ts_ms", "open", "high", "low", "close", "volume"])
                        w.writerows(rows)
                    os.replace(tmp, path)
                    rec.update(rows=len(rows),
                               first=rows[0][0] if rows else None,
                               last=rows[-1][0] if rows else None)
                else:
                    rec["rows"] = 0
                rec["seconds"] = round(time.time() - t0, 1)
                with open(log_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
