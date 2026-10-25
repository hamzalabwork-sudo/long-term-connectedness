"""Completeness summary of the full download (reads data/raw/{exchange}/download_log.jsonl)."""
import glob
import json
import os
import sys

raw = sys.argv[1] if len(sys.argv) > 1 else "data/raw"
for ex in ["binance", "okx", "bitget", "bitfinex", "bybit"]:
    recs = {}
    with open(os.path.join(raw, ex, "download_log.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            recs[r["series"]] = r                      # last record per series-month wins
    files = len(glob.glob(os.path.join(raw, ex, "*.csv.gz")))
    short = sorted(s for s, r in recs.items() if 0 < r.get("rows", 0) < 0.99 * r["expected"])
    empty = sorted(s for s, r in recs.items() if r.get("rows", 0) == 0)
    errs = sorted(s for s, r in recs.items() if r.get("error"))
    cov = sum(r.get("rows", 0) for r in recs.values()) / sum(r["expected"] for r in recs.values())
    print(f"{ex:9s} files {files}/276  minute coverage {100 * cov:.2f}%  "
          f"short(<99%) {len(short)}  empty {len(empty)}  errors {len(errs)}")
    if ex != "bitfinex":                               # Bitfinex omits no-trade minutes by design
        for s in short[:8]:
            print(f"     short: {s} {recs[s]['rows']}/{recs[s]['expected']}")
    for s in empty[:8]:
        print(f"     empty: {s}")
    for s in errs[:8]:
        print(f"     error: {s} {recs[s]['error'][:120]}")
