"""Direct h-step forecasts of average realised variance (pre-registration, sections 3-6).

For every series i, horizon h and model, OLS is re-estimated at the start of each month of
the evaluation period on all rows whose target is fully observed at that time (expanding
window), and the month's hourly forecasts are produced with those coefficients.

Models (the regressor sets build on each other):
  M0  HAR: own log RV at t and its means over the last 24, 168, 720 h
  M1  M0 + cross-market: log RV at t and 24-h mean of the 6 network partners  (benchmark)
  M2  M1 + time-domain connectedness  (instrument- and exchange-network TCI, FROM_i, TO_i)
  M3_<band>  M1 + the same six measures for one frequency band
  M4  M1 + all bands
Connectedness comes from one or more run directories of run_connectedness.py; the first is
the primary version, the others get M3_<band>@<run> and M4@<run> only.

Losses: QLIKE on levels with F = exp(zhat + s2/2), and squared error on z = log of the
h-hour average RV. Saved per (model, h) as (T_eval, 16) arrays for evaluate.py.

Usage:
    python forecast.py --split validation --conn rt_ewma15d_true rt_window_true pkg_full_true
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd

INSTRUMENTS = ["BTCS", "BTCP", "ETHS", "ETHP"]
EXCHANGES = ["BN", "OKX", "BG", "BF"]
SERIES = [f"{i}_{e}" for e in EXCHANGES for i in INSTRUMENTS]
SPLITS = {"validation": ("2024-01-01 00:00", "2024-12-31 23:00"),
          "test": ("2025-01-01 00:00", "2026-09-30 23:00")}


def partners(c):
    inst, ex = c.split("_")
    return ([f"{inst}_{e}" for e in EXCHANGES if e != ex],
            [f"{i}_{ex}" for i in INSTRUMENTS if i != inst])


def load_conn(root, run):
    meta = json.load(open(os.path.join(root, run, "meta.json")))
    dyn = {}
    for spec in meta["networks"]:
        dyn[spec] = pd.read_csv(os.path.join(root, run, spec.replace(":", "_") + "_dynamic.csv.gz"),
                                index_col=0, parse_dates=True)
    return dyn, ["time"] + meta["band_names"]


def ts_from_int(a):
    """Integer timestamps -> UTC index; the unit depends on the pandas version that wrote
    them (pandas 3 parses CSV dates at microsecond resolution), so infer it from magnitude."""
    m = np.abs(np.asarray(a)).max()
    unit = "s" if m < 1e11 else "ms" if m < 1e14 else "us" if m < 1e17 else "ns"
    return pd.to_datetime(a, unit=unit, utc=True)


def load_ct(root, run):
    """Pairwise FEVD shares per network: {spec: (cols, index, {domain: (T, k, k)})}."""
    meta = json.load(open(os.path.join(root, run, "meta.json")))
    out = {}
    for spec in meta["networks"]:
        z = np.load(os.path.join(root, run, spec.replace(":", "_") + ".npz"))
        ct = {"time": z["time_CT"]}
        for b, name in enumerate(meta["band_names"]):
            ct[name] = z["freq_CT"][:, b]
        out[spec] = (list(z["cols"]), ts_from_int(z["index"]), ct)
    return out


def weighted_partners(ct_net, c, lrv_f, domain, idx):
    """Partner log RV weighted by spillover shares to c (row c of the FEVD, own share
    excluded), and the equal-weighted average, at lags 1 h and 24 h mean."""
    cols, index, ct = ct_net
    i = cols.index(c)
    others = [j for j in range(len(cols)) if j != i]
    w = ct[domain][:, i, others].astype(float)
    w = pd.DataFrame(w / w.sum(1, keepdims=True), index=index).reindex(idx).to_numpy()
    if np.isfinite(w).mean() < 0.9:
        raise ValueError(f"spillover weights for {c} do not align with the panel index")
    x1 = lrv_f[[cols[j] for j in others]]
    x24 = x1.rolling(24).mean()
    return {"sw_1": (w * x1.to_numpy()).sum(1), "sw_24": (w * x24.to_numpy()).sum(1),
            "ew_1": x1.mean(1).to_numpy(), "ew_24": x24.mean(1).to_numpy()}


def conn_block(dyn, c, domain):
    inst, ex = c.split("_")
    out = {}
    for kind, net in (("inst", dyn[f"instrument:{inst}"]), ("exch", dyn[f"exchange:{ex}"])):
        out[f"{kind}_TCI"] = net[f"TCI_{domain}"]
        out[f"{kind}_FROM"] = net[f"FROM_{domain}_{c}"]
        out[f"{kind}_TO"] = net[f"TO_{domain}_{c}"]
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="data/processed/logrv_ds.csv.gz")
    ap.add_argument("--conn-root", default="results/conn")
    ap.add_argument("--conn", nargs="+", default=["rt_ewma15d_true"])
    ap.add_argument("--split", choices=list(SPLITS), default="validation")
    ap.add_argument("--train-start", default="2021-01-31 00:00")
    ap.add_argument("--horizons", nargs="+", type=int, default=[1, 8, 24, 168])
    ap.add_argument("--out", default="results/forecast")
    args = ap.parse_args()
    if args.split == "test" and not os.environ.get("FRL_UNSEAL_TEST"):
        raise SystemExit("test period is sealed; set FRL_UNSEAL_TEST=1 once the spec is frozen")

    tz = "UTC"
    lrv = pd.read_csv(args.panel, index_col=0, parse_dates=True)[SERIES]
    lrv_f = lrv.ffill().bfill()            # features: last value carried forward (no look-ahead)
    rv = np.exp(lrv)                                                  # targets keep NaN
    idx = lrv.index
    e0, e1 = (pd.Timestamp(s, tz=tz) for s in SPLITS[args.split])
    t_start = pd.Timestamp(args.train_start, tz=tz)
    eval_mask = (idx >= e0) & (idx <= e1)
    eval_idx = idx[eval_mask]
    months = pd.date_range(e0, e1, freq="MS")
    if months[0] != e0:
        months = months.insert(0, e0)

    conns = [(run, *load_conn(args.conn_root, run)) for run in args.conn]
    primary = args.conn[0]

    # model specifications: name -> list of feature-block keys
    specs = {"M0": ["own"], "M1": ["own", "cross"], "M2": ["own", "cross", f"time@{primary}"]}
    bands_primary = conns[0][2][1:]
    # exploratory (logged 2026-10-21): connectedness on top of the univariate HAR, which is the
    # comparison most of the literature reports; contrasts with M3 (on top of cross-market lags)
    for d in conns[0][2]:
        specs[f"M0+{d}"] = ["own", f"{d}@{primary}"]
    # added before unsealing the test (2026-10-21): spillover-weighted versus equal-weighted
    # partner volatility, same number of regressors (H4 in the pre-registration change log)
    ct_primary = load_ct(args.conn_root, primary)
    specs["E0"] = ["own", "ew"]
    for d in conns[0][2]:
        specs[f"S_{d}"] = ["own", f"sw@{d}"]
    for run, _, domains in conns:
        bands = domains[1:]
        suffix = "" if run == primary else f"@{run}"
        for b in bands:
            specs[f"M3_{b}{suffix}"] = ["own", "cross", f"{b}@{run}"]
        specs[f"M4{suffix}"] = ["own", "cross"] + [f"{b}@{run}" for b in bands]

    out_dir = os.path.join(args.out, args.split)
    os.makedirs(out_dir, exist_ok=True)
    arrays, t0 = {}, time.time()
    for h in args.horizons:
        z_all = np.log(rv.rolling(h, min_periods=h).mean().shift(-h))   # mean of t+1..t+h
        r_all = rv.rolling(h, min_periods=h).mean().shift(-h)
        for name in specs:
            for kind in ("qlike", "se", "zhat", "F"):
                arrays[f"{name}|h{h}|{kind}"] = np.full((len(eval_idx), len(SERIES)), np.nan)
        for j, c in enumerate(SERIES):
            s = lrv_f[c]
            blocks = {"own": pd.DataFrame({"own_1": s, "own_24": s.rolling(24).mean(),
                                           "own_168": s.rolling(168).mean(),
                                           "own_720": s.rolling(720).mean()})}
            p_inst, p_exch = partners(c)
            cross = {}
            for p in p_inst + p_exch:
                cross[f"{p}_1"] = lrv_f[p]
                cross[f"{p}_24"] = lrv_f[p].rolling(24).mean()
            blocks["cross"] = pd.DataFrame(cross)
            for run, dyn, domains in conns:
                for d in domains:
                    blocks[f"{d}@{run}"] = conn_block(dyn, c, d).reindex(idx)
            inst, ex = c.split("_")
            for d in conns[0][2]:
                wi = weighted_partners(ct_primary[f"instrument:{inst}"], c, lrv_f, d, idx)
                we = weighted_partners(ct_primary[f"exchange:{ex}"], c, lrv_f, d, idx)
                blocks[f"sw@{d}"] = pd.DataFrame({"i_sw_1": wi["sw_1"], "i_sw_24": wi["sw_24"],
                                                  "e_sw_1": we["sw_1"], "e_sw_24": we["sw_24"]}, index=idx)
                if "ew" not in blocks:
                    blocks["ew"] = pd.DataFrame({"i_ew_1": wi["ew_1"], "i_ew_24": wi["ew_24"],
                                                 "e_ew_1": we["ew_1"], "e_ew_24": we["ew_24"]}, index=idx)
            z, r = z_all[c], r_all[c]
            for name, keys in specs.items():
                X = pd.concat([blocks[k] for k in keys], axis=1).to_numpy()
                X = np.hstack([np.ones((len(X), 1)), X])
                ok_x = np.isfinite(X).all(1)
                ok_row = ok_x & np.isfinite(z.to_numpy()) & (idx >= t_start)
                zv, rvv = z.to_numpy(), r.to_numpy()
                for m0, m1 in zip(months, list(months[1:]) + [e1 + pd.Timedelta(hours=1)]):
                    train = ok_row & (idx <= m0 - pd.Timedelta(hours=h))
                    beta, *_ = np.linalg.lstsq(X[train], zv[train], rcond=None)
                    resid = zv[train] - X[train] @ beta
                    s2 = resid.var()
                    sel = (idx >= m0) & (idx < m1) & eval_mask & ok_x
                    zhat = X[sel] @ beta
                    rows = np.flatnonzero(sel[eval_mask])
                    F = np.exp(zhat + s2 / 2)
                    R = rvv[sel]
                    arrays[f"{name}|h{h}|zhat"][rows, j] = zhat
                    arrays[f"{name}|h{h}|F"][rows, j] = F
                    arrays[f"{name}|h{h}|se"][rows, j] = (zv[sel] - zhat) ** 2
                    arrays[f"{name}|h{h}|qlike"][rows, j] = R / F - np.log(R / F) - 1
        print(f"h={h}: done ({time.time() - t0:.0f}s)", flush=True)

    np.savez_compressed(os.path.join(out_dir, "losses.npz"), **arrays,
                        index=eval_idx.as_unit("ns").asi8, series=np.array(SERIES))
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump({"args": vars(args), "models": specs, "bands_primary": bands_primary,
                   "eval_hours": int(eval_mask.sum())}, f, indent=1)
    print("saved", out_dir)


if __name__ == "__main__":
    main()
