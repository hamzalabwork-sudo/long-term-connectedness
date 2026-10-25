"""Pre-registered inference on the forecast losses written by forecast.py (section 8).

  - pooled loss differentials d_t = mean_i (L_M1,i,t - L_M,i,t), Diebold-Mariano t-statistic
    with Newey-West lags max(h, floor(4 (T/100)^(2/9))), one-sided (H_A: M beats M1)
  - pooled Clark-West (2007) on z for nested models M vs M1
  - Holm correction over the 12 band tests (H1) and over the matched-vs-mismatched tests (H2)
  - band x horizon matrix of % QLIKE change vs M1

Usage:
    python evaluate.py --split validation
"""
import argparse
import json
import math
import os

import numpy as np
import pandas as pd

MATCHED = {1: "0-8", 24: "24-inf", 168: "24-inf"}        # pre-registered (H2)


def nw_mean_se(d, lags):
    """Mean of d and its Newey-West standard error."""
    d = d[np.isfinite(d)]
    T = len(d)
    u = d - d.mean()
    var = u @ u / T
    for l in range(1, lags + 1):
        var += 2 * (1 - l / (lags + 1)) * (u[l:] @ u[:-l]) / T
    return d.mean(), math.sqrt(var / T), T


def nw_tstat(d, lags):
    m, se, T = nw_mean_se(d, lags)
    return m / se, T


def p_one_sided(t):
    return 0.5 * math.erfc(t / math.sqrt(2))


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/forecast")
    ap.add_argument("--split", default="validation")
    args = ap.parse_args()
    d = os.path.join(args.root, args.split)
    L = np.load(os.path.join(d, "losses.npz"))
    meta = json.load(open(os.path.join(d, "meta.json")))
    models, bands = list(meta["models"]), meta["bands_primary"]
    horizons = meta["args"]["horizons"]

    rows, h1_rows = [], []
    for h in horizons:
        # common support: (t, i) cells finite for every model
        q = {m: L[f"{m}|h{h}|qlike"] for m in models}
        ok = np.logical_and.reduce([np.isfinite(v) for v in q.values()])
        T = ok.any(1).sum()
        lags = max(h, int(4 * (T / 100) ** (2 / 9)))
        pooled = {m: np.where(ok, v, np.nan) for m, v in q.items()}
        base = np.nanmean(pooled["M1"], 1)
        z1 = L[f"M1|h{h}|zhat"]
        se1 = L[f"M1|h{h}|se"]
        for m in models:
            lm = np.nanmean(pooled[m], 1)
            gain = 100 * (1 - np.nanmean(lm) / np.nanmean(base))
            t_dm, n = nw_tstat(base - lm, lags) if m != "M1" else (np.nan, T)
            # Clark-West: t > 0 means the larger of the two nested models (M vs M1) forecasts
            # better; for M0 (nested in M1) this tests M1 against M0
            zm, sem = L[f"{m}|h{h}|zhat"], L[f"{m}|h{h}|se"]
            if m == "M1":
                t_cw = np.nan
            elif m == "M0":
                f = np.where(ok, sem - (se1 - (zm - z1) ** 2), np.nan)
                t_cw = nw_tstat(np.nanmean(f, 1), lags)[0]
            else:
                f = np.where(ok, se1 - (sem - (z1 - zm) ** 2), np.nan)
                t_cw = nw_tstat(np.nanmean(f, 1), lags)[0]
            se_m = np.nanmean(np.where(ok, sem, np.nan), 1)
            se_b = np.nanmean(np.where(ok, se1, np.nan), 1)
            rows.append({"h": h, "model": m, "qlike": np.nanmean(lm), "gain_pct_vs_M1": gain,
                         "DM_t": t_dm, "DM_p": p_one_sided(t_dm) if np.isfinite(t_dm) else np.nan,
                         "mse_gain_pct_vs_M1": 100 * (1 - np.nanmean(se_m) / np.nanmean(se_b)),
                         "CW_t": t_cw, "CW_p": p_one_sided(t_cw) if np.isfinite(t_cw) else np.nan,
                         "T": n, "nw_lags": lags})
        # H2: matched band vs the other bands
        if h in MATCHED:
            mb = MATCHED[h]
            lmb = np.nanmean(pooled[f"M3_{mb}"], 1)
            scale = 100 / np.nanmean(base)
            for ob in bands:
                if ob == mb:
                    continue
                m_, se_, _ = nw_mean_se(np.nanmean(pooled[f"M3_{ob}"], 1) - lmb, lags)
                h1_rows.append({"h": h, "matched": mb, "other": ob, "DM_t": m_ / se_, "p": p_one_sided(m_ / se_),
                                "adv_pct": m_ * scale, "ci95_lo": (m_ - 1.96 * se_) * scale,
                                "ci95_hi": (m_ + 1.96 * se_) * scale})

    res = pd.DataFrame(rows)
    band_tests = res[res.model.isin([f"M3_{b}" for b in bands])].copy()
    res.loc[band_tests.index, "Holm_p_H1"] = holm(band_tests["DM_p"].to_numpy())
    h2 = pd.DataFrame(h1_rows)
    if len(h2):
        h2["Holm_p"] = holm(h2["p"].to_numpy())
    res.to_csv(os.path.join(d, "tests.csv"), index=False)
    h2.to_csv(os.path.join(d, "tests_H2.csv"), index=False)

    # exploratory: models built on M0 are compared with M0 (not pre-registered)
    explo = []
    for h in horizons:
        q0 = L[f"M0|h{h}|qlike"]
        ok = np.logical_and.reduce([np.isfinite(L[f"{m}|h{h}|qlike"]) for m in models])
        T = ok.any(1).sum()
        lags = max(h, int(4 * (T / 100) ** (2 / 9)))
        base = np.nanmean(np.where(ok, q0, np.nan), 1)
        for m in [m for m in models if m.startswith("M0+")] + ["M1"]:
            lm = np.nanmean(np.where(ok, L[f"{m}|h{h}|qlike"], np.nan), 1)
            t, _ = nw_tstat(base - lm, lags)
            explo.append({"h": h, "model": m, "gain_pct_vs_M0": 100 * (1 - np.nanmean(lm) / np.nanmean(base)),
                          "DM_t": t, "DM_p": p_one_sided(t)})
        # H4: spillover-weighted vs equal-weighted partner volatility (same regressor count)
        if "E0" in models:
            le = np.nanmean(np.where(ok, L[f"E0|h{h}|qlike"], np.nan), 1)
            for m in [m for m in models if m.startswith("S_")]:
                lm = np.nanmean(np.where(ok, L[f"{m}|h{h}|qlike"], np.nan), 1)
                t, _ = nw_tstat(le - lm, lags)
                explo.append({"h": h, "model": f"{m} vs E0",
                              "gain_pct_vs_M0": 100 * (1 - np.nanmean(lm) / np.nanmean(le)),
                              "DM_t": t, "DM_p": p_one_sided(t)})
            explo.append({"h": h, "model": "E0",
                          "gain_pct_vs_M0": 100 * (1 - np.nanmean(le) / np.nanmean(base)),
                          "DM_t": nw_tstat(base - le, lags)[0], "DM_p": np.nan})
    explo = pd.DataFrame(explo)
    if len(explo):
        h4 = explo.model.str.contains("vs E0")
        explo.loc[h4, "Holm_p_H4"] = holm(explo.loc[h4, "DM_p"].to_numpy())
        explo.to_csv(os.path.join(d, "tests_exploratory_vsM0.csv"), index=False)
        print("Exploratory: QLIKE gain (%) vs M0; rows 'S_<d> vs E0' are vs equal weights (E0):")
        print(explo.pivot(index="model", columns="h", values="gain_pct_vs_M0").round(2).to_string())
        print(explo.pivot(index="model", columns="h", values="DM_t").round(2).to_string())

    grid = res[res.model.str.startswith("M3_") | res.model.isin(["M2", "M4"])].pivot(
        index="model", columns="h", values="gain_pct_vs_M1")
    grid.to_csv(os.path.join(d, "gain_matrix.csv"))
    with pd.option_context("display.width", 200, "display.max_rows", 200, "display.float_format", "{:.3f}".format):
        print("QLIKE gain (%) vs M1 (positive = better than benchmark):")
        print(grid.round(2).to_string())
        print("\nAll tests:")
        print(res.round(4).to_string(index=False))
        print("\nH2 matched vs mismatched:")
        print(h2.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
