"""Pseudo-true long-band connectedness of the two centrings, computed exactly, versus the
letter's Monte Carlo (results/sim/*.csv of the letter).

World (as in the letter's sim_centering.py, Binance network):
    y_t = ybar + c L_t + x_t,   x_t a VAR (drift world: VAR(3) on de-levelled data; slow-spillover
                                world: HAR-type VAR(168) on the raw data, no level)
Estimators: TVP-VAR(3) with coefficient forgetting kappa (half-life 15 days).
  package : centred on the full-sample mean. Locally the filter sees d_t + x_t with d_t = c L_t, so
            its local least-squares limit is the VAR(3) projection of the forgetting-weighted second
            moments  M_l(t) = Gamma_x(l) + sum_j w_j d_{t-j} d_{t-j-l}'.
  ewma    : centred on a causal EWMA (half-life 15 days). The filter sees (1-G(L)) x_t + r_t with
            r_t = (1-G(L)) c L_t, so  M_l(t) = Gamma_{(1-G)x}(l) + sum_j w_j r_{t-j} r_{t-j-l}'.
Truth: population connectedness of x (H = infinity). Bands: true cycles <16 h, 16-48 h, >48 h
(the letter's package-labelled 1-8, 8-24, >24 h).

Usage (from iref/):  python code/pseudo_true.py --panel ../work/data/processed/logrv_ds.csv.gz
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "code"))
import spectral as sp  # noqa: E402
from sim_centering import fit_dgp  # noqa: E402

EDGES = [16, 48]
NLAG = 3


def to_lags(Phi, k):
    p = Phi.shape[1] // k
    return Phi.reshape(k, p, k).transpose(1, 0, 2)


def weighted_outer(z, t, w, maxlag):
    """sum_j w_j z_{t-j} z_{t-j-l}' for l = 0..maxlag (w truncated, renormalised)."""
    J = len(w)
    out = np.zeros((maxlag + 1, z.shape[1], z.shape[1]))
    a = z[t - J + 1:t + 1][::-1]                                         # z_t, z_{t-1}, ...
    for l in range(maxlag + 1):
        b = z[t - J + 1 - l:t + 1 - l][::-1]
        out[l] = np.einsum("j,ja,jb->ab", w, a, b)
    return out


def ewma_resid(z, halflife):
    """r_t = z_t - EWMA of z_1..z_{t-1} (causal), started at z_0."""
    lam = 0.5 ** (1 / halflife)
    m = np.empty_like(z)
    m[0] = z[0]
    for t in range(1, len(z)):
        m[t] = lam * m[t - 1] + (1 - lam) * z[t - 1]
    return z - m


def pseudo_true_path(Gx, Gfx, L, c, hl_coef, hl_ewma, step, start):
    kap = 0.5 ** (1 / hl_coef)
    J = int(10 * hl_coef)
    w = (1 - kap) * kap ** np.arange(J)
    w /= w.sum()
    d = c * L
    r = ewma_resid(d, hl_ewma)
    out = {"package": [], "ewma": []}
    for t in range(max(start, J + NLAG), len(L), step):
        Mp = Gx[:NLAG + 1] + weighted_outer(d, t, w, NLAG)
        Me = Gfx[:NLAG + 1] + weighted_outer(r, t, w, NLAG)
        out["package"].append(sp.conn_from_autocov(Mp, NLAG, EDGES, n=2 ** 12)["TCI"])
        out["ewma"].append(sp.conn_from_autocov(Me, NLAG, EDGES, n=2 ** 12)["TCI"])
    return {k: np.mean(v, 0) for k, v in out.items()}


def sim_bias(path, scale=None):
    s = pd.read_csv(path)
    tr = s[s.estimator == "TRUTH"].iloc[0]
    sim = s[s.estimator != "TRUTH"]
    if scale is not None:
        sim = sim[np.isclose(sim.scale, scale)]
    g = sim.groupby("estimator")[["short", "mid", "long"]].mean()
    return {e: (g.loc[e] - tr[["short", "mid", "long"]]).to_numpy(float) for e in ("package", "ewma15d") if e in g.index}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="../work/data/processed/logrv_ds.csv.gz")
    ap.add_argument("--cols", nargs="+", default=["BTCS_BN", "BTCP_BN", "ETHS_BN", "ETHP_BN"])
    ap.add_argument("--end", default="2025-12-31 23:00")
    ap.add_argument("--sim", default="../work/results/sim")
    ap.add_argument("--step", type=int, default=48)
    ap.add_argument("--out", default="results/pseudo_true.csv")
    args = ap.parse_args()
    p = pd.read_csv(args.panel, index_col=0, parse_dates=True)
    y = p[p.index <= pd.Timestamp(args.end, tz="UTC")][args.cols].interpolate().ffill().bfill().to_numpy()
    k = y.shape[1]
    n = 2 ** 17
    wgrid = sp.grid(n)
    h2 = np.abs(sp.ewma_transfer(wgrid, 360)) ** 2
    rows = []
    for world, dgp, scales, simfile in [("drift (observed level)", "drift", [0.0, 0.5, 1.0], "centering_level30d.csv"),
                                        ("slow spillovers (HAR-VAR)", "harvar", [0.0], "centering_harvar.csv")]:
        L, Phi, Sigma = fit_dgp(y, NLAG, 720, dgp)
        A = to_lags(Phi, k)
        fx = sp.var_spectrum(A, Sigma, wgrid)
        Gx = sp.autocov_from_spectrum(fx, NLAG)
        Gfx = sp.autocov_from_spectrum(fx * h2[:, None, None], NLAG)
        truth = sp.freq_connectedness(A, Sigma, EDGES, n=n)["TCI"]
        for c in scales:
            pt = pseudo_true_path(Gx, Gfx, L, c, 360, 360, args.step, 720)
            sb = sim_bias(os.path.join(args.sim, simfile), c)
            for est, key in (("package", "package"), ("ewma", "ewma15d")):
                b = pt[est] - truth
                rows.append({"world": world, "scale": c, "estimator": est, "truth_long": truth[2],
                             "pt_short": b[0], "pt_mid": b[1], "pt_long": b[2],
                             "sim_short": sb[key][0], "sim_mid": sb[key][1], "sim_long": sb[key][2]})
                print(f"{world:<27} c={c:3.1f} {est:<8} truth long {truth[2]:5.1f} | bias long: "
                      f"pseudo-true {b[2]:+6.1f}  simulated {sb[key][2]:+6.1f} | short {b[0]:+5.1f}/{sb[key][0]:+5.1f}",
                      flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
