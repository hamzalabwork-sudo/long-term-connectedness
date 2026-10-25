"""Indirect-inference estimate of the long band (Binance network).

Binding function: mean simulated estimator outputs m(c) (long and short band of the six estimators
in sim_indirect.py), linearly interpolated on a fine grid of c in [0, 1]. Weighting: inverse of the
across-seed variance of each output (the sampling noise of one realisation). Estimate:
c_hat = argmin_c (m(c) - m_real)' W (m(c) - m_real); the implied long band is truth(c_hat).
Also reports, for each estimator alone, the c (and implied long band) at which its simulated long
band equals the real one, and a J-type misfit at c_hat.

Usage (from iref/):  python code/match_indirect.py
"""
import argparse

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--network", default="exchange:BN")
ap.add_argument("--sim", default="results/sim_indirect_exchange_BN/estimates.csv")
ap.add_argument("--truth", default="results/sim_indirect_exchange_BN/truth.csv")
args = ap.parse_args()
sim = pd.read_csv(args.sim)
truth = pd.read_csv(args.truth).set_index("c")
real = pd.read_csv("results/real_estimates.csv")
real = real[real.network == args.network].set_index(["estimator", "centring", "hl_coef"])

keys = sorted(set(map(tuple, sim[["estimator", "centring", "hl_coef"]].to_numpy())))
cs = np.sort(sim.c.unique())
grid = np.linspace(0, 1, 401)
M, V, R, names = [], [], [], []
for est, cent, hl in keys:
    for band in ("long", "short"):
        g = sim[(sim.estimator == est) & (sim.centring == cent) & (sim.hl_coef == hl)].groupby("c")[band]
        M.append(np.interp(grid, cs, g.mean().loc[cs].to_numpy()))
        V.append(np.mean(g.var().to_numpy()))
        R.append(real.loc[(est, cent, hl), "letter_l" if band == "long" else "letter_s"])
        names.append(f"{est}-{cent}-{int(hl)}d-{band}")
M, V, R = np.array(M), np.array(V), np.array(R)
tl = np.interp(grid, truth.index.to_numpy(), truth["long"].to_numpy())
ts = np.interp(grid, truth.index.to_numpy(), truth["short"].to_numpy())

Q = (((M - R[:, None]) ** 2) / V[:, None]).sum(0)
i = int(np.argmin(Q))
print(f"joint match: c_hat = {grid[i]:.3f}, implied true long = {tl[i]:.1f}, short = {ts[i]:.1f}, "
      f"misfit Q = {Q[i]:.1f} on {len(R)} moments (1 parameter)")
ok = Q <= Q[i] + 3.84                                      # likelihood-ratio-type 95% set for c
print(f"  c set (Q <= Qmin + 3.84): [{grid[ok].min():.3f}, {grid[ok].max():.3f}] -> long in "
      f"[{tl[ok].min():.1f}, {tl[ok].max():.1f}]")
# misfit-adjusted set: when the one-parameter family does not fit all moments (Q_min > df), scale the
# threshold by the over-dispersion factor Q_min / df (quasi-likelihood correction)
disp = max(1.0, Q[i] / (len(R) - 1))
oka = Q <= Q[i] + 3.84 * disp
print(f"  misfit-adjusted set (dispersion {disp:.1f}): [{grid[oka].min():.3f}, {grid[oka].max():.3f}] -> long in "
      f"[{tl[oka].min():.1f}, {tl[oka].max():.1f}]")
print("\nfit at c_hat (real vs simulated, sim sd):")
for n, r, m, v in zip(names, R, M[:, i], V):
    print(f"  {n:<26} real {r:5.1f}  sim {m:5.1f}  sd {np.sqrt(v):4.1f}")
print("\nper-estimator inversion of the long band:")
for k, (est, cent, hl) in enumerate(keys):
    m = M[2 * k]
    r = R[2 * k]
    j = int(np.argmin(np.abs(m - r)))
    print(f"  {est:<6} {cent:<7} {int(hl):>3}d  real {r:5.1f}  -> c {grid[j]:.2f}, true long {tl[j]:5.1f}"
          f"{'  (outside simulated range)' if not (m.min() <= r <= m.max()) else ''}")
print(f"\nsimulated truth: long {truth['long'].round(1).to_dict()}")
pd.DataFrame([{"network": args.network, "c_hat": grid[i], "c_lo": grid[ok].min(), "c_hi": grid[ok].max(),
               "long": tl[i], "long_lo": tl[ok].min(), "long_hi": tl[ok].max(), "short": ts[i], "Q": Q[i],
               "disp": disp, "c_lo_adj": grid[oka].min(), "c_hi_adj": grid[oka].max(),
               "long_lo_adj": tl[oka].min(), "long_hi_adj": tl[oka].max(),
               "n_moments": len(R), "seeds": int(sim.seed.nunique())}]).to_csv(
    f"results/indirect_match_{args.network.replace(':', '_')}.csv", index=False)
