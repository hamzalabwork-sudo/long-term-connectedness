"""Is the remaining gap in the slow-spillover world a local-sampling (Jensen) effect?
Simulate the HAR-VAR once, form the forgetting-weighted sample second moments at each date
(full-sample centring, as the package does), project to VAR(3), and average connectedness.
Compare with connectedness at the population moments (the pseudo-true value).

Usage (from iref/):  python code/check_local_sampling.py
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "code"))
import spectral as sp  # noqa: E402
from pseudo_true import EDGES, NLAG, to_lags, weighted_outer  # noqa: E402
from sim_centering import fit_dgp, simulate  # noqa: E402

p = pd.read_csv("../work/data/processed/logrv_ds.csv.gz", index_col=0, parse_dates=True)
y = p[p.index <= pd.Timestamp("2025-12-31 23:00", tz="UTC")][["BTCS_BN", "BTCP_BN", "ETHS_BN", "ETHP_BN"]]
y = y.interpolate().ffill().bfill().to_numpy()
k = y.shape[1]
rows = []
for dgp in ["drift", "harvar"]:
    _, Phi, Sigma = fit_dgp(y, NLAG, 720, dgp)
    A = to_lags(Phi, k)
    truth = sp.freq_connectedness(A, Sigma, EDGES, n=2 ** 16)["TCI"][2]
    kap = 0.5 ** (1 / 360); J = 3600
    w = (1 - kap) * kap ** np.arange(J); w /= w.sum()
    res = []
    for seed in range(3):
        x = simulate(Phi, Sigma, len(y), seed)
        x = x - x.mean(0)                                   # full-sample centring
        vals = [sp.conn_from_autocov(weighted_outer(x, t, w, NLAG), NLAG, EDGES, n=2 ** 12)["TCI"][2]
                for t in range(J + 720, len(x), 48)]
        res.append(np.mean(vals))
    rows.append({"world": dgp, "truth_long": truth, "local_sample_bias_package": np.mean(res) - truth,
                 "seed_sd": float(np.std(np.array(res)))})
    print(f"{dgp:<7} truth long {truth:5.1f} | local-sample package bias {np.mean(res) - truth:+5.1f} "
          f"(seeds {np.round(np.array(res) - truth, 1)})", flush=True)
pd.DataFrame(rows).to_csv("results/local_sampling.csv", index=False)
