"""Checks of spectral.py.

1. Population frequency connectedness of a constant VAR equals the letter's validated
   connectedness() (DFT over a long horizon H) as H grows.
2. VAR -> spectrum -> autocovariances -> Yule-Walker recovers the VAR.
3. A long VAR(p) projection of a VARMA-type spectrum converges as p grows.

Usage (from iref/):  python code/check_spectral.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "code"))
import spectral as sp  # noqa: E402
from connectedness import connectedness, partition_from_periods  # noqa: E402

rng = np.random.default_rng(1)
k, p = 4, 2
while True:
    A = rng.normal(scale=0.25, size=(p, k, k))
    A[0] += 0.5 * np.eye(k)
    if sp.is_stable(A):
        break
L = rng.normal(size=(k, k))
Sigma = L @ L.T / k + np.eye(k) * 0.5
edges = [16, 48]                                  # true cycle lengths: <16, 16-48, >48 obs

pop = sp.freq_connectedness(A, Sigma, edges)
print("1. population vs connectedness.py (DFT over H)")
print(f"   population: bands {np.round(pop['TCI'], 5)}  total {pop['total']:.5f}")
B = np.concatenate(list(A), axis=1)[None]          # (1, k, k*p): [A_1 A_2]
for H in [100, 1000, 8000]:
    r = connectedness(B, Sigma[None], H, partition=partition_from_periods(edges), device="cpu")
    print(f"   H={H:5d}:     bands {np.round(r['freq_TCI'][0], 5)}  total {r['freqtotal_TCI'][0]:.5f}")

print("2. Yule-Walker on exact autocovariances")
G = sp.var_autocov(A, Sigma, 5)
A2, S2 = sp.yule_walker(G, p)
print(f"   max |A - A_yw| = {np.abs(A - A2).max():.2e}, max |Sigma - Sigma_yw| = {np.abs(Sigma - S2).max():.2e}")
f = sp.var_spectrum(A, Sigma, sp.grid(2 ** 12))
G2 = sp.autocov_from_spectrum(f, 5)
print(f"   autocov from spectrum vs Lyapunov: {np.abs(G - G2).max():.2e}")

print("3. VAR(p) projection of a VARMA(1,1) spectrum as p grows")
M = 0.6 * np.eye(k)
w = sp.grid(2 ** 14)
z = np.exp(-1j * w)
Ps = sp.psi(A[:1], w) @ (np.eye(k)[None] + M[None] * z[:, None, None])
fv = Ps @ Sigma @ np.conj(np.swapaxes(Ps, -1, -2)) / (2 * np.pi)
for pp in [1, 3, 10, 30, 80]:
    c = sp.conn_from_spectrum(fv, pp, edges)
    print(f"   p={pp:3d}: bands {np.round(c['TCI'], 3)}  total {c['total']:.3f}")
