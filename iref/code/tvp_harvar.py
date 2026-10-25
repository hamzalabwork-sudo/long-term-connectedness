"""TVP-HAR-VAR: an hourly time-varying VAR that can represent daily and weekly dynamics.

    x_t = sum_{j=1}^{3} A_{j,t} x_{t-j} + D_t m24_{t-1} + W_t m168_{t-1} + e_t,
    m24_{t-1} = mean(x_{t-1..t-24}),  m168_{t-1} = mean(x_{t-1..t-168}),
estimated with the same forgetting-factor Kalman recursion as the TVP-VAR (coefficient forgetting
kappa1, EWMA measurement covariance kappa2), after causal EWMA centring. The HAR terms make it a
restricted VAR(168), so its moving-average representation, hence its connectedness, follows from
  Phi_j = A_j 1[j<=3] + D/24 1[j<=24] + W/168,  j = 1..168.
Prior: OLS on the first `prior_size` observations (after the 168-hour burn-in).

Functions return the one-step predictive log score (same definition as tvp_predictive.py) and the
VAR(168) coefficients at a subsample of dates, for connectedness.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "code"))
from connectedness import causal_ewma_mean  # noqa: E402

LOG2PI = np.log(2 * np.pi)
P = 3
LONG = 168


def regressors(x):
    """r_t (T, 5k): [x_{t-1}, x_{t-2}, x_{t-3}, m24_{t-1}, m168_{t-1}]; rows t < 168 are nan."""
    T, k = x.shape
    c = np.vstack([np.zeros((1, k)), np.cumsum(x, 0)])          # c[t] = sum x_0..x_{t-1}
    r = np.full((T, 5 * k), np.nan)
    for t in range(LONG, T):
        r[t] = np.concatenate([x[t - 1], x[t - 2], x[t - 3], (c[t] - c[t - 24]) / 24, (c[t] - c[t - LONG]) / LONG])
    return r


def to_var168(beta, k):
    """beta (k*5k,) equation-major [e, regressor] -> Phi (k, 168k) in the letter's B layout."""
    C = beta.reshape(k, 5 * k)
    A = [C[:, j * k:(j + 1) * k] for j in range(P)]
    D, W = C[:, 3 * k:4 * k], C[:, 4 * k:5 * k]
    Phi = np.zeros((k, LONG * k))
    for j in range(LONG):
        blk = W / LONG + (D / 24 if j < 24 else 0) + (A[j] if j < P else 0)
        Phi[:, j * k:(j + 1) * k] = blk
    return Phi


def tvp_harvar(x, kappa1, kappa2, prior_size, demean_halflife, keep_every=24, keep_from=0, demean="ewma"):
    """demean: "ewma" (causal, half-life demean_halflife) or "full" (full-sample mean, as the package)."""
    x = np.asarray(x, float)
    T, k = x.shape
    mu = causal_ewma_mean(x, demean_halflife, x[:prior_size].mean(0)) if demean == "ewma" else x.mean(0)
    x = x - mu
    R = regressors(x)
    m = 5 * k
    # prior: OLS of x_t on r_t over t in [168, 168 + prior_size)
    s0, s1 = LONG, LONG + prior_size
    Rr, Y = R[s0:s1], x[s0:s1]
    G = np.linalg.pinv(Rr.T @ Rr)
    C = G @ Rr.T @ Y                                            # (m, k)
    U = Y - Rr @ C
    Q = U.T @ U / len(U)
    beta = C.T.reshape(-1)                                     # equation-major
    S = np.kron(np.eye(k), G)
    score = np.full(T, np.nan)
    keep_t, keep_B, keep_Q = [], [], []
    Qprev = Q
    for t in range(LONG + 1, T):
        Z = np.kron(np.eye(k), R[t][None, :])                  # (k, k*m)
        Rb = S / kappa1
        e = x[t] - Z @ beta
        Rx = Rb @ Z.T
        Pm = Qprev + Z @ Rx
        sign, logdet = np.linalg.slogdet(Pm)
        if sign > 0:
            score[t] = -0.5 * (k * LOG2PI + logdet + e @ np.linalg.solve(Pm, e))
        Qnew = kappa2 * Qprev + (1 - kappa2) * np.outer(e, e)
        KG = Rx @ np.linalg.pinv(Qnew + Z @ Rx)
        beta = beta + KG @ e
        S = Rb - KG @ (Z @ Rb)
        Qprev = Qnew
        if t >= keep_from and (t - keep_from) % keep_every == 0:
            keep_t.append(t)
            keep_B.append(to_var168(beta, k))
            keep_Q.append(Qnew.copy())
    return {"score": score, "t": np.array(keep_t), "B": np.array(keep_B), "Q": np.array(keep_Q)}
