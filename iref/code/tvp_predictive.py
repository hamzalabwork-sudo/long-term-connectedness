"""One-step-ahead predictive log density of the forgetting-factor TVP-VAR (TVPVAR.R recursion,
as in the letter's connectedness.tvp_var), for choosing decay rates by forecast performance.

At date t, before x_t is seen, the filter's predictive distribution is
    x_t | F_{t-1} ~ N( Z_t beta_{t|t-1},  Q_{t-1} + Z_t R_t Z_t' ),   R_t = S_{t-1} / kappa1,
where Q_{t-1} is the EWMA measurement covariance (kappa2) and S_{t-1} the state covariance.
Centring is causal (EWMA of past values or the initial-window mean), so the log score of x_t is
that of the observed series (the centring constant is known at t-1).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "code"))
from connectedness import _coef_matrix, _companion, bayes_prior, causal_ewma_mean  # noqa: E402

LOG2PI = np.log(2 * np.pi)


def tvp_var_logscore(x, nlag, kappa1, kappa2, prior_size, demean="ewma", demean_halflife=None):
    """Returns (T,) array of one-step predictive log densities (nan where undefined)."""
    x = np.asarray(x, float)
    T, k = x.shape
    b0, V0, Q0 = bayes_prior(x, nlag, prior_size)
    if demean == "ewma":
        mu = causal_ewma_mean(x, demean_halflife, x[:prior_size].mean(0))
    elif demean == "window":
        mu = x[:prior_size].mean(0)
    else:
        raise ValueError(demean)
    x = x - mu
    ms = nlag * k * k
    eye_k = np.arange(k)
    score = np.full(T, np.nan)
    beta_prev = Sb_prev = Q_prev = Bc = None
    for t in range(T):
        irep = t + 1
        if irep == 1:
            Q = Q0.copy()
        else:
            if irep <= nlag + 1:
                G = 0.1 * np.outer(x[t], x[t])
            else:
                lags = x[t - nlag:t][::-1].reshape(-1)
                u = x[t] - Bc[:k, :k * nlag] @ lags
                G = np.outer(u, u)
            Q = kappa2 * Q_prev + (1 - kappa2) * G
        if irep <= nlag + 1:
            beta_pred, Rb = b0.copy(), V0.copy()
            beta = beta_pred
            Sb = np.zeros((ms, ms)) if irep <= nlag else None
        else:
            beta_pred, Rb = beta_prev, Sb_prev / kappa1
        if irep >= nlag + 1:
            Z = np.zeros((k, nlag, k, k))
            for j in range(nlag):
                Z[eye_k, j, eye_k, :] = x[t - 1 - j]
            Z = Z.reshape(k, ms)
            Rx = Rb @ Z.T
            if irep >= nlag + 2:                                   # predictive density of x_t
                e = x[t] - Z @ beta_pred
                P = Q_prev + Z @ Rx
                sign, logdet = np.linalg.slogdet(P)
                if sign > 0:
                    score[t] = -0.5 * (k * LOG2PI + logdet + e @ np.linalg.solve(P, e))
            KG = Rx @ np.linalg.pinv(Q + Z @ Rx)
            beta = beta_pred + KG @ (x[t] - Z @ beta_pred)
            Sb = Rb - KG @ (Z @ Rb)
        B = _coef_matrix(beta, k, nlag)
        Bc = _companion(B, k, nlag)
        if not (irep == 1 or np.max(np.abs(np.linalg.eigvals(Bc))) <= 1):
            beta = 0.99 * beta_prev
            B = _coef_matrix(beta, k, nlag)
            Bc = _companion(B, k, nlag)
        beta_prev, Sb_prev, Q_prev = beta, Sb, Q
    return score


def tvp_var_kf(x, nlag, q, kappa2, prior_size, demean="ewma", demean_halflife=None):
    """Standard Kalman-filter TVP-VAR: random-walk coefficients beta_t = beta_{t-1} + eta_t with a
    constant state covariance Q_beta = q * (per-observation OLS coefficient variance from the
    prior window), in place of the forgetting factor (R_t = S_{t-1} + Q_beta instead of S_{t-1}/kappa1).
    The measurement covariance is the EWMA (kappa2) of one-step errors, as in the forgetting filter.
    Returns dict with B_t (T, k, k*nlag), Q_t (T, k, k), score (T,) one-step predictive log densities.
    """
    x = np.asarray(x, float)
    T, k = x.shape
    b0, V0, Q0 = bayes_prior(x, nlag, prior_size)
    n0 = prior_size - nlag
    ms = nlag * k * k
    eq = np.tile(np.repeat(np.arange(k), k), nlag)                 # equation of each coefficient
    Qb = np.diag(q * n0 * np.diag(V0) * np.diag(Q0)[eq])
    if demean == "ewma":
        mu = causal_ewma_mean(x, demean_halflife, x[:prior_size].mean(0))
    else:
        mu = x[:prior_size].mean(0)
    x = x - mu
    eye_k = np.arange(k)
    B_t = np.zeros((T, k, k * nlag))
    Q_t = np.zeros((T, k, k))
    score = np.full(T, np.nan)
    beta_prev = Sb_prev = Q_prev = Bc = None
    for t in range(T):
        irep = t + 1
        if irep == 1:
            Q = Q0.copy()
        else:
            if irep <= nlag + 1:
                G = 0.1 * np.outer(x[t], x[t])
            else:
                lags = x[t - nlag:t][::-1].reshape(-1)
                u = x[t] - Bc[:k, :k * nlag] @ lags
                G = np.outer(u, u)
            Q = kappa2 * Q_prev + (1 - kappa2) * G
        if irep <= nlag + 1:
            beta_pred, Rb = b0.copy(), V0.copy()
            beta = beta_pred
            Sb = np.zeros((ms, ms)) if irep <= nlag else None
        else:
            beta_pred, Rb = beta_prev, Sb_prev + Qb
        if irep >= nlag + 1:
            Z = np.zeros((k, nlag, k, k))
            for j in range(nlag):
                Z[eye_k, j, eye_k, :] = x[t - 1 - j]
            Z = Z.reshape(k, ms)
            Rx = Rb @ Z.T
            if irep >= nlag + 2:
                e = x[t] - Z @ beta_pred
                P = Q_prev + Z @ Rx
                sign, logdet = np.linalg.slogdet(P)
                if sign > 0:
                    score[t] = -0.5 * (k * LOG2PI + logdet + e @ np.linalg.solve(P, e))
            KG = Rx @ np.linalg.pinv(Q + Z @ Rx)
            beta = beta_pred + KG @ (x[t] - Z @ beta_pred)
            Sb = Rb - KG @ (Z @ Rb)
        B = _coef_matrix(beta, k, nlag)
        Bc = _companion(B, k, nlag)
        if not (irep == 1 or np.max(np.abs(np.linalg.eigvals(Bc))) <= 1):
            beta = 0.99 * beta_prev
            B = _coef_matrix(beta, k, nlag)
            Bc = _companion(B, k, nlag)
        B_t[t] = B
        Q_t[t] = Q
        beta_prev, Sb_prev, Q_prev = beta, Sb, Q
    return {"B_t": B_t, "Q_t": Q_t, "score": score}
