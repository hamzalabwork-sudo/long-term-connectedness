"""TVP-VAR with generalized time- and frequency-domain connectedness.

Follows the R package ConnectednessApproach (Gabauer; TVPVAR.R, BayesPrior.R, IRF.R,
FEVD.R, FrequencyConnectedness.R, ConnectednessTable.R) step by step, so that output can be
checked against it, and adds a real-time mode for forecasting use:

  mode       demeaning                          prior (bprior, Vprior, Q)
  "package"  full-sample means (TVPVAR.R:69)    BayesPrior on the first `prior_size` rows
                                                 (package default: all rows)
  "realtime" means of the first `prior_size`    BayesPrior on the first `prior_size` rows
             rows only

The Kalman filter is forward only (no smoother) in both modes, so in "realtime" mode B_t and
Q_t depend on data up to t plus the initial window.

Memory: the package stores the m x m state covariance for every t (m = nlag * k^2), which is
~236 GB per array at k=16, nlag=3, T=44k. Only the current one is kept here.
"""
import math

import numpy as np
import torch


# ----------------------------------------------------------------------------- prior

def bayes_prior(x, nlag, size=None, intercept=False):
    """OLS VAR(nlag) with intercepts on x[:size] (BayesPrior.R). Returns (bprior, Vprior, Q).

    Coefficient order follows the package: for lag j, equation e, variable v the index is
    j*k*k + e*k + v. Vprior is the coefficient block of ginv(sum Z'Z), not scaled by Q.
    The package drops the intercepts; intercept=True keeps them as the first k entries.
    """
    x = np.asarray(x, float)
    k = x.shape[1]
    size = x.shape[0] if size is None else size
    n = size - nlag
    Y = x[nlag:size]
    W = np.hstack([np.ones((n, 1))] + [x[nlag - j - 1:size - j - 1] for j in range(nlag)])
    A = np.linalg.pinv(W.T @ W)
    C = A @ (W.T @ Y)                                  # (1 + nlag*k, k): column e = equation e
    U = Y - W @ C
    Q = U.T @ U / n
    # parameter p belongs to equation eq[p] and regressor reg[p] (0 = constant)
    eq, reg = ([*range(k)], [0] * k) if intercept else ([], [])
    for j in range(nlag):
        for e in range(k):
            eq += [e] * k
            reg += [1 + j * k + v for v in range(k)]
    eq, reg = np.array(eq), np.array(reg)
    b = C[reg, eq]
    V = (eq[:, None] == eq[None, :]) * A[reg[:, None], reg[None, :]]
    return b, V, Q


# ----------------------------------------------------------------------------- TVP-VAR

def _coef_matrix(beta, k, nlag):
    """beta (j, e, v) -> B (k, k*nlag) with B[e, j*k + v] (the package's `biga`)."""
    return beta.reshape(nlag, k, k).transpose(1, 0, 2).reshape(k, nlag * k)


def _companion(B, k, nlag):
    if nlag == 1:
        return B
    low = np.hstack([np.eye(k * (nlag - 1)), np.zeros((k * (nlag - 1), k))])
    return np.vstack([B, low])


def causal_ewma_mean(x, halflife, init):
    """mu_t = EWMA of x_1..x_{t-1} (so x_t - mu_t uses no information from t onwards)."""
    lam = 0.5 ** (1 / halflife)
    mu = np.empty_like(x)
    m = np.asarray(init, float)
    for t in range(len(x)):
        mu[t] = m
        m = lam * m + (1 - lam) * x[t]
    return mu


def tvp_var(x, nlag, kappa1, kappa2, mode="package", prior_size=None, prior=None,
            demean=None, demean_halflife=None, intercept=False, verbose=False):
    """Kalman-filtered TVP-VAR (TVPVAR.R). Returns dict with B_t (T, k, k*nlag), Q_t (T, k, k)
    and `unstable` (T,) flags where the package's stability fallback was applied.

    demean: "full" (package), "window" (first prior_size rows) or "ewma" (causal EWMA mean with
    half-life `demean_halflife` hours, started from the window mean). Default follows `mode`.
    intercept: add a time-varying intercept to the state (not in the package), so that a
    drifting level is absorbed by the model instead of by a centring constant. The intercepts
    do not enter the moving-average representation, hence not the connectedness measures.
    """
    x = np.asarray(x, float)
    T, k = x.shape
    if prior is None:
        prior = bayes_prior(x, nlag, prior_size, intercept=intercept)
    b0, V0, Q0 = prior
    ni = k if intercept else 0
    if demean is None:
        demean = {"package": "full", "realtime": "window"}[mode]
    if demean in ("window", "ewma") and prior_size is None:
        raise ValueError(f"demean={demean} needs prior_size")
    if demean == "full":
        mu = x.mean(0)
    elif demean == "window":
        mu = x[:prior_size].mean(0)
    elif demean == "ewma":
        mu = causal_ewma_mean(x, demean_halflife, x[:prior_size].mean(0))
    else:
        raise ValueError(demean)
    x = x - mu
    ms = nlag * k * k
    m = ni + ms
    eye_k = np.arange(k)

    B_t = np.zeros((T, k, k * nlag))
    C_t = np.zeros((T, k))
    Q_t = np.zeros((T, k, k))
    unstable = np.zeros(T, bool)
    beta_prev = Sb_prev = Q_prev = Bc = None
    c = np.zeros(k)

    for t in range(T):                      # t = irep - 1 in the package's 1-based loop
        irep = t + 1
        # variance of the measurement equation (EWMA of one-step errors)
        if irep == 1:
            Q = Q0.copy()
        else:
            if irep <= nlag + 1:
                G = 0.1 * np.outer(x[t], x[t])
            else:
                lags = x[t - nlag:t][::-1].reshape(-1)          # [x_{t-1}, ..., x_{t-nlag}]
                u = x[t] - c - Bc[:k, :k * nlag] @ lags
                G = np.outer(u, u)
            Q = kappa2 * Q_prev + (1 - kappa2) * G
        # state prediction
        if irep <= nlag + 1:
            beta_pred, Rb = b0.copy(), V0.copy()
            beta = beta_pred
            Sb = np.zeros((m, m)) if irep <= nlag else None
        else:
            beta_pred, Rb = beta_prev, Sb_prev / kappa1
        # state update
        if irep >= nlag + 1:
            Z = np.zeros((k, nlag, k, k))
            for j in range(nlag):
                Z[eye_k, j, eye_k, :] = x[t - 1 - j]
            Z = Z.reshape(k, ms)
            if intercept:
                Z = np.hstack([np.eye(k), Z])
            Rx = Rb @ Z.T
            KG = Rx @ np.linalg.pinv(Q + Z @ Rx)
            beta = beta_pred + KG @ (x[t] - Z @ beta_pred)
            Sb = Rb - KG @ (Z @ Rb)
        # coefficients, with the package's stability fallback
        c = beta[:ni] if intercept else c
        B = _coef_matrix(beta[ni:], k, nlag)
        Bc = _companion(B, k, nlag)
        if irep == 1 or np.max(np.abs(np.linalg.eigvals(Bc))) <= 1:
            B_t[t] = B
            C_t[t] = c
        else:
            B_t[t] = B_t[t - 1]
            C_t[t] = C_t[t - 1]
            beta = 0.99 * beta_prev
            unstable[t] = True
        Q_t[t] = Q
        beta_prev, Sb_prev, Q_prev = beta, Sb, Q
        if verbose and (t % 5000 == 0):
            print(f"  tvp_var t={t}/{T}", flush=True)
    return {"B_t": B_t, "C_t": C_t, "Q_t": Q_t, "unstable": unstable, "mu": mu}


# ----------------------------------------------------------------------------- FEVD

def band_indices(partition, n):
    """frequencyConnectedness::getPartition(partition, n - 1) as 0-based index arrays.

    `partition` is decreasing in radians, e.g. [pi + 1e-5, pi/8, pi/24, 0]; band i holds
    the DFT bins of an n-point transform whose (package-mapped) frequency is in
    [partition[i+1], partition[i]).
    """
    half = n // 2
    space = np.arange(half + 1) / half * math.pi
    out = []
    for up, down in zip(partition[:-1], partition[1:]):
        o = ((space >= down) & (space < up)).astype(int)
        if n % 2 == 0:
            o = np.concatenate([o, o[1:-1][::-1]])
        else:
            o = np.concatenate([o, o[1:][::-1]])
        idx = np.flatnonzero(o)
        if len(idx) == 0:
            raise ValueError(f"empty band ({down}, {up}) for n={n}")
        out.append(idx)
    return out


def partition_from_periods(periods_hours, package_convention=False):
    """Band edges in hours -> decreasing radian partition.

    True periods use omega = 2*pi/P. The package convention (used to label bands in
    ConnectednessApproach output) is omega = pi/P, i.e. a band labelled "1-8" covers true
    cycles of 2-16 observations.
    """
    c = math.pi if package_convention else 2 * math.pi
    return [math.pi + 1e-5] + [c / p for p in sorted(periods_hours)] + [0.0]


def _table(ct):
    """Connectedness measures from row-normalized FEVD shares ct (..., k, k) in percent."""
    k = ct.shape[-1]
    off = ct * (1 - torch.eye(k, dtype=ct.dtype, device=ct.device))
    to = off.sum(-2)
    frm = off.sum(-1)
    return {"TO": to, "FROM": frm, "NET": to - frm, "TCI": to.mean(-1)}


@torch.no_grad()
def connectedness(B_t, Q_t, nfore, partition=None, device="cuda", chunk=None, dtype=torch.float64):
    """Generalized connectedness, time domain (FEVD.R type='time') and, if `partition` is
    given, frequency domain (FrequencyConnectedness.R, scenario 'ABS').

    Returns numpy arrays: time CT (T, k, k) in percent and TCI/TO/FROM/NET; frequency CT
    (T, nb, k, k) and TCI (T, nb), TO/FROM/NET (T, nb, k), one slice per band, plus the
    band sum as the frequency-domain total.
    """
    T, k, kp = B_t.shape
    nlag = kp // k
    n = nfore + 1
    if chunk is None:   # ~1.5 GB peak on the device (about 56 bytes per Psi element)
        chunk = max(16, int(4e8 // (n * k * k * 16)))
    bands = band_indices(partition, n) if partition is not None else None
    res = {"time_CT": np.zeros((T, k, k), np.float32)}
    if bands is not None:
        res["freq_CT"] = np.zeros((T, len(bands), k, k), np.float32)

    eye = torch.eye(k, dtype=dtype, device=device)
    for s in range(0, T, chunk):
        B = torch.as_tensor(B_t[s:s + chunk], dtype=dtype, device=device)
        S = torch.as_tensor(Q_t[s:s + chunk], dtype=dtype, device=device)
        c = B.shape[0]
        Phi = B.reshape(c, k, nlag, k).permute(0, 2, 1, 3)             # Phi[:, j] = lag j+1
        Psi = torch.empty((c, n, k, k), dtype=dtype, device=device)
        Psi[:, 0] = eye
        for h in range(1, n):
            acc = Phi[:, 0] @ Psi[:, h - 1]
            for j in range(1, min(h, nlag)):
                acc = acc + Phi[:, j] @ Psi[:, h - 1 - j]
            Psi[:, h] = acc
        sd = torch.diagonal(S, dim1=-2, dim2=-1)                        # (c, k)

        # time domain: horizons 0..nfore-1
        PS = Psi[:, :nfore] @ S.unsqueeze(1)                            # (c, nfore, k, k)
        theta = (PS ** 2).sum(1) / sd.unsqueeze(-2)                     # [j, l] / Sigma_ll
        res["time_CT"][s:s + c] = (100 * theta / theta.sum(-1, keepdim=True)).float().cpu().numpy()

        if bands is not None:
            # frequency domain: DFT over horizons 0..nfore
            M = torch.fft.fft(Psi, dim=1)                               # (c, n, k, k) complex
            MS = M @ S.to(M.dtype).unsqueeze(1)
            thf = (MS.abs() ** 2) / sd.unsqueeze(1).unsqueeze(-2)       # (c, n, k, k)
            all_idx = np.unique(np.concatenate(bands))
            tot = thf[:, all_idx].sum(1).sum(-1)                        # (c, k)
            for b, idx in enumerate(bands):
                ctb = thf[:, idx].sum(1) / tot.unsqueeze(-1)
                res["freq_CT"][s:s + c, b] = (100 * ctb).float().cpu().numpy()
            del M, MS, thf
        del Psi, PS

    tm = _table(torch.as_tensor(res["time_CT"], dtype=torch.float64))
    for key, val in tm.items():
        res[f"time_{key}"] = val.numpy()
    if bands is not None:
        fm = _table(torch.as_tensor(res["freq_CT"], dtype=torch.float64))
        for key, val in fm.items():
            res[f"freq_{key}"] = val.numpy()                            # band-specific
            res[f"freqtotal_{key}"] = val.sum(1).numpy()                # ABS: bands add up
    return res
