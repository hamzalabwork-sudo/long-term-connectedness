"""Frequency connectedness as a functional of second-order structure.

Population (H -> infinity) version of the Barunik-Krehlik decomposition of generalized
connectedness, with the band normalization of ConnectednessApproach ("ABS": the bands add up
to total frequency connectedness):

    theta_jl(w) = |[Psi(w) Sigma]_jl|^2 / Sigma_ll,      Psi(w) = (I - sum_m A_m e^{-imw})^{-1}
    C_b         = 100/k * sum_{j != l} int_b theta_jl / sum_l int_{-pi}^{pi} theta_jl

All functions work with population objects (VAR coefficients, autocovariances, spectra), so
that the pseudo-true value of an estimator is computed exactly rather than simulated:
  - a (local) VAR(p) fitted by least squares converges to the Yule-Walker projection of the
    autocovariances it sees (`yule_walker`);
  - full-sample centring of a series whose local level differs from the centring constant by d
    adds d d' to every autocovariance (`offset_autocov`);
  - causal EWMA centring applies the filter 1 - G(L), G(L) = (1-lam) sum_{j>=1} lam^{j-1} L^j
    (`ewma_filter_autocov`).

Bands are given by true cycle lengths (in observations): edges [P_1 < P_2 < ...] give bands
(cycles shorter than P_1), [P_1, P_2), ..., (longer than P_last), ordered short to long.
"""
import numpy as np


# ---------------------------------------------------------------------------- VAR objects
def companion(A):
    """A: (p, k, k) lag matrices -> companion (kp, kp)."""
    p, k, _ = A.shape
    F = np.zeros((k * p, k * p))
    F[:k] = np.concatenate(list(A), axis=1)
    if p > 1:
        F[k:, :-k] = np.eye(k * (p - 1))
    return F


def is_stable(A):
    return np.max(np.abs(np.linalg.eigvals(companion(A)))) < 1


def var_autocov(A, Sigma, maxlag):
    """Autocovariances Gamma_0..Gamma_maxlag (maxlag+1, k, k), Gamma_j = E x_t x_{t-j}'."""
    p, k, _ = A.shape
    F = companion(A)
    Q = np.zeros((k * p, k * p))
    Q[:k, :k] = Sigma
    # vec(V) = (I - F kron F)^{-1} vec(Q)
    n = k * p
    V = np.linalg.solve(np.eye(n * n) - np.kron(F, F), Q.reshape(-1)).reshape(n, n)
    G = np.zeros((maxlag + 1, k, k))
    Fj = np.eye(n)
    for j in range(maxlag + 1):
        G[j] = (Fj @ V)[:k, :k]
        Fj = F @ Fj
    return G


def var_spectrum(A, Sigma, w):
    """Spectral density f(w) = Psi Sigma Psi^* / (2 pi) on the grid w: (n, k, k) complex."""
    Psi = psi(A, w)
    return Psi @ Sigma @ np.conj(np.swapaxes(Psi, -1, -2)) / (2 * np.pi)


def psi(A, w):
    p, k, _ = A.shape
    z = np.exp(-1j * np.outer(w, np.arange(1, p + 1)))                 # (n, p)
    Aw = np.eye(k) - np.einsum("np,pij->nij", z, A)
    return np.linalg.inv(Aw)


# ---------------------------------------------------------------- spectra <-> autocovariances
def grid(n):
    """n equally spaced frequencies on [0, 2pi), the DFT grid."""
    return 2 * np.pi * np.arange(n) / n


def autocov_from_spectrum(f, maxlag):
    """f: (n, k, k) on grid(n) -> Gamma_0..Gamma_maxlag, Gamma_j = int f(w) e^{ijw} dw."""
    n = f.shape[0]
    G = np.fft.ifft(f, axis=0) * 2 * np.pi            # sum_w f(w) e^{i j w} * (2pi/n)
    G = np.real(G[:maxlag + 1])
    return G


def yule_walker(G, p):
    """Least-squares VAR(p) projection from autocovariances G (>= p+1 lags).
    Returns A (p, k, k) and innovation covariance Sigma."""
    k = G.shape[1]

    def gam(j):                                       # E x_t x_{t-j}', negative j by symmetry
        return G[j] if j >= 0 else G[-j].T
    R = np.block([[gam(j - i) for j in range(p)] for i in range(p)])  # E X X', X = [x_{t-1};..]
    r = np.concatenate([gam(i + 1) for i in range(p)], axis=1)          # E x_t X'
    B = np.linalg.solve(R.T, r.T).T                                     # (k, kp)
    Sigma = G[0] - B @ r.T
    A = B.reshape(k, p, k).transpose(1, 0, 2)
    return A, (Sigma + Sigma.T) / 2


def offset_autocov(G, d):
    """Autocovariances (second moments) of d + x_t for a constant offset d: G_j + d d'."""
    return G + np.outer(d, d)[None]


def ewma_transfer(w, halflife):
    """1 - G(e^{-iw}) for the causal EWMA of past values with the given half-life."""
    lam = 0.5 ** (1.0 / halflife)
    z = np.exp(-1j * w)
    return (1 - z) / (1 - lam * z)


def ewma_filter_autocov(f, halflife, maxlag):
    """Autocovariances of (1 - G(L)) y_t, given the spectrum f of y on grid(n)."""
    w = grid(f.shape[0])
    h2 = np.abs(ewma_transfer(w, halflife)) ** 2
    return autocov_from_spectrum(f * h2[:, None, None], maxlag)


# ------------------------------------------------------------------- frequency connectedness
def band_masks(w, edges):
    """Boolean masks (nb, n) for bands defined by true cycle lengths `edges` (observations),
    ordered from short to long cycles. Uses |w| folded to [0, pi]."""
    a = np.minimum(w % (2 * np.pi), 2 * np.pi - w % (2 * np.pi))
    cuts = [np.pi + 1e-12] + [2 * np.pi / P for P in sorted(edges)] + [-1e-12]
    return np.array([(a < hi) & (a >= lo) for hi, lo in zip(cuts[:-1], cuts[1:])])


def freq_connectedness(A, Sigma, edges, n=2 ** 14):
    """Population frequency connectedness of a VAR. Returns dict with
    CT (nb, k, k) in percent, TCI (nb,), total (time-domain H=inf generalized TCI)."""
    w = grid(n)
    PS = psi(A, w) @ Sigma
    th = np.abs(PS) ** 2 / np.diag(Sigma)[None, None, :]               # (n, k, k)
    tot = th.sum(0).sum(-1)                                             # (k,)
    masks = band_masks(w, edges)
    k = Sigma.shape[0]
    off = 1 - np.eye(k)
    CT = np.array([100 * th[m].sum(0) / tot[:, None] for m in masks])
    TCI = (CT * off).sum((-1, -2)) / k
    return {"CT": CT, "TCI": TCI, "total": TCI.sum()}


def conn_from_autocov(G, p, edges, n=2 ** 14):
    """Connectedness of the VAR(p) projection of autocovariances G (a pseudo-true value)."""
    A, S = yule_walker(G, p)
    return freq_connectedness(A, S, edges, n)


def conn_from_spectrum(f, p, edges, n=2 ** 14):
    """Connectedness of a process given by its spectrum, through a long VAR(p) projection
    (converges to the Wold-based value as p grows when f is regular)."""
    return conn_from_autocov(autocov_from_spectrum(f, p + 1), p, edges, n)
