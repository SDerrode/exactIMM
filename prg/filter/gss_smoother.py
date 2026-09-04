#!/usr/bin/env python3
"""
prg/filter/gss_smoother.py
==========================
Exact fixed-interval smoothers for the GSS model on the exactness domains
(the smoothing companion of the filtering CNS paper, docs/CNS-exactness).

On the two uniform families where finite-dimensional *filtering* is exact —
``{C≡0}`` (CGO-MSM) and the AB constraint (NGH-MSM), plus the intermediate
slaving-(A) family ``{A≡MC}`` — fixed-interval smoothing is also exact with an
O(N K²) two-pass algorithm, but the *mechanism* differs per family:

``regime_smoother(params, ys, kernel=...)``
    Smoothed regime posteriors ``p(r_n | y_{1:N})`` and pair posteriors
    ``p(r_n, r_{n+1} | y_{1:N})`` from an HMM forward-backward on the observed
    chain: under AB the pair (R, Y) is Markov with kernel ``Q`` (kernel
    ``"ab"``); under C≡0 the Y-chain is Markov on its own (kernel ``"c0"``);
    under slaving (A) only, the lag-1 chain (R, (Y_{n-1}, Y_n)) is Markov
    (kernel ``"lag1"``). Exact on the matching family.

``reweight_smoother(params, ys)``
    The ``{C≡0}`` state smoother: **reweighting**. Because the future informs
    the state only through the regimes (p(x_n | r_{1:n}, y_{1:N}) =
    p(x_n | r_{1:n}, y_{1:n}) when C≡0), the smoothed law is the *filtered*
    per-regime summaries reweighted by the smoothed regime posterior — no
    backward state correction at all. Exact (mean and variance) iff C≡0.

``constant_gain_smoother(params, ys)``
    The AB state smoother: **one-step constant-gain correction**. Under AB the
    fresh state noise ξ_n influences the future only through Y_{n+1}, so the
    exact backward correction is one step and pair-indexed, with gains

        W_{jk} = Γ_j C_kᵀ (C_k Γ_j C_kᵀ + Σ_V(k))⁻¹

    precomputed from the model blocks (no covariance recursion):

        X̂_{n|N} = Σ_{j,k} γ_n(j,k) [ M_j y_n + c_j + W_{jk} (y_{n+1} − m_{jk}(y_n)) ]

    with γ_n(j,k) the smoothed pairs and, at n = N, the filtered read-out.
    Exact iff AB holds (uniformly over regimes).

``lag1_constant_gain_smoother(params, ys)``
    The slaving-(A) extension on ``{A≡MC}``: same one-step correction on the
    lag-1 chain, with read-out ``M_j y_n + (B_j − M_j D_j) y_{n−1} + c_j`` and
    the same gains. Reduces to ``constant_gain_smoother`` when B = MD too.

The two state mechanisms are *not* interchangeable: reweighting with the exact
smoothed γ fails under AB (the backward ξ-correction is the essential part,
not a refinement), and the constant-gain read-out has no meaning under C≡0.
Each smoother is exact exactly on its family — the same domains as filtering.

All smoothers are batch functions on ``(params, ys)`` mirroring
:mod:`prg.experiments.reference_filters`; the observation model is the exact
read-out ``Y_n = H Z_n`` with ``H = [0, I_s]``. The first time slice uses the
initial per-regime law (no stationarity assumption). Exogenous inputs
(``params.p > 0`` with a driven simulation) are not supported yet.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "regime_smoother",
    "reweight_smoother",
    "constant_gain_smoother",
    "lag1_constant_gain_smoother",
    "chain_log_likelihood",
    "family_lrt",
    "SMOOTHER_KERNELS",
]

SMOOTHER_KERNELS = ("ab", "c0", "lag1")

_LOG_2PI = float(np.log(2.0 * np.pi))


# ---------------------------------------------------------------------------
# Small local helpers (kept local: prg.filter must not import prg.experiments)
# ---------------------------------------------------------------------------
def _obs_matrix(q: int, s: int) -> np.ndarray:
    """H = [0_{s×q}, I_s] — selects Y from Z = [X; Y]."""
    return np.hstack([np.zeros((s, q)), np.eye(s)])


def _gauss_logpdf(nu: np.ndarray, S: np.ndarray) -> float:
    """log N(nu; 0, S) for a column vector nu and SPD S."""
    s = nu.shape[0]
    sign, logdet = np.linalg.slogdet(S)
    if sign <= 0:
        S = S + 1e-12 * np.eye(s)
        sign, logdet = np.linalg.slogdet(S)
    quad = float((nu.T @ np.linalg.solve(S, nu)).item())
    return -0.5 * (s * _LOG_2PI + logdet + quad)


def _kalman_exact_y_update(
    z_pred: np.ndarray, P_pred: np.ndarray, y: np.ndarray, H: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """One exact-observation Kalman update; returns (z_post, P_post, log_lik)."""
    nu = y - H @ z_pred
    S = H @ P_pred @ H.T
    log_lik = _gauss_logpdf(nu, S)
    Kg = P_pred @ H.T @ np.linalg.inv(S)
    z_post = z_pred + Kg @ nu
    P_post = (np.eye(P_pred.shape[0]) - Kg @ H) @ P_pred
    return z_post, 0.5 * (P_post + P_post.T), log_lik


def _logsumexp(a: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    m = float(a.max())
    if not np.isfinite(m):
        return m
    return m + float(np.log(np.exp(a - m).sum()))


def _pi0(params) -> np.ndarray:
    pi0 = params.pi0
    if pi0 is None:
        return params.stationary_distribution()
    return np.asarray(pi0, dtype=float).ravel()


def _blocks(params):
    """Per-regime blocks (A, B, C, D, Σ_V, bX, bY) as lists of arrays."""
    K, q, s = params.K, params.q, params.s
    A = [params.f_matrix.A(k) for k in range(K)]
    B = [params.f_matrix.B(k) for k in range(K)]
    C = [params.f_matrix.C(k) for k in range(K)]
    D = [params.f_matrix.D(k) for k in range(K)]
    SV = [params.noise_cov.Sigma_V(k) for k in range(K)]
    b = [params.b(k).reshape(q + s, 1) for k in range(K)]
    bX = [bk[:q] for bk in b]
    bY = [bk[q:] for bk in b]
    return A, B, C, D, SV, bX, bY


def _slaving(params):
    """Per-regime slaving quantities M_k = Δ_k Σ_V(k)⁻¹, Γ_k (Schur complement)
    and read-out offset c_k = bX_k − M_k bY_k."""
    K, q, s = params.K, params.q, params.s
    M, Gam, c = [], [], []
    for k in range(K):
        SV = params.noise_cov.Sigma_V(k)
        Dt = params.noise_cov.Delta(k)
        SU = params.noise_cov.Sigma_U(k)
        Mk = np.linalg.solve(SV.T, Dt.T).T  # Δ Σ_V⁻¹, via a solve
        bk = params.b(k).reshape(q + s, 1)
        M.append(Mk)
        Gam.append(SU - Mk @ SV @ Mk.T)
        c.append(bk[:q] - Mk @ bk[q:])
    return M, Gam, c


def _first_slice(params, ys):
    """Initial slice: per-regime x-block posterior at n = 0 and log α_1.

    Returns ``(m0 (K,q,1), P0 (K,q,q), la (K,))`` with ``m0[k], P0[k]`` the
    moments of p(X_1 | r_1 = k, y_1) from the initial law (μ_z0, Σ_z0), and
    ``la[k] = log π_0(k) + log p(y_1 | r_1 = k)`` (unnormalized log-forward).
    """
    K, q, s = params.K, params.q, params.s
    dim = q + s
    H = _obs_matrix(q, s)
    pi0 = _pi0(params)
    y0 = np.asarray(ys, dtype=float).reshape(-1, s)[0].reshape(s, 1)
    m0 = np.zeros((K, q, 1))
    P0 = np.zeros((K, q, q))
    la = np.zeros(K)
    for k in range(K):
        z, P, ll = _kalman_exact_y_update(
            params.mu_z0(k).reshape(dim, 1), params.Sigma_z0(k), y0, H
        )
        m0[k], P0[k] = z[:q], P[:q, :q]
        la[k] = float(np.log(pi0[k] + 1e-300) + ll)
    return m0, P0, la


# ---------------------------------------------------------------------------
# Scaled forward-backward on the observed chain
# ---------------------------------------------------------------------------
def _forward_backward(la, log_ker):
    """Forward-backward from log α_1 (K,) and log kernels (N−1, K, K).

    Returns ``(gamma (N,K), gamma_pair (N−1,K,K), log_lik)`` with
    γ_n(k) = p(r_n = k | y_{1:N}), γ_n(j,k) = p(r_n = j, r_{n+1} = k | y_{1:N})
    and log_lik = log p(y_{1:N}). Per-step rescaling keeps everything finite;
    the common factors cancel in γ and the pairs.
    """
    Nm1, K, _ = log_ker.shape
    N = Nm1 + 1
    lse0 = _logsumexp(la)
    log_lik = lse0
    alpha = np.zeros((N, K))
    alpha[0] = np.exp(la - lse0)
    Ker = np.zeros_like(log_ker)
    for n in range(Nm1):
        m = float(log_ker[n].max())
        Ker[n] = np.exp(log_ker[n] - m)
        v = alpha[n] @ Ker[n]
        cn = float(v.sum())
        alpha[n + 1] = v / cn
        log_lik += np.log(cn) + m
    beta = np.zeros((N, K))
    beta[N - 1] = 1.0
    for n in range(N - 2, -1, -1):
        beta[n] = Ker[n] @ beta[n + 1]
        beta[n] /= beta[n].max()
    gamma = alpha * beta
    gamma /= gamma.sum(axis=1, keepdims=True)
    gamma_pair = np.zeros((Nm1, K, K))
    for n in range(Nm1):
        gp = (alpha[n][:, None] * Ker[n]) * beta[n + 1][None, :]
        gamma_pair[n] = gp / gp.sum()
    return gamma, gamma_pair, log_lik


# ---------------------------------------------------------------------------
# Kernels of the observed chain, per family
# ---------------------------------------------------------------------------
def _log_kernel_c0(params, ys_):
    """{C≡0} kernel: Q0[n,j,k] = log P[j,k] + log N(y_{n+1}; D_k y_n + bY_k, Σ_V(k)).
    The Y-chain is Markov on its own — no state read-out enters."""
    K, s = params.K, params.s
    N = ys_.shape[0]
    _, _, _, D, SV, _, bY = _blocks(params)
    logP = np.log(params.P + 1e-300)
    log_ker = np.zeros((N - 1, K, K))
    for n in range(N - 1):
        y0 = ys_[n].reshape(s, 1)
        y1 = ys_[n + 1].reshape(s, 1)
        for k in range(K):
            nu = y1 - (D[k] @ y0 + bY[k])
            lg = _gauss_logpdf(nu, SV[k])
            for j in range(K):
                log_ker[n, j, k] = logP[j, k] + lg
    return log_ker


def _readout_ab(params, ys_, m0):
    """AB read-out means: m_read[0,j] init-based, m_read[n,j] = M_j y_n + c_j."""
    K, q, s = params.K, params.q, params.s
    N = ys_.shape[0]
    M, _, c = _slaving(params)
    m_read = np.zeros((N, K, q, 1))
    m_read[0] = m0
    for n in range(1, N):
        yn = ys_[n].reshape(s, 1)
        for j in range(K):
            m_read[n, j] = M[j] @ yn + c[j]
    return m_read


def _readout_lag1(params, ys_, m0):
    """Slaving-(A) read-out means: M_j y_n + (B_j − M_j D_j) y_{n−1} + c_j (n ≥ 1)."""
    K, q, s = params.K, params.q, params.s
    N = ys_.shape[0]
    M, _, c = _slaving(params)
    _, B, _, D, _, _, _ = _blocks(params)
    E = [B[j] - M[j] @ D[j] for j in range(K)]
    m_read = np.zeros((N, K, q, 1))
    m_read[0] = m0
    for n in range(1, N):
        yp = ys_[n - 1].reshape(s, 1)
        yn = ys_[n].reshape(s, 1)
        for j in range(K):
            m_read[n, j] = M[j] @ yn + E[j] @ yp + c[j]
    return m_read


def _pair_gains(params, P_slices):
    """Per (slice t, pair j→k): innovation covariance S, gain W and corrected
    covariance P_corr of the one-step ξ-regression. Slice 0 is the init-based
    read-out covariance, slice 1 the constant Γ_j — so for n ≥ 1 the gains are
    the paper's precomputed constants W_{jk}."""
    K = params.K
    _, _, C, _, SV, _, _ = _blocks(params)
    T = len(P_slices)
    S = [[[None] * K for _ in range(K)] for _ in range(T)]
    W = [[[None] * K for _ in range(K)] for _ in range(T)]
    P_corr = [[[None] * K for _ in range(K)] for _ in range(T)]
    for t in range(T):
        for j in range(K):
            Pj = P_slices[t][j]
            for k in range(K):
                Sjk = C[k] @ Pj @ C[k].T + SV[k]
                Wjk = Pj @ C[k].T @ np.linalg.inv(Sjk)
                S[t][j][k] = Sjk
                W[t][j][k] = Wjk
                P_corr[t][j][k] = Pj - Wjk @ Sjk @ Wjk.T
    return S, W, P_corr


def _log_kernel_pairs(params, ys_, m_read, S):
    """Log kernels of the observed pair chain, from the per-regime read-out
    means ``m_read`` and the per-slice innovation covariances ``S``."""
    K, s = params.K, params.s
    N = ys_.shape[0]
    _, _, C, D, _, _, bY = _blocks(params)
    logP = np.log(params.P + 1e-300)
    log_ker = np.zeros((N - 1, K, K))
    for n in range(N - 1):
        t = min(n, 1)
        y0 = ys_[n].reshape(s, 1)
        y1 = ys_[n + 1].reshape(s, 1)
        for j in range(K):
            for k in range(K):
                nu = y1 - (C[k] @ m_read[n, j] + D[k] @ y0 + bY[k])
                log_ker[n, j, k] = logP[j, k] + _gauss_logpdf(nu, S[t][j][k])
    return log_ker


def _pair_smoother(params, ys_, m_read, P_slices, la):
    """Shared core of the constant-gain smoothers (AB and lag-1).

    ``m_read (N,K,q,1)`` are the per-regime read-out means, ``P_slices`` the
    read-out covariances per time slice (index min(n, 1)). Builds the observed
    chain kernel, runs the forward-backward, then applies the pair-indexed
    one-step correction. Returns ``(E_x, Var_x, gamma, log_lik)``.
    """
    K, q, s = params.K, params.q, params.s
    N = ys_.shape[0]
    _, _, C, D, _, _, bY = _blocks(params)
    S, W, P_corr = _pair_gains(params, P_slices)
    log_ker = _log_kernel_pairs(params, ys_, m_read, S)
    gamma, gamma_pair, log_lik = _forward_backward(la, log_ker)

    E_x = np.zeros((N, q))
    Var_x = np.zeros((N, q, q))
    for n in range(N - 1):
        t = min(n, 1)
        y0 = ys_[n].reshape(s, 1)
        y1 = ys_[n + 1].reshape(s, 1)
        means = np.zeros((K, K, q, 1))
        ex = np.zeros((q, 1))
        for j in range(K):
            for k in range(K):
                nu = y1 - (C[k] @ m_read[n, j] + D[k] @ y0 + bY[k])
                means[j, k] = m_read[n, j] + W[t][j][k] @ nu
                ex += gamma_pair[n, j, k] * means[j, k]
        vx = np.zeros((q, q))
        for j in range(K):
            for k in range(K):
                d = means[j, k] - ex
                vx += gamma_pair[n, j, k] * (P_corr[t][j][k] + d @ d.T)
        E_x[n] = ex.ravel()
        Var_x[n] = vx
    # n = N : no future — filtered read-out under the smoothed (= filtered) marginal
    t = min(N - 1, 1)
    ex = np.zeros((q, 1))
    for j in range(K):
        ex += gamma[N - 1, j] * m_read[N - 1, j]
    vx = np.zeros((q, q))
    for j in range(K):
        d = m_read[N - 1, j] - ex
        vx += gamma[N - 1, j] * (P_slices[t][j] + d @ d.T)
    E_x[N - 1] = ex.ravel()
    Var_x[N - 1] = vx
    return E_x, Var_x, gamma, log_lik


# ---------------------------------------------------------------------------
# Public smoothers
# ---------------------------------------------------------------------------
def _build_chain(params, ys_, kernel):
    """(log α_1 (K,), log kernels (N−1,K,K)) of the observed chain for a
    family kernel (``"ab"``, ``"c0"`` or ``"lag1"``)."""
    if kernel not in SMOOTHER_KERNELS:
        raise ValueError(f"unknown kernel {kernel!r}; expected one of {SMOOTHER_KERNELS}")
    m0, P0, la = _first_slice(params, ys_)
    if ys_.shape[0] == 1:
        return la, np.zeros((0, params.K, params.K))
    if kernel == "c0":
        return la, _log_kernel_c0(params, ys_)
    _, Gam, _ = _slaving(params)
    P_slices = [P0, np.stack(Gam)]
    m_read = _readout_ab(params, ys_, m0) if kernel == "ab" else _readout_lag1(params, ys_, m0)
    S, _, _ = _pair_gains(params, P_slices)
    return la, _log_kernel_pairs(params, ys_, m_read, S)


def regime_smoother(params, ys, kernel: str = "ab"):
    """Smoothed regime posteriors p(r_n | y_{1:N}) and pairs, O(N K²).

    ``kernel`` selects the observed-chain Markov structure: ``"ab"`` (pair
    (R, Y) Markov under AB), ``"c0"`` (Y-chain Markov under C≡0) or ``"lag1"``
    (lag-1 chain Markov under slaving (A)). Exact on the matching family.

    Returns ``(gamma (N,K), gamma_pair (N−1,K,K), log_lik)``.
    """
    ys_ = np.asarray(ys, dtype=float).reshape(-1, params.s)
    la, log_ker = _build_chain(params, ys_, kernel)
    if ys_.shape[0] == 1:
        gamma = np.exp(la - _logsumexp(la))[None, :]
        gamma /= gamma.sum()
        return gamma, log_ker, _logsumexp(la)
    return _forward_backward(la, log_ker)


def chain_log_likelihood(params, ys, kernel: str = "ab"):
    """Exact log p(y_{1:N}) on the matching family, by the O(N K²) forward
    pass of the observed chain (no backward pass, no state recursion).

    This is the likelihood the membership tests build on: on its family the
    value coincides with the true model evidence (e.g. with the GPB2 / exact
    filter likelihood), and it stays finite and cheap for long records.
    """
    ys_ = np.asarray(ys, dtype=float).reshape(-1, params.s)
    la, log_ker = _build_chain(params, ys_, kernel)
    if ys_.shape[0] == 1:
        return _logsumexp(la)
    lse0 = _logsumexp(la)
    log_lik = lse0
    alpha = np.exp(la - lse0)
    for n in range(log_ker.shape[0]):
        m = float(log_ker[n].max())
        v = alpha @ np.exp(log_ker[n] - m)
        cn = float(v.sum())
        alpha = v / cn
        log_lik += np.log(cn) + m
    return log_lik


def family_lrt(params_h0, params_h1, ys, kernel_h0: str = "c0", kernel_h1: str = "lag1"):
    """Exact Neyman–Pearson log likelihood ratio between two family models.

    Both hypotheses lying in exactness domains, their evidences are exactly
    computable in O(N K²): the returned statistic is

        λ(y) = log p(y_{1:N} | H1) − log p(y_{1:N} | H0),

    the log of the NP ratio (twice λ is the usual deviance scaling). Defaults
    match the map's family test: H0 in ``{C≡0}`` vs H1 in ``{A≡MC}``.
    """
    return chain_log_likelihood(params_h1, ys, kernel_h1) - chain_log_likelihood(
        params_h0, ys, kernel_h0
    )


def reweight_smoother(params, ys):
    """{C≡0} fixed-interval smoother: smoothed-regime **reweighting** of the
    *filtered* per-regime summaries — no backward state correction.

    The forward pass is the (exact-under-C≡0) IMM bank keeping the per-regime
    posterior summaries; the backward pass only re-weights them with the
    smoothed regime posterior from the ``"c0"`` forward-backward. Exact (mean
    and variance) iff C≡0; fails under AB, where the ξ-correction of
    :func:`constant_gain_smoother` is the essential ingredient.

    Returns ``(E_x (N,q), Var_x (N,q,q), gamma (N,K), log_lik)``.
    """
    K, q, s = params.K, params.q, params.s
    dim = q + s
    ys_ = np.asarray(ys, dtype=float).reshape(-1, s)
    N = ys_.shape[0]
    H = _obs_matrix(q, s)
    P = params.P
    F = [params.f_matrix.F(k) for k in range(K)]
    b = [params.b(k).reshape(dim, 1) for k in range(K)]
    SW = [params.noise_cov.Sigma_W(k) for k in range(K)]

    # forward: IMM bank of per-regime filtered summaries (exact under C≡0)
    zs, Ps = [None] * K, [None] * K
    la = np.zeros(K)
    pi0 = _pi0(params)
    y0 = ys_[0].reshape(s, 1)
    for k in range(K):
        z, Pc, ll = _kalman_exact_y_update(
            params.mu_z0(k).reshape(dim, 1), params.Sigma_z0(k), y0, H
        )
        zs[k], Ps[k] = z, Pc
        la[k] = float(np.log(pi0[k] + 1e-300) + ll)
    al = np.exp(la - la.max())
    al /= al.sum()
    xs = np.zeros((N, K, q))
    Pxs = np.zeros((N, K, q, q))
    for k in range(K):
        xs[0, k] = zs[k][:q, 0]
        Pxs[0, k] = Ps[k][:q, :q]
    for n in range(1, N):
        yn = ys_[n].reshape(s, 1)
        nzs, nPs, nla = [None] * K, [None] * K, np.zeros(K)
        for k in range(K):
            w = np.array([P[j, k] * al[j] for j in range(K)])
            cbar = float(w.sum())
            w = w / cbar if cbar > 1e-300 else np.full(K, 1.0 / K)
            z0 = sum(w[j] * zs[j] for j in range(K))
            P0 = sum(w[j] * (Ps[j] + (zs[j] - z0) @ (zs[j] - z0).T) for j in range(K))
            z_pred = F[k] @ z0 + b[k]
            P_pred = F[k] @ P0 @ F[k].T + SW[k]
            nzs[k], nPs[k], ll = _kalman_exact_y_update(z_pred, 0.5 * (P_pred + P_pred.T), yn, H)
            nla[k] = float(np.log(cbar + 1e-300) + ll)
        zs, Ps = nzs, nPs
        al = np.exp(nla - nla.max())
        al /= al.sum()
        for k in range(K):
            xs[n, k] = zs[k][:q, 0]
            Pxs[n, k] = Ps[k][:q, :q]

    # backward: smoothed regime posterior, then reweight
    if N == 1:
        gamma = np.exp(la - _logsumexp(la))[None, :]
        gamma /= gamma.sum()
        log_lik = _logsumexp(la)
    else:
        gamma, _, log_lik = _forward_backward(la, _log_kernel_c0(params, ys_))
    E_x = np.einsum("nk,nkq->nq", gamma, xs)
    Var_x = np.zeros((N, q, q))
    for n in range(N):
        for k in range(K):
            d = xs[n, k] - E_x[n]
            Var_x[n] += gamma[n, k] * (Pxs[n, k] + np.outer(d, d))
    return E_x, Var_x, gamma, log_lik


def constant_gain_smoother(params, ys):
    """AB fixed-interval smoother: pair-weighted one-step correction with
    precomputed constant gains W_{jk} — no covariance recursion, O(N K²).

    Mechanism: under AB the state is slaved to the current observation up to
    the fresh noise ξ_n ~ N(0, Γ_j), and ξ_n influences the future *only*
    through Y_{n+1}; the exact backward correction is therefore the one-step
    Gaussian regression of ξ_n on the pair innovation, weighted by the
    smoothed pair posterior. Exact (mean and variance) iff AB holds.

    Returns ``(E_x (N,q), Var_x (N,q,q), gamma (N,K), log_lik)``.
    """
    s = params.s
    ys_ = np.asarray(ys, dtype=float).reshape(-1, s)
    m0, P0, la = _first_slice(params, ys_)
    if ys_.shape[0] == 1:
        return _single_step_result(params, m0, P0, la)
    _, Gam, _ = _slaving(params)
    m_read = _readout_ab(params, ys_, m0)
    return _pair_smoother(params, ys_, m_read, [P0, np.stack(Gam)], la)


def lag1_constant_gain_smoother(params, ys):
    """Slaving-(A) fixed-interval smoother on ``{A≡MC}``: the lag-1 extension.

    Same pair-indexed one-step correction as :func:`constant_gain_smoother`,
    on the lag-1 observed chain (R_n, (Y_{n−1}, Y_n)) with read-out
    ``M_j y_n + (B_j − M_j D_j) y_{n−1} + c_j`` and the same gains W_{jk}.
    Exact on ``{A≡MC}`` (volet B free); coincides with the AB smoother when
    B = MD as well.

    Returns ``(E_x (N,q), Var_x (N,q,q), gamma (N,K), log_lik)``.
    """
    s = params.s
    ys_ = np.asarray(ys, dtype=float).reshape(-1, s)
    m0, P0, la = _first_slice(params, ys_)
    if ys_.shape[0] == 1:
        return _single_step_result(params, m0, P0, la)
    _, Gam, _ = _slaving(params)
    m_read = _readout_lag1(params, ys_, m0)
    return _pair_smoother(params, ys_, m_read, [P0, np.stack(Gam)], la)


def _single_step_result(params, m0, P0, la):
    """N = 1 degenerate case: smoothing is filtering on the first slice."""
    q = params.q
    gamma = np.exp(la - _logsumexp(la))[None, :]
    gamma /= gamma.sum()
    ex = np.zeros((q, 1))
    for k in range(params.K):
        ex += gamma[0, k] * m0[k]
    vx = np.zeros((q, q))
    for k in range(params.K):
        d = m0[k] - ex
        vx += gamma[0, k] * (P0[k] + d @ d.T)
    return ex.T.copy(), vx[None, :, :], gamma, _logsumexp(la)
