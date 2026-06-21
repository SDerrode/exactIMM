#!/usr/bin/env python3
"""
prg/experiments/reference_filters.py
====================================
Reference / baseline filters used by the simulation study, for comparison
against the proposed fast exact filter (``GSSFilter`` mode ``h5_exact``).

The observation model is exact: ``Y_n`` is the lower ``s`` block of the state
``Z_n = [X_n; Y_n]``, observed without noise. Each filter below conditions on
``Y_{1:n}`` via the degenerate measurement ``Y_n = H Z_n`` with ``H = [0, I_s]``
(no measurement noise), which is well posed because Σ_V(k) ≻ 0.

Filters
-------
exact_mixture_filter(params, ys)
    The **exact** Bayesian filter: a hypothesis tree over all Kᴺ regime
    histories (no pruning), each a conditional Kalman filter, recombined by
    posterior weight. Ground truth for short sequences (cost ~Kᴺ).
single_kalman_filter(params, ys)
    A single Kalman filter on the π-stationary-averaged linear model — the
    natural "ignore the switching" baseline.
oracle_filter(params, rs, ys)
    A Kalman filter that switches with the *known* regime sequence ``rs`` — the
    best achievable X-estimate (regimes given). Lower bound on the error.

Helpers
-------
stationary_moments(params) -> (mu_list, Sigma_list)
    Per-regime stationary mean/covariance E[Z|r=k], Var[Z|r=k] (the prior the
    exact-under-(H5) filter starts from).
with_stationary_init(params) -> GSSParams
    Copy of ``params`` whose μ_z0/Σ_z0 are the stationary moments, so the exact
    filter and ``h5_exact`` share the same prior.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "exact_mixture_filter",
    "single_kalman_filter",
    "oracle_filter",
    "stationary_moments",
    "with_stationary_init",
]


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
    return -0.5 * (s * np.log(2.0 * np.pi) + logdet + quad)


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


# ---------------------------------------------------------------------------
# Stationary moments and stationary-init params
# ---------------------------------------------------------------------------
def stationary_moments(params) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Per-regime stationary E[Z|r=k] and Var[Z|r=k] (fixed point of the
    time-reversed moment recursion used by ``GSSFilter``)."""
    import warnings

    from prg.filter.gss_filter import GSSFilter

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        filt = GSSFilter(params, mode="h5_exact")
    mu = [m.copy() for m in filt._mu_z]
    Sigma = [S.copy() for S in filt._Sigma]
    return mu, Sigma


def with_stationary_init(params):
    """Return a copy of ``params`` whose initial law is the stationary one."""
    from prg.classes.GSSParams import GSSParams

    mu, Sigma = stationary_moments(params)
    K = params.K
    return GSSParams(
        K=K,
        q=params.q,
        s=params.s,
        P=params.P,
        f_matrix=params.f_matrix,
        noise_cov=params.noise_cov,
        pi0=params.pi0,
        mu_z0_list=mu,
        Sigma_z0_list=[0.5 * (S + S.T) for S in Sigma],
        b_list=[params.b(k) for k in range(K)],
    )


# ---------------------------------------------------------------------------
# Exact Bayesian filter — Kᴺ hypothesis tree (ground truth)
# ---------------------------------------------------------------------------
def _collect(hyps, q, K):
    """Normalise hypothesis weights; return (E_x (q,), Var_x (q,q), pi (K,))."""
    logws = np.array([h["logw"] for h in hyps])
    w = np.exp(logws - logws.max())
    w /= w.sum()
    ex = np.zeros((q, 1))
    for wi, h in zip(w, hyps):
        ex += wi * h["z"][:q]
    vx = np.zeros((q, q))
    for wi, h in zip(w, hyps):
        d = h["z"][:q] - ex
        vx += wi * (h["P"][:q, :q] + d @ d.T)
    pr = np.zeros(K)
    for wi, h in zip(w, hyps):
        pr[h["r"]] += wi
    return ex.ravel(), vx, pr


def exact_mixture_filter(params, ys):
    """Exact filtered posterior over all Kᴺ regime histories (no pruning).

    Returns ``(E_x, Var_x, pi)`` with shapes ``(N, q)``, ``(N, q, q)``,
    ``(N, K)``: ``E[X_n|Y_{1:n}]``, ``Var[X_n|Y_{1:n}]`` and ``p(r_n|Y_{1:n})``.
    Cost grows as Kᴺ — use short sequences (N ≲ 12 for K=2).
    """
    K, q, s = params.K, params.q, params.s
    dim = q + s
    ys = np.asarray(ys, dtype=float).reshape(-1, s)
    N = ys.shape[0]
    P = params.P
    pi0 = params.pi0
    F = [params.f_matrix.F(k) for k in range(K)]
    b = [params.b(k).reshape(dim, 1) for k in range(K)]
    SW = [params.noise_cov.Sigma_W(k) for k in range(K)]
    mu0 = [params.mu_z0(k).reshape(dim, 1) for k in range(K)]
    Sig0 = [params.Sigma_z0(k) for k in range(K)]
    H = _obs_matrix(q, s)

    E_x = np.zeros((N, q))
    Var_x = np.zeros((N, q, q))
    pis = np.zeros((N, K))

    # n = 0 : prior p(Z_1|r_1) = N(mu0, Sig0), weight pi0; then observe Y_1.
    y1 = ys[0].reshape(s, 1)
    hyps = []
    for k in range(K):
        z, Pc, ll = _kalman_exact_y_update(mu0[k], Sig0[k], y1, H)
        hyps.append({"z": z, "P": Pc, "logw": float(np.log(pi0[k] + 1e-300) + ll), "r": k})
    E_x[0], Var_x[0], pis[0] = _collect(hyps, q, K)

    # n >= 1 : branch each hypothesis by the K next regimes.
    for n in range(1, N):
        yn = ys[n].reshape(s, 1)
        new = []
        for h in hyps:
            j = h["r"]
            for k in range(K):
                z_pred = F[k] @ h["z"] + b[k]
                P_pred = F[k] @ h["P"] @ F[k].T + SW[k]
                z, Pc, ll = _kalman_exact_y_update(z_pred, P_pred, yn, H)
                logw = h["logw"] + float(np.log(P[j, k] + 1e-300)) + ll
                new.append({"z": z, "P": Pc, "logw": logw, "r": k})
        hyps = new
        E_x[n], Var_x[n], pis[n] = _collect(hyps, q, K)

    return E_x, Var_x, pis


# ---------------------------------------------------------------------------
# Single Kalman on the π-averaged model ("ignore the switching")
# ---------------------------------------------------------------------------
def single_kalman_filter(params, ys):
    """Kalman filter on the stationary-π-averaged linear model.

    Uses F̄ = Σ_k π_k F_k, Σ̄_W = Σ_k π_k Σ_W(k), b̄ = Σ_k π_k b(k), started from
    the averaged stationary moment — a fair "no regime modelling" baseline.
    Returns ``(E_x, Var_x)`` of shapes ``(N, q)``, ``(N, q, q)``.
    """
    K, q, s = params.K, params.q, params.s
    dim = q + s
    ys = np.asarray(ys, dtype=float).reshape(-1, s)
    N = ys.shape[0]
    H = _obs_matrix(q, s)
    pi = params.stationary_distribution()

    Fbar = sum(pi[k] * params.f_matrix.F(k) for k in range(K))
    SWbar = sum(pi[k] * params.noise_cov.Sigma_W(k) for k in range(K))
    bbar = sum(pi[k] * params.b(k).reshape(dim, 1) for k in range(K))
    mu, Sigma = stationary_moments(params)
    z = sum(pi[k] * mu[k] for k in range(K))
    Pc = sum(pi[k] * (Sigma[k] + (mu[k] - z) @ (mu[k] - z).T) for k in range(K))

    E_x = np.zeros((N, q))
    Var_x = np.zeros((N, q, q))
    for n in range(N):
        if n > 0:
            z = Fbar @ z + bbar
            Pc = Fbar @ Pc @ Fbar.T + SWbar
        z, Pc, _ = _kalman_exact_y_update(z, Pc, ys[n].reshape(s, 1), H)
        E_x[n] = z[:q].ravel()
        Var_x[n] = Pc[:q, :q]
    return E_x, Var_x


# ---------------------------------------------------------------------------
# Oracle Kalman — known regime sequence (best achievable X-estimate)
# ---------------------------------------------------------------------------
def oracle_filter(params, rs, ys):
    """Kalman filter that switches with the *known* regime sequence ``rs``.

    Returns ``(E_x, Var_x)``: the minimum-mean-square X-estimate attainable when
    the regimes are observed — a lower bound on any regime-blind filter.
    """
    K, q, s = params.K, params.q, params.s
    dim = q + s
    ys = np.asarray(ys, dtype=float).reshape(-1, s)
    rs = np.asarray(rs, dtype=int).ravel()
    N = ys.shape[0]
    H = _obs_matrix(q, s)
    mu, Sigma = stationary_moments(params)

    r0 = int(rs[0])
    z = mu[r0].reshape(dim, 1).copy()
    Pc = Sigma[r0].copy()
    E_x = np.zeros((N, q))
    Var_x = np.zeros((N, q, q))
    for n in range(N):
        k = int(rs[n])
        if n > 0:
            z = params.f_matrix.F(k) @ z + params.b(k).reshape(dim, 1)
            Pc = params.f_matrix.F(k) @ Pc @ params.f_matrix.F(k).T + params.noise_cov.Sigma_W(k)
        z, Pc, _ = _kalman_exact_y_update(z, Pc, ys[n].reshape(s, 1), H)
        E_x[n] = z[:q].ravel()
        Var_x[n] = Pc[:q, :q]
    return E_x, Var_x
