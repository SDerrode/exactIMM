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
    A pairwise (coupled) Kalman filter on the π-stationary-averaged model — the
    regime-blind "ignore the switching" baseline (a Kalman couple on Z=[X;Y]).
oracle_filter(params, rs, ys)
    A Kalman filter that switches with the *known* regime sequence ``rs`` — the
    best achievable X-estimate (regimes given). Lower bound on the error.
imm_filter(params, ys)
    Blom-Bar-Shalom Interacting Multiple Model — O(K) Kalman updates/step.
gpb2_filter(params, ys)
    Generalized Pseudo-Bayesian order 2 — O(K²) Kalman updates/step.
rbpf_filter(params, ys, n_particles, seed)
    Rao-Blackwellised particle filter — converges to the exact posterior.

The three approximate switching baselines (``imm_filter``, ``gpb2_filter``,
``rbpf_filter``) all return ``(E_x, Var_x, pi, log_lik)`` so they can be scored
uniformly against the exact filter (see issue #5).

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
    "imm_filter",
    "gpb2_filter",
    "rbpf_filter",
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
# Pairwise (coupled) Kalman on the π-averaged model ("ignore the switching")
# ---------------------------------------------------------------------------
def single_kalman_filter(params, ys):
    """Pairwise (coupled) Kalman filter on the stationary-π-averaged model.

    Uses F̄ = Σ_k π_k F_k, Σ̄_W = Σ_k π_k Σ_W(k), b̄ = Σ_k π_k b(k), started from
    the averaged stationary moment — a fair regime-blind baseline. It runs on the
    couple Z=[X;Y] with the full averaged coupling F̄=[[A,B],[C,D]] (a Kalman
    *couple*, not a traditional reduced-state filter; name kept for compatibility).
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


# ---------------------------------------------------------------------------
# Approximate switching baselines (IMM, GPB2, RBPF) — issue #5
# ---------------------------------------------------------------------------
def _logsumexp(a: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    m = float(a.max())
    if not np.isfinite(m):
        return m
    return m + float(np.log(np.exp(a - m).sum()))


def _moments_from_mix(weights, zs, Ps, regimes, q, K):
    """Collapse a weighted Gaussian Z-mixture to (E_x (q,), Var_x (q,q), pi (K,))."""
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    ex = np.zeros((q, 1))
    for wi, zi in zip(w, zs):
        ex += wi * zi[:q]
    vx = np.zeros((q, q))
    for wi, zi, Pi in zip(w, zs, Ps):
        d = zi[:q] - ex
        vx += wi * (Pi[:q, :q] + d @ d.T)
    pr = np.zeros(K)
    for wi, ri in zip(w, regimes):
        pr[int(ri)] += wi
    return ex.ravel(), vx, pr


def _model_arrays(params):
    """(F, b, Σ_W, μ_z0, Σ_z0, π0, H, K, q, s, dim) extracted from a GSSParams."""
    K, q, s = params.K, params.q, params.s
    dim = q + s
    F = [params.f_matrix.F(k) for k in range(K)]
    b = [params.b(k).reshape(dim, 1) for k in range(K)]
    SW = [params.noise_cov.Sigma_W(k) for k in range(K)]
    mu0 = [params.mu_z0(k).reshape(dim, 1) for k in range(K)]
    Sig0 = [params.Sigma_z0(k) for k in range(K)]
    pi0 = params.pi0
    pi0 = (
        np.asarray(pi0, dtype=float).ravel()
        if pi0 is not None
        else params.stationary_distribution()
    )
    return F, b, SW, mu0, Sig0, pi0, _obs_matrix(q, s), K, q, s, dim


def imm_filter(params, ys):
    """Blom-Bar-Shalom Interacting Multiple Model (IMM) filter.

    Keeps one regime-conditional Kalman state and a mode probability per
    regime; each step mixes the K states (interaction), runs K mode-matched
    Kalman updates, and re-weights the modes by likelihood. An O(K)-per-step
    approximation of the exact Kᴺ posterior.

    Returns ``(E_x (N,q), Var_x (N,q,q), pi (N,K), log_lik)``.
    """
    F, b, SW, mu0, Sig0, pi0, H, K, q, s, dim = _model_arrays(params)
    P = params.P
    ys = np.asarray(ys, dtype=float).reshape(-1, s)
    N = ys.shape[0]
    E_x = np.zeros((N, q))
    Var_x = np.zeros((N, q, q))
    pis = np.zeros((N, K))
    loglik = 0.0

    # n = 0 : condition each regime prior on Y_1, weight by π0.
    y0 = ys[0].reshape(s, 1)
    z = [_kalman_exact_y_update(mu0[k], Sig0[k], y0, H) for k in range(K)]
    ll = np.array([h[2] for h in z])
    z, Pc = [h[0] for h in z], [h[1] for h in z]
    logmu = np.log(pi0 + 1e-300) + ll
    lse = _logsumexp(logmu)
    loglik += lse
    mu = np.exp(logmu - lse)
    E_x[0], Var_x[0], pis[0] = _moments_from_mix(mu, z, Pc, range(K), q, K)

    for n in range(1, N):
        yn = ys[n].reshape(s, 1)
        cbar = P.T @ mu  # predicted mode prob c_k = Σ_j P[j,k] μ_j
        z0, P0 = [None] * K, [None] * K
        for k in range(K):
            wjk = (
                np.array([P[j, k] * mu[j] for j in range(K)]) / cbar[k]
                if cbar[k] > 1e-300
                else np.full(K, 1.0 / K)
            )
            z0k = sum(wjk[j] * z[j] for j in range(K))
            P0k = sum(wjk[j] * (Pc[j] + (z[j] - z0k) @ (z[j] - z0k).T) for j in range(K))
            z0[k], P0[k] = z0k, 0.5 * (P0k + P0k.T)
        ll = np.zeros(K)
        for k in range(K):
            z_pred = F[k] @ z0[k] + b[k]
            P_pred = F[k] @ P0[k] @ F[k].T + SW[k]
            z[k], Pc[k], ll[k] = _kalman_exact_y_update(z_pred, 0.5 * (P_pred + P_pred.T), yn, H)
        logmu = np.log(cbar + 1e-300) + ll
        lse = _logsumexp(logmu)
        loglik += lse
        mu = np.exp(logmu - lse)
        E_x[n], Var_x[n], pis[n] = _moments_from_mix(mu, z, Pc, range(K), q, K)

    return E_x, Var_x, pis, loglik


def gpb2_filter(params, ys):
    """Generalized Pseudo-Bayesian order 2 (GPB2) filter.

    Keeps one Gaussian per current regime; each step expands to the K² regime
    pairs (previous, current), runs K² Kalman updates, then collapses back to K
    by merging over the previous regime. More accurate but K× costlier per step
    than IMM.

    Returns ``(E_x (N,q), Var_x (N,q,q), pi (N,K), log_lik)``.
    """
    F, b, SW, mu0, Sig0, pi0, H, K, q, s, dim = _model_arrays(params)
    P = params.P
    ys = np.asarray(ys, dtype=float).reshape(-1, s)
    N = ys.shape[0]
    E_x = np.zeros((N, q))
    Var_x = np.zeros((N, q, q))
    pis = np.zeros((N, K))
    loglik = 0.0

    y0 = ys[0].reshape(s, 1)
    z = [_kalman_exact_y_update(mu0[k], Sig0[k], y0, H) for k in range(K)]
    ll = np.array([h[2] for h in z])
    z, Pc = [h[0] for h in z], [h[1] for h in z]
    logmu = np.log(pi0 + 1e-300) + ll
    lse = _logsumexp(logmu)
    loglik += lse
    mu = np.exp(logmu - lse)
    E_x[0], Var_x[0], pis[0] = _moments_from_mix(mu, z, Pc, range(K), q, K)

    for n in range(1, N):
        yn = ys[n].reshape(s, 1)
        zjk, Pjk = {}, {}
        logw = np.full((K, K), -np.inf)
        for j in range(K):
            for k in range(K):
                z_pred = F[k] @ z[j] + b[k]
                P_pred = F[k] @ Pc[j] @ F[k].T + SW[k]
                zk, Pk, lll = _kalman_exact_y_update(z_pred, 0.5 * (P_pred + P_pred.T), yn, H)
                zjk[(j, k)], Pjk[(j, k)] = zk, Pk
                logw[j, k] = np.log(mu[j] + 1e-300) + np.log(P[j, k] + 1e-300) + lll
        loglik += _logsumexp(logw.ravel())
        w = np.exp(logw - logw.max())
        w /= w.sum()
        mu = w.sum(axis=0)  # collapsed current-mode probabilities
        for k in range(K):
            sk = w[:, k].sum()
            wjk = w[:, k] / sk if sk > 1e-300 else np.full(K, 1.0 / K)
            zk = sum(wjk[j] * zjk[(j, k)] for j in range(K))
            Pk = sum(
                wjk[j] * (Pjk[(j, k)] + (zjk[(j, k)] - zk) @ (zjk[(j, k)] - zk).T) for j in range(K)
            )
            z[k], Pc[k] = zk, 0.5 * (Pk + Pk.T)
        E_x[n], Var_x[n], pis[n] = _moments_from_mix(mu, z, Pc, range(K), q, K)

    return E_x, Var_x, pis, loglik


def rbpf_filter(params, ys, n_particles=1000, seed=0, resample_threshold=0.5):
    """Rao-Blackwellised particle filter (Doucet, de Freitas, Murphy & Russell, 2000).

    Particles over the discrete regime path, each carrying an exact
    regime-conditional Kalman state for Z. Regimes are proposed from the
    transition prior, weighted by the observation likelihood, and resampled
    (systematic) when the effective sample size falls below
    ``resample_threshold × n_particles``. Converges to the exact posterior as
    ``n_particles → ∞``.

    Returns ``(E_x (N,q), Var_x (N,q,q), pi (N,K), log_lik)``.
    """
    F, b, SW, mu0, Sig0, pi0, H, K, q, s, dim = _model_arrays(params)
    P = params.P
    ys = np.asarray(ys, dtype=float).reshape(-1, s)
    N = ys.shape[0]
    rng = np.random.default_rng(seed)
    M = int(n_particles)

    E_x = np.zeros((N, q))
    Var_x = np.zeros((N, q, q))
    pis = np.zeros((N, K))
    loglik = 0.0

    # n = 0 : sample r_0 ~ π0, init each particle's Kalman state, weight by Λ.
    y0 = ys[0].reshape(s, 1)
    r = rng.choice(K, size=M, p=pi0 / pi0.sum())
    z, Pc, incr = [None] * M, [None] * M, np.zeros(M)
    for p in range(M):
        z[p], Pc[p], incr[p] = _kalman_exact_y_update(mu0[r[p]], Sig0[r[p]], y0, H)
    loglik += _logsumexp(incr) - np.log(M)  # uniform prior weights 1/M
    logw = incr - _logsumexp(incr)
    W = np.exp(logw)
    E_x[0], Var_x[0], pis[0] = _moments_from_mix(W, z, Pc, r, q, K)

    for n in range(1, N):
        yn = ys[n].reshape(s, 1)
        r = np.array([rng.choice(K, p=P[int(r[p])]) for p in range(M)])  # transition prior
        incr = np.zeros(M)
        for p in range(M):
            k = int(r[p])
            z_pred = F[k] @ z[p] + b[k]
            P_pred = F[k] @ Pc[p] @ F[k].T + SW[k]
            z[p], Pc[p], incr[p] = _kalman_exact_y_update(z_pred, 0.5 * (P_pred + P_pred.T), yn, H)
        loglik += _logsumexp(logw + incr)  # log Σ_p W_{n-1}[p] Λ_p
        logw = logw + incr
        logw -= _logsumexp(logw)
        W = np.exp(logw)
        E_x[n], Var_x[n], pis[n] = _moments_from_mix(W, z, Pc, r, q, K)
        # systematic resampling if the effective sample size is too low
        if 1.0 / np.sum(W**2) < resample_threshold * M:
            pos = (rng.random() + np.arange(M)) / M
            idx = np.clip(np.searchsorted(np.cumsum(W), pos), 0, M - 1)
            z = [z[i].copy() for i in idx]
            Pc = [Pc[i].copy() for i in idx]
            r = r[idx]
            logw = np.full(M, -np.log(M))
            W = np.exp(logw)

    return E_x, Var_x, pis, loglik
