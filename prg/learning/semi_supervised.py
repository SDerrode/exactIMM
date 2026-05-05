#!/usr/bin/env python3
"""
prg/learning/semi_supervised.py
================================
Semi-supervised estimation of a GSS model from (X, Y) data — *the regime
sequence R is unknown*.

Approach
--------
The model is treated as an HMM where:
  - hidden state                    : R_n  ∈ {0, …, K-1}
  - observation at step n+1         : the pair (Z_n, Z_{n+1})  (Z = [X; Y])
  - emission                        : N(F(k) Z_n + b(k), Σ_W(k))
  - transition                      : the Markov matrix P

Parameters are estimated by **Expectation-Maximisation** (Baum-Welch):

  E-step   forward / backward in log-domain → posteriors γ_n(k), ξ_n(j,k)
           and log-likelihood log p(Z_{0:N})

  M-step   closed-form weighted updates of P, π₀, F(k), b(k), Σ_W(k)
           (weighted OLS with weights γ_{n+1}(k))

By default the optional H5 projection on A / B / Σ_U (and Δ=0) is applied
**only once** on the converged parameters at the end of EM — log-likelihood
remains monotonically non-decreasing during EM iterations.

With ``--constraint-each-iter`` the projection is applied at *every*
M-step (Generalized-EM): the constraint is satisfied throughout the
optimisation but log-likelihood monotonicity is no longer guaranteed.

Initialisation
--------------
K-means on the first differences ΔZ_n = Z_{n+1} − Z_n produces a hard
regime assignment, which is fed to the supervised OLS to get F(k), b(k),
Σ_W(k), and P.  Multiple random restarts are run; the one with the
highest final log-likelihood is kept (`--n-inits`, default 10).

After convergence regimes are reordered by the first diagonal entry of A
(descending) to mitigate label-switching across runs.

Usage
-----
    python -m prg.learning.semi_supervised <csv> -K <K> [OPTIONS]

Options
-------
    csv                    Path to the (R,X,Y) or (X,Y) CSV.  If 'r' is
                           present it is *ignored*.
    -K, --K                Number of regimes (required)
    --constraint {a,b,su}  H5 projection target (mutually exclusive)
    --delta-zero           Force Δ(k)=0 before the H5 step
    --constraint-each-iter Apply the projection at every M-step (GEM mode);
                           otherwise it is applied only once at the end of EM
    --n-inits              Number of random restarts (default 10)
    --max-iter             Maximum EM iterations per run (default 100)
    --tol                  Convergence tolerance on Δ log L (default 1e-5)
    --seed                 Random seed for k-means restarts
    --output PATH          Output .py path
    --model-name NAME      File / class base name
    -v, --verbose          Print per-iteration log-likelihood
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import sys

import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.linalg import solve_triangular
from scipy.special import logsumexp

from prg.learning.supervised import (
    _class_name_from_stem,
    _fit_regime,
    _generate_model_code,
    _nearest_spd,
    _read_csv,
)

__all__ = ["fit_semi_supervised"]

_log = logging.getLogger("exactIMM.learning.semi_supervised")

_LOG_FLOOR = 1e-300  # avoids log(0) in transition / initial probabilities


# ---------------------------------------------------------------------------
# Multivariate-normal log-pdf (vectorised, Cholesky-based)
# ---------------------------------------------------------------------------


def _log_mvn_batch(X: np.ndarray, mu: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
    """
    log N(x_i; mu, Sigma) for each row x_i of X.

    Parameters
    ----------
    X     : (N, d)
    mu    : (d,) or (d, 1)
    Sigma : (d, d)  symmetric positive-definite

    Returns
    -------
    (N,) array of log-densities.
    """
    N, d = X.shape
    mu = mu.reshape(-1)
    diff = X - mu[None, :]  # (N, d)
    try:
        L = np.linalg.cholesky(Sigma)
    except np.linalg.LinAlgError:
        # Tiny ridge as last resort
        L = np.linalg.cholesky(Sigma + 1e-8 * np.eye(d))
    # Solve L y = diff^T  →  y has shape (d, N)
    y = solve_triangular(L, diff.T, lower=True, check_finite=False)
    quad = (y * y).sum(axis=0)  # (N,)
    log_det = 2.0 * np.log(np.diag(L)).sum()
    return -0.5 * (d * np.log(2.0 * np.pi) + log_det + quad)


# ---------------------------------------------------------------------------
# Forward / backward in log-domain
# ---------------------------------------------------------------------------


def _compute_log_emissions(
    Z: np.ndarray,  # (N, dim_z)
    F_list: list[np.ndarray],
    b_list: list[np.ndarray],
    SigW_list: list[np.ndarray],
) -> np.ndarray:
    """
    Compute ``log_emis[n-1, k] = log N(Z_n; F(k) Z_{n-1} + b(k), Σ_W(k))``
    for n = 1, …, N-1.

    Returns array of shape (N-1, K).
    """
    N, dim_z = Z.shape
    K = len(F_list)
    log_emis = np.empty((N - 1, K))
    Z_prev = Z[:-1]
    Z_curr = Z[1:]
    for k in range(K):
        # Predicted mean for each transition: F Z_{n-1} + b
        means = Z_prev @ F_list[k].T + b_list[k].reshape(-1)[None, :]  # (N-1, d)
        log_emis[:, k] = _log_mvn_batch(Z_curr - means, np.zeros(dim_z), SigW_list[k])
    return log_emis


def _forward(
    log_emis: np.ndarray,  # (N-1, K)
    log_init: np.ndarray,  # (K,)   log p(Z_0 | R_0)
    log_P: np.ndarray,  # (K, K)
    log_pi0: np.ndarray,  # (K,)
) -> tuple[np.ndarray, float]:
    """
    Forward recursion in log-domain.

    Returns
    -------
    log_alpha : (N, K)
    log_lik   : float = log p(Z_{0:N})
    """
    N = log_emis.shape[0] + 1
    K = log_pi0.shape[0]
    log_alpha = np.empty((N, K))
    log_alpha[0] = log_pi0 + log_init
    for n in range(1, N):
        log_alpha[n] = log_emis[n - 1] + logsumexp(log_alpha[n - 1][:, None] + log_P, axis=0)
    log_lik = float(logsumexp(log_alpha[N - 1]))
    return log_alpha, log_lik


def _backward(
    log_emis: np.ndarray,  # (N-1, K)
    log_P: np.ndarray,  # (K, K)
) -> np.ndarray:
    """Backward recursion. Returns log_beta (N, K)."""
    N = log_emis.shape[0] + 1
    K = log_P.shape[0]
    log_beta = np.zeros((N, K))
    for n in range(N - 2, -1, -1):
        log_beta[n] = logsumexp(log_P + (log_emis[n] + log_beta[n + 1])[None, :], axis=1)
    return log_beta


def _compute_xi(
    log_alpha: np.ndarray,  # (N, K)
    log_beta: np.ndarray,  # (N, K)
    log_emis: np.ndarray,  # (N-1, K)
    log_P: np.ndarray,  # (K, K)
    log_lik: float,
) -> np.ndarray:
    """
    Compute ``log_xi[m, j, k] = log p(R_m=j, R_{m+1}=k | Z_{0:N})`` for
    m = 0, …, N-2.

    Returns array of shape (N-1, K, K).
    """
    # (N-1, K, K) sum
    return (
        log_alpha[:-1, :, None]
        + log_P[None, :, :]
        + log_emis[:, None, :]
        + log_beta[1:, None, :]
        - log_lik
    )


# ---------------------------------------------------------------------------
# Constraint projection (delta_zero + H5) — shared by GEM and post-hoc paths
# ---------------------------------------------------------------------------


def _apply_constraints(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    D: np.ndarray,
    SU: np.ndarray,
    Dt: np.ndarray,
    SV: np.ndarray,
    q: int,
    s: int,
    constraint: str | None,
    delta_zero: bool,
    where: str = "M-step",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply optional Δ=0 and H5 projection to one regime's blocks.

    Used both inside the M-step (Generalized-EM) and as a one-shot
    post-hoc projection on the final converged parameters.

    Parameters
    ----------
    where : str — short tag inserted in warning messages ("M-step", "post-hoc").
    """
    if delta_zero:
        Dt = np.zeros((q, s))
    SU = _nearest_spd(SU)
    SV = _nearest_spd(SV)

    if constraint == "b":
        from prg.utils.h5_constraint import compute_B_from_h5

        try:
            B = compute_B_from_h5(A, C, D, SU, Dt, SV)
        except ValueError as exc:
            _log.warning("H5 (B) failed in %s: %s — keeping unconstrained B.", where, exc)
    elif constraint == "a":
        from prg.utils.h5_constraint import compute_A_from_h5

        try:
            A = compute_A_from_h5(B, C, D, SU, Dt, SV)
        except ValueError as exc:
            _log.warning("H5 (A) failed in %s: %s — keeping unconstrained A.", where, exc)
    elif constraint == "su":
        from prg.utils.h5_constraint import compute_SU_from_h5

        try:
            SU = _nearest_spd(compute_SU_from_h5(A, B, C, D, Dt, SV))
        except ValueError as exc:
            _log.warning("H5 (Σ_U) failed in %s: %s — keeping unconstrained Σ_U.", where, exc)
    return A, B, C, D, SU, Dt, SV


# ---------------------------------------------------------------------------
# Weighted M-step for one regime
# ---------------------------------------------------------------------------


def _weighted_fit(
    Z_curr: np.ndarray,  # (N-1, dim_z)  Z_n
    Z_next: np.ndarray,  # (N-1, dim_z)  Z_{n+1}
    w: np.ndarray,  # (N-1,)        γ_{n+1}(k)
    q: int,
    s: int,
    constraint: str | None,
    delta_zero: bool,
    floor_w: float = 1e-12,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,  # A, B, C, D
    np.ndarray,
    np.ndarray,
    np.ndarray,  # SigU, Delta, SigV
    np.ndarray,  # b
]:
    """
    Weighted least-squares fit for one regime, plus optional H5 projection.

    sqrt-w trick: solve  diag(√w) [Z_curr | 1] Θ ≈ diag(√w) Z_next
    via :func:`numpy.linalg.lstsq`.

    Returns the same 8-tuple as :func:`prg.learning.supervised._fit_regime`.
    """
    N_pairs, dim_z = Z_curr.shape
    Wsum = w.sum()
    if Wsum < floor_w:
        # Degenerate regime — return identity dynamics with large noise
        _log.warning("Regime has total weight ≈ 0; resetting to identity dynamics.")
        F = np.eye(dim_z)
        b = np.zeros((dim_z, 1))
        SigW = np.eye(dim_z)
    else:
        sqrt_w = np.sqrt(np.clip(w, 0.0, None))
        Z_aug = np.hstack([Z_curr, np.ones((N_pairs, 1))])  # (N-1, dim_z+1)
        Z_aug_w = sqrt_w[:, None] * Z_aug
        Z_next_w = sqrt_w[:, None] * Z_next

        Theta, _, _, _ = np.linalg.lstsq(Z_aug_w, Z_next_w, rcond=None)
        F = Theta[:dim_z, :].T
        b = Theta[dim_z, :].reshape(dim_z, 1)

        residuals = Z_next - Z_aug @ Theta
        # Weighted MLE covariance: Σ = Σₙ wₙ rₙ rₙᵀ / Σₙ wₙ
        SigW = (residuals.T @ (w[:, None] * residuals)) / Wsum
        SigW = (SigW + SigW.T) / 2.0

    A = F[:q, :q]
    B = F[:q, q:]
    C = F[q:, :q]
    D = F[q:, q:]
    SU = SigW[:q, :q]
    Dt = SigW[:q, q:]
    SV = SigW[q:, q:]

    A, B, C, D, SU, Dt, SV = _apply_constraints(
        A,
        B,
        C,
        D,
        SU,
        Dt,
        SV,
        q,
        s,
        constraint,
        delta_zero,
        where="M-step",
    )

    return A, B, C, D, SU, Dt, SV, b


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


def _initialize_kmeans(
    Z: np.ndarray,
    K: int,
    seed: int,
) -> np.ndarray:
    """
    K-means on the first differences ΔZ_n = Z_{n+1} − Z_n.

    Returns
    -------
    R_init : (N,) int — hard regime assignment for each time step.
             R_init[n+1] = cluster of ΔZ_n; R_init[0] = R_init[1].
    """
    N, dim_z = Z.shape
    dZ = Z[1:] - Z[:-1]
    # kmeans2: k-means with k++ init for reproducibility across seeds
    try:
        _, labels = kmeans2(dZ, K, seed=seed, minit="++", missing="warn")
    except Exception:
        # Fallback to random init if k-means++ collapses
        _, labels = kmeans2(dZ, K, seed=seed, minit="random", missing="warn")

    R_init = np.empty(N, dtype=int)
    R_init[1:] = labels
    R_init[0] = int(labels[0])

    # If a regime ends up empty, randomly reassign one sample
    rng = np.random.default_rng(seed)
    for k in range(K):
        if (R_init == k).sum() == 0:
            j = rng.integers(N)
            R_init[j] = k
    return R_init


def _initialize_params_from_R(
    R: np.ndarray,
    Z: np.ndarray,
    K: int,
    q: int,
    s: int,
) -> tuple[
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],  # F, b, SigW
    np.ndarray,
    np.ndarray,  # P, pi0
    list[np.ndarray],
    list[np.ndarray],  # mu_z0, Sigma_z0
]:
    """
    Run a quick supervised OLS on the hard assignment R to seed EM.

    Falls back to identity / unit dynamics for regimes that lack data.
    """
    dim_z = q + s
    N = Z.shape[0]
    xs = Z[:, :q]
    ys = Z[:, q:]

    # P from transition counts (with Laplace smoothing for missing rows)
    P = np.zeros((K, K))
    for n in range(N - 1):
        P[R[n], R[n + 1]] += 1
    row_sums = P.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    P = P / row_sums
    # Replace any all-zero row with uniform
    for k in range(K):
        if P[k].sum() == 0.0:
            P[k] = np.ones(K) / K

    # π_0: empirical regime frequencies (with mild smoothing)
    counts = np.array([(R == k).sum() for k in range(K)], dtype=float)
    pi0 = (counts + 1.0) / (counts.sum() + K)

    # F, b, Σ_W via supervised OLS per regime (no constraint at init)
    F_list, b_list, SigW_list = [], [], []
    mu_z0_list, Sigma_z0_list = [], []

    Z_global_mean = Z.mean(axis=0).reshape(dim_z, 1)
    Z_global_cov = _nearest_spd(np.cov(Z, rowvar=False))

    for k in range(K):
        mask = R[1:] == k
        idx = np.where(mask)[0]
        if idx.size >= dim_z + 1:
            Z_curr = Z[idx]
            Z_next = Z[idx + 1]
            A, B, C, D, SU, Dt, SV, b = _fit_regime(
                Z_curr, Z_next, q, s, constraint=None, delta_zero=False
            )
            F = np.block([[A, B], [C, D]])
            SigW = np.block([[SU, Dt], [Dt.T, SV]])
        else:
            # Fallback for tiny clusters
            F = np.eye(dim_z)
            b = np.zeros((dim_z, 1))
            SigW = Z_global_cov + 0.1 * np.eye(dim_z)

        F_list.append(F)
        b_list.append(b)
        SigW_list.append(_nearest_spd(SigW))

        # μ_z0, Σ_z0: per-regime sample moments, fallback to global
        in_k = R == k
        if in_k.sum() >= 2:
            Z_k = Z[in_k]
            mu_z0_list.append(Z_k.mean(axis=0).reshape(dim_z, 1))
            Sigma_z0_list.append(_nearest_spd(np.cov(Z_k, rowvar=False)))
        else:
            mu_z0_list.append(Z_global_mean.copy())
            Sigma_z0_list.append(Z_global_cov.copy())

    return F_list, b_list, SigW_list, P, pi0, mu_z0_list, Sigma_z0_list


# ---------------------------------------------------------------------------
# One EM run
# ---------------------------------------------------------------------------


def _em_run(
    Z: np.ndarray,
    K: int,
    q: int,
    s: int,
    init_seed: int,
    constraint: str | None,
    delta_zero: bool,
    max_iter: int,
    tol: float,
    verbose: bool,
    constraint_each_iter: bool = False,
) -> tuple[dict, dict]:
    """
    Single EM run from one k-means initialisation.

    Parameters
    ----------
    constraint_each_iter : bool
        If True, apply the constraint (Δ=0 and/or H5) at every M-step
        (Generalized-EM, log-lik may not be monotone).
        If False (default), run EM unconstrained and apply the constraint
        once on the final converged parameters.

    Returns
    -------
    params : dict   — same keys as fit_supervised
    info   : dict   — {log_lik, log_lik_history, n_iter, converged, init_seed}
    """
    dim_z = q + s
    N = Z.shape[0]

    # --- Init ---
    R_init = _initialize_kmeans(Z, K, seed=init_seed)
    (F_list, b_list, SigW_list, P, pi0, mu_z0_list, Sigma_z0_list) = _initialize_params_from_R(
        R_init, Z, K, q, s
    )

    log_lik_history: list[float] = []
    converged = False

    for it in range(max_iter):
        log_P = np.log(np.clip(P, _LOG_FLOOR, None))
        log_pi0 = np.log(np.clip(pi0, _LOG_FLOOR, None))

        # Initial-state emission term: log N(Z_0; μ_z0(k), Σ_z0(k))
        log_init = np.array(
            [float(_log_mvn_batch(Z[0:1], mu_z0_list[k], Sigma_z0_list[k])[0]) for k in range(K)]
        )

        # E-step
        log_emis = _compute_log_emissions(Z, F_list, b_list, SigW_list)  # (N-1, K)
        log_alpha, log_lik = _forward(log_emis, log_init, log_P, log_pi0)
        log_beta = _backward(log_emis, log_P)
        log_gamma = log_alpha + log_beta - log_lik
        log_xi = _compute_xi(log_alpha, log_beta, log_emis, log_P, log_lik)

        log_lik_history.append(log_lik)
        if verbose:
            print(f"  iter {it:3d}   log L = {log_lik:.4f}")

        # Convergence check
        if it > 0:
            delta = log_lik - log_lik_history[-2]
            if abs(delta) < tol:
                converged = True
                break

        # M-step
        gamma = np.exp(log_gamma)  # (N, K)
        xi = np.exp(log_xi)  # (N-1, K, K)

        # P: row-normalised transition counts
        denom = gamma[:-1].sum(axis=0)  # (K,)
        denom_safe = np.where(denom > _LOG_FLOOR, denom, 1.0)
        P_new = xi.sum(axis=0) / denom_safe[:, None]
        # Sanitise
        P_new = np.clip(P_new, _LOG_FLOOR, 1.0)
        P_new = P_new / P_new.sum(axis=1, keepdims=True)
        # If a row had no support, replace with previous row
        for k in range(K):
            if denom[k] <= _LOG_FLOOR:
                P_new[k] = P[k]
        P = P_new

        # π_0
        pi0 = gamma[0] / gamma[0].sum()

        # F(k), b(k), Σ_W(k) — weighted per regime
        # When constraint_each_iter is False, run unconstrained inside the
        # EM loop; the constraint is applied once at the end.
        ms_constraint = constraint if constraint_each_iter else None
        ms_delta_zero = delta_zero if constraint_each_iter else False
        Z_curr = Z[:-1]
        Z_next = Z[1:]
        for k in range(K):
            w_k = gamma[1:, k]  # γ_{n+1}(k) for n=0..N-2
            A, B, C, D, SU, Dt, SV, b_k = _weighted_fit(
                Z_curr, Z_next, w_k, q, s, ms_constraint, ms_delta_zero
            )
            F_list[k] = np.block([[A, B], [C, D]])
            SigW_list[k] = np.block([[SU, Dt], [Dt.T, SV]])
            b_list[k] = b_k

        # μ_z0(k), Σ_z0(k) — weighted by γ_n(k)
        for k in range(K):
            w = gamma[:, k]
            denom_k = w.sum()
            if denom_k > _LOG_FLOOR:
                mu = (w[:, None] * Z).sum(axis=0) / denom_k
                centered = Z - mu
                cov = (centered.T @ (w[:, None] * centered)) / denom_k
                mu_z0_list[k] = mu.reshape(dim_z, 1)
                Sigma_z0_list[k] = _nearest_spd(cov)

    # ----- Decompose final F / Σ_W into A,B,C,D / Σ_U,Δ,Σ_V -----
    # When constraint_each_iter is False and (constraint or delta_zero) is
    # set, apply the projection here, ONCE, on the converged parameters.
    apply_post_hoc = (not constraint_each_iter) and (constraint is not None or delta_zero)
    A_list, B_list, C_list, D_list = [], [], [], []
    SU_list, Dt_list, SV_list = [], [], []
    for k in range(K):
        F = F_list[k]
        S = SigW_list[k]
        A = F[:q, :q]
        B = F[:q, q:]
        C = F[q:, :q]
        D = F[q:, q:]
        SU = S[:q, :q]
        Dt = S[:q, q:]
        SV = S[q:, q:]
        if apply_post_hoc:
            A, B, C, D, SU, Dt, SV = _apply_constraints(
                A,
                B,
                C,
                D,
                SU,
                Dt,
                SV,
                q,
                s,
                constraint,
                delta_zero,
                where="post-hoc",
            )
        A_list.append(A)
        B_list.append(B)
        C_list.append(C)
        D_list.append(D)
        SU_list.append(SU)
        Dt_list.append(Dt)
        SV_list.append(SV)

    params = {
        "K": K,
        "q": q,
        "s": s,
        "P": P,
        "A_list": A_list,
        "B_list": B_list,
        "C_list": C_list,
        "D_list": D_list,
        "Sigma_U_list": SU_list,
        "Delta_list": Dt_list,
        "Sigma_V_list": SV_list,
        "pi0": pi0,
        "mu_z0_list": mu_z0_list,
        "Sigma_z0_list": Sigma_z0_list,
        "b_list": b_list,
    }
    info = {
        "log_lik": log_lik_history[-1],
        "log_lik_history": log_lik_history,
        "n_iter": len(log_lik_history),
        "converged": converged,
        "init_seed": init_seed,
    }
    return params, info


# ---------------------------------------------------------------------------
# Label-switching mitigation
# ---------------------------------------------------------------------------


def _reorder_regimes(params: dict) -> dict:
    """Reorder regimes by A[0,0] (descending) for canonical labelling."""
    K = params["K"]
    keys_per_regime = [
        "A_list",
        "B_list",
        "C_list",
        "D_list",
        "Sigma_U_list",
        "Delta_list",
        "Sigma_V_list",
        "mu_z0_list",
        "Sigma_z0_list",
        "b_list",
    ]
    order = sorted(range(K), key=lambda k: -params["A_list"][k][0, 0])
    perm = np.array(order)

    new = dict(params)
    for key in keys_per_regime:
        new[key] = [params[key][order[k]] for k in range(K)]
    # P: rows and columns permuted
    P_old = params["P"]
    new["P"] = P_old[np.ix_(perm, perm)]
    # π_0
    new["pi0"] = params["pi0"][perm]
    return new


# ---------------------------------------------------------------------------
# Public API: multi-start fit
# ---------------------------------------------------------------------------


def fit_semi_supervised(
    xs: np.ndarray,
    ys: np.ndarray,
    K: int,
    *,
    constraint: str | None = None,
    delta_zero: bool = False,
    constraint_each_iter: bool = False,
    n_inits: int = 10,
    max_iter: int = 100,
    tol: float = 1e-5,
    seed: int | None = None,
    verbose: bool = False,
) -> tuple[dict, dict]:
    """
    Estimate GSS parameters from (X, Y) data — the regime sequence is hidden.

    Parameters
    ----------
    xs, ys : ndarrays of shape (N, q) and (N, s)
    K      : int — number of regimes
    constraint : None | 'a' | 'b' | 'su'
        H5 projection target.
    delta_zero : if True, force Δ(k)=0 before the H5 step.
    constraint_each_iter : bool, default False
        If True, apply ``constraint`` and ``delta_zero`` at *every* M-step
        (Generalized-EM — log-likelihood may not be monotone).
        If False (default), EM runs unconstrained and the projection is
        applied **once** on the converged parameters of the best run.
    n_inits  : number of independent EM runs from different k-means seeds.
    max_iter : maximum EM iterations per run.
    tol      : convergence threshold on |Δ log L|.
    seed     : base random seed (different k-means seeds are derived from it).
    verbose  : print per-iteration log-likelihood.

    Returns
    -------
    params : dict  — best-likelihood parameter set (regimes reordered)
    info   : dict  — {best_log_lik, best_init_seed, all_log_liks (list[float]),
                      all_n_iters (list[int]), all_converged (list[bool])}

    Notes
    -----
    With ``constraint_each_iter=True`` the procedure becomes a *Generalized
    EM*: the log-likelihood is *not* guaranteed to be monotonically
    non-decreasing.  Convergence is monitored on |Δ log L| (absolute value).
    With the default ``constraint_each_iter=False``, the EM iterations are
    standard (monotone log-likelihood) and the constraint is enforced as a
    post-hoc projection — matching the supervised estimator's behaviour.
    """
    q = xs.shape[1]
    s = ys.shape[1]
    Z = np.hstack([xs, ys])

    rng = np.random.default_rng(seed)
    init_seeds = rng.integers(low=0, high=2**31 - 1, size=n_inits).tolist()

    best_params: dict | None = None
    best_info: dict | None = None
    all_logL = []
    all_iters = []
    all_conv = []

    for run, s_init in enumerate(init_seeds):
        if verbose:
            print(f"\n=== EM run {run + 1}/{n_inits}  (init_seed={s_init}) ===")
        try:
            params, info = _em_run(
                Z,
                K,
                q,
                s,
                s_init,
                constraint,
                delta_zero,
                max_iter,
                tol,
                verbose,
                constraint_each_iter=constraint_each_iter,
            )
        except (np.linalg.LinAlgError, ValueError) as exc:
            _log.warning("EM run %d failed (%s) — skipping.", run + 1, exc)
            continue

        all_logL.append(info["log_lik"])
        all_iters.append(info["n_iter"])
        all_conv.append(info["converged"])

        if best_info is None or info["log_lik"] > best_info["log_lik"]:
            best_params = params
            best_info = info

    if best_params is None:
        raise RuntimeError("All EM runs failed.  Try lowering K, increasing N, or changing seed.")

    # Canonical regime ordering
    best_params = _reorder_regimes(best_params)

    return best_params, {
        "best_log_lik": best_info["log_lik"],
        "best_init_seed": best_info["init_seed"],
        "best_n_iter": best_info["n_iter"],
        "best_converged": best_info["converged"],
        "all_log_liks": all_logL,
        "all_n_iters": all_iters,
        "all_converged": all_conv,
        "log_lik_history": best_info["log_lik_history"],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m prg.learning.semi_supervised",
        description=(
            "Estimate a GSS model from (X, Y) data when the regime "
            "sequence R is unknown.  Uses Baum-Welch EM with k-means "
            "initialisation and multiple random restarts."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "csv", metavar="CSV", help="Path to the CSV (the 'r' column is ignored if present)."
    )
    p.add_argument("-K", "--K", type=int, required=True, metavar="K", help="Number of regimes.")
    p.add_argument(
        "--constraint",
        choices=["a", "b", "su"],
        default=None,
        metavar="TARGET",
        help="H5 projection target.  By default applied once at "
        "the end of EM; with --constraint-each-iter applied "
        "at every M-step (GEM, log-lik may not be monotone).",
    )
    p.add_argument("--delta-zero", action="store_true", help="Force Δ(k)=0 before the H5 step.")
    p.add_argument(
        "--constraint-each-iter",
        action="store_true",
        dest="constraint_each_iter",
        help="Apply --constraint / --delta-zero at every M-step "
        "(Generalized-EM).  Default: apply once at the end.",
    )
    p.add_argument("--n-inits", type=int, default=10, metavar="N", help="Number of EM restarts.")
    p.add_argument(
        "--max-iter", type=int, default=100, metavar="N", help="Maximum EM iterations per run."
    )
    p.add_argument(
        "--tol", type=float, default=1e-5, metavar="EPS", help="Convergence threshold on |Δ log L|."
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        metavar="SEED",
        help="Base random seed for k-means restarts.",
    )
    p.add_argument(
        "--output",
        default=None,
        metavar="PATH",
        help="Output .py file (default: prg/models/<auto>.py).",
    )
    p.add_argument(
        "--model-name",
        default=None,
        metavar="NAME",
        help="File / class base name (default: model_em_K<K>_q<q>_s<s>).",
    )
    p.add_argument(
        "-v", "--verbose", action="store_true", help="Print per-iteration log-likelihood."
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-8s %(name)s — %(message)s",
        stream=sys.stdout,
    )

    csv_path = pathlib.Path(args.csv).resolve()
    if not csv_path.exists():
        print(f"Error: CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    # _read_csv returns rs too — we ignore it here
    try:
        _, xs, ys, _, q, s = _read_csv(csv_path)
    except ValueError as exc:
        print(f"Error reading CSV: {exc}", file=sys.stderr)
        sys.exit(1)

    N = xs.shape[0]
    K = args.K
    if args.verbose:
        print(f"Data: N={N}, q={q}, s={s}, K={K}")

    try:
        params, info = fit_semi_supervised(
            xs,
            ys,
            K,
            constraint=args.constraint,
            delta_zero=args.delta_zero,
            constraint_each_iter=args.constraint_each_iter,
            n_inits=args.n_inits,
            max_iter=args.max_iter,
            tol=args.tol,
            seed=args.seed,
            verbose=args.verbose,
        )
    except RuntimeError as exc:
        print(f"Estimation failed: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"\nBest log L     : {info['best_log_lik']:.4f}")
    print(f"Best init seed : {info['best_init_seed']}")
    print(
        f"Iterations     : {info['best_n_iter']} "
        f"({'converged' if info['best_converged'] else 'max-iter reached'})"
    )
    print(f"All log Ls     : {[f'{x:.2f}' for x in info['all_log_liks']]}")

    # --- Resolve output path / class name ---
    # Stem precedence: --model-name > --output stem > auto
    if args.model_name is not None:
        stem = pathlib.Path(args.model_name).stem
    elif args.output is not None:
        stem = pathlib.Path(args.output).stem
    else:
        stem = f"model_em_K{K}_q{q}_s{s}"
    class_name = _class_name_from_stem(stem)

    if args.output is not None:
        out_path = pathlib.Path(args.output).resolve()
    else:
        models_dir = pathlib.Path(__file__).resolve().parent.parent / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        out_path = models_dir / f"{stem}.py"

    code = _generate_model_code(
        params=params,
        class_name=class_name,
        file_stem=stem,
        source_csv=str(csv_path),
        constraint=args.constraint,
        delta_zero=args.delta_zero,
    )
    # Insert an EM-specific note at the top of the docstring
    em_note = (
        f"  (semi-supervised EM, K={K}, "
        f"log L={info['best_log_lik']:.4f}, "
        f"{info['best_n_iter']} iters)"
    )
    code = code.replace(
        "Estimated by supervised OLS from fully-observed (R,X,Y) data.",
        f"Estimated by Baum-Welch EM from (X,Y) data — R hidden.{em_note}",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(code, encoding="utf-8")

    print(f"\nModel saved   : {out_path}")
    print(f"Class name    : {class_name}")
    print(f"Use with      : python -m prg.simulate --model {stem} -N 1000 --seed 42")


if __name__ == "__main__":
    main()
