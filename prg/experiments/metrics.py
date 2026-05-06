#!/usr/bin/env python3
"""
prg/experiments/metrics.py
==========================
Scalar performance metrics for the Monte-Carlo simulation study (§6).

All functions operate on numpy arrays collected over one Monte-Carlo run
(N time steps).  They are intentionally stateless and side-effect free so
that the Monte-Carlo runner can call them from any parallelisation backend.

Functions
---------
dof_h5                — free-parameter count d_{H5}(K, q, s) under the AB constraint
compute_rmse          — root mean squared filtering error (scalar)
compute_nees          — average normalised estimation error squared (ANEES)
compute_ljung_box     — Ljung–Box test p-value for innovation whiteness
compute_jarque_bera   — Jarque–Bera normality test p-value on innovations
compute_bic           — BIC for an H5-constrained GSS model
"""

from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats

__all__ = [
    "dof_h5",
    "compute_rmse",
    "compute_nees",
    "compute_ljung_box",
    "compute_jarque_bera",
    "compute_bic",
]

# ---------------------------------------------------------------------------
# Free-parameter count for BIC
# ---------------------------------------------------------------------------


def dof_h5(K: int, q: int, s: int) -> int:
    """
    Number of free parameters of an (H5)-compatible GSS(K, q, s) model
    under the closed-form AB constraint.

    Under (H5) the closed-form AB constraint determines A(k) and B(k)
    *both* from (C, D, Δ, Σ_V) via

        A(k) = Δ(k) Σ_V(k)⁻¹ C(k),
        B(k) = Δ(k) Σ_V(k)⁻¹ D(k),

    so they contribute *zero* free parameters. The count per regime is

        qs + s²            — C(k) (s×q) and D(k) (s×s)
        (q+s)(q+s+1)/2     — Σ_W(k) (symmetric (q+s)×(q+s))
        (q+s)              — drift bias b(k)

    summed over K regimes, plus K(K-1) for the off-diagonal entries of
    the transition matrix P (rows sum to 1) and K-1 for the initial
    distribution π_0 (sums to 1), giving

        d_{H5}(K, q, s) = K [ qs + s² + (q+s)(q+s+1)/2 + (q+s) ]
                         + K² - 1.

    Parameters
    ----------
    K, q, s : int
        Number of regimes, hidden-state dimension, observation dimension.

    Returns
    -------
    int

    Examples
    --------
    >>> dof_h5(2, 1, 1)
    17
    >>> dof_h5(3, 1, 1)
    29
    """
    dim_z = q + s
    per_regime = q * s + s**2 + dim_z * (dim_z + 1) // 2 + dim_z
    return K * per_regime + K**2 - 1


# ---------------------------------------------------------------------------
# RMSE
# ---------------------------------------------------------------------------


def compute_rmse(x_true: np.ndarray, x_est: np.ndarray) -> float:
    """
    Root mean squared filtering error, normalised by the state dimension q.

    RMSE = sqrt( (1 / (N·q)) · Σ_{n=0}^{N-1} ‖x_n − x̂_n‖² )

    For q=1 this reduces to the standard scalar RMSE.

    Parameters
    ----------
    x_true : ndarray of shape (N, q)
        True hidden states.
    x_est : ndarray of shape (N, q)
        Filtered estimates.

    Returns
    -------
    float
    """
    x_true = np.asarray(x_true, dtype=float)
    x_est = np.asarray(x_est, dtype=float)
    N, q = x_true.shape
    if N == 0:
        return float("nan")
    return float(np.sqrt(np.sum((x_true - x_est) ** 2) / (N * q)))


# ---------------------------------------------------------------------------
# NEES
# ---------------------------------------------------------------------------


def compute_nees(
    errors: np.ndarray,
    var_x_arr: np.ndarray,
) -> float:
    """
    Average Normalised Estimation Error Squared (ANEES), normalised by q.

    ANEES = (1 / (N·q)) · Σ_{n=0}^{N-1}  eₙᵀ Pₙ⁻¹ eₙ

    where eₙ = x_n − x̂_n and Pₙ = Var[X_n | y_{1:n}].  For a consistent
    filter, each eₙᵀ Pₙ⁻¹ eₙ ~ χ²(q), so E[ANEES] = 1.

    Parameters
    ----------
    errors : ndarray of shape (N, q)
        Estimation errors x_true − x_est at each step.
    var_x_arr : ndarray of shape (N, q, q)
        Posterior variance Var[X_n | y_{1:n}] at each step.

    Returns
    -------
    float
        ANEES; expected value ≈ 1 for a calibrated filter.
    """
    errors = np.asarray(errors, dtype=float)
    var_x_arr = np.asarray(var_x_arr, dtype=float)
    N, q = errors.shape
    if N == 0 or q == 0:
        return float("nan")

    nees_vals = np.empty(N)
    for n in range(N):
        e = errors[n]  # (q,)
        P = var_x_arr[n]  # (q, q)
        try:
            # P e  via solve (numerically more stable than P^{-1} e)
            Pe = np.linalg.solve(P, e)
            nees_vals[n] = float(e @ Pe)
        except np.linalg.LinAlgError:
            # Singular posterior covariance — use pseudo-inverse
            Pe, *_ = np.linalg.lstsq(P, e, rcond=None)
            nees_vals[n] = float(e @ Pe)

    return float(np.nanmean(nees_vals)) / q


# ---------------------------------------------------------------------------
# Ljung–Box (innovation whiteness)
# ---------------------------------------------------------------------------


def compute_ljung_box(innovations: np.ndarray, lags: int = 20) -> float:
    """
    Ljung–Box test p-value for innovation whiteness.

    For a scalar innovation (s=1) the test is applied directly.
    For multivariate innovations (s>1) the test is applied to each
    component independently and the *minimum* p-value is returned
    (conservative: rejection if any component is autocorrelated).

    A high p-value (> 0.05) indicates the innovations are consistent
    with a white-noise sequence.

    Parameters
    ----------
    innovations : ndarray of shape (N, s)
        Sequence of filter innovations ν_n = y_n − ŷ_{n|n−1}.
    lags : int, default 20
        Maximum lag to test (paper uses 20).

    Returns
    -------
    float
        Minimum p-value across all observation components.
    """
    from statsmodels.stats.diagnostic import acorr_ljungbox

    innovations = np.asarray(innovations, dtype=float)
    if innovations.ndim == 1:
        innovations = innovations[:, None]
    N, s = innovations.shape

    # Need at least lags+1 observations; cap if necessary
    effective_lags = min(lags, N // 2 - 1)
    if effective_lags < 1:
        return float("nan")

    min_pval = 1.0
    for i in range(s):
        col = innovations[:, i]
        # Drop NaN values
        col = col[np.isfinite(col)]
        if len(col) <= effective_lags + 1:
            continue
        try:
            res = acorr_ljungbox(col, lags=effective_lags, return_df=True)
            # Take the p-value at the *last* lag (most powerful test)
            pval = float(res["lb_pvalue"].iloc[-1])
        except Exception:
            pval = float("nan")
        if np.isfinite(pval):
            min_pval = min(min_pval, pval)

    return min_pval


# ---------------------------------------------------------------------------
# Jarque–Bera (innovation normality)
# ---------------------------------------------------------------------------


def compute_jarque_bera(innovations: np.ndarray) -> float:
    """
    Jarque–Bera normality test p-value applied to filter innovations.

    For multivariate innovations (s>1) the test is applied component-wise
    and the *minimum* p-value is returned.

    Parameters
    ----------
    innovations : ndarray of shape (N, s)
        Sequence of filter innovations.

    Returns
    -------
    float
        Minimum p-value across components.
    """
    innovations = np.asarray(innovations, dtype=float)
    if innovations.ndim == 1:
        innovations = innovations[:, None]
    N, s = innovations.shape

    min_pval = 1.0
    for i in range(s):
        col = innovations[:, i]
        col = col[np.isfinite(col)]
        if len(col) < 8:  # JB requires N ≥ 8
            continue
        try:
            _, pval = sp_stats.jarque_bera(col)
            pval = float(pval)
        except Exception:
            pval = float("nan")
        if np.isfinite(pval):
            min_pval = min(min_pval, pval)

    return min_pval


# ---------------------------------------------------------------------------
# BIC
# ---------------------------------------------------------------------------


def compute_bic(
    log_lik: float,
    N: int,
    K: int,
    q: int,
    s: int,
) -> float:
    """
    Bayesian Information Criterion for an H5-constrained GSS(K, q, s) model.

        BIC = d_{H5}(K, q, s) · log(N) − 2 · log p(y_{1:N})

    Lower BIC indicates a better model.

    Parameters
    ----------
    log_lik : float
        Total log-likelihood log p(y_{1:N}) = Σ_{n=1}^{N} log p(y_n | y_{1:n-1}).
    N : int
        Sequence length.
    K, q, s : int
        Number of regimes, hidden-state dimension, observation dimension.

    Returns
    -------
    float
    """
    d = dof_h5(K, q, s)
    return float(d * np.log(N) - 2.0 * log_lik)
