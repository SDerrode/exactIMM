#!/usr/bin/env python3
"""
prg/gui/diagnostics.py
======================
Pure-NumPy statistical diagnostics shared by the GUI (no Qt dependency).

Extracted verbatim from ``prg/gui/main_window.py`` so that the window
module can focus on widget/layout logic. These helpers cover the
Ljung-Box white-noise test, sample shape diagnostics (skewness, kurtosis,
Jarque-Bera), innovation standardisation, and the stationary distribution
of a row-stochastic matrix.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2 as _chi2_dist
from scipy.stats import jarque_bera as _jb


def _ljung_box(x: np.ndarray, lags: int | None = None) -> tuple[float, float, int]:
    """
    Ljung-Box portmanteau test for autocorrelation.

    H₀: no autocorrelation up to lag *h*.

    Returns (Q_stat, p_value, h_used).
    High p-value → fail to reject H₀ → innovation looks like white noise.
    """
    n = len(x)
    h = lags if lags is not None else min(20, max(5, n // 10))
    h = min(h, n - 1)
    x = x - x.mean()
    var = float(np.var(x))
    if var < 1e-30:  # constant series
        return 0.0, 1.0, h
    # Sample autocorrelations ρ_k, k = 1…h
    rho_sq = np.array([(np.dot(x[k:], x[:-k]) / (n * var)) ** 2 for k in range(1, h + 1)])
    Q = float(n * (n + 2) * np.sum(rho_sq / (n - np.arange(1, h + 1))))
    p = float(1.0 - _chi2_dist.cdf(Q, df=h))
    return Q, p, h


def _shape_diagnostics(x: np.ndarray) -> tuple[float, float, float, float]:
    """
    Compute sample shape diagnostics for an innovation series.

    Returns ``(skewness, excess_kurtosis, JB_stat, p_value)``.

    - Skewness  S        : 3rd standardised moment.   |S| ≈ 0 ⇔ symmetric.
    - Excess kurtosis K  : 4th std. moment − 3.       |K| ≈ 0 ⇔ Gaussian tails.
    - Jarque–Bera (JB)   : combined test, JB ~ χ²(2) under H₀ "Gaussian".
                           Computed via SciPy's reference implementation.

    Applied to *standardised* innovations these diagnostics are meaningful for
    GSS models: if S ≈ 0 and K ≈ 0 the filter is well-calibrated.
    """
    n = len(x)
    if n < 4:
        return 0.0, 0.0, 0.0, 1.0
    xc = np.asarray(x, dtype=float) - float(np.mean(x))
    sigma2 = float(np.mean(xc**2))
    if sigma2 < 1e-30:  # constant series
        return 0.0, 0.0, 0.0, 1.0
    sigma = float(np.sqrt(sigma2))
    z = xc / sigma
    S = float(np.mean(z**3))
    Kurt = float(np.mean(z**4) - 3.0)
    try:
        jb_res = _jb(x)  # SciPy returns (statistic, pvalue)
        JB = float(jb_res[0])
        p = float(jb_res[1])
    except Exception:  # noqa: BLE001
        # Manual fallback (matches SciPy formula exactly)
        JB = n * (S**2 / 6.0 + Kurt**2 / 24.0)
        p = float(1.0 - _chi2_dist.cdf(JB, df=2))
    return S, Kurt, JB, p


# ---------------------------------------------------------------------------
# Innovation standardisation (A12 / D1)
# ---------------------------------------------------------------------------


def _standardise_innovations(
    innovations: np.ndarray,
    mix_w: np.ndarray | None,
    Gamma: list | None,
    mu_Y_jk: list | None,
) -> np.ndarray:
    """
    Whiten innovations by the (approximate) marginal innovation covariance S.

    Two modes:

    * ngh_kf (all three extra arguments provided):
      S = Σ_{j,k} w_{jk} [Γ(j,k) + δ_{jk} δ_{jk}ᵀ]
      where δ_{jk} = μ_{Y,jk} − Σ w μ_{Y} is the deviation of the
      component mean from the mixture mean.  S is the *stationary*
      marginal innovation covariance.

    * gpb2 (extra arguments None):
      S is estimated from the sample covariance of the innovations.

    Returns  ν̃ = L⁻¹ ν   (shape same as *innovations*), where S = L Lᵀ.
    Under a well-calibrated filter each ν̃ᵢ is approximately N(0, 1).
    """
    s = innovations.shape[1]

    if mix_w is not None and Gamma is not None and mu_Y_jk is not None:
        # Stationary marginal covariance
        K_mix = mix_w.shape[0]
        mu_marg = np.zeros((s, 1))
        for j in range(K_mix):
            for k in range(K_mix):
                mu_marg += float(mix_w[j, k]) * mu_Y_jk[j][k]

        S = np.zeros((s, s))
        for j in range(K_mix):
            for k in range(K_mix):
                w = float(mix_w[j, k])
                if w < 1e-12:
                    continue
                delta = mu_Y_jk[j][k] - mu_marg  # (s, 1)
                S += w * (Gamma[j][k] + delta @ delta.T)
    else:
        # Sample covariance fallback (gpb2 mode)
        raw = innovations.T  # (s, N)
        S = np.cov(raw) if s > 1 else np.array([[float(np.var(innovations[:, 0]))]])

    # Cholesky + solve: ν̃ = L⁻¹ ν  →  each column has unit variance
    try:
        L = np.linalg.cholesky(_sym_reg(S, 1e-10))
        return np.linalg.solve(L, innovations.T).T  # (N, s)
    except np.linalg.LinAlgError:
        return innovations  # give up and return raw


def _sym_reg(M: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Return (M + Mᵀ)/2 + eps·I  (symmetrised + regularised)."""
    n = M.shape[0]
    return 0.5 * (M + M.T) + eps * np.eye(n)


# ---------------------------------------------------------------------------
# Stationary distribution helper
# ---------------------------------------------------------------------------


def _stationary_dist(P: np.ndarray) -> np.ndarray | None:
    """Stationary distribution of a row-stochastic matrix (left eigenvector)."""
    try:
        vals, vecs = np.linalg.eig(P.T)
        idx = int(np.argmin(np.abs(vals - 1.0)))
        pi = np.real(vecs[:, idx])
        pi = np.maximum(pi, 0.0)
        s = pi.sum()
        return pi / s if s > 1e-12 else None
    except np.linalg.LinAlgError:
        return None
