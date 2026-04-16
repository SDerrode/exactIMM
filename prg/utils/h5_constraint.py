#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prg/utils/h5_constraint.py
==========================
Enforce the H5 constraint (eq. 4.8) on a GSSParams object.

The constraint says that, given A(k), C(k), D(k), Σ_U(k), Δ(k), Σ_V(k),
the block B(k) of the transition matrix F(k) is uniquely determined by:

    B(k)ᵀ = L(k)⁻¹ · (P(k) M(k)⁻¹ (Q(k) A(k)ᵀ + Δ(k)ᵀ) − Δ(k)ᵀ A(k))

where (subscript k omitted):
    P = Δᵀ Cᵀ + Σ_V Dᵀ        (s × s)
    Q = C Σ_U + D Δᵀ           (s × q)
    R = C Δ + D Σ_V  (= Pᵀ)   (s × s)
    M = Q Cᵀ + R Dᵀ + Σ_V     (s × s)   symmetric, > 0 if Σ_U, Σ_V > 0
    L = Σ_V − P M⁻¹ Pᵀ        (s × s)   Schur complement of M

Public API
----------
apply_h5_constraint(params, *, logger=None) -> GSSParams
    Return a **new** GSSParams whose B(k) blocks satisfy eq. (4.8).
    Raises ValueError if the system is singular for any k.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from prg.classes.GSSParams import GSSParams

__all__ = ["apply_h5_constraint", "compute_B_from_h5"]


# ---------------------------------------------------------------------------
# Core formula
# ---------------------------------------------------------------------------

def compute_B_from_h5(
    A:  np.ndarray,   # (q, q)
    C:  np.ndarray,   # (s, q)
    D:  np.ndarray,   # (s, s)
    SU: np.ndarray,   # (q, q)  Σ_U
    Dt: np.ndarray,   # (q, s)  Δ
    SV: np.ndarray,   # (s, s)  Σ_V
) -> np.ndarray:
    """
    Compute B (q × s) from the H5 constraint (eq. 4.8).

    Returns
    -------
    ndarray of shape (q, s)

    Raises
    ------
    ValueError
        If M or L (Schur complement) is singular or numerically ill-conditioned.
    """
    P = Dt.T @ C.T + SV @ D.T
    Q = C @ SU + D @ Dt.T
    R = C @ Dt + D @ SV
    M = Q @ C.T + R @ D.T + SV   # s × s, symmetric

    cond_M = np.linalg.cond(M)
    if cond_M > 1e12:
        raise ValueError(
            f"M is ill-conditioned (cond = {cond_M:.3e}); "
            "cannot reliably solve for B."
        )

    # PM_inv = P M⁻¹  — solved as (Mᵀ \ Pᵀ)ᵀ to avoid forming explicit inverse
    try:
        PM_inv = np.linalg.solve(M.T, P.T).T
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"M is singular: {exc}") from exc

    L   = SV - PM_inv @ R                       # Schur complement, s × s
    rhs = PM_inv @ (Q @ A.T + Dt.T) - Dt.T @ A  # s × q

    cond_L = np.linalg.cond(L)
    if cond_L > 1e12:
        raise ValueError(
            f"L is ill-conditioned (cond = {cond_L:.3e}); "
            "cannot reliably solve for B."
        )

    try:
        B_T = np.linalg.solve(L, rhs)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"L (Schur complement) is singular: {exc}") from exc

    if not np.isfinite(B_T).all():
        raise ValueError("B_T contains non-finite values after solving.")

    return B_T.T   # q × s


# ---------------------------------------------------------------------------
# High-level function
# ---------------------------------------------------------------------------

def apply_h5_constraint(
    params: "GSSParams",
    *,
    logger: logging.Logger | None = None,
) -> "GSSParams":
    """
    Return a **new** GSSParams whose B(k) blocks satisfy the H5 constraint.

    For each regime k, B(k) is replaced by the solution of eq. (4.8).
    All other parameters (A, C, D, noise covariances, biases, …) are
    preserved unchanged.

    Parameters
    ----------
    params : GSSParams
        Original parameter set.
    logger : logging.Logger, optional
        If provided, INFO messages report the per-regime B correction and
        the Frobenius-norm distance ‖B_new − B_old‖_F.

    Returns
    -------
    GSSParams
        New parameter set with corrected B matrices.

    Raises
    ------
    ValueError
        If the constraint system is singular for any regime k.
    """
    # Import here to avoid circular imports at module level
    from prg.classes.FMatrix import FMatrix
    from prg.classes.GSSParams import GSSParams

    log = logger or logging.getLogger("fofgss.h5_constraint")
    K, q, s = params.K, params.q, params.s

    A_list: list[np.ndarray] = []
    B_list: list[np.ndarray] = []
    C_list: list[np.ndarray] = []
    D_list: list[np.ndarray] = []

    for k in range(K):
        A  = params.f_matrix.A(k)
        C  = params.f_matrix.C(k)
        D  = params.f_matrix.D(k)
        SU = params.noise_cov.Sigma_U(k)
        Dt = params.noise_cov.Delta(k)
        SV = params.noise_cov.Sigma_V(k)

        try:
            B_new = compute_B_from_h5(A, C, D, SU, Dt, SV)
        except ValueError as exc:
            raise ValueError(
                f"H5 constraint cannot be satisfied for regime k={k}: {exc}"
            ) from exc

        B_old = params.f_matrix.B(k)
        delta = float(np.linalg.norm(B_new - B_old, "fro"))
        log.info(
            "k=%d  B corrected  ‖ΔB‖_F = %.4g  "
            "(old %s → new %s)",
            k, delta,
            np.array2string(B_old.ravel(), precision=4, suppress_small=True),
            np.array2string(B_new.ravel(), precision=4, suppress_small=True),
        )
        A_list.append(A)
        B_list.append(B_new)
        C_list.append(C)
        D_list.append(D)

    # Build updated FMatrix
    new_f_matrix = FMatrix(
        K=K, q=q, s=s,
        A_list=A_list,
        B_list=B_list,
        C_list=C_list,
        D_list=D_list,
    )

    # Rebuild GSSParams (all other fields unchanged)
    return GSSParams(
        K=K, q=q, s=s,
        P=params.P,
        f_matrix=new_f_matrix,
        noise_cov=params.noise_cov,
        pi0=params.pi0,
        mu_z0_list=[params.mu_z0(k) for k in range(K)],
        Sigma_z0_list=[params.Sigma_z0(k) for k in range(K)],
        b_list=[params.b(k) for k in range(K)],
    )
