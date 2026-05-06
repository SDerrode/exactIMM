#!/usr/bin/env python3
"""
prg/utils/h5_constraint.py
==========================
Closed-form (H5)-compatible "AB constraint" parametrisation.

The (H5) algebraic constraint, derived from the Markovianity of (R, Y)
(paper appendix B), reads

    Δᵀ Aᵀ + Σ_V Bᵀ  =  P M⁻¹ (Q Aᵀ + R Bᵀ + Δᵀ),

with P = Δᵀ Cᵀ + Σ_V Dᵀ, Q = C Σ_U + D Δᵀ, R = C Δ + D Σ_V,
M = Q Cᵀ + R Dᵀ + Σ_V. Requiring this to hold *uniformly* in the
joint covariance Σ(r₁) = [[Σ_U, Δ], [Δᵀ, Σ_V]] collapses A and B
onto the closed form

    A = Δ Σ_V⁻¹ C,        B = Δ Σ_V⁻¹ D.

This is the *only* parametrisation compatible with (H5) for arbitrary
Σ(r₁); the K² regime-pair equations of (H5) are then trivially
satisfied. (C, D, Σ_U, Σ_V, Δ) are free.

Public API
----------
compute_AB(C, D, Dt, SV) -> (A, B)
    Closed-form A, B from C, D, Δ, Σ_V.
apply_AB_constraint(params, *, logger=None) -> GSSParams
    Return a new GSSParams with each regime's (A_k, B_k) replaced by
    the closed form.
compute_h5_residual(A, B, C, D, SU, Dt, SV) -> ndarray (s, q)
    Frobenius-norm-zero ⇔ (H5) holds.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from prg.classes.GSSParams import GSSParams

__all__ = [
    "apply_AB_constraint",
    "compute_AB",
    "compute_h5_residual",
]


# ---------------------------------------------------------------------------
# (H5) residual
# ---------------------------------------------------------------------------
def compute_h5_residual(
    A: np.ndarray,  # (q, q)
    B: np.ndarray,  # (q, s)
    C: np.ndarray,  # (s, q)
    D: np.ndarray,  # (s, s)
    SU: np.ndarray,  # (q, q)  Σ_U
    Dt: np.ndarray,  # (q, s)  Δ
    SV: np.ndarray,  # (s, s)  Σ_V
) -> np.ndarray:
    """
    Evaluate the (H5) algebraic constraint residual

        F = (Δᵀ Aᵀ + Σ_V Bᵀ) − P M⁻¹ (Q Aᵀ + R Bᵀ + Δᵀ),

    with the auxiliary blocks

        P = Δᵀ Cᵀ + Σ_V Dᵀ          (s × s)
        Q = C Σ_U + D Δᵀ            (s × q)
        R = C Δ  + D Σ_V  (= Pᵀ)    (s × s)
        M = Q Cᵀ + R Dᵀ + Σ_V       (s × s, symmetric, ≻ 0 if Σ_U, Σ_V ≻ 0).

    Returns
    -------
    F : ndarray of shape (s, q).  ``‖F‖ = 0`` ⇔ (H5) holds exactly.

    Raises
    ------
    numpy.linalg.LinAlgError
        If M is singular.
    """
    P = Dt.T @ C.T + SV @ D.T
    Q = C @ SU + D @ Dt.T
    R = C @ Dt + D @ SV
    M = Q @ C.T + R @ D.T + SV
    W = Q @ A.T + R @ B.T + Dt.T
    Z = Dt.T @ A.T + SV @ B.T
    return Z - P @ np.linalg.solve(M, W)


# ---------------------------------------------------------------------------
# Closed-form AB constraint
# ---------------------------------------------------------------------------
def compute_AB(
    C: np.ndarray,   # (s, q)
    D: np.ndarray,   # (s, s)
    Dt: np.ndarray,  # (q, s)  Δ
    SV: np.ndarray,  # (s, s)  Σ_V (symmetric ≻ 0)
) -> tuple[np.ndarray, np.ndarray]:
    """
    Closed-form (H5)-compatible AB parametrisation:

        A = Δ Σ_V⁻¹ C,        B = Δ Σ_V⁻¹ D.

    A model with these blocks satisfies (H5) for every regime pair
    (r₁, r₂), independently of Σ_U.

    Parameters
    ----------
    C : ndarray (s, q)
    D : ndarray (s, s)
    Dt : ndarray (q, s)  -- Δ
    SV : ndarray (s, s)  -- Σ_V (symmetric positive definite)

    Returns
    -------
    (A, B) : tuple of ndarrays of shapes (q, q) and (q, s).

    Raises
    ------
    ValueError
        If Σ_V is singular or ill-conditioned.
    """
    cond_SV = np.linalg.cond(SV)
    if cond_SV > 1e12:
        raise ValueError(
            f"Σ_V is ill-conditioned (cond = {cond_SV:.3e}); "
            "cannot reliably solve A = Δ Σ_V⁻¹ C."
        )
    try:
        SV_inv_C = np.linalg.solve(SV, C)  # Σ_V⁻¹ C  (s × q)
        SV_inv_D = np.linalg.solve(SV, D)  # Σ_V⁻¹ D  (s × s)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"Σ_V is singular: {exc}") from exc
    A = Dt @ SV_inv_C  # (q, q)
    B = Dt @ SV_inv_D  # (q, s)
    return A, B


# ---------------------------------------------------------------------------
# Application to a GSSParams object
# ---------------------------------------------------------------------------
def apply_AB_constraint(
    params: GSSParams,
    *,
    logger: logging.Logger | None = None,
) -> GSSParams:
    """
    Return a new GSSParams whose A(k), B(k) blocks are replaced by the
    closed form ``A_k = Δ_k Σ_V_k⁻¹ C_k``, ``B_k = Δ_k Σ_V_k⁻¹ D_k``
    for every regime k. C, D, Σ_U, Σ_V, Δ, Π, π₀, μ_z₀, Σ_z₀, b are
    preserved unchanged.

    Parameters
    ----------
    params : GSSParams
    logger : logging.Logger, optional
        If provided, INFO messages report the per-regime ‖A_new − A_old‖_F
        and ‖B_new − B_old‖_F.

    Returns
    -------
    GSSParams
        A *new* object with updated A, B blocks.

    Raises
    ------
    ValueError
        If Σ_V(k) is singular for any k.
    """
    # Import here to avoid circular imports at module level
    from prg.classes.FMatrix import FMatrix
    from prg.classes.GSSParams import GSSParams as _GSSParams

    log = logger or logging.getLogger("exactIMM.h5_constraint")
    K, q, s = params.K, params.q, params.s

    A_list: list[np.ndarray] = []
    B_list: list[np.ndarray] = []
    C_list: list[np.ndarray] = []
    D_list: list[np.ndarray] = []

    for k in range(K):
        C_k = params.f_matrix.C(k)
        D_k = params.f_matrix.D(k)
        Dt_k = params.noise_cov.Delta(k)
        SV_k = params.noise_cov.Sigma_V(k)

        try:
            A_new, B_new = compute_AB(C_k, D_k, Dt_k, SV_k)
        except ValueError as exc:
            raise ValueError(
                f"AB constraint cannot be applied for regime k={k}: {exc}"
            ) from exc

        A_old = params.f_matrix.A(k)
        B_old = params.f_matrix.B(k)
        log.info(
            "k=%d  AB-constraint  ‖ΔA‖_F = %.4g, ‖ΔB‖_F = %.4g",
            k,
            float(np.linalg.norm(A_new - A_old, "fro")),
            float(np.linalg.norm(B_new - B_old, "fro")),
        )
        A_list.append(A_new)
        B_list.append(B_new)
        C_list.append(C_k)
        D_list.append(D_k)

    new_f_matrix = FMatrix(
        K=K, q=q, s=s,
        A_list=A_list, B_list=B_list, C_list=C_list, D_list=D_list,
    )

    return _GSSParams(
        K=K, q=q, s=s,
        P=params.P,
        f_matrix=new_f_matrix,
        noise_cov=params.noise_cov,
        pi0=params.pi0,
        mu_z0_list=[params.mu_z0(k) for k in range(K)],
        Sigma_z0_list=[params.Sigma_z0(k) for k in range(K)],
        b_list=[params.b(k) for k in range(K)],
    )
