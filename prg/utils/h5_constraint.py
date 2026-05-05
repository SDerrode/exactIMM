#!/usr/bin/env python3
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

__all__ = [
    "apply_h5_constraint",
    "compute_B_from_h5",
    "compute_A_from_h5",
    "compute_SU_from_h5",
    "compute_C_from_h5",
    "compute_h5_residual",
]


# ---------------------------------------------------------------------------
# Residual of the (H5) algebraic constraint
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
    Evaluate the (H5) algebraic constraint residual (paper eq. 4.4).

    The constraint is

        Δᵀ A + Σ_V Bᵀ  =  P M⁻¹ (Q Aᵀ + R Bᵀ + Δᵀ)

    with the auxiliary matrices

        P = Δᵀ Cᵀ + Σ_V Dᵀ          (s × s)
        Q = C Σ_U + D Δᵀ            (s × q)
        R = C Δ  + D Σ_V  (= Pᵀ)    (s × s)
        M = Q Cᵀ + R Dᵀ + Σ_V       (s × s, symmetric, > 0 if Σ_U, Σ_V > 0)
        W = Q Aᵀ + R Bᵀ + Δᵀ        (s × q)

    Returns
    -------
    F : ndarray of shape (s, q)
        The residual ``F = Z − P M⁻¹ W`` with ``Z = Δᵀ A + Σ_V Bᵀ``.
        ``‖F‖ = 0`` ⇔ (H5) holds exactly.

    Raises
    ------
    numpy.linalg.LinAlgError
        If M is singular (cannot invert).
    """
    P = Dt.T @ C.T + SV @ D.T
    Q = C @ SU + D @ Dt.T
    R = C @ Dt + D @ SV
    M = Q @ C.T + R @ D.T + SV
    W = Q @ A.T + R @ B.T + Dt.T
    Z = Dt.T @ A + SV @ B.T
    X = np.linalg.solve(M, W)  # X = M⁻¹ W
    return Z - P @ X


# ---------------------------------------------------------------------------
# Core formula
# ---------------------------------------------------------------------------


def compute_B_from_h5(
    A: np.ndarray,  # (q, q)
    C: np.ndarray,  # (s, q)
    D: np.ndarray,  # (s, s)
    SU: np.ndarray,  # (q, q)  Σ_U
    Dt: np.ndarray,  # (q, s)  Δ
    SV: np.ndarray,  # (s, s)  Σ_V
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
    M = Q @ C.T + R @ D.T + SV  # s × s, symmetric

    cond_M = np.linalg.cond(M)
    if cond_M > 1e12:
        raise ValueError(
            f"M is ill-conditioned (cond = {cond_M:.3e}); cannot reliably solve for B."
        )

    # PM_inv = P M⁻¹  — solved as (Mᵀ \ Pᵀ)ᵀ to avoid forming explicit inverse
    try:
        PM_inv = np.linalg.solve(M.T, P.T).T
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"M is singular: {exc}") from exc

    L = SV - PM_inv @ R  # Schur complement, s × s
    rhs = PM_inv @ (Q @ A.T + Dt.T) - Dt.T @ A  # s × q

    cond_L = np.linalg.cond(L)
    if cond_L > 1e12:
        raise ValueError(
            f"L is ill-conditioned (cond = {cond_L:.3e}); cannot reliably solve for B."
        )

    try:
        B_T = np.linalg.solve(L, rhs)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"L (Schur complement) is singular: {exc}") from exc

    if not np.isfinite(B_T).all():
        raise ValueError("B_T contains non-finite values after solving.")

    return B_T.T  # q × s


def compute_A_from_h5(
    B: np.ndarray,  # (q, s)
    C: np.ndarray,  # (s, q)
    D: np.ndarray,  # (s, s)
    SU: np.ndarray,  # (q, q)  Σ_U
    Dt: np.ndarray,  # (q, s)  Δ
    SV: np.ndarray,  # (s, s)  Σ_V
) -> np.ndarray:
    """
    Compute A (q × q) from the H5 constraint with B fixed (eq. 4.8).

    Rearranging the constraint for A gives the linear system:
        G Aᵀ = rhs_A
    with  G = PM⁻¹Q − Δᵀ  (s × q)
          rhs_A = L Bᵀ − PM⁻¹Δᵀ  (s × q)

    Returns
    -------
    ndarray of shape (q, q)

    Raises
    ------
    ValueError
        If M or L is ill-conditioned, or A is not uniquely determined.
    """
    P = Dt.T @ C.T + SV @ D.T
    Q = C @ SU + D @ Dt.T
    R = C @ Dt + D @ SV
    M = Q @ C.T + R @ D.T + SV

    cond_M = np.linalg.cond(M)
    if cond_M > 1e12:
        raise ValueError(
            f"M is ill-conditioned (cond = {cond_M:.3e}); cannot reliably solve for A."
        )

    try:
        PM_inv = np.linalg.solve(M.T, P.T).T
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"M is singular: {exc}") from exc

    L = SV - PM_inv @ R

    cond_L = np.linalg.cond(L)
    if cond_L > 1e12:
        raise ValueError(
            f"L is ill-conditioned (cond = {cond_L:.3e}); cannot reliably solve for A."
        )

    G = PM_inv @ Q - Dt.T  # s × q
    rhs_A = L @ B.T - PM_inv @ Dt.T  # s × q

    # Solve G @ A^T = rhs_A (lstsq handles s ≠ q)
    A_T, _, rank, _ = np.linalg.lstsq(G, rhs_A, rcond=None)

    if rank < min(G.shape):
        raise ValueError(f"G is rank-deficient (rank={rank}); A is not uniquely determined.")

    if not np.isfinite(A_T).all():
        raise ValueError("A contains non-finite values after solving.")

    return A_T.T  # q × q


def compute_SU_from_h5(
    A: np.ndarray,  # (q, q)
    B: np.ndarray,  # (q, s)
    C: np.ndarray,  # (s, q)
    D: np.ndarray,  # (s, s)
    Dt: np.ndarray,  # (q, s)  Δ
    SV: np.ndarray,  # (s, s)  Σ_V
) -> np.ndarray:
    """
    Compute Σ_U (q × q) from the H5 constraint with A, B, C, D, Δ, Σ_V fixed.

    Multiplying the constraint through by M eliminates M⁻¹ and yields
    a linear equation in Σ_U:
        C Σ_U E − (PC) Σ_U Aᵀ = RHS
    where  E = CᵀZ,  Z = Σ_V Bᵀ + ΔᵀA,  PC = P C.

    Vectorising with vec(XYZ) = (Zᵀ⊗X) vec(Y) gives the (qs × q²) system:
        [(Eᵀ⊗C) − (A⊗PC)] vec(Σ_U) = vec(RHS)

    Returns
    -------
    ndarray of shape (q, q), symmetric positive definite

    Raises
    ------
    ValueError
        If the system is rank-deficient or Σ_U is not positive definite.
    """
    P = Dt.T @ C.T + SV @ D.T  # s × s
    Q0 = D @ Dt.T  # s × q  (Q = C Σ_U + Q0)
    R = C @ Dt + D @ SV  # s × s
    M0 = Q0 @ C.T + R @ D.T + SV  # s × s  (M = C Σ_U Cᵀ + M0)

    Z = SV @ B.T + Dt.T @ A  # s × q
    W = Q0 @ A.T + R @ B.T + Dt.T  # s × q
    RHS = P @ W - M0 @ Z  # s × q

    E = C.T @ Z  # q × q
    PC = P @ C  # s × q
    KronMat = np.kron(E.T, C) - np.kron(A, PC)  # (qs × q²)
    rhs_vec = RHS.ravel(order="F")  # (qs,)

    SU_vec, _, rank, _ = np.linalg.lstsq(KronMat, rhs_vec, rcond=None)

    q = A.shape[0]
    if rank < q * q:
        raise ValueError(
            f"Kronecker system is rank-deficient (rank={rank} < {q * q}); "
            "Σ_U is not uniquely determined."
        )

    SU = SU_vec.reshape(q, q, order="F")
    SU = (SU + SU.T) / 2  # enforce symmetry

    if not np.isfinite(SU).all():
        raise ValueError("Σ_U contains non-finite values after solving.")

    try:
        np.linalg.cholesky(SU)
    except np.linalg.LinAlgError:
        raise ValueError("Computed Σ_U is not positive definite.")

    return SU


def compute_C_from_h5(
    A: np.ndarray,  # (q, q)
    B: np.ndarray,  # (q, s)
    D: np.ndarray,  # (s, s)
    SU: np.ndarray,  # (q, q)  Σ_U
    Dt: np.ndarray,  # (q, s)  Δ
    SV: np.ndarray,  # (s, s)  Σ_V
    *,
    C_init: np.ndarray | None = None,
    max_iter: int = 50,
    tol: float = 1e-9,
    lambda_relax: float = 0.5,
) -> np.ndarray:
    """
    Compute C (s × q) from the H5 constraint (eq. 4.20) by fixed-point iteration.

    Unlike A, B, and Σ_U — which give linear systems — the H5 constraint is
    *quadratic* in C (eq. 4.20).  This function uses Scheme 2 (eq. 4.23/4.24):
    at each iteration C is kept only in W = C K + W₀ while M̃ and P̃ are
    frozen at the previous iterate, giving the linear system

        P̃  C  K  =  M̃ Z − P̃ W₀

    solved by lstsq and then relaxed:

        C^(k+1)  ←  (1 − λ) C̃  +  λ C_lstsq

    Convergence is monitored on the full non-linearised residual

        F(C) = M(C) Z − P(C) W(C)

    Initialising at C_init = 0 selects the branch closest to the CGOMSM
    solution (C = 0 corresponds to hypothesis H4).

    Parameters
    ----------
    A, B, D, SU, Dt, SV : ndarray
        The six fixed matrices (shapes as in compute_B_from_h5).
    C_init : ndarray of shape (s, q), optional
        Initial iterate.  Default: zeros (CGOMSM branch).
    max_iter : int
        Maximum number of iterations (default 50).
    tol : float
        Convergence threshold on ‖F(C)‖_F / max(1, ‖Z‖_F) (default 1e-9).
    lambda_relax : float
        Relaxation factor ∈ (0, 1] (default 0.5).

    Returns
    -------
    ndarray of shape (s, q)

    Raises
    ------
    ValueError
        If M̃ or the Kronecker system is ill-conditioned, if C is not
        uniquely determined, or if the iteration does not converge.
    """
    q = A.shape[0]
    s = D.shape[0]
    C = np.zeros((s, q)) if C_init is None else np.array(C_init, dtype=float)

    # Quantities constant in C
    Z = Dt.T @ A + SV @ B.T  # s × q  (= lhs of H5)
    K = SU @ A.T + Dt @ B.T  # q × q
    W0 = D @ Dt.T @ A.T + D @ SV @ B.T + Dt.T  # s × q  (= W at C=0)
    Z_norm = max(float(np.linalg.norm(Z, "fro")), 1.0)

    def _residual(C_: np.ndarray) -> np.ndarray:
        """F(C) = Z − P M⁻¹ W  (the H5 constraint in residual form)."""
        Q_ = C_ @ SU + D @ Dt.T
        R_ = C_ @ Dt + D @ SV
        P_ = Dt.T @ C_.T + SV @ D.T
        M_ = Q_ @ C_.T + R_ @ D.T + SV
        W_ = Q_ @ A.T + R_ @ B.T + Dt.T
        # Solve M_ X = W_  ⟹  X = M_⁻¹ W_
        X, _, _, _ = np.linalg.lstsq(M_, W_, rcond=1e-12)
        return Z - P_ @ X

    res_norm = float(np.linalg.norm(_residual(C), "fro")) / Z_norm

    for _it in range(max_iter):
        if res_norm < tol:
            break

        # Freeze current C in M̃ and P̃; keep C in W = C K + W₀
        Q_t = C @ SU + D @ Dt.T
        R_t = C @ Dt + D @ SV
        P_t = Dt.T @ C.T + SV @ D.T
        M_t = Q_t @ C.T + R_t @ D.T + SV  # s × s

        # G̃ = P̃ M̃⁻¹  (s × s), computed via lstsq for stability
        # G̃ X = P_t  ⟺  M_t^T G̃^T = P_t^T
        G_t = np.linalg.lstsq(M_t.T, P_t.T, rcond=1e-12)[0].T  # s × s

        # T̃ = Z − G̃ W₀  (constant part of the rhs for this frozen iterate)
        T_t = Z - G_t @ W0  # s × q

        # Solve  G̃ C K = T̃  ⟺  (Kᵀ ⊗ G̃) vec(C) = vec(T̃)
        KronMat = np.kron(K.T, G_t)  # (sq × sq)
        rhs_vec = T_t.ravel(order="F")  # (sq,)
        C_vec, _, rank, _ = np.linalg.lstsq(KronMat, rhs_vec, rcond=1e-12)

        if rank < s * q:
            raise ValueError(
                f"Kronecker system is rank-deficient (rank={rank} < {s * q}) "
                f"at iteration {_it}; C is not uniquely determined."
            )

        C_new = C_vec.reshape(s, q, order="F")

        # Armijo-style line search: halve λ until residual decreases
        lam = lambda_relax
        for _ in range(8):
            C_try = (1.0 - lam) * C + lam * C_new
            res_try = float(np.linalg.norm(_residual(C_try), "fro")) / Z_norm
            if res_try <= res_norm or lam < 1e-4:
                break
            lam *= 0.5

        C = C_try
        res_norm = res_try

    if res_norm >= tol:
        raise ValueError(
            f"compute_C_from_h5 did not converge after {max_iter} iterations "
            f"(residual = {res_norm:.3e} > tol = {tol:.3e}). "
            "Try increasing max_iter, reducing lambda_relax, or providing C_init."
        )

    if not np.isfinite(C).all():
        raise ValueError("C contains non-finite values after iteration.")

    return C


# ---------------------------------------------------------------------------
# High-level function
# ---------------------------------------------------------------------------


def apply_h5_constraint(
    params: GSSParams,
    *,
    logger: logging.Logger | None = None,
) -> GSSParams:
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

    log = logger or logging.getLogger("exactIMM.h5_constraint")
    K, q, s = params.K, params.q, params.s

    A_list: list[np.ndarray] = []
    B_list: list[np.ndarray] = []
    C_list: list[np.ndarray] = []
    D_list: list[np.ndarray] = []

    for k in range(K):
        A = params.f_matrix.A(k)
        C = params.f_matrix.C(k)
        D = params.f_matrix.D(k)
        SU = params.noise_cov.Sigma_U(k)
        Dt = params.noise_cov.Delta(k)
        SV = params.noise_cov.Sigma_V(k)

        try:
            B_new = compute_B_from_h5(A, C, D, SU, Dt, SV)
        except ValueError as exc:
            raise ValueError(f"H5 constraint cannot be satisfied for regime k={k}: {exc}") from exc

        B_old = params.f_matrix.B(k)
        delta = float(np.linalg.norm(B_new - B_old, "fro"))
        log.info(
            "k=%d  B corrected  ‖ΔB‖_F = %.4g  (old %s → new %s)",
            k,
            delta,
            np.array2string(B_old.ravel(), precision=4, suppress_small=True),
            np.array2string(B_new.ravel(), precision=4, suppress_small=True),
        )
        A_list.append(A)
        B_list.append(B_new)
        C_list.append(C)
        D_list.append(D)

    # Build updated FMatrix
    new_f_matrix = FMatrix(
        K=K,
        q=q,
        s=s,
        A_list=A_list,
        B_list=B_list,
        C_list=C_list,
        D_list=D_list,
    )

    # Rebuild GSSParams (all other fields unchanged)
    return GSSParams(
        K=K,
        q=q,
        s=s,
        P=params.P,
        f_matrix=new_f_matrix,
        noise_cov=params.noise_cov,
        pi0=params.pi0,
        mu_z0_list=[params.mu_z0(k) for k in range(K)],
        Sigma_z0_list=[params.Sigma_z0(k) for k in range(K)],
        b_list=[params.b(k) for k in range(K)],
    )
