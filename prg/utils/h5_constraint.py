#!/usr/bin/env python3
"""
prg/utils/h5_constraint.py
==========================
Closed-form (H5)-compatible "AB constraint" parametrisation.

The (H5) algebraic constraint, derived from the Markovianity of (R, Y)
(paper appendix B), reads

    Δᵀ Aᵀ + Σ_V Bᵀ  =  P M⁻¹ (Q Aᵀ + R Bᵀ + Δᵀ),

with P = Δᵀ Cᵀ + Σ_V Dᵀ, Q = C Σ_U + D Δᵀ, R = C Δ + D Σ_V,
M = Q Cᵀ + R Dᵀ + Σ_V. The closed form

    A = Δ Σ_V⁻¹ C,        B = Δ Σ_V⁻¹ D

is **sufficient** for (H5): a model with these blocks satisfies the
K² regime-pair equations of (H5) by construction, for any choice of
(C, D, Σ_U, Σ_V, Δ).

Necessity is more subtle. Under the physical hypothesis that the K
regime-noise covariances Σ(r) = [[Σ_U,r, Δ_r]; [Δᵀ_r, Σ_V,r]] are
all positive definite, an elimination argument shows that AB is also
**necessary** — i.e., the unique (H5)-compatible parametrisation —
generically when K·s ≥ q + s. In the sub-determined regime
K·s < q + s, (H5)-compatible models exist that are not of the AB
form: AB is one specific point in a (q+s−Ks)·q-dimensional affine
space of solutions per regime. Most practical configurations
(K=2 or 3, q=s) fall in the over- or exactly-determined regime
where AB ≡ (H5).

Public API
----------
compute_AB(C, D, Dt, SV) -> (A, B)
    Closed-form A, B from C, D, Δ, Σ_V.
apply_AB_constraint(params, *, logger=None) -> GSSParams
    Return a new GSSParams with each regime's (A_k, B_k) replaced by
    the closed form.
compute_h5_residual(A, B, C, D, SU, Dt, SV) -> ndarray (s, q)
    Same-regime (k, k) residual. Frobenius-norm-zero is *necessary* for
    (H5) but not sufficient (it ignores the cross pairs j != k).
compute_h5_pair_residual(A_k, B_k, C_k, D_k, Dt_k, SV_k, SU_j, Dt_j, SV_j) -> (q, s)
    Pairwise residual beta_1(j, k) for the ordered pair (source j -> target k);
    zero for all K^2 pairs <=> (H5) holds.
h5_residual_max(params) -> (max_resid, (j, k))
    Complete (H5) check: max pairwise residual over all ordered pairs.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from prg.classes.GSSParams import GSSParams

__all__ = [
    "NGH_MSM_RESID_TOL",
    "apply_AB_constraint",
    "compute_AB",
    "compute_h5_pair_residual",
    "compute_h5_residual",
    "h5_residual_max",
    "is_ngh_msm",
    "validate_ngh_msm",
]

# Default tolerance on the *relative* pairwise (H5) residual used by
# validate_ngh_msm / is_ngh_msm. Mirrors GSSFilter.H5_TOL so that a model
# accepted here is also accepted (no warning) by mode="h5_exact".
NGH_MSM_RESID_TOL = 1e-6


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

    This is the **same-regime** constraint (the regime pair ``(k, k)``). It is
    a *necessary* condition for (H5), but **not sufficient**: when the regimes
    have different joint covariances, (H5) also constrains the cross pairs
    ``(j, k)``, ``j ≠ k``. Use :func:`h5_residual_max` (or
    :func:`compute_h5_pair_residual`) for the complete, all-pairs check.

    Returns
    -------
    F : ndarray of shape (s, q).  ``‖F‖ = 0`` ⇔ the same-regime ``(k, k)``
        condition holds (necessary for (H5)).

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
    residual: np.ndarray = Z - P @ np.linalg.solve(M, W)
    return residual


# ---------------------------------------------------------------------------
# Pairwise (H5) residual — the *complete* (H5)-compatibility check
# ---------------------------------------------------------------------------
def _h5_pair_beta(
    A_k: np.ndarray,  # (q, q)  target-regime k dynamics
    B_k: np.ndarray,  # (q, s)
    C_k: np.ndarray,  # (s, q)
    D_k: np.ndarray,  # (s, s)
    Delta_k: np.ndarray,  # (q, s)  Cov(W_X, W_Y | r=k)
    SV_k: np.ndarray,  # (s, s)   Var(W_Y | r=k)
    SU_j: np.ndarray,  # (q, q)  source-regime j joint-covariance blocks
    Delta_j: np.ndarray,  # (q, s)
    SV_j: np.ndarray,  # (s, s)
) -> np.ndarray:
    """Regression of X_{n+1} on (Y_n, Y_{n+1}) given r_n=j, r_{n+1}=k.

    Returns ``beta`` of shape (q, 2s); the first s columns are the loading on
    Y_n (``beta_1``), the last s the loading on Y_{n+1}.

    Raises numpy.linalg.LinAlgError if the joint (Y_n, Y_{n+1}) covariance is
    singular.
    """
    cov_Xn1_Yn = A_k @ Delta_j + B_k @ SV_j
    cov_Xn1_Yn1 = (
        A_k @ SU_j @ C_k.T
        + A_k @ Delta_j @ D_k.T
        + B_k @ Delta_j.T @ C_k.T
        + B_k @ SV_j @ D_k.T
        + Delta_k
    )
    cov_Yn_Yn1 = Delta_j.T @ C_k.T + SV_j @ D_k.T
    var_Yn1 = (
        C_k @ SU_j @ C_k.T
        + C_k @ Delta_j @ D_k.T
        + D_k @ Delta_j.T @ C_k.T
        + D_k @ SV_j @ D_k.T
        + SV_k
    )
    V = np.block([[SV_j, cov_Yn_Yn1], [cov_Yn_Yn1.T, var_Yn1]])  # (2s, 2s)
    cov_X = np.hstack([cov_Xn1_Yn, cov_Xn1_Yn1])  # (q, 2s)
    beta: np.ndarray = np.linalg.solve(V, cov_X.T).T  # (q, 2s)
    return beta


def compute_h5_pair_residual(
    A_k: np.ndarray,
    B_k: np.ndarray,
    C_k: np.ndarray,
    D_k: np.ndarray,
    Delta_k: np.ndarray,
    SV_k: np.ndarray,
    SU_j: np.ndarray,
    Delta_j: np.ndarray,
    SV_j: np.ndarray,
) -> np.ndarray:
    """Pairwise (H5) residual ``β₁(j, k)`` for the ordered pair (source j → target k).

    (H5) — the Markovianity of ``(R, Y)`` — holds for the pair (j, k) iff,
    conditionally on ``r_n = j`` and ``r_{n+1} = k``, the regression of
    ``X_{n+1}`` on ``(Y_n, Y_{n+1})`` does *not* load on ``Y_n``. This returns
    that loading ``β₁`` (shape ``(q, s)``); ``‖β₁‖ = 0`` ⇔ (H5) holds for the
    pair. The model is (H5)-compatible iff ``β₁(j, k) = 0`` for **all** K²
    ordered pairs.

    The single-regime :func:`compute_h5_residual` only covers the same-regime
    pairs ``(k, k)``: that is **necessary but not sufficient** for (H5) whenever
    the regimes have different joint covariances. Use :func:`h5_residual_max`
    for the complete check.

    Target-regime arguments ``A_k, B_k, C_k, D_k`` (dynamics), ``Delta_k``
    (``Cov(W_X, W_Y | k)``), ``SV_k`` (``Var(W_Y | k)``); source-regime joint
    noise-covariance blocks ``SU_j, Delta_j, SV_j``.

    Raises numpy.linalg.LinAlgError if the joint (Y_n, Y_{n+1}) covariance is
    singular.
    """
    s = SV_j.shape[0]
    return _h5_pair_beta(A_k, B_k, C_k, D_k, Delta_k, SV_k, SU_j, Delta_j, SV_j)[:, :s]


def h5_residual_max(
    params: GSSParams,
    *,
    relative: bool = True,
) -> tuple[float, tuple[int, int]]:
    """Largest pairwise (H5) residual over all K² ordered regime pairs.

    Returns ``(max_resid, (j, k))`` where ``max_resid`` is the maximum over all
    ordered pairs (j, k) of ``‖β₁(j, k)‖_F`` — relative to the regression scale
    ``‖[β₁ β₂](j, k)‖_F`` when ``relative=True``. ``max_resid = 0`` ⇔ the model
    is fully (H5)-compatible; ``inf`` is returned if some pair's (Y_n, Y_{n+1})
    covariance is singular.

    This is the **complete** (H5) check. :func:`compute_h5_residual` only tests
    the same-regime pairs (necessary, not sufficient).
    """
    K = params.K
    A = [params.f_matrix.A(k) for k in range(K)]
    B = [params.f_matrix.B(k) for k in range(K)]
    C = [params.f_matrix.C(k) for k in range(K)]
    D = [params.f_matrix.D(k) for k in range(K)]
    SU = [params.noise_cov.Sigma_U(k) for k in range(K)]
    Dt = [params.noise_cov.Delta(k) for k in range(K)]
    SV = [params.noise_cov.Sigma_V(k) for k in range(K)]

    s = params.s
    worst = 0.0
    arg: tuple[int, int] = (0, 0)
    for j in range(K):
        for k in range(K):
            try:
                beta = _h5_pair_beta(A[k], B[k], C[k], D[k], Dt[k], SV[k], SU[j], Dt[j], SV[j])
            except np.linalg.LinAlgError:
                return float("inf"), (j, k)
            r = float(np.linalg.norm(beta[:, :s], "fro"))
            if relative:
                r /= max(float(np.linalg.norm(beta, "fro")), 1e-300)
            if r > worst:
                worst = r
                arg = (j, k)
    return worst, arg


# ---------------------------------------------------------------------------
# NGH-MSM validity (the corrected CNS of Proposition 2)
# ---------------------------------------------------------------------------
def _min_eig_sym(M: np.ndarray) -> float:
    """Smallest eigenvalue of the symmetric part of M (real, finite or -inf)."""
    Msym = 0.5 * (M + M.T)
    try:
        return float(np.linalg.eigvalsh(Msym)[0])
    except np.linalg.LinAlgError:
        return float("-inf")


def validate_ngh_msm(
    params: GSSParams,
    *,
    tol: float = NGH_MSM_RESID_TOL,
) -> list[str]:
    """List the ways ``params`` violates the corrected NGH-MSM condition (Prop. 2).

    Returns an empty list iff ``params`` is a valid NGH-MSM: a model of the AB
    family whose structural hypotheses all hold. Each non-empty entry is a
    human-readable description of one violated condition. Nothing is raised — the
    caller decides what to do (cf. :meth:`GSSParams.check_ngh_msm`).

    Conditions checked (per the corrected Proposition 2)
    ----------------------------------------------------
    1. ``C_k ≠ 0``            ∀ k        — this is a genuine NGH-MSM (the new
       family), not the degenerate ``C_k = 0`` case, which is a classical CMS-HLM
       (the (H4) family). NB: ``C_k ≠ 0`` is a *family-membership* requirement, not
       a mathematical prerequisite of the AB / NSC. AB is the necessary-and-
       sufficient (H5) parametrisation for *any* ``C_k`` given only ``Σ_V_k ≻ 0``,
       ``C_k = 0`` included — where it gives the CMS-HLM's ``A_k = 0``,
       ``B_k = Δ_k Σ_V_k⁻¹ D_k``. Full column rank of ``C_k`` (hence ``s ≥ q``) and
       invertibility of ``D_k`` are likewise *not* required: the corrected
       Proposition 2 needs only ``Σ_V_k ≻ 0`` (see the module docstring).
    2. ``Σ_V_k ≻ 0``          ∀ k        — symmetric positive definite;
    3. ``Γ_k = Σ_U_k − Δ_k Σ_V_k⁻¹ Δ_k^T ⪰ 0`` ∀ k — the Schur complement is a
       genuine covariance (⇔ the joint noise covariance Σ_W(k) is PSD);
    4. AB / (H5) constraint   — ``max`` relative pairwise residual ``≤ tol``,
       i.e. ``A_k = Δ_k Σ_V_k⁻¹ C_k`` and ``B_k = Δ_k Σ_V_k⁻¹ D_k`` (up to tol).

    Conditions (1)–(4) are exactly what makes the exact fast filter of
    Proposition 4 (closed form ``M_k = Δ_k Σ_V_k⁻¹``, ``Γ_k`` constant in n)
    applicable. They are *not* imposed at ``GSSParams`` construction, because the
    same class also serves non-(H5) models handled by ``mode="imm_general"``.
    """
    issues: list[str] = []
    K = params.K

    for k in range(K):
        C = params.f_matrix.C(k)
        SU = params.noise_cov.Sigma_U(k)
        Dt = params.noise_cov.Delta(k)
        SV = params.noise_cov.Sigma_V(k)

        # ``C_k != 0`` is a *family-membership* check, NOT a mathematical validity
        # condition: AB is the NSC (Prop. 2) for ANY ``C_k`` given only ``Σ_V ≻ 0``
        # — ``C_k = 0`` included (the Lehmann matrix-inversion argument needs no
        # condition on C, and none on D; verified numerically). At ``C_k = 0`` AB
        # gives ``A_k = 0``, ``B_k = Δ_k Σ_V_k⁻¹ D_k``: a perfectly filterable model,
        # but a classical CMS-HLM (the (H4) family), not the new NGH-MSM family. We
        # flag it so callers do not mistake the old family for the new. (Full column
        # rank of ``C_k``, hence ``s >= q``, is likewise not required. Whether AB is
        # moreover the *unique* (H5) parametrisation is a separate identifiability
        # question, governed by ``K·s >= q + s`` — see the module docstring.)
        if float(np.linalg.norm(C, "fro")) <= 1e-12:
            issues.append(
                f"regime {k}: C = 0 — this is a classical CMS-HLM (the (H4) family), "
                "not a NGH-MSM (the new family)."
            )

        ev_SV = _min_eig_sym(SV)
        if ev_SV <= 0.0:
            issues.append(f"regime {k}: Σ_V is not positive definite (min eig = {ev_SV:.3e}).")
        else:
            Gamma = SU - Dt @ np.linalg.solve(SV, Dt.T)
            ev_G = _min_eig_sym(Gamma)
            psd_tol = 1e-9 * max(1.0, float(np.linalg.norm(SU, "fro")))
            if ev_G < -psd_tol:
                issues.append(
                    f"regime {k}: Γ = Σ_U − Δ Σ_V⁻¹ Δ^T is not PSD (min eig = {ev_G:.3e}); "
                    f"the joint noise covariance Σ_W is not PSD."
                )

    try:
        max_rel, (j, k) = h5_residual_max(params, relative=True)
    except (np.linalg.LinAlgError, ValueError) as exc:  # pragma: no cover - defensive
        issues.append(f"could not evaluate the (H5) residual: {exc}")
    else:
        if not np.isfinite(max_rel):
            issues.append(
                "(H5) residual is infinite (a regime pair has a singular "
                "(Y_n, Y_{n+1}) covariance)."
            )
        elif max_rel > tol:
            issues.append(
                f"AB / (H5) constraint violated: max relative pairwise residual = "
                f"{max_rel:.3e} > tol = {tol:.1e} (worst pair j={j}, k={k}). Enforce it "
                f"with apply_AB_constraint (A = Δ Σ_V⁻¹ C, B = Δ Σ_V⁻¹ D)."
            )

    return issues


def is_ngh_msm(params: GSSParams, *, tol: float = NGH_MSM_RESID_TOL) -> bool:
    """``True`` iff ``params`` is a valid NGH-MSM (cf. :func:`validate_ngh_msm`)."""
    return not validate_ngh_msm(params, tol=tol)


# ---------------------------------------------------------------------------
# Closed-form AB constraint
# ---------------------------------------------------------------------------
def compute_AB(
    C: np.ndarray,  # (s, q)
    D: np.ndarray,  # (s, s)
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
            f"Σ_V is ill-conditioned (cond = {cond_SV:.3e}); cannot reliably solve A = Δ Σ_V⁻¹ C."
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
        A *new* object with updated A, B blocks. Because the result satisfies
        the AB constraint by construction, it is returned as the stronger type
        :class:`~prg.classes.GSSParams.NGHMSMParams` **when** the structural
        conditions also hold (C_k ≠ 0, Σ_V_k ≻ 0, Γ_k ⪰ 0). If one fails (e.g.
        C_k = 0 — the degenerate CMS-HLM family — or a non-PSD Schur complement,
        for which the projected model has no valid exact filter), a base
        :class:`GSSParams` is returned and a warning is logged, preserving the
        "project what you can" contract for CLI/script paths.

    Raises
    ------
    ValueError
        If Σ_V(k) is singular for any k (the AB blocks cannot be computed).
    """
    # Import here to avoid circular imports at module level
    from prg.classes.FMatrix import FMatrix
    from prg.classes.GSSParams import GSSParams as _GSSParams
    from prg.classes.GSSParams import NGHMSMParams as _NGHMSMParams
    from prg.utils.exceptions import ParamError as _ParamError

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
            raise ValueError(f"AB constraint cannot be applied for regime k={k}: {exc}") from exc

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
        K=K,
        q=q,
        s=s,
        A_list=A_list,
        B_list=B_list,
        C_list=C_list,
        D_list=D_list,
    )

    kwargs = dict(
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
    # A, B are AB-correct by construction, so only the *structural* CNS can
    # fail. Prefer the stronger NGHMSMParams type; if a structural condition is
    # violated, fall back to a base GSSParams (warn, do not raise) so callers
    # that "project what they can" (CLI/script --constraint) keep working.
    try:
        return _NGHMSMParams(**kwargs)
    except _ParamError as exc:
        log.warning(
            "AB constraint applied, but the result is not a fully valid NGH-MSM "
            "(%s); returning a base GSSParams.",
            exc,
        )
        return _GSSParams(**kwargs)
