#!/usr/bin/env python3
"""
prg/utils/exactness.py
======================
Exactness domains of the three filters, as block-level residuals.

The companion paper *When Are the IMM and GPB2 Filters Exact?* characterizes,
within Gaussian pairwise Markov switching models, the models on which the
classical collapse filters reproduce the exact Bayes filter:

* order-1 IMM      -- exact iff  C_r = 0 for every regime r  (conditional
                      autonomy of the observation);
* order-2 GPB2     -- exact iff  C_k (A_j - M_j C_j) = 0 for every ordered pair
                      of regimes (j, k)  (*cross-annihilation*: every
                      observation row kills every residual state memory
                      N_j = A_j - M_j C_j, with M_j = Delta_j Sigma_V,j^-1); for a
                      scalar state this is exactly "C = 0 at every regime, or
                      A = M C at every regime", and in higher dimension it also
                      admits a memory confined to the kernel of the channel;
* constant gain    -- exact iff the AB constraint holds
                      (:func:`prg.utils.ab_constraint.ab_residual_max`).

All three are stated under the non-degeneracy assumption (G) of the paper. The
functions below return *relative* residuals (0 on the domain, up to ~1e-15 in
floating point) so that a user can test where a fitted or designed model sits.
"""

from __future__ import annotations

import numpy as np

from prg.classes.GSSParams import GSSParams
from prg.utils.ab_constraint import ab_residual_max

__all__ = [
    "residual_memory",
    "cross_annihilation_residual",
    "imm_domain_residual",
    "gpb2_domain_residual",
    "exactness_domains",
]


def residual_memory(params: GSSParams, j: int) -> np.ndarray:
    """``N_j = A_j - M_j C_j``: the state memory left once the observation
    channel has been regressed out (shape (q, q))."""
    fm = params.f_matrix
    return fm.A(j) - params.noise_cov.M(j) @ fm.C(j)


def cross_annihilation_residual(
    params: GSSParams,
    *,
    relative: bool = True,
) -> tuple[float, tuple[int, int]]:
    """Largest ``‖C_k N_j‖_F`` over all ordered regime pairs (j, k).

    Returns ``(max_resid, (j, k))``. ``max_resid = 0`` ⇔ the model lies in the
    cross-annihilation family, i.e. GPB2 is exact. With ``relative=True`` each
    term is divided by ``‖C_k‖_F ‖N_j‖_F`` (pairs with a zero factor contribute
    0, they are trivially annihilated).
    """
    K = params.K
    C = [params.f_matrix.C(k) for k in range(K)]
    N = [residual_memory(params, j) for j in range(K)]
    worst, arg = 0.0, (0, 0)
    for j in range(K):
        nN = float(np.linalg.norm(N[j], "fro"))
        for k in range(K):
            nC = float(np.linalg.norm(C[k], "fro"))
            r = float(np.linalg.norm(C[k] @ N[j], "fro"))
            if relative:
                r = 0.0 if nC * nN == 0.0 else r / (nC * nN)
            if r > worst:
                worst, arg = r, (j, k)
    return worst, arg


def imm_domain_residual(params: GSSParams, *, relative: bool = True) -> float:
    """Largest ``‖C_r‖_F`` over the regimes (relative to ``‖F_r‖_F``): 0 ⇔ the
    observation is conditionally autonomous ⇔ the order-1 IMM is exact."""
    fm = params.f_matrix
    worst = 0.0
    for r in range(params.K):
        v = float(np.linalg.norm(fm.C(r), "fro"))
        if relative:
            v /= max(float(np.linalg.norm(fm.F(r), "fro")), 1e-300)
        worst = max(worst, v)
    return worst


def gpb2_domain_residual(params: GSSParams, *, relative: bool = True) -> float:
    """Cross-annihilation residual, scalar form of
    :func:`cross_annihilation_residual`: 0 ⇔ GPB2 is exact."""
    return cross_annihilation_residual(params, relative=relative)[0]


def exactness_domains(params: GSSParams, *, tol: float = 1e-10) -> dict[str, bool]:
    """Which classical filters are exact on ``params`` (under assumption (G)).

    Returns ``{"imm": ..., "gpb2": ..., "constant_gain": ...}``; the residuals
    behind the booleans are :func:`imm_domain_residual`,
    :func:`gpb2_domain_residual` and
    :func:`prg.utils.ab_constraint.ab_residual_max`.
    """
    return {
        "imm": imm_domain_residual(params) <= tol,
        "gpb2": gpb2_domain_residual(params) <= tol,
        "constant_gain": ab_residual_max(params)[0] <= tol,
    }
