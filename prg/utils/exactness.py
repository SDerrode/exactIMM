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
    "assumption_g",
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
    """Which classical filters are exact on ``params``, and whether the
    non-degeneracy Assumption (G) under which the theorems hold is satisfied.

    Returns ``{"imm": ..., "gpb2": ..., "constant_gain": ..., "assumption_g": ...}``;
    the residuals behind the booleans are :func:`imm_domain_residual`,
    :func:`gpb2_domain_residual`, :func:`prg.utils.ab_constraint.ab_residual_max`
    and :func:`assumption_g`. When ``assumption_g`` is False the three domain
    flags are still the block conditions of the theorems, but their *necessity*
    is no longer guaranteed (a filter may be exact off its domain).
    """
    return {
        "imm": imm_domain_residual(params) <= tol,
        "gpb2": gpb2_domain_residual(params) <= tol,
        "constant_gain": ab_residual_max(params)[0] <= tol,
        "assumption_g": assumption_g(params)["g"],
    }


def assumption_g(params: GSSParams, *, tol: float = 1e-12) -> dict:
    """Non-degeneracy Assumption (G) of the exactness paper, checked on the blocks.

    * **(G1) full support** -- ``p_jk > 0`` for every pair of regimes.
    * **(G2) observational separation of the regime** -- the rows of the
      transition matrix are not all identical, *or* the state-informed part of
      the observation row, i.e. the four quantities ``Sigma_V^-1 C``,
      ``C^T Sigma_V^-1 C``, ``C^T Sigma_V^-1 D`` and ``C^T Sigma_V^-1 b^Y``,
      depends on the regime.

    Returns a dict with the residuals ``g1_min_p`` (smallest transition
    probability), ``g2_rows`` (largest relative difference between two rows of
    ``P``), ``g2_channel`` (largest relative difference, over regime pairs, of
    the four state-informed quantities), and the booleans ``g1``, ``g2``,
    ``g`` (= ``g1 and g2``). Both theorems of the paper are stated under (G);
    off it, a collapse filter can be exact for reasons unrelated to its
    collapse rule (Remark "Assumption (G2) is sharp", experiment E8).

    Note: the paper reads (G2) on classes of *observational twins*; this check
    is the plain, per-regime version, which is what the experiments of the
    committed suite rely on.
    """
    K, q = params.K, params.q
    P = np.asarray(params.P, dtype=float)
    g1_min_p = float(P.min())
    # (G2a) rows of P
    row_diff = 0.0
    for j in range(K):
        for jp in range(j + 1, K):
            row_diff = max(row_diff, float(np.max(np.abs(P[j] - P[jp]))))
    g2_rows = row_diff  # rows are probability vectors: already a relative scale
    # (G2b) state-informed part of the observation row
    fm, nc = params.f_matrix, params.noise_cov
    quantities = []
    for r in range(K):
        C, D = fm.C(r), fm.D(r)
        SVi = np.linalg.inv(nc.Sigma_V(r))
        bY = np.asarray(params.b(r), dtype=float).reshape(-1)[q:]
        quantities.append([SVi @ C, C.T @ SVi @ C, C.T @ SVi @ D, C.T @ SVi @ bY])
    g2_channel = 0.0
    for idx in range(4):
        scale = max(float(np.linalg.norm(quantities[r][idx])) for r in range(K))
        if scale == 0.0:
            continue
        for r in range(K):
            for rp in range(r + 1, K):
                d = float(np.linalg.norm(quantities[r][idx] - quantities[rp][idx])) / scale
                g2_channel = max(g2_channel, d)
    g1 = g1_min_p > tol
    g2 = (g2_rows > tol) or (g2_channel > tol)
    return {
        "g1_min_p": g1_min_p,
        "g2_rows": g2_rows,
        "g2_channel": g2_channel,
        "g1": g1,
        "g2": g2,
        "g": g1 and g2,
    }
