#!/usr/bin/env python3
"""
scripts/verify_h5_compat.py
============================
Numerical verification of the closed-form (H5)-compatible AB constraint:

    A(r) = Δ(r) Σ_V(r)⁻¹ C(r),
    B(r) = Δ(r) Σ_V(r)⁻¹ D(r).                              (★)

For each random draw of (C, D, Σ_U, Σ_V, Δ) per regime r ∈ Ω = {1, …, K}:
    1. Build A, B via formula (★).
    2. For every pair (r_1, r_2) ∈ Ω², evaluate the (H5) residual
            F  =  (Δ_{r_1}ᵀ A_{r_2}ᵀ + Σ_V_{r_1} B_{r_2}ᵀ)
                − P_{r_1,r_2} M_{r_1,r_2}⁻¹
                    (Q_{r_1,r_2} A_{r_2}ᵀ + R_{r_1,r_2} B_{r_2}ᵀ + Δ_{r_2}ᵀ)
       and report ‖F‖_F.
    3. As a negative control, replace A, B by random matrices and
       show residuals become Ω(1).

Convention
----------
We use the form

    Δᵀ Aᵀ + Σ_V Bᵀ  =  P M⁻¹ (Q Aᵀ + R Bᵀ + Δᵀ)

obtained from the appendix-B derivation `A Δ + B Σ_V = T M⁻¹ R` by
transposing both sides.

Usage
-----
    python scripts/verify_h5_compat.py [-K K] [-q q] [-s s] [-n N_DRAWS]
                                       [--seed SEED] [--tol TOL]
"""

from __future__ import annotations

import argparse
import sys

import numpy as np


# ---------------------------------------------------------------------------
# Random model generation
# ---------------------------------------------------------------------------
def make_pd(size: int, rng: np.random.Generator, scale: float = 0.5) -> np.ndarray:
    """Return a random symmetric positive-definite (size × size) matrix."""
    L = rng.standard_normal((size, size)) * scale
    return L @ L.T + np.eye(size) * 0.2


def random_model(K: int, q: int, s: int, rng: np.random.Generator):
    """
    Draw K independent regimes.

    For each regime, the joint noise covariance
        Σ_W = [[Σ_U,  Δ ],
               [Δᵀ,  Σ_V]]
    is built as a random symmetric positive-definite matrix and split
    into blocks; this guarantees Σ_V ≻ 0 and Σ_W ≻ 0 by construction.
    C and D are unconstrained Gaussian matrices.
    """
    C_list, D_list, SU_list, SV_list, Dt_list = [], [], [], [], []
    for _ in range(K):
        SW = make_pd(q + s, rng, scale=0.4)
        SU_list.append(SW[:q, :q].copy())
        Dt_list.append(SW[:q, q:].copy())
        SV_list.append(SW[q:, q:].copy())
        C_list.append(rng.standard_normal((s, q)) * 0.5)
        D_list.append(rng.standard_normal((s, s)) * 0.5)
    return C_list, D_list, SU_list, SV_list, Dt_list


# ---------------------------------------------------------------------------
# (A, B) constructors
# ---------------------------------------------------------------------------
def build_AB_from_c(C_list, D_list, SV_list, Dt_list):
    """
    Construct A(r), B(r) from formula (c):
        A(r) = Δ(r) Σ_V(r)⁻¹ C(r),
        B(r) = Δ(r) Σ_V(r)⁻¹ D(r).
    """
    K = len(C_list)
    A_list, B_list = [], []
    for k in range(K):
        SV_inv = np.linalg.solve(SV_list[k], np.eye(SV_list[k].shape[0]))
        A_list.append(Dt_list[k] @ SV_inv @ C_list[k])
        B_list.append(Dt_list[k] @ SV_inv @ D_list[k])
    return A_list, B_list


def random_AB(K: int, q: int, s: int, rng: np.random.Generator):
    """Draw arbitrary A, B matrices (NOT consistent with (H5))."""
    A_list = [rng.standard_normal((q, q)) * 0.4 for _ in range(K)]
    B_list = [rng.standard_normal((q, s)) * 0.4 for _ in range(K)]
    return A_list, B_list


# ---------------------------------------------------------------------------
# (H5) residual
# ---------------------------------------------------------------------------
def h5_residual(
    A2: np.ndarray, B2: np.ndarray,                       # at r_2
    C2: np.ndarray, D2: np.ndarray,                       # at r_2
    SU1: np.ndarray, SV1: np.ndarray, Dt1: np.ndarray,    # at r_1
    SV2: np.ndarray, Dt2: np.ndarray,                     # at r_2
) -> np.ndarray:
    """
    Evaluate the (H5) residual at the pair (r_1, r_2):

        F  =  (Δ_{r_1}ᵀ A_{r_2}ᵀ + Σ_V_{r_1} B_{r_2}ᵀ)
            − P M⁻¹ (Q A_{r_2}ᵀ + R B_{r_2}ᵀ + Δ_{r_2}ᵀ)

    with the auxiliary blocks

        P  = Δ_{r_1}ᵀ C_{r_2}ᵀ + Σ_V_{r_1} D_{r_2}ᵀ      (s × s)
        Q  = C_{r_2} Σ_U_{r_1} + D_{r_2} Δ_{r_1}ᵀ        (s × q)
        R  = C_{r_2} Δ_{r_1}   + D_{r_2} Σ_V_{r_1}       (s × s)  = Pᵀ
        M  = Q C_{r_2}ᵀ + R D_{r_2}ᵀ + Σ_V_{r_2}         (s × s)

    Returns
    -------
    F : ndarray of shape (s, q)   — ‖F‖_F = 0 ⇔ (H5) holds at (r_1, r_2).
    """
    P = Dt1.T @ C2.T + SV1 @ D2.T
    Q = C2 @ SU1 + D2 @ Dt1.T
    R = C2 @ Dt1 + D2 @ SV1
    M = Q @ C2.T + R @ D2.T + SV2
    LHS = Dt1.T @ A2.T + SV1 @ B2.T
    rhs_inner = Q @ A2.T + R @ B2.T + Dt2.T
    return LHS - P @ np.linalg.solve(M, rhs_inner)


def evaluate_all_pairs(C, D, SU, SV, Dt, A, B) -> np.ndarray:
    """
    Compute the residual norm matrix R[r_1, r_2] = ‖F_{r_1, r_2}‖_F
    for every (r_1, r_2) ∈ Ω².
    """
    K = len(C)
    norms = np.zeros((K, K))
    for r1 in range(K):
        for r2 in range(K):
            F = h5_residual(
                A[r2], B[r2], C[r2], D[r2],
                SU[r1], SV[r1], Dt[r1],
                SV[r2], Dt[r2],
            )
            norms[r1, r2] = float(np.linalg.norm(F, "fro"))
    return norms


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Verify the closed-form (H5)-compatible AB constraint "
        "parametrisation A = Δ Σ_V⁻¹ C, B = Δ Σ_V⁻¹ D."
    )
    ap.add_argument("-K", type=int, default=3, help="Number of regimes (default 3).")
    ap.add_argument("-q", type=int, default=2, help="Dim. of X (default 2).")
    ap.add_argument("-s", type=int, default=2, help="Dim. of Y (default 2).")
    ap.add_argument("-n", "--n-draws", type=int, default=20,
                    help="Number of independent random draws (default 20).")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed (default 0).")
    ap.add_argument("--tol", type=float, default=1e-9,
                    help="Pass tolerance on max ‖F‖_F (default 1e-9).")
    ap.add_argument("--show-pair-table", action="store_true",
                    help="Also print the (K × K) residual table for the first draw.")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    print("Verification of AB constraint: A(r) = Δ Σ_V⁻¹ C, B(r) = Δ Σ_V⁻¹ D")
    print(f"Setup: K={args.K}, q={args.q}, s={args.s}, "
          f"n_draws={args.n_draws}, seed={args.seed}")
    print()

    max_res_c = np.empty(args.n_draws)
    max_res_rand = np.empty(args.n_draws)
    first_norms_c = None
    first_norms_rand = None

    for trial in range(args.n_draws):
        C, D, SU, SV, Dt = random_model(args.K, args.q, args.s, rng)
        # AB constraint — should give zero residuals
        A_c, B_c = build_AB_from_c(C, D, SV, Dt)
        norms_c = evaluate_all_pairs(C, D, SU, SV, Dt, A_c, B_c)
        max_res_c[trial] = norms_c.max()
        # Negative control — random A, B (no constraint)
        A_r, B_r = random_AB(args.K, args.q, args.s, rng)
        norms_r = evaluate_all_pairs(C, D, SU, SV, Dt, A_r, B_r)
        max_res_rand[trial] = norms_r.max()
        if trial == 0:
            first_norms_c = norms_c
            first_norms_rand = norms_r

    print(f"{'Strategy':<32} {'min ‖F‖':>12} {'med ‖F‖':>12} {'max ‖F‖':>12}")
    print("-" * 72)
    print(f"{'AB constraint (closed-form)':<32} "
          f"{max_res_c.min():12.3e} {np.median(max_res_c):12.3e} {max_res_c.max():12.3e}")
    print(f"{'random A, B (negative control)':<32} "
          f"{max_res_rand.min():12.3e} {np.median(max_res_rand):12.3e} "
          f"{max_res_rand.max():12.3e}")
    print()

    if args.show_pair_table:
        print("Residual table for draw 0 — AB constraint (max should be 0):")
        print(_format_table(first_norms_c))
        print()
        print("Residual table for draw 0 — random A, B (max should be Ω(1)):")
        print(_format_table(first_norms_rand))
        print()

    pass_c = max_res_c.max() < args.tol
    # We expect non-zero residuals for the negative control on every draw.
    pass_rand = max_res_rand.min() > args.tol

    if pass_c and pass_rand:
        print(f"OK  : AB constraint gives ‖F‖ < {args.tol:.0e} for every (r_1, r_2) "
              f"and every draw.")
        print("OK  : random A, B yields strictly positive residuals "
              "(negative control sane).")
        return 0
    else:
        print(f"FAIL: (c).max  = {max_res_c.max():.3e}  (tol={args.tol:.0e})")
        print(f"FAIL: rand.min = {max_res_rand.min():.3e}  "
              f"(expected > {args.tol:.0e})")
        return 1


def _format_table(M: np.ndarray) -> str:
    K = M.shape[0]
    header = "        " + "  ".join(f"r_2={j:<2d}" for j in range(K))
    lines = [header]
    for i in range(K):
        row = "  ".join(f"{M[i, j]:7.2e}" for j in range(K))
        lines.append(f"r_1={i:<2d}  {row}")
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
