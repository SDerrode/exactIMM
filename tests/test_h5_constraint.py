#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/test_h5_constraint.py
===========================
Unit tests for prg/utils/h5_constraint.py.

Coverage
--------
- compute_B_from_h5 : output shape, constraint satisfaction, degenerate cases,
  singular M, idempotency
- apply_h5_constraint : return type, constraint satisfaction for all regimes,
  preservation of A/C/D/noise_cov/b, idempotency
"""

from __future__ import annotations

import importlib
import inspect

import numpy as np
import pytest

from prg.classes.GSSParams import GSSParams
from prg.models.base_gss_model import BaseGSSModel
from prg.utils.h5_constraint import (
    apply_h5_constraint, compute_B_from_h5, compute_C_from_h5,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check_constraint(A, C, D, SU, Dt, SV, B, atol=1e-10):
    """
    Verify the H5 constraint (eq. 4.4):

        Δᵀ A + Σ_V Bᵀ = (Δᵀ Cᵀ + Σ_V Dᵀ) M⁻¹ ((C Σ_U + D Δᵀ) Aᵀ
                          + (C Δ + D Σ_V) Bᵀ + Δᵀ)

    Returns True when the equation holds up to *atol*.
    """
    P = Dt.T @ C.T + SV @ D.T          # s × s
    Q = C @ SU + D @ Dt.T              # s × q
    R = C @ Dt + D @ SV                # s × s
    M = Q @ C.T + R @ D.T + SV        # s × s

    M_inv = np.linalg.inv(M)
    lhs = Dt.T @ A + SV @ B.T                        # s × q
    rhs = P @ M_inv @ (Q @ A.T + R @ B.T + Dt.T)    # s × q

    return np.allclose(lhs, rhs, atol=atol)


def _make_pd(size, rng, scale=0.5):
    """Return a random positive-definite (size × size) matrix."""
    L = rng.standard_normal((size, size)) * scale
    return L @ L.T + np.eye(size) * 0.2


def _random_inputs(q, s, rng):
    """Build (A, C, D, SU, Dt, SV) with compatible shapes and positive-definite
    covariance matrices."""
    A  = rng.standard_normal((q, q)) * 0.5
    C  = rng.standard_normal((s, q)) * 0.3
    D  = rng.standard_normal((s, s)) * 0.3
    SU = _make_pd(q, rng)
    Dt = rng.standard_normal((q, s)) * 0.1
    SV = _make_pd(s, rng)
    return A, C, D, SU, Dt, SV


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def rng():
    """Shared random generator (fixed seed) for reproducible tests."""
    return np.random.default_rng(42)


@pytest.fixture(scope="module")
def params_K2_q1_s1() -> GSSParams:
    """GSSParams loaded dynamically from model_gss_K2_q1_s1."""
    mod = importlib.import_module("prg.models.model_gss_K2_q1_s1")
    cls = next(
        c
        for _, c in inspect.getmembers(mod, inspect.isclass)
        if issubclass(c, BaseGSSModel) and c is not BaseGSSModel
    )
    return GSSParams.from_model(cls())


# ---------------------------------------------------------------------------
# Tests for compute_B_from_h5
# ---------------------------------------------------------------------------


def test_compute_B_output_shape(rng):
    """B must have shape (q, s) for arbitrary q=2, s=3."""
    q, s = 2, 3
    A, C, D, SU, Dt, SV = _random_inputs(q, s, rng)
    B = compute_B_from_h5(A, C, D, SU, Dt, SV)
    assert B.shape == (q, s), f"expected ({q}, {s}), got {B.shape}"


def test_compute_B_satisfies_constraint(rng):
    """B returned by compute_B_from_h5 must satisfy the H5 constraint (eq. 4.4)."""
    q, s = 2, 3
    A, C, D, SU, Dt, SV = _random_inputs(q, s, rng)
    B = compute_B_from_h5(A, C, D, SU, Dt, SV)
    assert _check_constraint(A, C, D, SU, Dt, SV, B, atol=1e-10), (
        "H5 constraint (eq. 4.4) not satisfied by compute_B_from_h5 output"
    )


def test_compute_B_zero_delta_zero_C(rng):
    """Simplified case Δ=0, C=0: constraint must still be satisfied."""
    q, s = 2, 2
    A  = rng.standard_normal((q, q)) * 0.5
    C  = np.zeros((s, q))
    D  = rng.standard_normal((s, s)) * 0.4
    SU = _make_pd(q, rng)
    Dt = np.zeros((q, s))
    SV = _make_pd(s, rng)

    # With C=0, Δ=0:
    #   P = Σ_V Dᵀ,   Q = 0,   R = D Σ_V,   M = D Σ_V Dᵀ + Σ_V
    B = compute_B_from_h5(A, C, D, SU, Dt, SV)
    assert _check_constraint(A, C, D, SU, Dt, SV, B, atol=1e-10), (
        "H5 constraint not satisfied for the C=0, Δ=0 special case"
    )


def test_compute_B_singular_M_raises():
    """compute_B_from_h5 must raise ValueError when M is singular (C=D=Σ_V=0)."""
    q, s = 2, 2
    A  = np.eye(q) * 0.5
    C  = np.zeros((s, q))
    D  = np.zeros((s, s))
    SU = np.eye(q) * 0.1
    Dt = np.zeros((q, s))
    SV = np.zeros((s, s))   # makes M = 0 → singular

    with pytest.raises(ValueError):
        compute_B_from_h5(A, C, D, SU, Dt, SV)


def test_compute_B_idempotent(rng):
    """Applying compute_B_from_h5 on a B that already satisfies the constraint
    must return a matrix whose distance from B is below 1e-10."""
    q, s = 2, 3
    A, C, D, SU, Dt, SV = _random_inputs(q, s, rng)

    # First application produces a solution B*
    B_star = compute_B_from_h5(A, C, D, SU, Dt, SV)

    # Second application (B_star is already a solution, so the formula is fixed-point)
    B_star2 = compute_B_from_h5(A, C, D, SU, Dt, SV)

    assert np.linalg.norm(B_star2 - B_star, "fro") < 1e-10, (
        "compute_B_from_h5 is not idempotent (two calls on same inputs differ)"
    )


# ---------------------------------------------------------------------------
# Tests for apply_h5_constraint
# ---------------------------------------------------------------------------


def test_apply_h5_constraint_output_type(params_K2_q1_s1):
    """apply_h5_constraint must return a GSSParams instance."""
    result = apply_h5_constraint(params_K2_q1_s1)
    assert isinstance(result, GSSParams), (
        f"expected GSSParams, got {type(result)}"
    )


def test_apply_h5_constraint_satisfies_for_all_regimes(params_K2_q1_s1):
    """For every regime k, the updated B(k) must satisfy the H5 constraint."""
    constrained = apply_h5_constraint(params_K2_q1_s1)
    K = constrained.K

    for k in range(K):
        A  = constrained.f_matrix.A(k)
        B  = constrained.f_matrix.B(k)
        C  = constrained.f_matrix.C(k)
        D  = constrained.f_matrix.D(k)
        SU = constrained.noise_cov.Sigma_U(k)
        Dt = constrained.noise_cov.Delta(k)
        SV = constrained.noise_cov.Sigma_V(k)

        assert _check_constraint(A, C, D, SU, Dt, SV, B, atol=1e-10), (
            f"H5 constraint not satisfied for regime k={k}"
        )


def test_apply_h5_constraint_preserves_A_C_D(params_K2_q1_s1):
    """apply_h5_constraint must not modify A(k), C(k), or D(k) for any k."""
    constrained = apply_h5_constraint(params_K2_q1_s1)
    K = params_K2_q1_s1.K

    for k in range(K):
        np.testing.assert_array_equal(
            constrained.f_matrix.A(k),
            params_K2_q1_s1.f_matrix.A(k),
            err_msg=f"A(k={k}) was modified",
        )
        np.testing.assert_array_equal(
            constrained.f_matrix.C(k),
            params_K2_q1_s1.f_matrix.C(k),
            err_msg=f"C(k={k}) was modified",
        )
        np.testing.assert_array_equal(
            constrained.f_matrix.D(k),
            params_K2_q1_s1.f_matrix.D(k),
            err_msg=f"D(k={k}) was modified",
        )


def test_apply_h5_constraint_preserves_noise_cov(params_K2_q1_s1):
    """apply_h5_constraint must not modify any noise-covariance block."""
    constrained = apply_h5_constraint(params_K2_q1_s1)
    K = params_K2_q1_s1.K

    for k in range(K):
        np.testing.assert_array_equal(
            constrained.noise_cov.Sigma_U(k),
            params_K2_q1_s1.noise_cov.Sigma_U(k),
            err_msg=f"Sigma_U(k={k}) was modified",
        )
        np.testing.assert_array_equal(
            constrained.noise_cov.Delta(k),
            params_K2_q1_s1.noise_cov.Delta(k),
            err_msg=f"Delta(k={k}) was modified",
        )
        np.testing.assert_array_equal(
            constrained.noise_cov.Sigma_V(k),
            params_K2_q1_s1.noise_cov.Sigma_V(k),
            err_msg=f"Sigma_V(k={k}) was modified",
        )


def test_apply_h5_constraint_preserves_b_bias(params_K2_q1_s1):
    """apply_h5_constraint must not modify the drift bias b(k) for any k."""
    constrained = apply_h5_constraint(params_K2_q1_s1)
    K = params_K2_q1_s1.K

    for k in range(K):
        np.testing.assert_array_equal(
            constrained.b(k),
            params_K2_q1_s1.b(k),
            err_msg=f"b(k={k}) was modified",
        )


# ---------------------------------------------------------------------------
# Tests for compute_C_from_h5
# ---------------------------------------------------------------------------


def _residual_h5(A, C, D, SU, Dt, SV, B):
    """Full residual F(C) = Z − P M⁻¹ W  (H5 constraint in residual form)."""
    P = Dt.T @ C.T + SV @ D.T
    Q = C @ SU + D @ Dt.T
    R = C @ Dt + D @ SV
    M = Q @ C.T + R @ D.T + SV
    Z = Dt.T @ A + SV @ B.T
    W = Q @ A.T + R @ B.T + Dt.T
    X = np.linalg.solve(M, W)   # M⁻¹ W
    return Z - P @ X


def _make_consistent_inputs_with_C(q, s, rng):
    """Return (A, B, C_true, D, SU, Dt, SV) where B was computed from H5 using
    C_true, guaranteeing that C_true is a valid solution of compute_C_from_h5."""
    A, C_true, D, SU, Dt, SV = _random_inputs(q, s, rng)
    B = compute_B_from_h5(A, C_true, D, SU, Dt, SV)
    return A, B, C_true, D, SU, Dt, SV


def test_compute_C_output_shape(rng):
    """C must have shape (s, q)."""
    q, s = 2, 3
    A, B, C_true, D, SU, Dt, SV = _make_consistent_inputs_with_C(q, s, rng)
    # Warm-start from C_true so the iteration is already at a solution
    C = compute_C_from_h5(A, B, D, SU, Dt, SV, C_init=C_true)
    assert C.shape == (s, q), f"expected ({s}, {q}), got {C.shape}"


def test_compute_C_satisfies_constraint(rng):
    """C returned by compute_C_from_h5 (warm-started at C_true) must satisfy H5."""
    q, s = 2, 3
    A, B, C_true, D, SU, Dt, SV = _make_consistent_inputs_with_C(q, s, rng)
    C = compute_C_from_h5(A, B, D, SU, Dt, SV, C_init=C_true)
    res = _residual_h5(A, C, D, SU, Dt, SV, B)
    assert np.linalg.norm(res, "fro") < 1e-8, (
        f"H5 constraint not satisfied: residual = {np.linalg.norm(res, 'fro'):.3e}"
    )


def test_compute_C_zero_init_is_fixed_point_when_C0_zero(rng):
    """When B is computed from H5 with C=0, compute_C_from_h5 initialized at 0
    must return C ≈ 0 (C=0 is a fixed point of the iteration)."""
    q, s = 2, 2
    A  = rng.standard_normal((q, q)) * 0.4
    D  = rng.standard_normal((s, s)) * 0.3
    SU = _make_pd(q, rng)
    Dt = rng.standard_normal((q, s)) * 0.1
    SV = _make_pd(s, rng)
    B  = compute_B_from_h5(A, np.zeros((s, q)), D, SU, Dt, SV)  # B consistent with C=0

    C = compute_C_from_h5(A, B, D, SU, Dt, SV, C_init=np.zeros((s, q)))
    assert np.linalg.norm(C, "fro") < 1e-6, (
        f"Expected C ≈ 0 (fixed point), got ‖C‖_F = {np.linalg.norm(C, 'fro'):.3e}"
    )


def test_compute_C_singular_raises():
    """compute_C_from_h5 must raise ValueError when P = 0 (D=0, Δ=0) but Z ≠ 0.

    With D=0 and Δ=0: P = Δᵀ Cᵀ + Σ_V Dᵀ = 0 for any C (since both terms vanish).
    Hence G̃ = P̃ M̃⁻¹ = 0, the Kronecker matrix is zero but the rhs Z ≠ 0, so
    the system is inconsistent and rank-deficient → must raise ValueError.
    """
    q, s = 2, 2
    rng2 = np.random.default_rng(7)
    A   = rng2.standard_normal((q, q)) * 0.5
    B   = rng2.standard_normal((q, s)) * 0.5   # B ≠ 0 ensures Z = SV B^T ≠ 0
    D   = np.zeros((s, s))                      # forces P = 0
    SU  = _make_pd(q, rng2)
    Dt  = np.zeros((q, s))                      # forces P = 0
    SV  = _make_pd(s, rng2)                     # M = SV > 0, non-singular

    with pytest.raises(ValueError):
        compute_C_from_h5(A, B, D, SU, Dt, SV)


def test_compute_C_warm_start_is_fixed_point(rng):
    """Starting exactly at a known solution (C_true) must return C ≈ C_true."""
    q, s = 2, 3
    A, B, C_true, D, SU, Dt, SV = _make_consistent_inputs_with_C(q, s, rng)
    C = compute_C_from_h5(A, B, D, SU, Dt, SV, C_init=C_true, max_iter=5)
    diff = np.linalg.norm(C - C_true, "fro")
    assert diff < 1e-8, (
        f"C_true is not a fixed point: ‖C - C_true‖_F = {diff:.3e}"
    )


def test_apply_h5_constraint_idempotent(params_K2_q1_s1):
    """Applying apply_h5_constraint twice must give the same B matrices
    (Frobenius-norm difference < 1e-10 for every regime)."""
    once  = apply_h5_constraint(params_K2_q1_s1)
    twice = apply_h5_constraint(once)
    K = params_K2_q1_s1.K

    for k in range(K):
        delta = np.linalg.norm(twice.f_matrix.B(k) - once.f_matrix.B(k), "fro")
        assert delta < 1e-10, (
            f"apply_h5_constraint is not idempotent: ‖ΔB(k={k})‖_F = {delta:.3e}"
        )
