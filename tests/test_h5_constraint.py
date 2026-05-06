#!/usr/bin/env python3
"""
tests/test_h5_constraint.py
===========================
Unit tests for prg/utils/h5_constraint.py.

Coverage
--------
- compute_AB      : output shape, residual zero, Σ_V singular
- apply_AB_constraint : return type, residual zero for all regimes,
                             preservation of (C, D, Σ_W, biases),
                             idempotency
- compute_h5_residual     : zero on the AB manifold, non-zero off it
"""

from __future__ import annotations

import importlib
import inspect

import numpy as np
import pytest

from prg.classes.GSSParams import GSSParams
from prg.models.base_gss_model import BaseGSSModel
from prg.utils.h5_constraint import (
    apply_AB_constraint,
    compute_AB,
    compute_h5_residual,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_pd(size, rng, scale=0.5):
    """Return a random positive-definite (size × size) matrix."""
    L = rng.standard_normal((size, size)) * scale
    return L @ L.T + np.eye(size) * 0.2


def _random_inputs(q, s, rng):
    """Build (C, D, Σ_U, Δ, Σ_V) with compatible shapes and PD covariances."""
    C = rng.standard_normal((s, q)) * 0.3
    D = rng.standard_normal((s, s)) * 0.3
    SU = _make_pd(q, rng)
    Dt = rng.standard_normal((q, s)) * 0.1
    SV = _make_pd(s, rng)
    return C, D, SU, Dt, SV


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def rng():
    """Shared RNG (fixed seed) for reproducible tests."""
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
# compute_AB
# ---------------------------------------------------------------------------
class TestComputeAB:
    def test_output_shapes(self, rng):
        q, s = 2, 3
        C, D, _, Dt, SV = _random_inputs(q, s, rng)
        A, B = compute_AB(C, D, Dt, SV)
        assert A.shape == (q, q), f"expected ({q},{q}), got {A.shape}"
        assert B.shape == (q, s), f"expected ({q},{s}), got {B.shape}"

    def test_residual_zero(self, rng):
        """A model parametrised by the AB constraint must satisfy (H5) for any Σ_U."""
        q, s = 2, 3
        C, D, SU, Dt, SV = _random_inputs(q, s, rng)
        A, B = compute_AB(C, D, Dt, SV)
        F = compute_h5_residual(A, B, C, D, SU, Dt, SV)
        assert np.linalg.norm(F, "fro") < 1e-10, (
            f"residual at AB-constraint = {np.linalg.norm(F, 'fro'):.3e}"
        )

    def test_residual_zero_independent_of_SU(self, rng):
        """The (H5) residual under the AB constraint must remain zero for any Σ_U."""
        q, s = 2, 3
        C, D, _, Dt, SV = _random_inputs(q, s, rng)
        A, B = compute_AB(C, D, Dt, SV)
        for _ in range(5):
            SU_alt = _make_pd(q, rng)
            F = compute_h5_residual(A, B, C, D, SU_alt, Dt, SV)
            assert np.linalg.norm(F, "fro") < 1e-10

    def test_singular_SV_raises(self):
        """Σ_V = 0 must raise ValueError."""
        q, s = 2, 2
        C = np.zeros((s, q))
        D = np.zeros((s, s))
        Dt = np.zeros((q, s))
        SV = np.zeros((s, s))  # singular
        with pytest.raises(ValueError):
            compute_AB(C, D, Dt, SV)

    def test_zero_delta_yields_zero_AB(self, rng):
        """Δ = 0 forces A = B = 0 (degenerate but valid)."""
        q, s = 2, 3
        C = rng.standard_normal((s, q))
        D = rng.standard_normal((s, s))
        Dt = np.zeros((q, s))
        SV = _make_pd(s, rng)
        A, B = compute_AB(C, D, Dt, SV)
        np.testing.assert_array_equal(A, np.zeros((q, q)))
        np.testing.assert_array_equal(B, np.zeros((q, s)))


# ---------------------------------------------------------------------------
# apply_AB_constraint
# ---------------------------------------------------------------------------
class TestApplyABConstraint:
    def test_output_type(self, params_K2_q1_s1):
        result = apply_AB_constraint(params_K2_q1_s1)
        assert isinstance(result, GSSParams)

    def test_residual_zero_for_all_regimes(self, params_K2_q1_s1):
        """Every regime k of the projected model satisfies (H5)."""
        constrained = apply_AB_constraint(params_K2_q1_s1)
        for k in range(constrained.K):
            A = constrained.f_matrix.A(k)
            B = constrained.f_matrix.B(k)
            C = constrained.f_matrix.C(k)
            D = constrained.f_matrix.D(k)
            SU = constrained.noise_cov.Sigma_U(k)
            Dt = constrained.noise_cov.Delta(k)
            SV = constrained.noise_cov.Sigma_V(k)
            F = compute_h5_residual(A, B, C, D, SU, Dt, SV)
            assert np.linalg.norm(F, "fro") < 1e-10, (
                f"k={k}: residual = {np.linalg.norm(F, 'fro'):.3e}"
            )

    def test_preserves_C_D(self, params_K2_q1_s1):
        constrained = apply_AB_constraint(params_K2_q1_s1)
        for k in range(constrained.K):
            np.testing.assert_array_equal(
                constrained.f_matrix.C(k), params_K2_q1_s1.f_matrix.C(k)
            )
            np.testing.assert_array_equal(
                constrained.f_matrix.D(k), params_K2_q1_s1.f_matrix.D(k)
            )

    def test_preserves_noise_cov(self, params_K2_q1_s1):
        constrained = apply_AB_constraint(params_K2_q1_s1)
        for k in range(constrained.K):
            np.testing.assert_array_equal(
                constrained.noise_cov.Sigma_U(k), params_K2_q1_s1.noise_cov.Sigma_U(k)
            )
            np.testing.assert_array_equal(
                constrained.noise_cov.Delta(k), params_K2_q1_s1.noise_cov.Delta(k)
            )
            np.testing.assert_array_equal(
                constrained.noise_cov.Sigma_V(k), params_K2_q1_s1.noise_cov.Sigma_V(k)
            )

    def test_preserves_bias(self, params_K2_q1_s1):
        constrained = apply_AB_constraint(params_K2_q1_s1)
        for k in range(constrained.K):
            np.testing.assert_array_equal(
                constrained.b(k), params_K2_q1_s1.b(k)
            )

    def test_idempotent(self, params_K2_q1_s1):
        """Applying twice yields the same A, B."""
        first = apply_AB_constraint(params_K2_q1_s1)
        second = apply_AB_constraint(first)
        for k in range(first.K):
            np.testing.assert_allclose(
                first.f_matrix.A(k), second.f_matrix.A(k), atol=1e-12
            )
            np.testing.assert_allclose(
                first.f_matrix.B(k), second.f_matrix.B(k), atol=1e-12
            )


# ---------------------------------------------------------------------------
# compute_h5_residual
# ---------------------------------------------------------------------------
class TestComputeH5Residual:
    def test_zero_at_AB(self, rng):
        """(H5) residual is exactly zero on the AB-constraint manifold."""
        q, s = 2, 3
        C, D, SU, Dt, SV = _random_inputs(q, s, rng)
        A, B = compute_AB(C, D, Dt, SV)
        F = compute_h5_residual(A, B, C, D, SU, Dt, SV)
        assert F.shape == (s, q)
        assert np.linalg.norm(F, "fro") < 1e-10

    def test_nonzero_for_random_AB(self, rng):
        """(H5) residual is generically non-zero for arbitrary A, B."""
        q, s = 2, 3
        C, D, SU, Dt, SV = _random_inputs(q, s, rng)
        A = rng.standard_normal((q, q)) * 0.5
        B = rng.standard_normal((q, s)) * 0.5
        F = compute_h5_residual(A, B, C, D, SU, Dt, SV)
        assert np.linalg.norm(F, "fro") > 1e-3
