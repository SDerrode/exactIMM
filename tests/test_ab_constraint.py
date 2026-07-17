#!/usr/bin/env python3
"""
tests/test_ab_constraint.py
===========================
Unit tests for prg/utils/ab_constraint.py.

Coverage
--------
- compute_AB      : output shape, residual zero, Σ_V singular
- apply_AB_constraint : return type, residual zero for all regimes,
                             preservation of (C, D, Σ_W, biases),
                             idempotency
- compute_ab_residual     : zero on the AB manifold, non-zero off it
"""

from __future__ import annotations

import importlib
import inspect

import numpy as np
import pytest

from prg.classes.GSSParams import GSSParams
from prg.models.base_gss_model import BaseGSSModel
from prg.utils.ab_constraint import (
    ab_residual_max,
    apply_AB_constraint,
    compute_AB,
    compute_ab_pair_residual,
    compute_ab_residual,
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
        """A model parametrised by the AB constraint must satisfy AB for any Σ_U."""
        q, s = 2, 3
        C, D, SU, Dt, SV = _random_inputs(q, s, rng)
        A, B = compute_AB(C, D, Dt, SV)
        F = compute_ab_residual(A, B, C, D, SU, Dt, SV)
        assert np.linalg.norm(F, "fro") < 1e-10, (
            f"residual at AB-constraint = {np.linalg.norm(F, 'fro'):.3e}"
        )

    def test_residual_zero_independent_of_SU(self, rng):
        """The AB residual under the AB constraint must remain zero for any Σ_U."""
        q, s = 2, 3
        C, D, _, Dt, SV = _random_inputs(q, s, rng)
        A, B = compute_AB(C, D, Dt, SV)
        for _ in range(5):
            SU_alt = _make_pd(q, rng)
            F = compute_ab_residual(A, B, C, D, SU_alt, Dt, SV)
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
        """Every regime k of the projected model satisfies AB."""
        constrained = apply_AB_constraint(params_K2_q1_s1)
        for k in range(constrained.K):
            A = constrained.f_matrix.A(k)
            B = constrained.f_matrix.B(k)
            C = constrained.f_matrix.C(k)
            D = constrained.f_matrix.D(k)
            SU = constrained.noise_cov.Sigma_U(k)
            Dt = constrained.noise_cov.Delta(k)
            SV = constrained.noise_cov.Sigma_V(k)
            F = compute_ab_residual(A, B, C, D, SU, Dt, SV)
            assert np.linalg.norm(F, "fro") < 1e-10, (
                f"k={k}: residual = {np.linalg.norm(F, 'fro'):.3e}"
            )

    def test_preserves_C_D(self, params_K2_q1_s1):
        constrained = apply_AB_constraint(params_K2_q1_s1)
        for k in range(constrained.K):
            np.testing.assert_array_equal(constrained.f_matrix.C(k), params_K2_q1_s1.f_matrix.C(k))
            np.testing.assert_array_equal(constrained.f_matrix.D(k), params_K2_q1_s1.f_matrix.D(k))

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
            np.testing.assert_array_equal(constrained.b(k), params_K2_q1_s1.b(k))

    def test_idempotent(self, params_K2_q1_s1):
        """Applying twice yields the same A, B."""
        first = apply_AB_constraint(params_K2_q1_s1)
        second = apply_AB_constraint(first)
        for k in range(first.K):
            np.testing.assert_allclose(first.f_matrix.A(k), second.f_matrix.A(k), atol=1e-12)
            np.testing.assert_allclose(first.f_matrix.B(k), second.f_matrix.B(k), atol=1e-12)


# ---------------------------------------------------------------------------
# compute_ab_residual
# ---------------------------------------------------------------------------
class TestComputeABResidual:
    def test_zero_at_AB(self, rng):
        """AB residual is exactly zero on the AB-constraint manifold."""
        q, s = 2, 3
        C, D, SU, Dt, SV = _random_inputs(q, s, rng)
        A, B = compute_AB(C, D, Dt, SV)
        F = compute_ab_residual(A, B, C, D, SU, Dt, SV)
        assert F.shape == (s, q)
        assert np.linalg.norm(F, "fro") < 1e-10

    def test_nonzero_for_random_AB(self, rng):
        """AB residual is generically non-zero for arbitrary A, B."""
        q, s = 2, 3
        C, D, SU, Dt, SV = _random_inputs(q, s, rng)
        A = rng.standard_normal((q, q)) * 0.5
        B = rng.standard_normal((q, s)) * 0.5
        F = compute_ab_residual(A, B, C, D, SU, Dt, SV)
        assert np.linalg.norm(F, "fro") > 1e-3


# ---------------------------------------------------------------------------
# Pairwise (all-pairs) AB check
# ---------------------------------------------------------------------------
class TestPairwiseAB:
    def test_ab_model_fully_compatible(self, params_K2_q1_s1):
        """An AB-constrained model is AB-constrained across every regime pair."""
        ab = apply_AB_constraint(params_K2_q1_s1)
        max_rel, _ = ab_residual_max(ab)
        assert max_rel < 1e-9, f"AB model not fully AB-constrained: {max_rel:.3e}"

    def test_pair_residual_zero_on_AB_manifold(self, rng):
        """β₁(j, k) vanishes for all pairs when every regime is AB-parametrised."""
        q, s = 2, 1
        reg = []
        for _ in range(2):
            C, D, SU, Dt, SV = _random_inputs(q, s, rng)
            A, B = compute_AB(C, D, Dt, SV)
            reg.append((A, B, C, D, SU, Dt, SV))
        for j in range(2):
            for k in range(2):
                Ak, Bk, Ck, Dk, _, Dtk, SVk = reg[k]
                _, _, _, _, SUj, Dtj, SVj = reg[j]
                b1 = compute_ab_pair_residual(Ak, Bk, Ck, Dk, Dtk, SVk, SUj, Dtj, SVj)
                assert np.linalg.norm(b1, "fro") < 1e-9

    def test_pairwise_strictly_stronger_than_same_regime(self):
        """A non-AB model can satisfy *every* same-regime residual yet violate
        AB on a cross pair (j != k); only the pairwise check catches it.

        Built in the sub-determined regime K=2, q=2, s=1 (Ks < q+s), where the
        same-regime residual has a non-trivial null space.
        """
        rng = np.random.default_rng(3)
        q, s = 2, 1

        def spd(n):
            M = rng.standard_normal((n, n))
            return M @ M.T + 0.5 * np.eye(n)

        reg = []
        for _ in range(2):
            C = rng.standard_normal((s, q))
            D = rng.standard_normal((s, s))
            SU, SV, Dt = spd(q), spd(s), 0.3 * rng.standard_normal((q, s))
            A, B = compute_AB(C, D, Dt, SV)
            reg.append([A, B, C, D, SU, Dt, SV])

        # Perturb regime 0's (A, B) inside the null space of its same-regime residual.
        r0 = reg[0]
        base = np.concatenate([r0[0].ravel(), r0[1].ravel()])
        nA = q * q

        def mono_of(vec):
            A = vec[:nA].reshape(q, q)
            B = vec[nA:].reshape(q, s)
            return compute_ab_residual(A, B, r0[2], r0[3], r0[4], r0[5], r0[6]).ravel()

        n = base.size
        J = np.zeros((s * q, n))
        rb = mono_of(base)
        for i in range(n):
            e = np.zeros(n)
            e[i] = 1.0
            J[:, i] = mono_of(base + e) - rb
        _, S, Vt = np.linalg.svd(J)
        null_dir = Vt[int((S > 1e-9).sum()) :][0]
        pert = base + null_dir
        reg[0][0] = pert[:nA].reshape(q, q)
        reg[0][1] = pert[nA:].reshape(q, s)

        # Same-regime residual stays ~0 for every regime (necessary condition holds) ...
        mono_max = max(np.linalg.norm(compute_ab_residual(*reg[k]), "fro") for k in range(2))
        assert mono_max < 1e-8, f"same-regime residual unexpectedly nonzero: {mono_max:.2e}"

        # ... yet the pairwise check detects the cross-pair AB violation.
        pair_max = 0.0
        for j in range(2):
            for k in range(2):
                b1 = compute_ab_pair_residual(
                    reg[k][0],
                    reg[k][1],
                    reg[k][2],
                    reg[k][3],
                    reg[k][5],
                    reg[k][6],
                    reg[j][4],
                    reg[j][5],
                    reg[j][6],
                )
                pair_max = max(pair_max, float(np.linalg.norm(b1, "fro")))
        assert pair_max > 1e-2, f"pairwise check missed the violation: {pair_max:.2e}"
