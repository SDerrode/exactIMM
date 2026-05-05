#!/usr/bin/env python3
"""
tests/test_noise_covariance.py
==============================
Unit tests for GSSNoiseCovariance.
"""

import numpy as np
import pytest

from prg.classes.NoiseCovariance import GSSNoiseCovariance
from prg.utils.exceptions import CovarianceError, ParamError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_noise_cov(K=2, q=1, s=1) -> GSSNoiseCovariance:
    """Return a valid GSSNoiseCovariance (diagonal noise for simplicity)."""
    Sigma_U_list = [np.eye(q) * (0.1 + 0.05 * k) for k in range(K)]
    Delta_list = [np.zeros((q, s)) for _ in range(K)]
    Sigma_V_list = [np.eye(s) * (0.1 + 0.05 * k) for k in range(K)]
    return GSSNoiseCovariance(
        K=K,
        q=q,
        s=s,
        Sigma_U_list=Sigma_U_list,
        Delta_list=Delta_list,
        Sigma_V_list=Sigma_V_list,
    )


# ---------------------------------------------------------------------------
# Construction — valid cases
# ---------------------------------------------------------------------------


class TestNoiseCovarianceConstruction:
    def test_scalar_case(self):
        nc = _make_noise_cov(K=2, q=1, s=1)
        assert nc.K == 2
        assert nc.q == 1
        assert nc.s == 1
        assert nc.dim_z == 2

    def test_multidimensional(self):
        nc = _make_noise_cov(K=3, q=2, s=3)
        assert nc.K == 3
        assert nc.q == 2
        assert nc.s == 3
        assert nc.dim_z == 5

    def test_correlated_noise(self):
        """A valid non-zero Delta should construct without error."""
        # Sigma_W = [[0.2, 0.05], [0.05, 0.1]] — det = 0.02 - 0.0025 > 0
        nc = GSSNoiseCovariance(
            K=2,
            q=1,
            s=1,
            Sigma_U_list=[np.array([[0.2]]), np.array([[0.2]])],
            Delta_list=[np.array([[0.05]]), np.array([[0.05]])],
            Sigma_V_list=[np.array([[0.1]]), np.array([[0.1]])],
        )
        assert nc is not None


# ---------------------------------------------------------------------------
# Accessors
# ---------------------------------------------------------------------------


class TestNoiseCovarianceAccessors:
    def test_block_shapes(self):
        K, q, s = 3, 2, 3
        nc = _make_noise_cov(K=K, q=q, s=s)
        for k in range(K):
            assert nc.Sigma_U(k).shape == (q, q)
            assert nc.Delta(k).shape == (q, s)
            assert nc.Sigma_V(k).shape == (s, s)
            assert nc.Sigma_W(k).shape == (q + s, q + s)
            assert nc.chol_W(k).shape == (q + s, q + s)

    def test_Sigma_W_block_structure(self):
        """Sigma_W(k) must equal the block matrix [[Sigma_U, Delta], [Delta^T, Sigma_V]]."""
        nc = _make_noise_cov(K=2, q=2, s=2)
        for k in range(2):
            expected = np.block(
                [
                    [nc.Sigma_U(k), nc.Delta(k)],
                    [nc.Delta(k).T, nc.Sigma_V(k)],
                ]
            )
            np.testing.assert_array_equal(nc.Sigma_W(k), expected)

    def test_Sigma_W_is_symmetric(self):
        nc = _make_noise_cov(K=2, q=2, s=3)
        for k in range(2):
            W = nc.Sigma_W(k)
            np.testing.assert_allclose(W, W.T, atol=1e-14)

    def test_chol_W_cached(self):
        """chol_W(k) should return the same object on repeated calls."""
        nc = _make_noise_cov(K=2, q=1, s=1)
        L1 = nc.chol_W(0)
        L2 = nc.chol_W(0)
        assert L1 is L2

    def test_chol_W_lower_triangular(self):
        nc = _make_noise_cov(K=2, q=2, s=2)
        for k in range(2):
            L = nc.chol_W(k)
            np.testing.assert_array_equal(L, np.tril(L))

    def test_chol_W_reconstruction(self):
        """L @ L.T must equal Sigma_W(k)."""
        nc = _make_noise_cov(K=2, q=2, s=2)
        for k in range(2):
            L = nc.chol_W(k)
            np.testing.assert_allclose(L @ L.T, nc.Sigma_W(k), atol=1e-12)


# ---------------------------------------------------------------------------
# Validation — invalid cases
# ---------------------------------------------------------------------------


class TestNoiseCovarianceValidation:
    def test_non_pd_sigma_W(self):
        """Sigma_W that is not positive definite must raise CovarianceError."""
        # Sigma_W = [[1, 2], [2, 1]] — not PD (det < 0)
        with pytest.raises(CovarianceError, match="Sigma_W"):
            GSSNoiseCovariance(
                K=2,
                q=1,
                s=1,
                Sigma_U_list=[np.array([[1.0]]), np.array([[1.0]])],
                Delta_list=[np.array([[2.0]]), np.array([[2.0]])],
                Sigma_V_list=[np.array([[1.0]]), np.array([[1.0]])],
            )

    def test_non_symmetric_sigma_W(self):
        """A Delta that breaks symmetry of Sigma_W must raise CovarianceError."""
        # Sigma_W = [[1, 0.5], [0.5, 1]] is PD
        # but if we force a non-symmetric Sigma_U we get a non-symmetric Sigma_W
        # We test the asymmetry check through the CovarianceMatrix diagnostic.
        # Here we test a simpler case: Sigma_U itself not symmetric.
        with pytest.raises((CovarianceError, ParamError)):
            GSSNoiseCovariance(
                K=2,
                q=2,
                s=1,
                Sigma_U_list=[
                    np.array([[1.0, 0.5], [0.0, 1.0]]),  # not symmetric
                    np.eye(2),
                ],
                Delta_list=[np.zeros((2, 1)), np.zeros((2, 1))],
                Sigma_V_list=[np.array([[1.0]]), np.array([[1.0]])],
            )

    def test_wrong_list_length(self):
        with pytest.raises(ParamError, match="Sigma_U_list"):
            GSSNoiseCovariance(
                K=2,
                q=1,
                s=1,
                Sigma_U_list=[np.array([[0.1]])],  # only 1, need 2
                Delta_list=[np.zeros((1, 1)), np.zeros((1, 1))],
                Sigma_V_list=[np.array([[0.1]]), np.array([[0.1]])],
            )

    def test_wrong_Delta_shape(self):
        with pytest.raises(ParamError, match="Delta_list"):
            GSSNoiseCovariance(
                K=2,
                q=1,
                s=2,
                Sigma_U_list=[np.array([[0.1]]), np.array([[0.1]])],
                Delta_list=[np.zeros((1, 1)), np.zeros((1, 1))],  # (1,1) not (1,2)
                Sigma_V_list=[np.eye(2), np.eye(2)],
            )

    def test_K_less_than_2(self):
        with pytest.raises(ParamError, match="K must be"):
            _make_noise_cov(K=1)


# ---------------------------------------------------------------------------
# Display (smoke tests)
# ---------------------------------------------------------------------------


class TestNoiseCovarianceDisplay:
    def test_repr(self):
        nc = _make_noise_cov()
        r = repr(nc)
        assert "GSSNoiseCovariance" in r

    def test_summary_no_crash(self, capsys):
        nc = _make_noise_cov(K=2, q=2, s=2)
        nc.summary()
        captured = capsys.readouterr()
        assert "GSSNoiseCovariance" in captured.out
