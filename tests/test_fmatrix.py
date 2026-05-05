#!/usr/bin/env python3
"""
tests/test_fmatrix.py
=====================
Unit tests for FMatrix.
"""

import numpy as np
import pytest

from prg.classes.FMatrix import FMatrix
from prg.utils.exceptions import ParamError

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_fmatrix(K=2, q=1, s=1) -> FMatrix:
    """Return a valid FMatrix for (K, q, s)."""
    A_list = [np.full((q, q), 0.1 * (k + 1)) for k in range(K)]
    B_list = [np.full((q, s), 0.2 * (k + 1)) for k in range(K)]
    C_list = [np.full((s, q), 0.3 * (k + 1)) for k in range(K)]
    D_list = [np.full((s, s), 0.4 * (k + 1)) for k in range(K)]
    return FMatrix(K=K, q=q, s=s, A_list=A_list, B_list=B_list, C_list=C_list, D_list=D_list)


# ---------------------------------------------------------------------------
# Construction — valid cases
# ---------------------------------------------------------------------------


class TestFMatrixConstruction:
    def test_scalar_case(self):
        fm = _make_fmatrix(K=2, q=1, s=1)
        assert fm.K == 2
        assert fm.q == 1
        assert fm.s == 1
        assert fm.dim_z == 2

    def test_multidimensional(self):
        fm = _make_fmatrix(K=3, q=2, s=3)
        assert fm.K == 3
        assert fm.q == 2
        assert fm.s == 3
        assert fm.dim_z == 5

    def test_float_conversion(self):
        """All arrays should be stored as float64."""
        A_list = [np.array([[1]], dtype=int)]
        B_list = [np.array([[0]], dtype=int)]
        C_list = [np.array([[0]], dtype=int)]
        D_list = [np.array([[1]], dtype=int)]
        fm = FMatrix(
            K=2,
            q=1,
            s=1,
            A_list=A_list * 2,
            B_list=B_list * 2,
            C_list=C_list * 2,
            D_list=D_list * 2,
        )
        assert fm.A(0).dtype == np.float64


# ---------------------------------------------------------------------------
# Sub-block accessors
# ---------------------------------------------------------------------------


class TestFMatrixAccessors:
    def test_sub_block_shapes(self):
        K, q, s = 3, 2, 3
        fm = _make_fmatrix(K=K, q=q, s=s)
        for k in range(K):
            assert fm.A(k).shape == (q, q)
            assert fm.B(k).shape == (q, s)
            assert fm.C(k).shape == (s, q)
            assert fm.D(k).shape == (s, s)

    def test_F_shape(self):
        K, q, s = 2, 2, 3
        fm = _make_fmatrix(K=K, q=q, s=s)
        for k in range(K):
            assert fm.F(k).shape == (q + s, q + s)

    def test_F_reconstruction(self):
        """F(k) must equal the block matrix built from A,B,C,D."""
        fm = _make_fmatrix(K=2, q=2, s=2)
        for k in range(2):
            expected = np.block(
                [
                    [fm.A(k), fm.B(k)],
                    [fm.C(k), fm.D(k)],
                ]
            )
            np.testing.assert_array_equal(fm.F(k), expected)

    def test_values_are_correct(self):
        """Check numerical values for k=0 and k=1 (scalar case)."""
        fm = _make_fmatrix(K=2, q=1, s=1)
        np.testing.assert_allclose(fm.A(0), [[0.1]])
        np.testing.assert_allclose(fm.A(1), [[0.2]])
        np.testing.assert_allclose(fm.D(0), [[0.4]])
        np.testing.assert_allclose(fm.D(1), [[0.8]])


# ---------------------------------------------------------------------------
# Construction — invalid cases (ParamError)
# ---------------------------------------------------------------------------


class TestFMatrixValidation:
    def test_K_less_than_2(self):
        with pytest.raises(ParamError, match="K must be"):
            _make_fmatrix(K=1)

    def test_K_not_int(self):
        with pytest.raises(ParamError, match="K must be"):
            FMatrix(
                K=2.0,
                q=1,
                s=1,  # type: ignore
                A_list=[np.eye(1), np.eye(1)],
                B_list=[np.eye(1), np.eye(1)],
                C_list=[np.eye(1), np.eye(1)],
                D_list=[np.eye(1), np.eye(1)],
            )

    def test_q_zero(self):
        with pytest.raises(ParamError, match="q must be"):
            _make_fmatrix(K=2, q=0, s=1)

    def test_s_zero(self):
        with pytest.raises(ParamError, match="s must be"):
            _make_fmatrix(K=2, q=1, s=0)

    def test_wrong_list_length(self):
        """A_list with wrong number of arrays should raise ParamError."""
        with pytest.raises(ParamError, match="A_list"):
            FMatrix(
                K=2,
                q=1,
                s=1,
                A_list=[np.eye(1)],  # only 1 element, need 2
                B_list=[np.eye(1), np.eye(1)],
                C_list=[np.eye(1), np.eye(1)],
                D_list=[np.eye(1), np.eye(1)],
            )

    def test_wrong_A_shape(self):
        """A_k with wrong shape should raise ParamError."""
        with pytest.raises(ParamError, match="A_list"):
            FMatrix(
                K=2,
                q=1,
                s=1,
                A_list=[np.eye(2), np.eye(1)],  # k=0 is (2,2) not (1,1)
                B_list=[np.eye(1), np.eye(1)],
                C_list=[np.eye(1), np.eye(1)],
                D_list=[np.eye(1), np.eye(1)],
            )

    def test_wrong_B_shape(self):
        """B_k with wrong shape (q,q) instead of (q,s)."""
        with pytest.raises(ParamError, match="B_list"):
            FMatrix(
                K=2,
                q=2,
                s=3,
                A_list=[np.eye(2), np.eye(2)],
                B_list=[np.eye(2), np.eye(2)],  # (2,2) not (2,3)
                C_list=[np.zeros((3, 2)), np.zeros((3, 2))],
                D_list=[np.eye(3), np.eye(3)],
            )


# ---------------------------------------------------------------------------
# Repr / summary (smoke tests — no crash)
# ---------------------------------------------------------------------------


class TestFMatrixDisplay:
    def test_repr(self):
        fm = _make_fmatrix()
        r = repr(fm)
        assert "FMatrix" in r
        assert "K=2" in r

    def test_summary_no_crash(self, capsys):
        fm = _make_fmatrix(K=2, q=2, s=2)
        fm.summary()
        captured = capsys.readouterr()
        assert "FMatrix" in captured.out
