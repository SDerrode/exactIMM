#!/usr/bin/env python3
"""
tests/test_gss_params.py
========================
Unit tests for GSSParams.
"""

import pathlib
import subprocess
import sys

import numpy as np
import pytest

from prg.classes.FMatrix import FMatrix
from prg.classes.GSSParams import GSSParams
from prg.classes.NoiseCovariance import GSSNoiseCovariance
from prg.models.model_gss_K2_q1_s1 import ModelGssK2Q1S1
from prg.utils.exceptions import CovarianceError, ParamError

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_valid_params(K=2, q=1, s=1) -> GSSParams:
    """Build a minimal but valid GSSParams for (K, q, s)."""
    dim_z = q + s

    # Row-stochastic P
    P = np.full((K, K), 1.0 / K)

    f_matrix = FMatrix(
        K=K,
        q=q,
        s=s,
        A_list=[np.eye(q) * 0.5 for _ in range(K)],
        B_list=[np.zeros((q, s)) for _ in range(K)],
        C_list=[np.zeros((s, q)) for _ in range(K)],
        D_list=[np.eye(s) * 0.5 for _ in range(K)],
    )
    noise_cov = GSSNoiseCovariance(
        K=K,
        q=q,
        s=s,
        Sigma_U_list=[np.eye(q) * 0.1 for _ in range(K)],
        Delta_list=[np.zeros((q, s)) for _ in range(K)],
        Sigma_V_list=[np.eye(s) * 0.1 for _ in range(K)],
    )
    pi0 = np.ones(K) / K
    mu_z0_list = [np.zeros((dim_z, 1)) for _ in range(K)]
    Sigma_z0_list = [np.eye(dim_z) for _ in range(K)]

    return GSSParams(
        K=K,
        q=q,
        s=s,
        P=P,
        f_matrix=f_matrix,
        noise_cov=noise_cov,
        pi0=pi0,
        mu_z0_list=mu_z0_list,
        Sigma_z0_list=Sigma_z0_list,
    )


# ---------------------------------------------------------------------------
# Construction — valid cases
# ---------------------------------------------------------------------------


class TestGSSParamsConstruction:
    def test_basic(self):
        params = _make_valid_params(K=2, q=1, s=1)
        assert params.K == 2
        assert params.q == 1
        assert params.s == 1
        assert params.dim_z == 2

    def test_from_model(self):
        """GSSParams.from_model must build cleanly from the toy model."""
        model = ModelGssK2Q1S1()
        params = GSSParams.from_model(model)
        assert params.K == 2
        assert params.q == 1
        assert params.s == 1

    def test_pi0_none_gives_stationary(self):
        """When pi0=None, the stored pi0 must be the stationary distribution."""
        K = 2
        dim_z = 2
        P = np.array([[0.9, 0.1], [0.2, 0.8]])
        f_matrix = FMatrix(
            K=K,
            q=1,
            s=1,
            A_list=[np.eye(1) * 0.5, np.eye(1) * 0.5],
            B_list=[np.zeros((1, 1)), np.zeros((1, 1))],
            C_list=[np.zeros((1, 1)), np.zeros((1, 1))],
            D_list=[np.eye(1) * 0.5, np.eye(1) * 0.5],
        )
        noise_cov = GSSNoiseCovariance(
            K=K,
            q=1,
            s=1,
            Sigma_U_list=[np.eye(1) * 0.1, np.eye(1) * 0.1],
            Delta_list=[np.zeros((1, 1)), np.zeros((1, 1))],
            Sigma_V_list=[np.eye(1) * 0.1, np.eye(1) * 0.1],
        )
        params = GSSParams(
            K=K,
            q=1,
            s=1,
            P=P,
            f_matrix=f_matrix,
            noise_cov=noise_cov,
            pi0=None,
            mu_z0_list=[np.zeros((dim_z, 1))] * K,
            Sigma_z0_list=[np.eye(dim_z)] * K,
        )
        # Stationary of [[0.9,0.1],[0.2,0.8]]: pi = [2/3, 1/3]
        np.testing.assert_allclose(params.pi0, [2 / 3, 1 / 3], atol=1e-6)

    def test_Sigma_z0_cholesky_cached(self):
        """chol_z0(k) must equal np.linalg.cholesky(Sigma_z0(k))."""
        params = _make_valid_params(K=2, q=1, s=1)
        for k in range(2):
            L = params.chol_z0(k)
            S = params.Sigma_z0(k)
            np.testing.assert_allclose(L @ L.T, S, atol=1e-12)


# ---------------------------------------------------------------------------
# Stationary distribution
# ---------------------------------------------------------------------------


class TestStationaryDistribution:
    def test_stationary_left_eigenvector(self):
        """pi @ P must equal pi (up to numerical tolerance)."""
        P = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.3, 0.3, 0.4]])
        pi = GSSParams._compute_stationary(P)
        np.testing.assert_allclose(pi @ P, pi, atol=1e-10)

    def test_stationary_sums_to_one(self):
        P = np.array([[0.9, 0.1], [0.4, 0.6]])
        pi = GSSParams._compute_stationary(P)
        assert abs(pi.sum() - 1.0) < 1e-10

    def test_stationary_nonnegative(self):
        P = np.array([[0.5, 0.5], [0.3, 0.7]])
        pi = GSSParams._compute_stationary(P)
        assert np.all(pi >= 0.0)


# ---------------------------------------------------------------------------
# Validation — invalid cases
# ---------------------------------------------------------------------------


class TestGSSParamsValidation:
    def test_K_too_small(self):
        with pytest.raises(ParamError, match="K must be"):
            _make_valid_params(K=1)

    def test_P_not_square(self):
        params_kw = dict(
            K=2,
            q=1,
            s=1,
            P=np.ones((2, 3)) / 3,  # not square
            f_matrix=FMatrix(
                K=2,
                q=1,
                s=1,
                A_list=[np.eye(1)] * 2,
                B_list=[np.zeros((1, 1))] * 2,
                C_list=[np.zeros((1, 1))] * 2,
                D_list=[np.eye(1)] * 2,
            ),
            noise_cov=GSSNoiseCovariance(
                K=2,
                q=1,
                s=1,
                Sigma_U_list=[np.eye(1) * 0.1] * 2,
                Delta_list=[np.zeros((1, 1))] * 2,
                Sigma_V_list=[np.eye(1) * 0.1] * 2,
            ),
            pi0=np.array([0.5, 0.5]),
            mu_z0_list=[np.zeros((2, 1))] * 2,
            Sigma_z0_list=[np.eye(2)] * 2,
        )
        with pytest.raises(ParamError):
            GSSParams(**params_kw)

    def test_P_rows_dont_sum_to_1(self):
        with pytest.raises(ParamError, match="row-stochastic"):
            K, q, s = 2, 1, 1
            GSSParams(
                K=K,
                q=q,
                s=s,
                P=np.array([[0.9, 0.2], [0.2, 0.8]]),  # rows sum to 1.1, 1.0
                f_matrix=FMatrix(
                    K=K,
                    q=q,
                    s=s,
                    A_list=[np.eye(1)] * 2,
                    B_list=[np.zeros((1, 1))] * 2,
                    C_list=[np.zeros((1, 1))] * 2,
                    D_list=[np.eye(1)] * 2,
                ),
                noise_cov=GSSNoiseCovariance(
                    K=K,
                    q=q,
                    s=s,
                    Sigma_U_list=[np.eye(1) * 0.1] * 2,
                    Delta_list=[np.zeros((1, 1))] * 2,
                    Sigma_V_list=[np.eye(1) * 0.1] * 2,
                ),
                pi0=np.array([0.5, 0.5]),
                mu_z0_list=[np.zeros((2, 1))] * 2,
                Sigma_z0_list=[np.eye(2)] * 2,
            )

    def test_pi0_wrong_length(self):
        with pytest.raises(ParamError, match="pi0"):
            K, q, s = 2, 1, 1
            GSSParams(
                K=K,
                q=q,
                s=s,
                P=np.eye(K) / K * K,
                f_matrix=FMatrix(
                    K=K,
                    q=q,
                    s=s,
                    A_list=[np.eye(1)] * 2,
                    B_list=[np.zeros((1, 1))] * 2,
                    C_list=[np.zeros((1, 1))] * 2,
                    D_list=[np.eye(1)] * 2,
                ),
                noise_cov=GSSNoiseCovariance(
                    K=K,
                    q=q,
                    s=s,
                    Sigma_U_list=[np.eye(1) * 0.1] * 2,
                    Delta_list=[np.zeros((1, 1))] * 2,
                    Sigma_V_list=[np.eye(1) * 0.1] * 2,
                ),
                pi0=np.array([0.5, 0.3, 0.2]),  # wrong length (3 instead of 2)
                mu_z0_list=[np.zeros((2, 1))] * 2,
                Sigma_z0_list=[np.eye(2)] * 2,
            )

    def test_pi0_doesnt_sum_to_1(self):
        with pytest.raises(ParamError, match="sum to 1"):
            K, q, s = 2, 1, 1
            GSSParams(
                K=K,
                q=q,
                s=s,
                P=np.full((K, K), 1.0 / K),
                f_matrix=FMatrix(
                    K=K,
                    q=q,
                    s=s,
                    A_list=[np.eye(1)] * 2,
                    B_list=[np.zeros((1, 1))] * 2,
                    C_list=[np.zeros((1, 1))] * 2,
                    D_list=[np.eye(1)] * 2,
                ),
                noise_cov=GSSNoiseCovariance(
                    K=K,
                    q=q,
                    s=s,
                    Sigma_U_list=[np.eye(1) * 0.1] * 2,
                    Delta_list=[np.zeros((1, 1))] * 2,
                    Sigma_V_list=[np.eye(1) * 0.1] * 2,
                ),
                pi0=np.array([0.3, 0.3]),  # sums to 0.6
                mu_z0_list=[np.zeros((2, 1))] * 2,
                Sigma_z0_list=[np.eye(2)] * 2,
            )

    def test_Sigma_z0_not_pd(self):
        with pytest.raises(CovarianceError, match="Sigma_z0"):
            K, q, s = 2, 1, 1
            GSSParams(
                K=K,
                q=q,
                s=s,
                P=np.full((K, K), 0.5),
                f_matrix=FMatrix(
                    K=K,
                    q=q,
                    s=s,
                    A_list=[np.eye(1)] * 2,
                    B_list=[np.zeros((1, 1))] * 2,
                    C_list=[np.zeros((1, 1))] * 2,
                    D_list=[np.eye(1)] * 2,
                ),
                noise_cov=GSSNoiseCovariance(
                    K=K,
                    q=q,
                    s=s,
                    Sigma_U_list=[np.eye(1) * 0.1] * 2,
                    Delta_list=[np.zeros((1, 1))] * 2,
                    Sigma_V_list=[np.eye(1) * 0.1] * 2,
                ),
                pi0=np.array([0.5, 0.5]),
                mu_z0_list=[np.zeros((2, 1))] * 2,
                Sigma_z0_list=[-np.eye(2), np.eye(2)],  # k=0 not PD
            )

    def test_mu_z0_wrong_shape(self):
        with pytest.raises(ParamError, match="mu_z0_list"):
            K, q, s = 2, 1, 1
            GSSParams(
                K=K,
                q=q,
                s=s,
                P=np.full((K, K), 0.5),
                f_matrix=FMatrix(
                    K=K,
                    q=q,
                    s=s,
                    A_list=[np.eye(1)] * 2,
                    B_list=[np.zeros((1, 1))] * 2,
                    C_list=[np.zeros((1, 1))] * 2,
                    D_list=[np.eye(1)] * 2,
                ),
                noise_cov=GSSNoiseCovariance(
                    K=K,
                    q=q,
                    s=s,
                    Sigma_U_list=[np.eye(1) * 0.1] * 2,
                    Delta_list=[np.zeros((1, 1))] * 2,
                    Sigma_V_list=[np.eye(1) * 0.1] * 2,
                ),
                pi0=np.array([0.5, 0.5]),
                mu_z0_list=[np.zeros((3, 1))] * 2,  # wrong dim (3 instead of 2)
                Sigma_z0_list=[np.eye(2)] * 2,
            )


# ---------------------------------------------------------------------------
# Display (smoke tests)
# ---------------------------------------------------------------------------


class TestGSSParamsDisplay:
    def test_repr(self):
        params = _make_valid_params()
        r = repr(params)
        assert "GSSParams" in r

    def test_summary_no_crash(self, capsys):
        params = _make_valid_params(K=2, q=1, s=1)
        params.summary()
        captured = capsys.readouterr()
        assert "GSSParams" in captured.out


# ---------------------------------------------------------------------------
# Validation must not be gated behind __debug__ (i.e. it must run under -O)
# ---------------------------------------------------------------------------
class TestValidationUnderOptimize:
    def test_validation_runs_under_O(self):
        """Constructing an invalid GSSParams must still raise under ``python -O``.

        Regression guard: the structural validation used to sit behind
        ``if __debug__:``, which silently disabled it under ``-O``.
        """
        root = pathlib.Path(__file__).resolve().parent.parent
        code = (
            "import numpy as np;"
            "from prg.classes.GSSParams import GSSParams;"
            "GSSParams(K=1, q=1, s=1, P=np.array([[1.0]]), f_matrix=None,"
            " noise_cov=None, pi0=None, mu_z0_list=[], Sigma_z0_list=[])"
        )
        res = subprocess.run(
            [sys.executable, "-O", "-c", code],
            cwd=root,
            capture_output=True,
            text=True,
        )
        assert res.returncode != 0, "GSSParams(K=1) must fail even under python -O"
        assert "K must be" in res.stderr, (
            f"structural validation did not fire under -O:\n{res.stderr}"
        )
