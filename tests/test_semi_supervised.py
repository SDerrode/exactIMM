#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/test_semi_supervised.py
==============================
Unit tests for prg.learning.semi_supervised.

Coverage
--------
- _log_mvn_batch: matches scipy reference, vectorisation correct
- _forward / _backward: shapes, log_lik agrees with brute-force enumeration
- _compute_xi: marginals consistent with γ
- _initialize_kmeans: returns valid hard assignment, all regimes present
- _weighted_fit: reduces to OLS when w is uniform; constraints applied
- _em_run: log_lik history non-decreasing without H5 constraint;
           shapes and SPD of outputs
- fit_semi_supervised: smoke test, multi-start, convergence,
                       statistical recovery (loose)
- _reorder_regimes: A[0,0] descending, P/π0 permuted consistently
- CLI: smoke test, model file generated and importable
"""

from __future__ import annotations

import importlib
import pathlib
import sys

import numpy as np
import pytest
from scipy.stats import multivariate_normal

from prg.classes.GSSParams import GSSParams
from prg.classes.GSSSimulator import GSSSimulator
from prg.learning.semi_supervised import (
    _backward,
    _compute_log_emissions,
    _compute_xi,
    _em_run,
    _forward,
    _initialize_kmeans,
    _initialize_params_from_R,
    _log_mvn_batch,
    _reorder_regimes,
    _weighted_fit,
    fit_semi_supervised,
)
from prg.models.model_gss_K2_q1_s1 import ModelGssK2Q1S1


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def true_params() -> GSSParams:
    return GSSParams.from_model(ModelGssK2Q1S1())


@pytest.fixture(scope="module")
def simulated(true_params, tmp_path_factory):
    """Simulate N=2000, seed=0 — return (xs, ys, rs_true, csv_path)."""
    tmp = tmp_path_factory.mktemp("data")
    sim = GSSSimulator(true_params, N=2000, seed=0)
    csv = sim.run(output_dir=tmp, model_name="model_gss_K2_q1_s1")

    from prg.learning.supervised import _read_csv as rc
    rs, xs, ys, K, q, s = rc(csv)
    return xs, ys, rs, csv


# ---------------------------------------------------------------------------
# _log_mvn_batch
# ---------------------------------------------------------------------------


class TestLogMVN:
    def test_matches_scipy_1d(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((50, 1))
        mu = np.array([0.5])
        Sigma = np.array([[1.5]])
        ours = _log_mvn_batch(X, mu, Sigma)
        ref = multivariate_normal(mean=mu, cov=Sigma).logpdf(X)
        np.testing.assert_allclose(ours, ref, atol=1e-10)

    def test_matches_scipy_2d(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((100, 2))
        mu = np.array([1.0, -2.0])
        Sigma = np.array([[2.0, 0.5], [0.5, 1.0]])
        ours = _log_mvn_batch(X, mu, Sigma)
        ref = multivariate_normal(mean=mu, cov=Sigma).logpdf(X)
        np.testing.assert_allclose(ours, ref, atol=1e-10)

    def test_handles_near_singular(self):
        X = np.array([[0.0]])
        # Should not crash even with singular Σ
        result = _log_mvn_batch(X, np.zeros(1), np.array([[0.0]]))
        assert np.isfinite(result[0])


# ---------------------------------------------------------------------------
# Forward / backward
# ---------------------------------------------------------------------------


class TestForwardBackward:
    @pytest.fixture
    def small_hmm(self):
        """Small HMM: K=2, N=4, dim_z=1 → tractable for brute force."""
        rng = np.random.default_rng(1)
        N, K, dim_z = 4, 2, 1
        Z = rng.standard_normal((N, dim_z))
        F = [np.array([[0.5]]), np.array([[-0.3]])]
        b = [np.array([[0.0]]), np.array([[1.0]])]
        SigW = [np.array([[0.5]]), np.array([[0.2]])]
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        pi0 = np.array([0.6, 0.4])
        log_emis = _compute_log_emissions(Z, F, b, SigW)
        log_init = np.array([
            float(_log_mvn_batch(Z[0:1], np.zeros(dim_z), np.eye(dim_z))[0]),
        ] * K)
        return Z, F, b, SigW, P, pi0, log_emis, log_init

    def test_forward_shapes(self, small_hmm):
        Z, F, b, SigW, P, pi0, log_emis, log_init = small_hmm
        log_alpha, log_lik = _forward(
            log_emis, log_init, np.log(P), np.log(pi0)
        )
        assert log_alpha.shape == (Z.shape[0], len(F))
        assert np.isfinite(log_lik)

    def test_backward_shapes(self, small_hmm):
        Z, F, b, SigW, P, _, log_emis, _ = small_hmm
        log_beta = _backward(log_emis, np.log(P))
        assert log_beta.shape == (Z.shape[0], len(F))
        # Last row must be zero
        np.testing.assert_array_equal(log_beta[-1], 0.0)

    def test_log_lik_brute_force(self, small_hmm):
        """Marginal log-likelihood = log Σ over all R sequences."""
        Z, F, b, SigW, P, pi0, log_emis, log_init = small_hmm
        N, K = Z.shape[0], len(F)

        log_alpha, log_lik = _forward(
            log_emis, log_init, np.log(P), np.log(pi0)
        )

        # Brute force enumeration over K^N sequences
        from itertools import product
        log_p_total = -np.inf
        for seq in product(range(K), repeat=N):
            lp = np.log(pi0[seq[0]]) + log_init[seq[0]]
            for n in range(1, N):
                lp += np.log(P[seq[n - 1], seq[n]]) + log_emis[n - 1, seq[n]]
            log_p_total = np.logaddexp(log_p_total, lp)

        np.testing.assert_allclose(log_lik, log_p_total, atol=1e-10)

    def test_gamma_xi_consistency(self, small_hmm):
        """Σ_k ξ_n(j,k) should equal γ_n(j)  (marginalisation)."""
        Z, F, b, SigW, P, pi0, log_emis, log_init = small_hmm
        log_alpha, log_lik = _forward(
            log_emis, log_init, np.log(P), np.log(pi0)
        )
        log_beta = _backward(log_emis, np.log(P))
        log_gamma = log_alpha + log_beta - log_lik
        log_xi = _compute_xi(log_alpha, log_beta, log_emis, np.log(P), log_lik)

        gamma = np.exp(log_gamma)
        xi    = np.exp(log_xi)

        # Σ_k ξ_n(j, k) = γ_n(j)  for n = 0, …, N-2
        np.testing.assert_allclose(
            xi.sum(axis=2), gamma[:-1], atol=1e-10
        )
        # Σ over j and k of ξ_n = 1 (probability)
        np.testing.assert_allclose(
            xi.sum(axis=(1, 2)), np.ones(Z.shape[0] - 1), atol=1e-10
        )


# ---------------------------------------------------------------------------
# K-means initialisation
# ---------------------------------------------------------------------------


class TestKmeansInit:
    def test_returns_valid_assignment(self, simulated):
        xs, ys, _, _ = simulated
        Z = np.hstack([xs, ys])
        R = _initialize_kmeans(Z, K=2, seed=0)
        assert R.shape == (Z.shape[0],)
        assert R.dtype.kind in "iu"
        assert (R >= 0).all() and (R < 2).all()

    def test_all_regimes_present(self, simulated):
        xs, ys, _, _ = simulated
        Z = np.hstack([xs, ys])
        R = _initialize_kmeans(Z, K=2, seed=0)
        assert set(np.unique(R)) == {0, 1}

    def test_reproducible(self, simulated):
        xs, ys, _, _ = simulated
        Z = np.hstack([xs, ys])
        R1 = _initialize_kmeans(Z, K=2, seed=42)
        R2 = _initialize_kmeans(Z, K=2, seed=42)
        np.testing.assert_array_equal(R1, R2)


# ---------------------------------------------------------------------------
# _initialize_params_from_R
# ---------------------------------------------------------------------------


class TestInitParamsFromR:
    def test_basic(self, simulated):
        xs, ys, rs, _ = simulated
        Z = np.hstack([xs, ys])
        K, q, s = 2, 1, 1
        F, b, S, P, pi0, mu, Sz = _initialize_params_from_R(rs, Z, K, q, s)
        assert len(F) == K
        assert P.shape == (K, K)
        np.testing.assert_allclose(P.sum(axis=1), np.ones(K), atol=1e-10)
        np.testing.assert_allclose(pi0.sum(), 1.0, atol=1e-10)
        for k in range(K):
            assert F[k].shape == (q + s, q + s)
            np.testing.assert_allclose(S[k], S[k].T, atol=1e-10)
            np.linalg.cholesky(S[k])  # SPD
            np.linalg.cholesky(Sz[k])


# ---------------------------------------------------------------------------
# Weighted fit
# ---------------------------------------------------------------------------


class TestWeightedFit:
    def test_uniform_weights_match_OLS(self, simulated):
        """Uniform weights → same answer as supervised _fit_regime."""
        from prg.learning.supervised import _fit_regime
        xs, ys, _, _ = simulated
        Z = np.hstack([xs, ys])
        Z_curr, Z_next = Z[:-1], Z[1:]
        N_pairs = Z_curr.shape[0]
        w = np.ones(N_pairs)

        A1, B1, C1, D1, SU1, Dt1, SV1, b1 = _weighted_fit(
            Z_curr, Z_next, w, q=1, s=1, constraint=None, delta_zero=False
        )
        A2, B2, C2, D2, SU2, Dt2, SV2, b2 = _fit_regime(
            Z_curr, Z_next, q=1, s=1, constraint=None, delta_zero=False
        )

        # F coefficients should match exactly (same OLS problem)
        np.testing.assert_allclose(A1, A2, atol=1e-8)
        np.testing.assert_allclose(B1, B2, atol=1e-8)
        np.testing.assert_allclose(b1, b2, atol=1e-8)
        # Σ_W blocks: weighted MLE with uniform w == unweighted MLE
        np.testing.assert_allclose(SU1, SU2, atol=1e-8)
        np.testing.assert_allclose(SV1, SV2, atol=1e-8)

    def test_zero_weight_falls_back(self):
        """Total weight ≈ 0 → returns identity dynamics, no crash."""
        Z = np.zeros((10, 2))
        w = np.zeros(10)
        A, B, C, D, SU, Dt, SV, b = _weighted_fit(
            Z, Z, w, q=1, s=1, constraint=None, delta_zero=False
        )
        # Identity blocks
        assert A.shape == (1, 1)
        np.testing.assert_allclose(np.block([[A, B], [C, D]]), np.eye(2))

    def test_delta_zero(self, simulated):
        xs, ys, _, _ = simulated
        Z = np.hstack([xs, ys])
        N_pairs = Z.shape[0] - 1
        w = np.ones(N_pairs)
        _, _, _, _, _, Dt, _, _ = _weighted_fit(
            Z[:-1], Z[1:], w, q=1, s=1,
            constraint=None, delta_zero=True,
        )
        np.testing.assert_array_equal(Dt, np.zeros((1, 1)))


# ---------------------------------------------------------------------------
# Single EM run
# ---------------------------------------------------------------------------


class TestEMRun:
    def test_log_lik_monotone_no_constraint(self, simulated):
        """Without H5 constraint, log L should be non-decreasing."""
        xs, ys, _, _ = simulated
        Z = np.hstack([xs, ys])
        params, info = _em_run(
            Z, K=2, q=1, s=1, init_seed=0,
            constraint=None, delta_zero=False,
            max_iter=20, tol=1e-8, verbose=False,
        )
        history = info["log_lik_history"]
        # Allow tiny numerical decreases (≤ 1e-6)
        for i in range(1, len(history)):
            assert history[i] >= history[i - 1] - 1e-6, (
                f"log L decreased at iter {i}: {history[i-1]:.4f} → {history[i]:.4f}"
            )

    def test_output_shapes(self, simulated):
        xs, ys, _, _ = simulated
        Z = np.hstack([xs, ys])
        params, info = _em_run(
            Z, K=2, q=1, s=1, init_seed=0,
            constraint=None, delta_zero=False,
            max_iter=10, tol=1e-5, verbose=False,
        )
        assert params["K"] == 2
        assert params["P"].shape == (2, 2)
        for k in range(2):
            assert params["A_list"][k].shape == (1, 1)
            np.linalg.cholesky(params["Sigma_U_list"][k])
            np.linalg.cholesky(params["Sigma_V_list"][k])

    def test_with_constraint_b(self, simulated):
        from prg.utils.h5_constraint import compute_B_from_h5
        xs, ys, _, _ = simulated
        Z = np.hstack([xs, ys])
        params, info = _em_run(
            Z, K=2, q=1, s=1, init_seed=0,
            constraint="b", delta_zero=False,
            max_iter=10, tol=1e-5, verbose=False,
        )
        for k in range(2):
            B_check = compute_B_from_h5(
                params["A_list"][k], params["C_list"][k],
                params["D_list"][k], params["Sigma_U_list"][k],
                params["Delta_list"][k], params["Sigma_V_list"][k],
            )
            np.testing.assert_allclose(params["B_list"][k], B_check, atol=1e-8)


# ---------------------------------------------------------------------------
# fit_semi_supervised
# ---------------------------------------------------------------------------


class TestFitSemiSupervised:
    def test_smoke(self, simulated):
        xs, ys, _, _ = simulated
        params, info = fit_semi_supervised(
            xs, ys, K=2, n_inits=3, max_iter=20, seed=0,
        )
        assert params["K"] == 2
        assert "best_log_lik" in info
        assert len(info["all_log_liks"]) == 3
        assert info["best_log_lik"] == max(info["all_log_liks"])

    def test_recovers_dynamics(self, simulated, true_params):
        """
        With N=2000 and 5 restarts, the estimated A coefficients should
        come close to the true ones (loose tolerance, label switching).
        """
        xs, ys, _, _ = simulated
        params, info = fit_semi_supervised(
            xs, ys, K=2, n_inits=5, max_iter=50, seed=0,
        )
        true_A_sorted = sorted(
            [true_params.f_matrix.A(k)[0, 0] for k in range(2)],
            reverse=True,
        )
        est_A_sorted  = sorted(
            [params["A_list"][k][0, 0] for k in range(2)],
            reverse=True,
        )
        for est, truth in zip(est_A_sorted, true_A_sorted):
            assert abs(est - truth) < 0.30, (
                f"A recovery failed: est={est:.3f}, true={truth:.3f}"
            )

    def test_n_inits_one(self, simulated):
        xs, ys, _, _ = simulated
        params, info = fit_semi_supervised(
            xs, ys, K=2, n_inits=1, max_iter=10, seed=0,
        )
        assert len(info["all_log_liks"]) == 1
        assert info["best_log_lik"] == info["all_log_liks"][0]

    def test_constraint_b_passes(self, simulated):
        xs, ys, _, _ = simulated
        params, info = fit_semi_supervised(
            xs, ys, K=2, constraint="b", n_inits=2,
            max_iter=15, seed=0,
        )
        from prg.utils.h5_constraint import compute_B_from_h5
        for k in range(2):
            B_check = compute_B_from_h5(
                params["A_list"][k], params["C_list"][k],
                params["D_list"][k], params["Sigma_U_list"][k],
                params["Delta_list"][k], params["Sigma_V_list"][k],
            )
            np.testing.assert_allclose(params["B_list"][k], B_check, atol=1e-8)


# ---------------------------------------------------------------------------
# Reorder regimes
# ---------------------------------------------------------------------------


class TestReorderRegimes:
    def test_descending_A00(self):
        params = {
            "K": 3, "q": 1, "s": 1,
            "A_list": [np.array([[0.2]]), np.array([[0.9]]), np.array([[0.5]])],
            "B_list": [np.array([[1.0]]), np.array([[2.0]]), np.array([[3.0]])],
            "C_list": [np.zeros((1, 1))] * 3,
            "D_list": [np.zeros((1, 1))] * 3,
            "Sigma_U_list": [np.eye(1)] * 3,
            "Delta_list":   [np.zeros((1, 1))] * 3,
            "Sigma_V_list": [np.eye(1)] * 3,
            "mu_z0_list":    [np.zeros((2, 1))] * 3,
            "Sigma_z0_list": [np.eye(2)] * 3,
            "b_list":        [np.zeros((2, 1))] * 3,
            "P": np.array([[0.7, 0.2, 0.1],
                           [0.1, 0.8, 0.1],
                           [0.2, 0.3, 0.5]]),
            "pi0": np.array([0.1, 0.6, 0.3]),
        }
        out = _reorder_regimes(params)
        # New order should be A=[0.9, 0.5, 0.2]
        assert out["A_list"][0][0, 0] == 0.9
        assert out["A_list"][1][0, 0] == 0.5
        assert out["A_list"][2][0, 0] == 0.2
        # B follows the same permutation
        assert out["B_list"][0][0, 0] == 2.0
        # P is permuted on both axes
        assert out["P"][0, 0] == 0.8     # was P[1, 1]
        # π_0 follows
        assert out["pi0"][0] == 0.6      # was pi0[1]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestCLI:
    def _run_main(self, argv):
        from prg.learning.semi_supervised import main
        saved = sys.argv
        sys.argv = ["semi_supervised"] + argv
        try:
            main()
        finally:
            sys.argv = saved

    def test_smoke(self, simulated, tmp_path):
        _, _, _, csv = simulated
        out = tmp_path / "model_em_smoke.py"
        self._run_main([
            str(csv), "-K", "2",
            "--n-inits", "2",
            "--max-iter", "10",
            "--seed", "0",
            "--output", str(out),
        ])
        assert out.exists()
        text = out.read_text(encoding="utf-8")
        assert "BaseGSSModel" in text
        assert "Baum-Welch EM" in text

    def test_generated_importable(self, simulated, tmp_path):
        _, _, _, csv = simulated
        out = tmp_path / "model_em_imp.py"
        self._run_main([
            str(csv), "-K", "2",
            "--n-inits", "2", "--max-iter", "10", "--seed", "0",
            "--output", str(out),
        ])
        sys.path.insert(0, str(tmp_path))
        try:
            mod = importlib.import_module("model_em_imp")
            cls = mod.ModelEmImp
            inst = cls()
            p = inst.get_params()
            assert p["K"] == 2
        finally:
            sys.path.pop(0)
            sys.modules.pop("model_em_imp", None)

    def test_missing_csv(self):
        with pytest.raises(SystemExit):
            self._run_main([
                "/nonexistent/path.csv", "-K", "2",
            ])
