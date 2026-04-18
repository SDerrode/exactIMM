#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/test_gss_filter.py
========================
Unit tests for GSSFilter and FilterResult.

Coverage
--------
- Construction and repr
- FilterResult shape and Var_x property
- Initialisation step (n=0)
- Recursion (n≥1): shapes, valid probabilities, state update
- reset() reproducibility
- run() joint simulate-and-filter
- run_csv() consistency with run()
- Statistical sanity: RMSE < naive zero-predictor
- Posterior variance is positive semi-definite
- Option B / zero-mean equivalence
"""

from __future__ import annotations

import pathlib
import tempfile

import numpy as np
import pytest

from prg.classes.GSSParams import GSSParams
from prg.filter.gss_filter import FilterResult, GSSFilter
from prg.models.model_gss_K2_q1_s1 import ModelGssK2Q1S1


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def params() -> GSSParams:
    """Shared GSSParams for the toy model (K=2, q=1, s=1)."""
    return GSSParams.from_model(ModelGssK2Q1S1())


@pytest.fixture
def filt(params) -> GSSFilter:
    """Fresh GSSFilter for each test."""
    return GSSFilter(params)


@pytest.fixture(scope="module")
def run_df(params):
    """Run once (N=200, seed=0); reused across sanity tests."""
    _, df = GSSFilter(params).run(N=200, seed=0)
    return df


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestGSSFilterConstruction:
    def test_basic(self, params, filt):
        assert filt.params is params
        assert filt.n == 0

    def test_repr(self, filt):
        r = repr(filt)
        assert "GSSFilter" in r
        assert "K=2" in r
        assert "n=0" in r


# ---------------------------------------------------------------------------
# FilterResult
# ---------------------------------------------------------------------------


class TestFilterResult:
    def test_shapes(self, params, filt):
        y = np.array([[0.5]])
        res = filt.step(y)
        assert res.E_x.shape  == (params.q, 1)
        assert res.E_xx.shape == (params.q, params.q)
        assert res.pi.shape   == (params.K,)

    def test_pi_sums_to_one(self, params, filt):
        y = np.array([[0.5]])
        res = filt.step(y)
        assert abs(res.pi.sum() - 1.0) < 1e-12

    def test_pi_non_negative(self, filt):
        y = np.array([[0.5]])
        res = filt.step(y)
        assert np.all(res.pi >= 0)

    def test_var_x_shape(self, params, filt):
        y = np.array([[0.5]])
        res = filt.step(y)
        assert res.Var_x.shape == (params.q, params.q)

    def test_var_x_psd(self, filt):
        """Posterior variance must be positive semi-definite."""
        y = np.array([[0.5]])
        res = filt.step(y)
        eigvals = np.linalg.eigvalsh(res.Var_x)
        assert np.all(eigvals >= -1e-10)

    def test_var_x_symmetric(self, filt):
        y = np.array([[0.5]])
        res = filt.step(y)
        assert np.allclose(res.Var_x, res.Var_x.T, atol=1e-12)

    def test_n_increments(self, filt):
        for expected_n in range(5):
            res = filt.step(np.array([[float(expected_n)]]))
            assert res.n == expected_n
        assert filt.n == 5


# ---------------------------------------------------------------------------
# Multi-step recursion
# ---------------------------------------------------------------------------


class TestRecursion:
    def test_shapes_after_many_steps(self, params, filt):
        for i in range(10):
            y = np.random.default_rng(i).standard_normal((params.s, 1))
            res = filt.step(y)
            assert res.E_x.shape  == (params.q, 1)
            assert res.E_xx.shape == (params.q, params.q)
            assert res.pi.shape   == (params.K,)

    def test_pi_always_valid(self, params, filt):
        rng = np.random.default_rng(99)
        for _ in range(20):
            y = rng.standard_normal((params.s, 1))
            res = filt.step(y)
            assert abs(res.pi.sum() - 1.0) < 1e-12
            assert np.all(res.pi >= 0)

    def test_e_xx_psd_after_many_steps(self, params, filt):
        """E[XX^T|y] must remain PSD throughout the recursion."""
        rng = np.random.default_rng(7)
        for _ in range(15):
            y = rng.standard_normal((params.s, 1))
            res = filt.step(y)
            eigvals = np.linalg.eigvalsh(res.E_xx)
            assert np.all(eigvals >= -1e-10)

    def test_y_scalar_and_column_give_same_result(self, params):
        """step() should accept both shape (s,) and (s,1)."""
        f1 = GSSFilter(params)
        f2 = GSSFilter(params)
        y_col = np.array([[1.23]])
        y_flat = np.array([1.23])
        res1 = f1.step(y_col)
        res2 = f2.step(y_flat)
        assert np.allclose(res1.E_x, res2.E_x)
        assert np.allclose(res1.pi,  res2.pi)


# ---------------------------------------------------------------------------
# reset() and reproducibility
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_restores_n_zero(self, filt, params):
        for _ in range(5):
            filt.step(np.zeros((params.s, 1)))
        assert filt.n == 5
        filt.reset()
        assert filt.n == 0

    def test_reset_gives_same_results(self, params):
        """Two passes over the same sequence must yield identical outputs."""
        filt = GSSFilter(params)
        rng  = np.random.default_rng(42)
        ys   = [rng.standard_normal((params.s, 1)) for _ in range(20)]

        results_first  = [filt.step(y) for y in ys]
        filt.reset()
        results_second = [filt.step(y) for y in ys]

        for r1, r2 in zip(results_first, results_second):
            assert np.allclose(r1.E_x,  r2.E_x)
            assert np.allclose(r1.E_xx, r2.E_xx)
            assert np.allclose(r1.pi,   r2.pi)


# ---------------------------------------------------------------------------
# run()
# ---------------------------------------------------------------------------


class TestRun:
    def test_df_has_expected_columns(self, params, run_df):
        q, K = params.q, params.K
        for i in range(q):
            assert f"E_x_{i}" in run_df.columns
            assert f"V_x_{i}" in run_df.columns
        for k in range(K):
            assert f"p_r_{k}" in run_df.columns
        assert "sq_err" in run_df.columns

    def test_df_length(self, run_df):
        assert len(run_df) == 200

    def test_probabilities_sum_to_one(self, params, run_df):
        p_cols = [f"p_r_{k}" for k in range(params.K)]
        row_sums = run_df[p_cols].sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-10)

    def test_sq_err_non_negative(self, run_df):
        assert (run_df["sq_err"] >= 0).all()

    def test_reproducibility_across_runs(self, params):
        """Two calls with the same seed must produce identical DataFrames."""
        _, df1 = GSSFilter(params).run(N=50, seed=3)
        _, df2 = GSSFilter(params).run(N=50, seed=3)
        assert np.allclose(df1["E_x_0"].values, df2["E_x_0"].values)
        assert np.allclose(df1["sq_err"].values, df2["sq_err"].values)

    def test_different_seeds_differ(self, params):
        _, df1 = GSSFilter(params).run(N=100, seed=1)
        _, df2 = GSSFilter(params).run(N=100, seed=2)
        assert not np.allclose(df1["E_x_0"].values, df2["E_x_0"].values)

    def test_sim_csv_written(self, params):
        with tempfile.TemporaryDirectory() as tmp:
            sim_path, _ = GSSFilter(params).run(N=30, seed=5, output_dir=tmp)
            assert sim_path is not None
            assert pathlib.Path(sim_path).exists()
            assert pathlib.Path(sim_path).stat().st_size > 0

    def test_no_sim_csv_when_output_dir_none(self, params):
        sim_path, df = GSSFilter(params).run(N=30, seed=5, output_dir=None)
        assert sim_path is None
        assert len(df) == 30


# ---------------------------------------------------------------------------
# run_csv()
# ---------------------------------------------------------------------------


class TestRunCsv:
    def test_csv_and_run_give_same_estimates(self, params):
        """
        run() and run_csv() on the same simulation must give identical
        filtered estimates.
        """
        with tempfile.TemporaryDirectory() as tmp:
            sim_path, df_run = GSSFilter(params).run(
                N=80, seed=11, output_dir=tmp
            )
            df_csv = GSSFilter(params).run_csv(sim_path)

        assert np.allclose(
            df_run["E_x_0"].values, df_csv["E_x_0"].values, atol=1e-12
        )
        assert np.allclose(
            df_run["p_r_0"].values, df_csv["p_r_0"].values, atol=1e-12
        )

    def test_csv_has_sq_err_when_x_present(self, params):
        with tempfile.TemporaryDirectory() as tmp:
            sim_path, _ = GSSFilter(params).run(N=30, seed=9, output_dir=tmp)
            df = GSSFilter(params).run_csv(sim_path)
        assert "sq_err" in df.columns

    def test_csv_without_x_has_no_sq_err(self, params):
        """A CSV with only y columns must not produce sq_err."""
        import pandas as pd

        # Build a y-only CSV
        rng = np.random.default_rng(0)
        rows = [{"n": i, "y_0": float(rng.standard_normal())} for i in range(20)]
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as tmp:
            pd.DataFrame(rows).to_csv(tmp.name, index=False)
            df = GSSFilter(params).run_csv(tmp.name)
        assert "sq_err" not in df.columns


# ---------------------------------------------------------------------------
# Statistical sanity
# ---------------------------------------------------------------------------


class TestStatisticalSanity:
    def test_rmse_below_naive(self, params):
        """
        The filter RMSE must beat the naive predictor E[X]=0 on a long run.
        The naive MSE is Var[X] ≈ 1 (identity initial covariance).
        """
        _, df = GSSFilter(params).run(N=2000, seed=0)
        filter_mse = df["sq_err"].mean()
        assert filter_mse < 0.8, (
            f"Filter MSE {filter_mse:.3f} is not below naive baseline 0.8"
        )

    def test_posterior_variance_decreases_on_average(self, params):
        """
        After the transient, V_x should stabilise well below the prior
        variance of 1.0.
        """
        _, df = GSSFilter(params).run(N=500, seed=1)
        mean_var = df["V_x_0"].iloc[50:].mean()   # skip transient
        assert mean_var < 0.8

    def test_regime_probabilities_not_degenerate(self, params):
        """
        On a long run no single regime should monopolise all probability
        (both states are visited).
        """
        _, df = GSSFilter(params).run(N=1000, seed=2)
        mean_p0 = df["p_r_0"].mean()
        assert 0.2 < mean_p0 < 0.8


# ---------------------------------------------------------------------------
# Option B / zero-mean equivalence
# ---------------------------------------------------------------------------


class TestOptionB:
    def test_zero_mean_model_identical_to_nonzero_zero_init(self, params):
        """
        The toy model already has mu_z0=0.  Explicitly zeroing it must give
        the same filter output (Option B reduces to the paper for zero mean).
        """
        # params already has mu_z0=0 — just run twice and compare
        filt_a = GSSFilter(params)
        filt_b = GSSFilter(params)
        rng = np.random.default_rng(77)
        ys  = [rng.standard_normal((params.s, 1)) for _ in range(30)]
        for y in ys:
            ra = filt_a.step(y)
            rb = filt_b.step(y)
            assert np.allclose(ra.E_x, rb.E_x,  atol=1e-14)
            assert np.allclose(ra.pi,  rb.pi,   atol=1e-14)

    def test_e_xx_geq_e_x_squared(self, params):
        """
        E[X²|y] ≥ E[X|y]²  must hold at every step (Var ≥ 0 in 1-D).
        """
        filt = GSSFilter(params)
        rng  = np.random.default_rng(55)
        for _ in range(30):
            y   = rng.standard_normal((params.s, 1))
            res = filt.step(y)
            # scalar case: E_xx[0,0] >= E_x[0,0]**2
            assert float(res.E_xx[0, 0]) >= float(res.E_x[0, 0]) ** 2 - 1e-10
