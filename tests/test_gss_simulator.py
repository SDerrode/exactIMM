#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/test_gss_simulator.py
============================
Unit tests for GSSSimulator.
"""

import csv
import pathlib
import tempfile

import numpy as np
import pytest

from prg.classes.GSSParams import GSSParams
from prg.classes.GSSSimulator import GSSSimulator
from prg.models.model_gss_K2_q1_s1 import ModelGssK2Q1S1
from prg.utils.exceptions import ParamError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def params_K2_q1_s1() -> GSSParams:
    """Pre-built GSSParams for the toy model (K=2, q=1, s=1)."""
    return GSSParams.from_model(ModelGssK2Q1S1())


@pytest.fixture
def sim(params_K2_q1_s1) -> GSSSimulator:
    """Fresh simulator (seed=0) for each test."""
    return GSSSimulator(params_K2_q1_s1, N=50, seed=0)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestGSSSimulatorConstruction:
    def test_basic(self, params_K2_q1_s1):
        sim = GSSSimulator(params_K2_q1_s1, N=100, seed=42)
        assert sim.N == 100
        assert sim.params is params_K2_q1_s1

    def test_N_not_positive(self, params_K2_q1_s1):
        with pytest.raises(ParamError, match="N must be"):
            GSSSimulator(params_K2_q1_s1, N=0)

    def test_N_not_int(self, params_K2_q1_s1):
        with pytest.raises(ParamError, match="N must be"):
            GSSSimulator(params_K2_q1_s1, N=10.5)  # type: ignore

    def test_repr(self, sim):
        r = repr(sim)
        assert "GSSSimulator" in r
        assert "N=50" in r


# ---------------------------------------------------------------------------
# Iterator protocol
# ---------------------------------------------------------------------------


class TestGSSSimulatorIterator:
    def test_first_yield_is_n0(self, sim):
        n, r, x, y = next(iter(sim))
        assert n == 0

    def test_yields_exactly_N_steps(self, params_K2_q1_s1):
        N = 37
        sim = GSSSimulator(params_K2_q1_s1, N=N, seed=1)
        steps = list(sim)
        assert len(steps) == N

    def test_indices_are_sequential(self, params_K2_q1_s1):
        N = 20
        sim = GSSSimulator(params_K2_q1_s1, N=N, seed=2)
        indices = [n for n, *_ in sim]
        assert indices == list(range(N))

    def test_stop_iteration(self, params_K2_q1_s1):
        sim = GSSSimulator(params_K2_q1_s1, N=5, seed=3)
        list(sim)    # exhaust
        with pytest.raises(StopIteration):
            next(sim)

    def test_r_range(self, params_K2_q1_s1):
        N = 200
        K = params_K2_q1_s1.K
        sim = GSSSimulator(params_K2_q1_s1, N=N, seed=4)
        for _, r, _, _ in sim:
            assert 0 <= r < K

    def test_x_shape(self, params_K2_q1_s1):
        q = params_K2_q1_s1.q
        sim = GSSSimulator(params_K2_q1_s1, N=10, seed=5)
        for _, _, x, _ in sim:
            assert x.shape == (q, 1)

    def test_y_shape(self, params_K2_q1_s1):
        s = params_K2_q1_s1.s
        sim = GSSSimulator(params_K2_q1_s1, N=10, seed=6)
        for _, _, _, y in sim:
            assert y.shape == (s, 1)

    def test_all_finite(self, params_K2_q1_s1):
        sim = GSSSimulator(params_K2_q1_s1, N=100, seed=7)
        for _, _, x, y in sim:
            assert np.all(np.isfinite(x))
            assert np.all(np.isfinite(y))


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


class TestGSSSimulatorReproducibility:
    def test_same_seed_same_sequence(self, params_K2_q1_s1):
        N = 50
        sim_a = GSSSimulator(params_K2_q1_s1, N=N, seed=99)
        sim_b = GSSSimulator(params_K2_q1_s1, N=N, seed=99)
        for (na, ra, xa, ya), (nb, rb, xb, yb) in zip(sim_a, sim_b):
            assert na == nb
            assert ra == rb
            np.testing.assert_array_equal(xa, xb)
            np.testing.assert_array_equal(ya, yb)

    def test_different_seeds_different_sequences(self, params_K2_q1_s1):
        N = 100
        sim_a = GSSSimulator(params_K2_q1_s1, N=N, seed=11)
        sim_b = GSSSimulator(params_K2_q1_s1, N=N, seed=22)
        diffs = sum(
            not np.array_equal(xa, xb)
            for (_, _, xa, _), (_, _, xb, _) in zip(sim_a, sim_b)
        )
        assert diffs > 0   # with overwhelming probability

    def test_reset_restores_sequence(self, params_K2_q1_s1):
        N = 30
        sim = GSSSimulator(params_K2_q1_s1, N=N, seed=55)
        first_run = list(sim)
        sim.reset()            # same seed
        second_run = list(sim)
        for (na, ra, xa, ya), (nb, rb, xb, yb) in zip(first_run, second_run):
            assert na == nb and ra == rb
            np.testing.assert_array_equal(xa, xb)
            np.testing.assert_array_equal(ya, yb)

    def test_reset_with_new_seed(self, params_K2_q1_s1):
        sim = GSSSimulator(params_K2_q1_s1, N=20, seed=1)
        run_a = list(sim)
        sim.reset(seed=999)
        run_b = list(sim)
        xs_a = [x for _, _, x, _ in run_a]
        xs_b = [x for _, _, x, _ in run_b]
        assert not all(np.array_equal(a, b) for a, b in zip(xs_a, xs_b))


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------


class TestGSSSimulatorRunAndSave:
    def test_csv_created(self, params_K2_q1_s1, tmp_path):
        sim = GSSSimulator(params_K2_q1_s1, N=10, seed=0)
        filepath = sim.run(output_dir=tmp_path, model_name="test_model")
        assert filepath.exists()

    def test_csv_row_count(self, params_K2_q1_s1, tmp_path):
        N = 42
        sim = GSSSimulator(params_K2_q1_s1, N=N, seed=0)
        filepath = sim.run(output_dir=tmp_path, model_name="test_model")
        with filepath.open() as fh:
            reader = csv.reader(fh)
            rows = list(reader)
        # 1 header + N data rows
        assert len(rows) == N + 1

    def test_csv_header(self, params_K2_q1_s1, tmp_path):
        sim = GSSSimulator(params_K2_q1_s1, N=5, seed=0)
        filepath = sim.run(output_dir=tmp_path, model_name="test_model")
        with filepath.open() as fh:
            header = next(csv.reader(fh))
        assert header[0] == "n"
        assert header[1] == "r"
        assert header[2] == "x_0"   # q=1 → only x_0
        assert header[3] == "y_0"   # s=1 → only y_0

    def test_csv_n_column_sequential(self, params_K2_q1_s1, tmp_path):
        N = 15
        sim = GSSSimulator(params_K2_q1_s1, N=N, seed=0)
        filepath = sim.run(output_dir=tmp_path, model_name="test_model")
        with filepath.open() as fh:
            reader = csv.reader(fh)
            next(reader)           # skip header
            ns = [int(row[0]) for row in reader]
        assert ns == list(range(N))

    def test_csv_r_column_in_range(self, params_K2_q1_s1, tmp_path):
        K = params_K2_q1_s1.K
        N = 50
        sim = GSSSimulator(params_K2_q1_s1, N=N, seed=0)
        filepath = sim.run(output_dir=tmp_path, model_name="test_model")
        with filepath.open() as fh:
            reader = csv.reader(fh)
            next(reader)
            rs = [int(row[1]) for row in reader]
        assert all(0 <= r < K for r in rs)

    def test_run_resets_before_saving(self, params_K2_q1_s1, tmp_path):
        """Calling run() twice with same seed must produce identical CSV files."""
        sim = GSSSimulator(params_K2_q1_s1, N=20, seed=77)
        p1 = sim.run(output_dir=tmp_path / "run1", model_name="m")
        p2 = sim.run(output_dir=tmp_path / "run2", model_name="m")
        assert p1.read_text() == p2.read_text()


# ---------------------------------------------------------------------------
# Statistical sanity check (large N)
# ---------------------------------------------------------------------------


class TestGSSSimulatorStatistics:
    def test_empirical_mean_near_zero(self, params_K2_q1_s1):
        """
        With zero initial mean and a stable model, the long-run empirical
        mean of Z_n should be close to 0.
        Tolerance is generous to avoid flakiness.
        """
        N = 5_000
        sim = GSSSimulator(params_K2_q1_s1, N=N, seed=123)
        xs = np.array([x.ravel() for _, _, x, _ in sim])
        mean_x = xs.mean(axis=0)
        np.testing.assert_allclose(mean_x, 0.0, atol=0.2)
