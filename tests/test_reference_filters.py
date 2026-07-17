#!/usr/bin/env python3
"""
tests/test_reference_filters.py
===============================
Guards the ground-truth filters of the simulation study. The exact
hypothesis-tree filter is the reference against which the paper's claim of
exactness is checked, so it must itself be correct: on an AB/AB model it has to
agree with ``GSSFilter`` mode ``ngh_kf`` to floating-point precision.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from prg.classes.GSSSimulator import GSSSimulator
from prg.experiments.models_paper import get_params
from prg.experiments.reference_filters import (
    exact_mixture_filter,
    gpb2_filter,
    imm_filter,
    oracle_filter,
    rbpf_filter,
    single_kalman_filter,
    with_stationary_init,
)
from prg.experiments.run_simulations import _params_from_dict
from prg.filter.gss_filter import GSSFilter


def _data(params, N, seed):
    rs, ys = [], []
    for _, r, _x, y in GSSSimulator(params, N=N, seed=seed):
        rs.append(int(r))
        ys.append(np.asarray(y, dtype=float).ravel())
    return np.array(rs), np.array(ys)


@pytest.mark.parametrize("name,N", [("M1", 8), ("M2", 7), ("M3", 6)])
def test_exact_mixture_equals_ngh_kf(name, N):
    """On an AB model, the exact Bayesian filter coincides with ngh_kf."""
    params = with_stationary_init(_params_from_dict(get_params(name)))
    _, ys = _data(params, N, seed=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        filt = GSSFilter(params, mode="ngh_kf")
        ExH = np.array([filt.step(y.reshape(-1, 1)).E_x.ravel() for y in ys])
        filt2 = GSSFilter(params, mode="ngh_kf")
        PiH = np.array([np.asarray(filt2.step(y.reshape(-1, 1)).pi).ravel() for y in ys])
    ExE, _VarE, PiE = exact_mixture_filter(params, ys)
    assert np.abs(ExH - ExE).max() < 1e-9
    assert np.abs(PiH - PiE).max() < 1e-9


def test_oracle_beats_single_kalman():
    """The regime-aware oracle should not be worse than the regime-blind Kalman."""
    params = with_stationary_init(_params_from_dict(get_params("M3")))
    rs, ys = _data(params, N=300, seed=4)
    # ground-truth X via the simulator (re-run identically for x)
    xs = np.array(
        [np.asarray(x, dtype=float).ravel() for _, _r, x, _y in GSSSimulator(params, N=300, seed=4)]
    )
    Eo, _ = oracle_filter(params, rs, ys)
    Ek, _ = single_kalman_filter(params, ys)
    rmse_o = float(np.sqrt(np.mean((Eo - xs) ** 2)))
    rmse_k = float(np.sqrt(np.mean((Ek - xs) ** 2)))
    assert rmse_o <= rmse_k + 1e-9
    assert Eo.shape == (300, params.q)


def test_rank_deficient_model_is_valid_and_exact():
    """E7: with a rank-deficient C (s < q, full column rank impossible), the model
    is still a valid NGH-MSM and ngh_kf still equals the exact Bayesian filter."""
    from prg.experiments.study import rank_deficient_model
    from prg.utils.ab_constraint import validate_ngh_msm

    params = rank_deficient_model()
    assert params.s < params.q  # under-observed: C cannot be full column rank
    assert all(np.linalg.matrix_rank(params.f_matrix.C(k)) < params.q for k in range(params.K))
    assert validate_ngh_msm(params) == []  # accepted by the relaxed (C≠0) gate

    _, ys = _data(params, N=9, seed=11)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        filt = GSSFilter(params, mode="ngh_kf")
        ExH = np.array([filt.step(y.reshape(-1, 1)).E_x.ravel() for y in ys])
        filt2 = GSSFilter(params, mode="ngh_kf")
        PiH = np.array([np.asarray(filt2.step(y.reshape(-1, 1)).pi).ravel() for y in ys])
    ExE, _VarE, PiE = exact_mixture_filter(params, ys)
    assert np.abs(ExH - ExE).max() < 1e-9
    assert np.abs(PiH - PiE).max() < 1e-9


# ---------------------------------------------------------------------------
# Approximate switching baselines — issue #5 (IMM / GPB2 / RBPF)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("filt", [imm_filter, gpb2_filter, rbpf_filter])
def test_baseline_output_contract(filt):
    """Each baseline returns (E_x, Var_x, pi, log_lik) with valid shapes."""
    params = with_stationary_init(_params_from_dict(get_params("M3")))
    _, ys = _data(params, N=10, seed=5)
    E_x, Var_x, pi, ll = filt(params, ys)
    N, K, q = 10, params.K, params.q
    assert E_x.shape == (N, q)
    assert Var_x.shape == (N, q, q)
    assert pi.shape == (N, K)
    assert np.allclose(pi.sum(axis=1), 1.0)
    assert np.all(pi >= -1e-9)
    assert np.isfinite(ll)


@pytest.mark.parametrize("name,N", [("M1", 8), ("M2", 7), ("M3", 7)])
def test_gpb2_equals_exact_on_ab_model(name, N):
    """On an AB/AB model the regime-conditional posterior depends only on the
    current regime (marginal Markovianity, Prop. 4), so GPB2's K-hypothesis
    collapse is lossless: it coincides with the exact Kᴺ filter."""
    params = with_stationary_init(_params_from_dict(get_params(name)))
    _, ys = _data(params, N, seed=1)
    ExE, _VarE, PiE = exact_mixture_filter(params, ys)
    Eg, _Vg, Pg, llg = gpb2_filter(params, ys)
    assert np.isfinite(llg)
    assert np.abs(Eg - ExE).max() < 1e-8
    assert np.abs(Pg - PiE).max() < 1e-8


@pytest.mark.parametrize("name", ["M1", "M2", "M3"])
def test_imm_close_to_exact(name):
    """IMM (single moment-matched mixing) tracks the exact filter closely."""
    params = with_stationary_init(_params_from_dict(get_params(name)))
    _, ys = _data(params, N=9, seed=2)
    ExE, _VarE, PiE = exact_mixture_filter(params, ys)
    Ei, _Vi, Pi_, lli = imm_filter(params, ys)
    assert np.isfinite(lli)
    assert np.abs(Ei - ExE).max() < 2e-3
    assert np.abs(Pi_ - PiE).max() < 2e-2


def test_rbpf_close_to_exact():
    """RBPF (many particles, fixed seed) approaches the exact posterior."""
    params = with_stationary_init(_params_from_dict(get_params("M1")))
    _, ys = _data(params, N=9, seed=3)
    ExE, _VarE, PiE = exact_mixture_filter(params, ys)
    Er, _Vr, Pr, llr = rbpf_filter(params, ys, n_particles=4000, seed=0)
    assert np.isfinite(llr)
    assert np.abs(Er - ExE).max() < 3e-2
    assert np.abs(Pr - PiE).max() < 5e-2
