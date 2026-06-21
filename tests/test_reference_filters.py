#!/usr/bin/env python3
"""
tests/test_reference_filters.py
===============================
Guards the ground-truth filters of the simulation study. The exact
hypothesis-tree filter is the reference against which the paper's claim of
exactness is checked, so it must itself be correct: on an (H5)/AB model it has to
agree with ``GSSFilter`` mode ``h5_exact`` to floating-point precision.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from prg.classes.GSSSimulator import GSSSimulator
from prg.experiments.models_paper import get_params
from prg.experiments.reference_filters import (
    exact_mixture_filter,
    oracle_filter,
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
def test_exact_mixture_equals_h5_exact(name, N):
    """On an AB model, the exact Bayesian filter coincides with h5_exact."""
    params = with_stationary_init(_params_from_dict(get_params(name)))
    _, ys = _data(params, N, seed=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        filt = GSSFilter(params, mode="h5_exact")
        ExH = np.array([filt.step(y.reshape(-1, 1)).E_x.ravel() for y in ys])
        filt2 = GSSFilter(params, mode="h5_exact")
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
