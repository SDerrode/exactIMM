#!/usr/bin/env python3
"""
tests/test_smoothers.py
=======================
Machine-precision guards for the O(N K²) fixed-interval smoothers of
``prg/filter/gss_smoother.py`` (the smoothing companion paper), against the
exact Kᴺ path-enumeration smoother of
``prg/experiments/reference_smoothers.py``.

The map being tested:
- regime smoothing (forward-backward on the observed chain) is exact on both
  families with the matching kernel;
- ``{C≡0}``: the state smoother is a *reweighting* of the filtered per-regime
  summaries — and that same formula fails under AB (discriminant);
- AB: the state smoother is the *one-step constant-gain correction* — and the
  naive smoothed read-out without the correction fails (the trap);
- ``{A≡MC}``: the lag-1 extension is exact and reduces to the AB smoother
  when B = MD as well.
"""

from __future__ import annotations

import numpy as np
import pytest

from prg.classes.GSSSimulator import GSSSimulator
from prg.experiments.cns_exactness import (
    ab_model,
    cgo_memory_model,
    three_regime_mixed_model,
)
from prg.experiments.models_paper import get_params
from prg.experiments.reference_filters import gpb2_filter
from prg.experiments.reference_smoothers import exact_mixture_smoother
from prg.experiments.run_simulations import _params_from_dict
from prg.filter.gss_smoother import (
    _slaving,
    constant_gain_smoother,
    lag1_constant_gain_smoother,
    regime_smoother,
    reweight_smoother,
)
from prg.utils.ab_constraint import apply_AB_constraint

N_STEPS = 9
SEEDS = (100, 101, 102)
TOL = 1e-9  # machine-precision budget after ~10 steps of small-matrix algebra


def _rel(a, b):
    """Normalized sup-norm deviation (same metric as the exactness studies)."""
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a - b)) / (np.max(np.abs(b)) + 1e-12))


def _data(params, N, seed):
    ys = []
    for _, _r, _x, y in GSSSimulator(params, N=N, seed=seed):
        ys.append(np.asarray(y, dtype=float).ravel())
    return np.array(ys)


# ---------------------------------------------------------------------------
# Regime smoothing — exact on both families with the matching kernel
# ---------------------------------------------------------------------------
def test_regime_smoother_exact_on_ab():
    params = ab_model(0.4, dB=0.0)
    for seed in SEEDS:
        ys = _data(params, N_STEPS, seed)
        _, _, pi_ex, pair_ex = exact_mixture_smoother(params, ys)
        gamma, gamma_pair, _ = regime_smoother(params, ys, kernel="ab")
        assert _rel(gamma, pi_ex) < TOL
        assert _rel(gamma_pair, pair_ex) < TOL


def test_regime_smoother_exact_on_c0():
    params = cgo_memory_model()
    for seed in SEEDS:
        ys = _data(params, N_STEPS, seed)
        _, _, pi_ex, pair_ex = exact_mixture_smoother(params, ys)
        gamma, gamma_pair, _ = regime_smoother(params, ys, kernel="c0")
        assert _rel(gamma, pi_ex) < TOL
        assert _rel(gamma_pair, pair_ex) < TOL


def test_regime_smoother_lag1_exact_on_slaving_A():
    params = ab_model(0.4, dB=0.15)  # {A≡MC} \ AB: volet B broken
    for seed in SEEDS:
        ys = _data(params, N_STEPS, seed)
        _, _, pi_ex, pair_ex = exact_mixture_smoother(params, ys)
        gamma, gamma_pair, _ = regime_smoother(params, ys, kernel="lag1")
        assert _rel(gamma, pi_ex) < TOL
        assert _rel(gamma_pair, pair_ex) < TOL


# ---------------------------------------------------------------------------
# {C≡0} — reweighting is the exact smoother there, and only there
# ---------------------------------------------------------------------------
def test_reweight_smoother_exact_on_c0():
    params = cgo_memory_model()
    for seed in SEEDS:
        ys = _data(params, N_STEPS, seed)
        Ex_ex, Vx_ex, pi_ex, _ = exact_mixture_smoother(params, ys)
        Ex, Vx, gamma, _ = reweight_smoother(params, ys)
        assert _rel(Ex, Ex_ex) < TOL
        assert _rel(Vx, Vx_ex) < TOL
        assert _rel(gamma, pi_ex) < TOL


def test_reweight_smoother_fails_under_ab():
    """Discriminant: the {C≡0} mechanism (reweighting of filtered summaries)
    is NOT the AB smoother — two families, two mechanisms."""
    params = ab_model(0.4, dB=0.0)
    gaps = []
    for seed in SEEDS:
        ys = _data(params, N_STEPS, seed)
        Ex_ex, _, _, _ = exact_mixture_smoother(params, ys)
        Ex_rw, _, _, _ = reweight_smoother(params, ys)
        gaps.append(_rel(Ex_rw, Ex_ex))
    assert np.median(gaps) > 0.05


def test_naive_readout_fails_under_ab():
    """The trap: reweighting the slaved read-out M_k y_n + c_k with the EXACT
    smoothed regime posterior still misses — the one-step backward correction
    is the essential part of the AB smoother, not a refinement."""
    params = ab_model(0.4, dB=0.0)
    gaps = []
    for seed in SEEDS:
        ys = _data(params, N_STEPS, seed)
        Ex_ex, _, _, _ = exact_mixture_smoother(params, ys)
        _, _, gamma_ab, _ = constant_gain_smoother(params, ys)  # exact gamma under AB
        M, _, c = _slaving(params)
        yv = ys.reshape(-1, params.s)
        Ex_naive = np.array(
            [
                sum(
                    gamma_ab[n, k] * (M[k] @ yv[n].reshape(-1, 1) + c[k]).ravel()
                    for k in range(params.K)
                )
                for n in range(len(yv))
            ]
        )
        gaps.append(_rel(Ex_naive, Ex_ex))
    assert np.median(gaps) > 0.05  # the backward correction is essential


# ---------------------------------------------------------------------------
# AB — the one-step constant-gain correction is the exact smoother
# ---------------------------------------------------------------------------
def test_constant_gain_smoother_exact_on_ab():
    params = ab_model(0.4, dB=0.0)
    for seed in SEEDS:
        ys = _data(params, N_STEPS, seed)
        Ex_ex, Vx_ex, pi_ex, _ = exact_mixture_smoother(params, ys)
        Ex, Vx, gamma, _ = constant_gain_smoother(params, ys)
        assert _rel(Ex, Ex_ex) < TOL
        assert _rel(Vx, Vx_ex) < TOL
        assert _rel(gamma, pi_ex) < TOL


def test_constant_gain_smoother_exact_with_nonstationary_init():
    """The first-slice handling: exact on an AB model whose initial law is
    *not* the stationary one (paper model M1, raw init)."""
    params = _params_from_dict(get_params("M1"))
    ys = _data(params, 8, seed=1)
    Ex_ex, Vx_ex, pi_ex, _ = exact_mixture_smoother(params, ys)
    Ex, Vx, gamma, _ = constant_gain_smoother(params, ys)
    assert _rel(Ex, Ex_ex) < TOL
    assert _rel(Vx, Vx_ex) < TOL
    assert _rel(gamma, pi_ex) < TOL


def test_constant_gain_smoother_exact_on_k3_ab_model():
    """Uniform AB with K = 3 regimes (one of them C = 0): still exact."""
    params = apply_AB_constraint(three_regime_mixed_model(0.5, 0.5))
    ys = _data(params, 7, seed=3)
    Ex_ex, Vx_ex, pi_ex, _ = exact_mixture_smoother(params, ys)
    Ex, Vx, gamma, _ = constant_gain_smoother(params, ys)
    assert _rel(Ex, Ex_ex) < TOL
    assert _rel(Vx, Vx_ex) < TOL
    assert _rel(gamma, pi_ex) < TOL


# ---------------------------------------------------------------------------
# {A≡MC} — the lag-1 extension, and its reduction to AB
# ---------------------------------------------------------------------------
def test_lag1_smoother_exact_on_slaving_A():
    params = ab_model(0.4, dB=0.15)  # volet B broken: outside AB, inside {A≡MC}
    for seed in SEEDS:
        ys = _data(params, N_STEPS, seed)
        Ex_ex, Vx_ex, pi_ex, _ = exact_mixture_smoother(params, ys)
        Ex, Vx, gamma, _ = lag1_constant_gain_smoother(params, ys)
        assert _rel(Ex, Ex_ex) < TOL
        assert _rel(Vx, Vx_ex) < TOL
        assert _rel(gamma, pi_ex) < TOL


def test_lag1_reduces_to_constant_gain_on_ab():
    params = ab_model(0.4, dB=0.0)
    ys = _data(params, N_STEPS, seed=100)
    Ex_ab, Vx_ab, g_ab, ll_ab = constant_gain_smoother(params, ys)
    Ex_l1, Vx_l1, g_l1, ll_l1 = lag1_constant_gain_smoother(params, ys)
    assert _rel(Ex_l1, Ex_ab) < TOL
    assert _rel(Vx_l1, Vx_ab) < TOL
    assert _rel(g_l1, g_ab) < TOL
    assert abs(ll_l1 - ll_ab) < 1e-8


# ---------------------------------------------------------------------------
# Contracts and cross-checks
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "smoother,params",
    [
        (constant_gain_smoother, ab_model(0.4, dB=0.0)),
        (lag1_constant_gain_smoother, ab_model(0.4, dB=0.15)),
        (reweight_smoother, cgo_memory_model()),
    ],
)
def test_smoother_output_contract(smoother, params):
    N, K, q = 10, params.K, params.q
    ys = _data(params, N, seed=5)
    E_x, Var_x, gamma, ll = smoother(params, ys)
    assert E_x.shape == (N, q)
    assert Var_x.shape == (N, q, q)
    assert gamma.shape == (N, K)
    assert np.allclose(gamma.sum(axis=1), 1.0)
    assert np.all(gamma >= -1e-12)
    assert np.isfinite(ll)
    for n in range(N):
        assert np.min(np.linalg.eigvalsh(0.5 * (Var_x[n] + Var_x[n].T))) > -1e-10


def test_smoother_single_observation():
    """N = 1 degenerate case: smoothing is filtering on the first slice."""
    params = ab_model(0.4, dB=0.0)
    ys = _data(params, 1, seed=7)
    for smoother in (constant_gain_smoother, lag1_constant_gain_smoother, reweight_smoother):
        E_x, Var_x, gamma, ll = smoother(params, ys)
        assert E_x.shape == (1, params.q)
        assert Var_x.shape == (1, params.q, params.q)
        assert np.allclose(gamma.sum(axis=1), 1.0)
        assert np.isfinite(ll)
    gamma, gamma_pair, ll = regime_smoother(params, ys, kernel="ab")
    assert gamma.shape == (1, params.K)
    assert gamma_pair.shape == (0, params.K, params.K)
    assert np.isfinite(ll)


def test_pair_marginals_consistent_with_marginals():
    params = ab_model(0.4, dB=0.0)
    ys = _data(params, N_STEPS, seed=100)
    gamma, gamma_pair, _ = regime_smoother(params, ys, kernel="ab")
    assert _rel(gamma_pair.sum(axis=2), gamma[:-1]) < 1e-12
    assert _rel(gamma_pair.sum(axis=1), gamma[1:]) < 1e-12


def test_log_lik_matches_gpb2_on_domains():
    """log p(y_{1:N}) from the observed-chain forward pass must agree with the
    (exact-on-domain) GPB2 likelihood: AB gauge and {C≡0} gauge."""
    p_ab = ab_model(0.4, dB=0.0)
    ys = _data(p_ab, N_STEPS, seed=100)
    _, _, _, ll = constant_gain_smoother(p_ab, ys)
    _, _, _, ll_gpb2 = gpb2_filter(p_ab, ys)
    assert abs(ll - ll_gpb2) < 1e-8

    p_c0 = cgo_memory_model()
    ys = _data(p_c0, N_STEPS, seed=100)
    _, _, _, ll = reweight_smoother(p_c0, ys)
    _, _, _, ll_gpb2 = gpb2_filter(p_c0, ys)
    assert abs(ll - ll_gpb2) < 1e-8


def test_regime_smoother_rejects_unknown_kernel():
    params = ab_model(0.4, dB=0.0)
    ys = _data(params, 4, seed=0)
    with pytest.raises(ValueError, match="kernel"):
        regime_smoother(params, ys, kernel="rts")
