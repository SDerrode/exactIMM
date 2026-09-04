#!/usr/bin/env python3
"""
tests/test_unsupervised.py
==========================
Guards for the Y-only exact EM on the ``{C≡0}`` family
(``prg/learning/unsupervised.py``), following the P5 validation protocol:

- T1: the E-step posteriors equal the exact Kᴺ path-enumeration posteriors;
- T2: the EM iterates are identical when the E-step is swapped for the exact
  enumeration one — this EM *is* the exact EM;
- T3: the observed-chain likelihood climbs monotonically;
- T4: the state block is a strict nuisance (Y-likelihood flat in it);
- recovery of contrasted chain parameters from a long record.
"""

from __future__ import annotations

import numpy as np

from prg.classes.GSSSimulator import GSSSimulator
from prg.experiments.cns_exactness import _params
from prg.experiments.reference_filters import imm_filter
from prg.experiments.reference_smoothers import exact_mixture_smoother
from prg.filter.gss_smoother import chain_log_likelihood
from prg.learning.unsupervised import (
    _estep,
    _mstep,
    c0_chain_to_gss,
    fit_em_c0,
)

TRUE_P = np.array([[0.9, 0.1], [0.1, 0.9]])
TRUE_D = [np.array([[0.3]]), np.array([[0.7]])]
TRUE_BY = [np.array([[0.1]]), np.array([[-0.2]])]
TRUE_SV = [np.array([[0.20]]), np.array([[0.60]])]
TRUE = (TRUE_P, TRUE_D, TRUE_BY, TRUE_SV)

# perturbed EM start (same shape, wrong values)
START = (
    np.array([[0.65, 0.35], [0.35, 0.65]]),
    [np.array([[0.5]]), np.array([[0.5]])],
    [np.array([[0.0]]), np.array([[0.0]])],
    [np.array([[0.35]]), np.array([[0.35]])],
)


def _data(params, N, seed):
    ys = []
    for _, _r, _x, y in GSSSimulator(params, N=N, seed=seed):
        ys.append(np.asarray(y, dtype=float).ravel())
    return np.array(ys)


def _exact_estep(theta, ys_):
    """E-step by Kᴺ path enumeration, via the reference smoother."""
    gss = c0_chain_to_gss(*theta)
    _, _, pi, pi_pair = exact_mixture_smoother(gss, ys_)
    return pi, pi_pair


def test_estep_posteriors_match_exact():
    gss = c0_chain_to_gss(*TRUE)
    for seed in (100, 101, 102):
        ys = _data(gss, 10, seed)
        gamma, xi, ll = _estep(ys.reshape(-1, 1), *TRUE)
        pi_ex, pair_ex = _exact_estep(TRUE, ys)
        assert np.max(np.abs(gamma - pi_ex)) < 1e-9
        assert np.max(np.abs(xi - pair_ex)) < 1e-9
        # and the E-step likelihood is the chain likelihood
        assert abs(ll - chain_log_likelihood(gss, ys, kernel="c0")) < 1e-8


def test_em_iterates_match_exact_estep():
    """T2 — swap the O(N K²) E-step for the exact Kᴺ one: identical EM."""
    gss = c0_chain_to_gss(*TRUE)
    ys = _data(gss, 10, seed=100)
    ys_ = ys.reshape(-1, 1)
    th_fast, th_exact = START, START
    for _ in range(4):
        g_f, x_f, _ = _estep(ys_, *th_fast)
        th_fast = _mstep(ys_, g_f, x_f, th_fast)
        g_e, x_e = _exact_estep(th_exact, ys)
        th_exact = _mstep(ys_, g_e, x_e, th_exact)
        gap = max(
            float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
            for pa, pb in zip(th_fast, th_exact)
            for a, b in zip(np.atleast_1d(pa), np.atleast_1d(pb))
        )
        assert gap < 1e-9


def test_loglik_monotone():
    gss = c0_chain_to_gss(*TRUE)
    ys = _data(gss, 300, seed=7)
    res = fit_em_c0(ys, K=2, theta0=START, max_iter=25, tol=0.0)
    assert len(res.log_liks) == 25
    assert np.min(np.diff(res.log_liks)) > -1e-8


def test_state_block_is_nuisance():
    """T4 — models sharing the chain (P, D, b^Y, Σ_V) but with different state
    blocks have identical Y-evidence (here via the exact-on-C≡0 IMM)."""
    models = [
        _params(
            A=list(A),
            B=[0.10, 0.10],
            C=[0.0, 0.0],
            D=[0.3, 0.7],
            SU=[0.40, 0.35],
            Dt=[0.15, -0.20],
            SV=[0.20, 0.60],
            p_switch=0.10,
        )
        for A in [(0.7, 0.4), (0.2, -0.3), (0.0, 0.0)]
    ]
    ys = _data(models[0], 30, seed=11)  # one record, scored under all three
    lls = [(imm_filter(p, ys)[3], chain_log_likelihood(p, ys, kernel="c0")) for p in models]
    full = [a for a, _ in lls]
    chain = [b for _, b in lls]
    assert np.ptp(full) < 1e-8  # flat in the state block
    assert max(abs(a - b) for a, b in lls) < 1e-8  # chain ll IS the evidence
    assert np.ptp(chain) < 1e-8


def test_recovery_from_long_record():
    gss = c0_chain_to_gss(*TRUE)
    ys = _data(gss, 1500, seed=42)
    res = fit_em_c0(ys, K=2, max_iter=100, tol=1e-4, seed=0)
    order = np.argsort([d.item() for d in res.D])  # label matching by D
    D_hat = np.array([res.D[k].item() for k in order])
    b_hat = np.array([res.bY[k].item() for k in order])
    SV_hat = np.array([res.SV[k].item() for k in order])
    P_hat = res.P[np.ix_(order, order)]
    assert np.max(np.abs(D_hat - [0.3, 0.7])) < 0.1
    assert np.max(np.abs(b_hat - [0.1, -0.2])) < 0.1
    assert np.max(np.abs(SV_hat - [0.20, 0.60])) < 0.15
    assert np.min(np.diag(P_hat)) > 0.85
    assert res.converged


def test_result_contract():
    gss = c0_chain_to_gss(*TRUE)
    ys = _data(gss, 120, seed=5)
    res = fit_em_c0(ys, K=2, theta0=START, max_iter=10, tol=1e-8)
    assert res.P.shape == (2, 2)
    assert np.allclose(res.P.sum(axis=1), 1.0)
    assert res.params.K == 2 and res.params.s == 1
    assert all(float(v) > 0 for v in np.ravel(res.SV))
    assert len(res.log_liks) == res.n_iter
    # the embedded GSSParams reproduces the fitted chain likelihood
    ll_embed = chain_log_likelihood(res.params, ys, kernel="c0")
    g, x, ll_direct = _estep(ys.reshape(-1, 1), res.P, res.D, res.bY, res.SV)
    assert abs(ll_embed - ll_direct) < 1e-8
