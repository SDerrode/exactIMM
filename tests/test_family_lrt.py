#!/usr/bin/env python3
"""
tests/test_family_lrt.py
========================
Guards for the exact observed-chain likelihood and the between-family
Neyman-Pearson LRT (applications of the smoothing companion paper):

- ``chain_log_likelihood`` is the *true* model evidence on each family
  (agreement with the exact-on-domain GPB2 likelihood, and with the
  forward-backward of ``regime_smoother``);
- the LRT built from two exact evidences separates the families with high
  power at realistic sizes (AUC sanity, cf. the P6 study: 0.61 → 0.99).
"""

from __future__ import annotations

import numpy as np

from prg.classes.GSSSimulator import GSSSimulator
from prg.experiments.cns_exactness import ab_model, cgo_memory_model, slaving_A_model
from prg.experiments.reference_filters import gpb2_filter
from prg.filter.gss_smoother import chain_log_likelihood, family_lrt, regime_smoother


def _data(params, N, seed):
    ys = []
    for _, _r, _x, y in GSSSimulator(params, N=N, seed=seed):
        ys.append(np.asarray(y, dtype=float).ravel())
    return np.array(ys)


def test_chain_log_likelihood_matches_regime_smoother():
    cases = [
        (ab_model(0.4, dB=0.0), "ab"),
        (cgo_memory_model(), "c0"),
        (ab_model(0.4, dB=0.15), "lag1"),
    ]
    for params, kernel in cases:
        ys = _data(params, 40, seed=2)
        ll_fwd = chain_log_likelihood(params, ys, kernel)
        _, _, ll_fb = regime_smoother(params, ys, kernel)
        assert abs(ll_fwd - ll_fb) < 1e-10


def test_chain_log_likelihood_is_the_evidence_on_domains():
    """On each family the observed-chain likelihood equals the true model
    evidence, here computed by the exact-on-domain GPB2 filter."""
    for params, kernel in [
        (ab_model(0.4, dB=0.0), "ab"),
        (ab_model(0.4, dB=0.15), "lag1"),
        (cgo_memory_model(), "c0"),
    ]:
        ys = _data(params, 30, seed=4)
        ll = chain_log_likelihood(params, ys, kernel)
        _, _, _, ll_gpb2 = gpb2_filter(params, ys)
        assert abs(ll - ll_gpb2) < 1e-8


def test_family_lrt_separates_the_families():
    """AUC of the LRT statistic between H0 = {C≡0} (slaving_A_model(0)) and
    H1 = slaving-A with coupling 0.4, N = 50 — the strong-coupling cell of the
    P6 power study."""
    h0, h1 = slaving_A_model(0.0), slaving_A_model(0.4)
    lrt0, lrt1 = [], []
    for sd in range(40):
        ys0 = _data(h0, 50, seed=1000 + sd)
        ys1 = _data(h1, 50, seed=1000 + sd)
        lrt0.append(family_lrt(h0, h1, ys0))
        lrt1.append(family_lrt(h0, h1, ys1))
    s0, s1 = np.asarray(lrt0), np.asarray(lrt1)
    auc = (s1[:, None] > s0[None, :]).mean() + 0.5 * (s1[:, None] == s0[None, :]).mean()
    assert auc > 0.8


def test_family_lrt_centered_when_hypotheses_coincide():
    """With H1 = H0 the statistic is identically zero."""
    h0 = slaving_A_model(0.0)
    ys = _data(h0, 50, seed=3)
    assert abs(family_lrt(h0, h0, ys, kernel_h0="c0", kernel_h1="c0")) < 1e-12
