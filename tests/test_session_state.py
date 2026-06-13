#!/usr/bin/env python3
"""
tests/test_session_state.py
===========================
Unit tests for prg/gui/session_state.py::_SessionState.

These are pure-Python (no Qt binding required), so they run in CI:
``_SessionState`` only depends on numpy and dataclasses.
"""

from __future__ import annotations

import numpy as np

from prg.gui.session_state import _SessionState


def _filled() -> _SessionState:
    """A session with both data and stale filter results populated."""
    st = _SessionState()
    st.store_data(np.arange(3), np.zeros(3, int), np.zeros((3, 1)), np.zeros((3, 1)), 0)
    st.store_filter_results(
        E_xs=np.ones((3, 1)),
        Var_xs=np.ones((3, 1)),
        pis=np.ones((3, 2)),
        log_lik_total=-1.0,
    )
    st.store_innovations(np.ones((3, 1)))
    return st


def _assert_no_filter(st: _SessionState) -> None:
    assert st.innovations is None
    assert st.filter_E_xs is None
    assert st.filter_Var_xs is None
    assert st.filter_pis is None
    assert st.filter_log_lik is None
    assert not st.has_filter()


def test_begin_simulation_clears_stale_filter():
    """Re-Simulate must drop previous filter results (else the regime-diagnostics
    confusion matrix would pair new ground truth with old posteriors)."""
    st = _filled()
    st.begin_simulation(params=object(), signature=None)
    _assert_no_filter(st)


def test_load_external_clears_stale_filter():
    """Loading a new CSV must likewise drop previous filter results."""
    st = _filled()
    st.load_external(
        np.arange(4),
        np.zeros(4, int),
        np.zeros((4, 1)),
        np.zeros((4, 1)),
        params=object(),
        signature=None,
    )
    _assert_no_filter(st)


def test_reset_clears_everything():
    st = _filled()
    st.reset()
    _assert_no_filter(st)
    assert st.data is None
    assert st.params is None
    assert not st.has_data()
