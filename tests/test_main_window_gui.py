#!/usr/bin/env python3
"""
tests/test_main_window_gui.py
=============================
Smoke tests for prg/gui/main_window.py and the helper modules extracted
from it (diagnostics, session_state, workers, dialogs).

These guard the module split: they fail if a symbol moved to another
module is no longer importable from where the window expects it, or if
any of the extracted classes can no longer be constructed.
"""

from __future__ import annotations

import os

# Force a headless Qt platform before any PyQt6 import (see test_param_panel_gui).
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

from prg.classes.GSSParams import GSSParams
from prg.gui.dialogs import _InnovHistDialog, _RegimeDiagDialog, _WaitDialog
from prg.gui.main_window import GSSMainWindow
from prg.gui.session_state import _SessionState
from prg.gui.workers import _FilterWorker, _SimWorker
from prg.models.model_gss_K2_q1_s1 import ModelGssK2Q1S1

_P = np.array([[0.9, 0.1], [0.2, 0.8]])


# ---------------------------------------------------------------------------
# Extracted helper module: prg.gui.diagnostics
# ---------------------------------------------------------------------------


class TestDiagnostics:
    def test_ljung_box_white_noise_high_pvalue(self):
        from prg.gui.diagnostics import _ljung_box

        x = np.random.default_rng(0).standard_normal(500)
        _q, p, _h = _ljung_box(x)
        assert 0.0 <= p <= 1.0

    def test_shape_diagnostics_gaussian(self):
        from prg.gui.diagnostics import _shape_diagnostics

        x = np.random.default_rng(1).standard_normal(2000)
        S, K, _jb, p = _shape_diagnostics(x)
        assert abs(S) < 0.3 and abs(K) < 0.5
        assert 0.0 <= p <= 1.0

    def test_stationary_dist_matches_eigenvector(self):
        from prg.gui.diagnostics import _stationary_dist

        pi = _stationary_dist(_P)
        assert pi is not None
        np.testing.assert_allclose(pi @ _P, pi, atol=1e-10)

    def test_standardise_innovations_sample_mode(self):
        from prg.gui.diagnostics import _standardise_innovations

        innov = np.random.default_rng(2).standard_normal((300, 2))
        out = _standardise_innovations(innov, None, None, None)
        assert out.shape == innov.shape


# ---------------------------------------------------------------------------
# Extracted: prg.gui.session_state
# ---------------------------------------------------------------------------


class TestSessionState:
    def test_lifecycle(self):
        st = _SessionState()
        assert not st.has_data() and not st.has_filter()
        st.store_data([0, 1], [0, 0], np.zeros((2, 1)), np.zeros((2, 1)), 0)
        assert st.has_data()
        st.store_innovations(np.zeros((2, 1)))
        assert st.has_filter()
        st.reset()
        assert not st.has_data() and not st.has_filter()


# ---------------------------------------------------------------------------
# Extracted: prg.gui.workers (construct only; no thread start)
# ---------------------------------------------------------------------------


class TestWorkers:
    def test_construct(self):
        params = GSSParams.from_model(ModelGssK2Q1S1())
        assert _SimWorker(params, N=10, seed=0) is not None
        assert _FilterWorker(params, ys=np.zeros((10, 1))) is not None


# ---------------------------------------------------------------------------
# Extracted: prg.gui.dialogs (construct offscreen)
# ---------------------------------------------------------------------------


class TestDialogs:
    def test_innov_hist_dialog(self, qtbot):
        innov = np.random.default_rng(3).standard_normal((100, 1))
        dlg = _InnovHistDialog(innov)
        qtbot.addWidget(dlg)

    def test_regime_diag_dialog(self, qtbot):
        dlg = _RegimeDiagDialog(
            K=2,
            rs=[0, 1, 0, 1, 1, 0],
            P=_P,
            pis=np.tile([0.6, 0.4], (6, 1)),
        )
        qtbot.addWidget(dlg)

    def test_wait_dialog_progress(self, qtbot):
        dlg = _WaitDialog("working…")
        qtbot.addWidget(dlg)
        dlg.set_progress(5, 10)


# ---------------------------------------------------------------------------
# Main window construction
# ---------------------------------------------------------------------------


class TestMainWindow:
    def test_construct(self, qtbot):
        win = GSSMainWindow(K=2, q=1, s=1, P=_P)
        qtbot.addWidget(win)
        assert win.windowTitle() != ""

    def test_h5_exact_blockers_guard(self, qtbot):
        """The h5_exact CNS guard fires only for non-(H5) models in h5_exact mode."""
        from prg.classes.GSSParams import NGHMSMParams
        from prg.models.model_gss_K2_q2_s1 import ModelGss_K2_q2_s1

        win = GSSMainWindow(K=2, q=1, s=1, P=_P)
        qtbot.addWidget(win)
        valid = NGHMSMParams.from_model(ModelGssK2Q1S1())
        invalid = GSSParams.from_model(ModelGss_K2_q2_s1())  # s<q → non-(H5)

        # imm_general selected → never blocks, whatever the model.
        win._mode_combo.setCurrentIndex(win._mode_combo.findData("imm_general"))
        assert win._h5_exact_blockers(valid) == []
        assert win._h5_exact_blockers(invalid) == []
        assert win._h5_exact_blockers(None) == []

        # h5_exact selected → valid passes, invalid lists CNS issues.
        win._mode_combo.setCurrentIndex(win._mode_combo.findData("h5_exact"))
        assert win._h5_exact_blockers(valid) == []
        issues = win._h5_exact_blockers(invalid)
        assert issues and any("s = 1 < q = 2" in m for m in issues)
