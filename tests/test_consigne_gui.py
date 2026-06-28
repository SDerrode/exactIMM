#!/usr/bin/env python3
"""
tests/test_consigne_gui.py
==========================
GUI tests for the exogenous-input ("consigne" uₙ) integration.

Coverage
--------
- ParamPanel exposes per-regime G_r editors when p>0; get_G_r_list() returns
  K arrays of shape (q+s, p), and None for the autonomous default (p=0).
- _SimWorker / _FilterWorker accept a u= keyword (backward-compatible default).
- GSSMainWindow._get_input_signal returns the right (N, p) shape for a
  generator mode and for the constant mode, and None when autonomous.
- PlotPanel reserves p extra u^j axes when p>0 and stays unchanged at p=0.
- MatrixTableWidget supports a rectangular (q+s, p) layout via cols=.
"""

from __future__ import annotations

import os

# Force a headless Qt platform before any PyQt6 import (see test_param_panel_gui).
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

from prg.gui.main_window import GSSMainWindow
from prg.gui.matrix_widget import MatrixTableWidget
from prg.gui.param_panel import ParamPanel
from prg.gui.plot_panel import PlotPanel
from prg.gui.session_state import _SessionState
from prg.gui.workers import _FilterWorker, _SimWorker

_P = np.array([[0.9, 0.1], [0.2, 0.8]])


# ---------------------------------------------------------------------------
# MatrixTableWidget — rectangular (q+s, p) support
# ---------------------------------------------------------------------------


class TestRectangularMatrixWidget:
    def test_square_is_unchanged(self, qtbot):
        w = MatrixTableWidget(2, 2, is_covariance=False, title="F")
        qtbot.addWidget(w)
        assert w.get_matrix().shape == (4, 4)

    def test_rectangular_rows_cols(self, qtbot):
        # q+s = 4 rows, p = 3 cols
        g = MatrixTableWidget(2, 2, cols=3, is_covariance=False, title="G_r")
        qtbot.addWidget(g)
        mat = g.get_matrix()
        assert mat.shape == (4, 3)
        assert np.allclose(mat, 0.0)  # default fill
        target = np.arange(12).reshape(4, 3).astype(float)
        g.set_matrix(target)
        assert np.allclose(g.get_matrix(), target)
        assert g.is_valid()


# ---------------------------------------------------------------------------
# ParamPanel — G_r editors
# ---------------------------------------------------------------------------


class TestParamPanelGr:
    def test_autonomous_default(self, qtbot):
        panel = ParamPanel(K=2, q=2, s=1)
        qtbot.addWidget(panel)
        # p=0 ⇒ no G_r, get_G_r_list() is None (autonomous, as before)
        assert panel.get_G_r_list() is None
        for tab in panel._state_tabs:
            assert tab._g_widget is None

    def test_set_input_dim_creates_g_editors(self, qtbot):
        panel = ParamPanel(K=2, q=2, s=1)
        qtbot.addWidget(panel)
        panel.set_input_dim(2)
        glist = panel.get_G_r_list()
        assert glist is not None
        assert len(glist) == 2
        for g in glist:
            assert g.shape == (3, 2)  # (q+s, p) = (3, 2)
        for tab in panel._state_tabs:
            assert tab._g_widget is not None

    def test_construct_with_p(self, qtbot):
        panel = ParamPanel(K=3, q=1, s=1, p=2)
        qtbot.addWidget(panel)
        glist = panel.get_G_r_list()
        assert glist is not None and len(glist) == 3
        assert all(g.shape == (2, 2) for g in glist)

    def test_set_and_get_g_r(self, qtbot):
        panel = ParamPanel(K=2, q=1, s=1, p=2)
        qtbot.addWidget(panel)
        target = np.array([[1.0, 2.0], [3.0, 4.0]])
        panel.set_state_params(0, np.eye(2) * 0.5, np.eye(2) * 0.1, G_r=target)
        assert np.allclose(panel.get_G_r_list()[0], target)

    def test_set_input_dim_back_to_zero(self, qtbot):
        panel = ParamPanel(K=2, q=1, s=1, p=2)
        qtbot.addWidget(panel)
        assert panel.get_G_r_list() is not None
        panel.set_input_dim(0)
        assert panel.get_G_r_list() is None
        assert panel.is_valid()
        for tab in panel._state_tabs:
            assert tab._g_widget is None


# ---------------------------------------------------------------------------
# Workers — accept u=
# ---------------------------------------------------------------------------


class TestWorkersAcceptU:
    def test_sim_worker_u_kwarg(self):
        # Construction with u= must not raise and must store it.
        u = np.ones((5, 2))
        w = _SimWorker(params=None, N=5, seed=0, u=u)
        assert w._u is u

    def test_sim_worker_u_default_none(self):
        w = _SimWorker(params=None, N=5, seed=0)
        assert w._u is None

    def test_filter_worker_u_kwarg(self):
        u = np.ones((5, 2))
        w = _FilterWorker(params=None, ys=np.zeros((5, 1)), u=u)
        assert w._u is u

    def test_filter_worker_u_default_none(self):
        w = _FilterWorker(params=None, ys=np.zeros((5, 1)))
        assert w._u is None

    def test_sim_worker_runs_with_u(self, qtbot):
        # End-to-end: build params via the GUI, run the worker synchronously.
        win = GSSMainWindow(K=2, q=1, s=1, P=_P)
        qtbot.addWidget(win)
        win._input_dim_spin.setValue(1)
        # Seed each regime's G_r so the model is well-defined.
        for k in range(2):
            win._param_panel.set_state_params(
                k,
                np.eye(2) * 0.5,
                np.eye(2) * 0.1,
                G_r=np.array([[0.3], [0.2]]),
            )
        params = win._build_gss_params()
        assert params.p == 1
        N = 30
        from prg.utils.input_signal import make_input

        u = make_input("step(10)", N, params.p)
        worker = _SimWorker(params, N, seed=0, u=u)
        # Run synchronously (QThread.run is a plain method).
        worker.run()


# ---------------------------------------------------------------------------
# Session state — stores u
# ---------------------------------------------------------------------------


class TestSessionStateU:
    def test_store_data_keeps_u(self):
        st = _SessionState()
        u = np.ones((3, 2))
        st.store_data([0, 1, 2], [0, 0, 0], None, np.zeros((3, 1)), 0, u=u)
        assert st.u is u

    def test_reset_clears_u(self):
        st = _SessionState()
        st.store_data([0], [0], None, np.zeros((1, 1)), 0, u=np.ones((1, 1)))
        st.reset()
        assert st.u is None


# ---------------------------------------------------------------------------
# GSSMainWindow._get_input_signal
# ---------------------------------------------------------------------------


class TestGetInputSignal:
    def test_autonomous_returns_none(self, qtbot):
        win = GSSMainWindow(K=2, q=1, s=1, P=_P)
        qtbot.addWidget(win)
        assert win._input_dim_spin.value() == 0
        assert win._get_input_signal(100) is None

    def test_generator_shape(self, qtbot):
        win = GSSMainWindow(K=2, q=1, s=1, P=_P)
        qtbot.addWidget(win)
        win._input_dim_spin.setValue(2)
        idx = win._input_mode_combo.findData("gen:step")
        win._input_mode_combo.setCurrentIndex(idx)
        win._input_param_edit.setText("20")
        u = win._get_input_signal(50)
        assert u is not None
        assert u.shape == (50, 2)
        # step(20): zero before index 20, one after
        assert np.allclose(u[:20], 0.0)
        assert np.allclose(u[20:], 1.0)

    def test_constant_mode_shape_and_values(self, qtbot):
        win = GSSMainWindow(K=2, q=1, s=1, P=_P)
        qtbot.addWidget(win)
        win._input_dim_spin.setValue(2)
        idx = win._input_mode_combo.findData("const")
        win._input_mode_combo.setCurrentIndex(idx)
        win._input_const_spins[0].setValue(1.5)
        win._input_const_spins[1].setValue(-2.0)
        u = win._get_input_signal(10)
        assert u is not None
        assert u.shape == (10, 2)
        assert np.allclose(u[:, 0], 1.5)
        assert np.allclose(u[:, 1], -2.0)

    def test_none_mode_when_pgt0(self, qtbot):
        win = GSSMainWindow(K=2, q=1, s=1, P=_P)
        qtbot.addWidget(win)
        win._input_dim_spin.setValue(2)
        idx = win._input_mode_combo.findData("none")
        win._input_mode_combo.setCurrentIndex(idx)
        assert win._get_input_signal(10) is None


# ---------------------------------------------------------------------------
# PlotPanel — u^j axes
# ---------------------------------------------------------------------------


class TestPlotPanelInputAxes:
    def test_p0_axis_count_unchanged(self, qtbot):
        panel = PlotPanel(q=1, s=1)  # p defaults to 0
        qtbot.addWidget(panel)
        # 1 (R) + 1 (pi) + 1 (X) + 1 (Y) + 1 (innov) = 5
        assert panel._n_axes == 5
        assert panel._p == 0

    def test_pgt0_adds_u_axes(self, qtbot):
        panel = PlotPanel(q=1, s=1, p=2)
        qtbot.addWidget(panel)
        # + 2 u rows = 7
        assert panel._n_axes == 7
        assert panel._p == 2
        # u axes sit between Y and innovations
        assert panel._u_offset == 2 + 1 + 1
        assert panel._innov_offset == panel._u_offset + 2

    def test_set_input_dim_rebuilds(self, qtbot):
        panel = PlotPanel(q=1, s=1)
        qtbot.addWidget(panel)
        assert panel._n_axes == 5
        panel.set_input_dim(3)
        assert panel._n_axes == 8
        assert panel._p == 3
        panel.set_input_dim(0)
        assert panel._n_axes == 5

    def test_update_plots_with_u(self, qtbot):
        panel = PlotPanel(q=1, s=1, p=2)
        qtbot.addWidget(panel)
        N = 8
        ns = list(range(N))
        rs = [0, 1] * (N // 2)
        xs = np.zeros((N, 1))
        ys = np.ones((N, 1))
        u = np.column_stack([np.arange(N), np.ones(N)])
        # Must not raise and must draw a curve on each u axis.
        panel.update_plots(ns, rs, xs, ys, K=2, u=u)
        for j in range(2):
            ax = panel._axes[panel._u_offset + j]
            assert len(ax.lines) >= 1

    def test_update_plots_backward_compatible(self, qtbot):
        # p=0 panel, no u argument → identical to legacy behaviour.
        panel = PlotPanel(q=1, s=1)
        qtbot.addWidget(panel)
        panel.update_plots([0, 1, 2], [0, 1, 0], np.zeros((3, 1)), np.ones((3, 1)), K=2)
