#!/usr/bin/env python3
"""
prg/gui/main_window.py
======================
GSSMainWindow — top-level application window.

Layout (C8: updated to reflect current UI)
-------------------------------------------
  ┌──────────────────────────┬──────────────────────────────────────────────┐
  │  Left panel              │  Right tabs                                  │
  │  ─────────────────────── │  ┌──────────────┬──────────────────────────┐ │
  │  [Preset combo]          │  │  Main plots  │  Predicted Y             │ │
  │  ParamPanel              │  │  (PlotPanel) │  (PredYPanel)            │ │
  │    tabs: F(k) / Σ_W(k)  │  │              │                          │ │
  │    per-state tabs        │  │  R_n step    │  Trajectory + Density    │ │
  │  ─────────────────────── │  │  π_n(k)      │  tabs                    │ │
  │  [P matrix widget]       │  │  X^i         │                          │ │
  │  ─────────────────────── │  │  Y^i         │                          │ │
  │  N  [spinbox]            │  │  ν^i innov.  │                          │ │
  │  Seed  [lineedit]        │  └──────────────┴──────────────────────────┘ │
  │  Filter mode [combo]     │                                              │
  │  [Simulate] [Filter]     │                                              │
  │  [Save CSV] [Load CSV]   │                                              │
  │  [Export model]          │                                              │
  │  [Export plots]          │                                              │
  │  [Innovation hist] [Reset]│                                             │
  │  ─────────────────────── │                                              │
  │  Filter quality frame    │                                              │
  │    log L, MSE, RMSE      │                                              │
  │  Innovation diagnostics  │                                              │
  │    Ljung-Box / shape badges│                                            │
  └──────────────────────────┴──────────────────────────────────────────────┘

Key internal classes
---------------------
  _SessionState     — single source of truth for simulation + filter data
  _SimWorker        — QThread: runs GSSSimulator
  _FilterWorker     — QThread: runs GSSFilter (emits progress signals)
  _WaitDialog       — modal progress dialog with Cancel button
  _InnovHistDialog  — modeless histogram/ACF/scatter dialog for innovations
"""

import csv
import datetime
import pathlib
import time

import numpy as np
from PyQt6.QtCore import QSettings, Qt, pyqtSignal
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

# ---------------------------------------------------------------------------
# C3 — Named constants for plot colours and diagnostic thresholds
# ---------------------------------------------------------------------------

# Matplotlib series colours (tab10 palette, first 5)
_COL_X = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]  # hidden components
_COL_Y = ["#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]  # observed components
_COL_R = ["#555555", "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]  # regime colours

# Diagnostic FWER target (used in Bonferroni correction for innovation tests)
_INNOV_ALPHA_FAM: float = 0.05  # family-wise error rate over 2·s tests

# Status "pill" stylesheets for the diagnostic badges (OK / warning / error).
_PILL_OK = (
    "font-size: 10px; padding: 2px 8px; border-radius: 3px;"
    "background: #d4edda; color: #155724; border: 1px solid #c3e6cb;"
)
_PILL_WARN = (
    "font-size: 10px; padding: 2px 8px; border-radius: 3px;"
    "background: #fff3cd; color: #856404; border: 1px solid #ffc107;"
)
_PILL_ERR = (
    "font-size: 10px; padding: 2px 8px; border-radius: 3px;"
    "background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb;"
)

from prg.classes.FMatrix import FMatrix
from prg.classes.GSSParams import GSSParams
from prg.classes.NoiseCovariance import GSSNoiseCovariance
from prg.gui.diagnostics import (
    _ljung_box,
    _shape_diagnostics,
    _standardise_innovations,
    _stationary_dist,
)
from prg.gui.dialogs import _InnovHistDialog, _RegimeDiagDialog, _WaitDialog
from prg.gui.matrix_widget import StochasticMatrixWidget
from prg.gui.param_panel import ParamPanel
from prg.gui.plot_panel import PlotPanel, PredYPanel
from prg.gui.session_state import _SessionState
from prg.gui.workers import _FilterWorker, _SimWorker
from prg.models.presets import PRESETS

# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------


class GSSMainWindow(QMainWindow):
    """Interactive GSS simulator window."""

    # Emitted when the user picks a preset with different K/q/s.
    # The slot in main.py closes this window and reopens with the new model.
    restart_with_model = pyqtSignal(object)

    def __init__(
        self,
        K: int,
        q: int,
        s: int,
        P: np.ndarray | None = None,
        model=None,
        parent=None,
    ):
        super().__init__(parent)
        self._K = K
        self._q = q
        self._s = s
        # Transition matrix (fixed, not editable)
        self._P = P if P is not None else np.full((K, K), 1.0 / K)

        # Single source of truth for everything Simulate / Filter / MC / Load
        # produces. See _SessionState above for the invariants.
        self._state = _SessionState()

        self._worker: _SimWorker | None = None
        self._filter_worker: _FilterWorker | None = None
        self._wait_dlg: _WaitDialog | None = None
        self._op_t0: float = time.perf_counter()  # C4: initialised in constructor

        self.setWindowTitle(f"exactIMM — GSS Simulator  (K={K}, q={q}, s={s})")

        # ── central widget ────────────────────────────────────────────
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self._splitter = splitter
        main_layout.addWidget(splitter)

        # ── left panel ───────────────────────────────────────────────
        left = QWidget()
        left.setMinimumWidth(540)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)

        # ── Preset selector ───────────────────────────────────────────
        self._presets = PRESETS
        preset_row = QHBoxLayout()
        preset_row.setSpacing(6)
        preset_row.addWidget(QLabel("Preset :"))
        self._preset_combo = QComboBox()
        self._preset_combo.addItem("— select a model —")
        for entry in PRESETS:
            self._preset_combo.addItem(entry.label)
            self._preset_combo.setItemData(
                self._preset_combo.count() - 1,
                entry.tooltip,
                Qt.ItemDataRole.ToolTipRole,
            )
        self._preset_combo.setToolTip("Load a predefined model")
        self._preset_combo.activated.connect(self._on_preset_selected)
        preset_row.addWidget(self._preset_combo, stretch=1)
        left_layout.addLayout(preset_row)

        self._param_panel = ParamPanel(K, q, s)
        self._param_panel.validity_changed.connect(self._on_validity_changed)
        self._param_panel.value_changed.connect(self._refresh_filter_button_drift_indicator)
        self._param_panel.constraint_toggled.connect(self._on_reset)
        # Stretch=1 so the State-k tabs grow with the window: with the natural
        # height alone, larger (q, s) configurations (e.g. M2 with q=s=2)
        # triggered an internal scrollbar that made the tab content hard to use.
        left_layout.addWidget(self._param_panel, stretch=1)

        # Transition matrix P
        p_section = QHBoxLayout()
        p_label = QLabel("Transition matrix:")
        p_label.setStyleSheet("font-size: 11px;")
        p_section.addWidget(p_label)
        self._p_widget = StochasticMatrixWidget(K, title=f"P  ({K}×{K})")
        self._p_widget.set_matrix(self._P)
        self._p_widget.validity_changed.connect(self._on_validity_changed)
        self._p_widget.value_changed.connect(self._update_stationary_display)
        self._p_widget.value_changed.connect(self._refresh_filter_button_drift_indicator)
        p_section.addWidget(self._p_widget)

        self._stationary_label = QLabel("")
        self._stationary_label.setStyleSheet("font-size: 10px; color: #444444;")
        p_section.addWidget(self._stationary_label)

        p_section.addStretch()
        left_layout.addLayout(p_section)
        self._update_stationary_display()  # initialise avec la valeur par défaut

        # N field
        n_row = QHBoxLayout()
        n_row.addWidget(QLabel("N (steps) :"))
        self._n_spin = QSpinBox()
        self._n_spin.setRange(1, 1_000_000)
        self._n_spin.setValue(1000)
        self._n_spin.setSingleStep(100)
        self._n_spin.valueChanged.connect(self._on_sim_params_changed)
        n_row.addWidget(self._n_spin)
        left_layout.addLayout(n_row)

        # Seed field
        seed_row = QHBoxLayout()
        seed_row.addWidget(QLabel("Seed:"))
        self._seed_edit = QLineEdit()
        self._seed_edit.setPlaceholderText("empty = random")
        self._seed_edit.setMaximumWidth(120)
        self._seed_edit.textChanged.connect(self._on_seed_text_changed)  # A2: validate
        self._seed_edit.editingFinished.connect(
            self._on_sim_params_changed
        )  # A1: not per-keystroke
        seed_row.addWidget(self._seed_edit)
        left_layout.addLayout(seed_row)

        # Auto-filter row
        auto_row = QHBoxLayout()
        self._auto_filter_check = QCheckBox("Auto-filter after simulate")
        self._auto_filter_check.setToolTip(
            "When enabled, the filter runs automatically after every single simulation.\n"
            "The filter uses the parameters captured at simulate time — editing the\n"
            "widgets afterwards does not affect the auto-filter run.\n"
            "The Filter mode (IMM / H5) and Joseph form settings below apply as usual.\n"
            "No effect in Monte-Carlo mode."
        )
        auto_row.addWidget(self._auto_filter_check)
        auto_row.addStretch()
        left_layout.addLayout(auto_row)

        # Filter-mode row
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Filter mode:"))
        self._mode_combo = QComboBox()
        self._mode_combo.addItem("Approximate IMM - H5 not required", "imm_general")
        self._mode_combo.addItem("exactIMM - Exact IMM - H5 required", "h5_exact")
        self._mode_combo.setToolTip(
            "Approximate IMM — per-step moment propagation from the filtered π_n.\n"
            "               Works for any GSS model, with or without (H5)\n"
            "               (matches exactIMM ≤ v0.9.0).\n"
            "Exact IMM    — stationary pre-computed moments. Exact when (H5)\n"
            "               holds: the algebraic constraint\n"
            "               Δᵀ Aᵀ + Σ_V Bᵀ = P M⁻¹ W linking the 7 blocks.\n"
            "               Emits a warning when the residual exceeds the\n"
            "               tolerance. Tick the per-regime 'AB constraint'\n"
            "               to enforce it (sets A = Δ Σ_V⁻¹ C, B = Δ Σ_V⁻¹ D)."
        )
        mode_row.addWidget(self._mode_combo)
        mode_row.addStretch()
        left_layout.addLayout(mode_row)

        # Joseph form row (only meaningful for h5_exact mode)
        joseph_row = QHBoxLayout()
        self._joseph_check = QCheckBox("Joseph form (covariance update)")
        self._joseph_check.setToolTip(
            "Only used in 'Exact IMM under (H5)' mode. When checked, the\n"
            "mode-conditional posterior covariance is computed via the Joseph\n"
            "form (numerically stable, symmetric and PSD by construction).\n"
            "Mathematically equivalent to the short form under stationarity.\n"
            "See paper Appendix E."
        )
        joseph_row.addWidget(self._joseph_check)
        joseph_row.addStretch()
        left_layout.addLayout(joseph_row)

        # Gray out the Joseph checkbox unless mode is h5_exact
        def _sync_joseph_enabled():
            self._joseph_check.setEnabled(self._mode_combo.currentData() == "h5_exact")

        self._mode_combo.currentIndexChanged.connect(lambda _: _sync_joseph_enabled())
        _sync_joseph_enabled()

        # No final stretch: the param-panel above already absorbs the extra
        # vertical space (stretch=1), and we want the buttons to sit just below
        # the control widgets rather than be pushed to the bottom of the window.

        # Buttons — grille 2×2
        #   [Simulate]        [Filter]
        #   [Save CSV]        [Export model…]
        #   [Reset ——————————————————————————]
        btn_grid = QGridLayout()
        btn_grid.setSpacing(6)

        self._btn_simulate = QPushButton("Simulate")
        self._btn_simulate.setFixedHeight(36)
        self._btn_simulate.setToolTip("Run a new simulation  (Ctrl+R)")  # B2
        self._btn_simulate.clicked.connect(self._on_simulate)

        self._btn_filter = QPushButton("Filter")
        self._btn_filter.setFixedHeight(36)
        self._btn_filter.setEnabled(False)
        self._btn_filter.setToolTip("Run the optimal filter on the simulation  (Ctrl+F)")  # B3
        self._btn_filter.clicked.connect(self._on_filter)

        self._btn_save = QPushButton("Save CSV")
        self._btn_save.setFixedHeight(36)
        self._btn_save.setEnabled(False)
        self._btn_save.setToolTip("Save simulation data to a CSV file  (Ctrl+S)")  # B4
        self._btn_save.clicked.connect(self._on_save)

        self._btn_load = QPushButton("Load CSV…")
        self._btn_load.setFixedHeight(36)
        self._btn_load.setToolTip("Load a previously saved simulation CSV  (Ctrl+O)")  # B5
        self._btn_load.clicked.connect(self._on_load_data)

        self._btn_export = QPushButton("Export model…")
        self._btn_export.setFixedHeight(36)
        self._btn_export.setEnabled(False)
        self._btn_export.setToolTip(
            "Export current parameters as a Python model file  (Ctrl+E)"
        )  # B6
        self._btn_export.clicked.connect(self._on_export_model)

        self._btn_export_plots = QPushButton("Export plots…")
        self._btn_export_plots.setFixedHeight(36)
        self._btn_export_plots.setEnabled(False)
        self._btn_export_plots.setToolTip(
            "Save current plots to PDF / PNG / SVG  (Ctrl+Shift+E)"
        )  # B6
        self._btn_export_plots.clicked.connect(self._on_export_plots)

        self._btn_innov_hist = QPushButton("📊 Innov. histograms…")
        self._btn_innov_hist.setFixedHeight(36)
        self._btn_innov_hist.setEnabled(False)
        self._btn_innov_hist.setToolTip(
            "Show innovation histograms, ACF, and scatter plots.  (Ctrl+I)"  # A8
        )
        self._btn_innov_hist.clicked.connect(self._on_innov_hist)

        # B7/B8: regime diagnostics (confusion matrix + duration histogram)
        self._btn_regime_diag = QPushButton("📊 Regime diagnostics…")
        self._btn_regime_diag.setFixedHeight(36)
        self._btn_regime_diag.setEnabled(False)
        self._btn_regime_diag.setToolTip(
            "Confusion matrix (argmax π_n vs r_n) and regime-duration histograms  (Ctrl+D)"
        )
        self._btn_regime_diag.clicked.connect(self._on_regime_diag)

        # D5/B10: session persistence (params + data + filter results)
        self._btn_save_session = QPushButton("💾 Save session…")
        self._btn_save_session.setFixedHeight(32)
        self._btn_save_session.setToolTip(
            "Save the complete session (parameters + data + filter results) "
            "to a .exactIMM file  (Ctrl+Shift+S)"
        )
        self._btn_save_session.clicked.connect(self._on_save_session)

        self._btn_load_session = QPushButton("📂 Load session…")
        self._btn_load_session.setFixedHeight(32)
        self._btn_load_session.setToolTip(
            "Restore a previously saved .exactIMM session  (Ctrl+Shift+O)"
        )
        self._btn_load_session.clicked.connect(self._on_load_session)

        self._btn_reset = QPushButton("⟳  Reset")
        self._btn_reset.setFixedHeight(36)
        self._btn_reset.setToolTip("Clear all results and reset the plots  (Ctrl+Shift+R)")  # B7
        self._btn_reset.clicked.connect(self._on_reset)

        btn_grid.addWidget(self._btn_simulate, 0, 0)
        btn_grid.addWidget(self._btn_filter, 0, 1)
        btn_grid.addWidget(self._btn_save, 1, 0)
        btn_grid.addWidget(self._btn_load, 1, 1)
        btn_grid.addWidget(self._btn_export, 2, 0)
        btn_grid.addWidget(self._btn_export_plots, 2, 1)
        btn_grid.addWidget(self._btn_innov_hist, 3, 0)
        btn_grid.addWidget(self._btn_regime_diag, 3, 1)
        btn_grid.addWidget(self._btn_save_session, 4, 0)
        btn_grid.addWidget(self._btn_load_session, 4, 1)
        btn_grid.addWidget(self._btn_reset, 5, 0, 1, 2)

        left_layout.addLayout(btn_grid)

        # B1: keyboard shortcut hint — small one-liner for discoverability
        _sc_hint = QLabel(
            "Ctrl+R Simulate  ·  Ctrl+F Filter  ·  Ctrl+I Innov.  ·  Ctrl+D Regimes  ·  Ctrl+Shift+R Reset"
        )
        _sc_hint.setStyleSheet("font-size: 9px; color: #888888; padding: 1px 0;")
        _sc_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_layout.addWidget(_sc_hint)

        # Thin separator between buttons and result panels (B11)
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        sep.setStyleSheet("color: #cccccc;")
        left_layout.addWidget(sep)

        # ── MSE display box ──────────────────────────────────────────
        self._mse_frame = QFrame()
        self._mse_frame.setFrameShape(QFrame.Shape.StyledPanel)
        self._mse_frame.setVisible(False)
        mse_layout = QVBoxLayout(self._mse_frame)
        mse_layout.setContentsMargins(8, 6, 8, 6)
        mse_layout.setSpacing(3)

        self._mse_title = QLabel("Filter quality")
        self._mse_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        mse_layout.addWidget(self._mse_title)

        self._loglik_label = QLabel("")
        self._loglik_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._loglik_label.setToolTip(
            "Total log-likelihood  log L = Σₙ log p(yₙ | y₁:ₙ₋₁)  (nats).\n"
            "Mean per step:  ℓ̄ = log L / N.\n\n"
            "Model-selection criteria (smaller = better fit adjusted for complexity):\n"
            "  BIC = −2 log L + d · log N\n"
            "  AIC = −2 log L + 2d\n"
            "where d = number of free parameters.\n\n"
            "Under the true model, ℓ̄ → −h  (entropy rate) as N → ∞."
        )
        mse_layout.addWidget(self._loglik_label)

        self._mse_global_label = QLabel("")
        self._mse_global_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        mse_layout.addWidget(self._mse_global_label)

        self._rmse_label = QLabel("")
        self._rmse_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        mse_layout.addWidget(self._rmse_label)

        # Per-component labels (created dynamically in _on_filter_finished)
        self._mse_comp_labels: list[QLabel] = []

        # ── Innovation diagnostics (Ljung-Box + Jarque-Bera) ─────────
        self._innov_frame = QFrame()
        self._innov_frame.setFrameShape(QFrame.Shape.StyledPanel)
        self._innov_frame.setVisible(False)
        innov_layout = QVBoxLayout(self._innov_frame)
        innov_layout.setContentsMargins(8, 6, 8, 6)
        innov_layout.setSpacing(4)

        innov_title = QLabel("Innovation diagnostics")
        innov_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        innov_title.setStyleSheet("font-size: 10px; font-weight: bold;")
        innov_title.setToolTip(
            "Ljung-Box: whiteness test on raw innovations\n"
            "           (no autocorrelation up to lag h).\n"
            "Shape:    skewness S and excess kurtosis K of STANDARDISED\n"
            "           innovations ν̃ = S^{-1/2} ν.\n"
            "           (h5_exact: whitened by stationary Γ mixture;\n"
            "            imm_general: whitened by sample covariance)\n"
            "           For a well-tuned filter: |S|≈0, |K|≈0."
        )
        innov_layout.addWidget(innov_title)

        innov_grid = QGridLayout()
        innov_grid.setSpacing(4)
        header_style = "font-size: 9px; color: #555555; font-style: italic;"
        h_lb = QLabel("Ljung-Box")
        h_jb = QLabel("Skew · Kurt")
        h_lb.setStyleSheet(header_style)
        h_lb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        h_jb.setStyleSheet(header_style)
        h_jb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        innov_grid.addWidget(h_lb, 0, 1)
        innov_grid.addWidget(h_jb, 0, 2)

        self._innov_lb_badges: list[QLabel] = []
        self._innov_jb_badges: list[QLabel] = []
        for i in range(s):
            name = QLabel(f"ν^{i}")
            name.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            name.setStyleSheet("font-size: 10px;")
            lb = QLabel()
            lb.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lb.setFixedHeight(20)
            jb = QLabel()
            jb.setAlignment(Qt.AlignmentFlag.AlignCenter)
            jb.setFixedHeight(20)
            innov_grid.addWidget(name, i + 1, 0)
            innov_grid.addWidget(lb, i + 1, 1)
            innov_grid.addWidget(jb, i + 1, 2)
            self._innov_lb_badges.append(lb)
            self._innov_jb_badges.append(jb)
        innov_grid.setColumnStretch(1, 1)
        innov_grid.setColumnStretch(2, 1)
        innov_layout.addLayout(innov_grid)

        # Wrap both result panels in a scrollable container with a capped
        # maximum height: once a filter run produces output, the Filter quality
        # and Innovation diagnostics frames can be tall (especially for q,s>1),
        # and would otherwise eat into the ParamPanel above. The scroll area
        # keeps all content reachable without compressing the parameter tabs.
        results_box = QWidget()
        results_layout = QVBoxLayout(results_box)
        results_layout.setContentsMargins(0, 0, 0, 0)
        results_layout.setSpacing(6)
        results_layout.addWidget(self._mse_frame)
        results_layout.addWidget(self._innov_frame)

        self._results_scroll = QScrollArea()
        self._results_scroll.setWidget(results_box)
        self._results_scroll.setWidgetResizable(True)
        self._results_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._results_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._results_scroll.setMaximumHeight(220)
        # Hidden until the first filter run produces something to display
        self._results_scroll.setVisible(False)
        left_layout.addWidget(self._results_scroll)

        # Re-wire the inner frames' setVisible so the surrounding scroll area
        # follows automatically: any caller still doing
        # ``self._mse_frame.setVisible(...)`` or ``self._innov_frame.setVisible(...)``
        # transparently keeps ``self._results_scroll`` in sync. We track the
        # intended visibility ourselves because ``QWidget.isVisible()`` returns
        # False when any ancestor is hidden, which would defeat the heuristic.
        self._mse_visible = False
        self._innov_visible = False

        def _wrap_setVisible(frame: QFrame, attr: str) -> None:
            original = frame.setVisible

            def patched(visible: bool, *, _orig=original, _self=self, _attr=attr) -> None:
                _orig(visible)
                setattr(_self, _attr, bool(visible))
                _self._results_scroll.setVisible(_self._mse_visible or _self._innov_visible)

            frame.setVisible = patched  # type: ignore[method-assign]

        _wrap_setVisible(self._mse_frame, "_mse_visible")
        _wrap_setVisible(self._innov_frame, "_innov_visible")

        self._status_bar = QLabel("")
        self._status_bar.setWordWrap(True)
        self._status_bar.setStyleSheet("font-size: 10px; color: #444444;")  # A10: neutral default
        left_layout.addWidget(self._status_bar)

        splitter.addWidget(left)

        # ── right panel — QTabWidget avec deux onglets ───────────────
        self._right_tabs = QTabWidget()
        self._right_tabs.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # Onglet 0 : trajectoires simulation / filtre (inchangé)
        self._plot_panel = PlotPanel(q, s)
        self._right_tabs.addTab(self._plot_panel, "Simulation / Filtre")

        # Onglet 1 : densité conditionnelle p(y_{n+1} | r_n, r_{n+1}, y_n)
        self._pred_y_panel = PredYPanel(K, q, s)
        self._right_tabs.addTab(
            self._pred_y_panel,
            "p(yₙ₊₁ | rₙ, rₙ₊₁, yₙ)",
        )
        self._right_tabs.setTabEnabled(1, False)
        self._right_tabs.setTabToolTip(
            1,
            "Exact conditional Gaussian density p(y_{n+1} | r_n=j, r_{n+1}=k, y_n).\n"
            "Available after filtering in 'Exact IMM – H5 required' mode.",
        )

        splitter.addWidget(self._right_tabs)

        # Left panel takes ~420px initially, right panel takes the rest
        splitter.setSizes([540, 800])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        # ── Menu bar + status bar + persisted settings ────────────────
        self._build_menu_bar()
        self._build_status_bar()

        # If a model was passed, load its F and Σ_W, and select it in the combobox
        _preset_loaded = False
        if model is not None:
            self._load_model(model)
            cls_name = type(model).__name__
            for idx, entry in enumerate(self._presets, start=1):
                if entry.class_name == cls_name:
                    self._preset_combo.setCurrentIndex(idx)
                    _preset_loaded = True
                    break

        self._refresh_simulate_button()
        self._load_settings()
        self._refresh_session_summary()

        # When a preset is loaded at startup: ensure h5_exact is the active
        # filter mode.  The AB constraint stays unchecked by default so
        # the user can opt in explicitly.
        if _preset_loaded:
            idx_h5 = self._mode_combo.findData("h5_exact")
            if idx_h5 >= 0:
                self._mode_combo.setCurrentIndex(idx_h5)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_validity_changed(self, _: bool) -> None:
        self._refresh_simulate_button()
        self._refresh_filter_button_drift_indicator()

    def _refresh_filter_button_drift_indicator(self) -> None:
        """Mark the Filter button when GUI params differ from those captured.

        A small ⚠ prefix on the button label, plus a tooltip, makes it
        obvious that re-clicking Filter will use the *captured* parameters,
        not the ones currently shown in the widgets. Cleared as soon as
        the user re-runs Simulate.
        """
        base_label = "Filter"
        if self._state.params_signature is None:
            self._btn_filter.setText(base_label)
            self._btn_filter.setToolTip("Run the optimal filter on the simulation  (Ctrl+F)")
            self._btn_filter.setStyleSheet("")
            return
        live_sig = self._params_signature()
        if live_sig is not None and live_sig != self._state.params_signature:
            self._btn_filter.setText("⚠ " + base_label)
            self._btn_filter.setToolTip(
                "Parameters in the panel have changed since the last Simulate.\n"
                "Filter will use the parameters captured at Simulate, not the\n"
                "current GUI values. Re-run Simulate to use the new ones."
            )
            # B11: amber border makes the drift state unmissable
            self._btn_filter.setStyleSheet(
                "QPushButton { border: 2px solid #e6a800; background-color: #fff8e1; }"
                "QPushButton:hover { background-color: #fff0b3; }"
            )
        else:
            self._btn_filter.setText(base_label)
            self._btn_filter.setToolTip("Run the optimal filter on the simulation  (Ctrl+F)")
            self._btn_filter.setStyleSheet("")

    def _on_sim_params_changed(self) -> None:
        """Called when N or Seed changes: invalidate current results."""
        if not self._state.has_data():
            return
        self._on_reset()

    def _on_seed_text_changed(self, text: str) -> None:
        """Show red border when seed text cannot be parsed as an integer (A2)."""
        stripped = text.strip()
        if not stripped:
            self._seed_edit.setStyleSheet("")  # empty → random, neutral
        else:
            try:
                int(stripped)
                self._seed_edit.setStyleSheet("")  # valid integer
            except ValueError:
                self._seed_edit.setStyleSheet(
                    "border: 1px solid #cc0000; background-color: #fff0f0;"
                )

    def _on_reset(self) -> None:
        """Clear all simulation / filter results and reset the interface."""
        # Stop in-flight workers FIRST so their late `finished` signal
        # cannot resurrect the state we are about to clear.
        self._cancel_active_workers()
        # Also dismiss any wait dialog from a cancelled operation
        if getattr(self, "_wait_dlg", None) is not None:
            self._wait_dlg.close()
            self._wait_dlg = None
        # Re-enable the simulate button (a cancelled MC may have left it disabled)
        self._btn_simulate.setEnabled(True)

        self._state.reset()
        self._refresh_filter_button_drift_indicator()
        self._btn_filter.setEnabled(False)
        self._btn_save.setEnabled(False)
        self._btn_export.setEnabled(False)
        self._btn_export_plots.setEnabled(False)
        self._btn_innov_hist.setEnabled(False)
        self._btn_regime_diag.setEnabled(False)
        self._sync_menu_actions()
        self._mse_frame.setVisible(False)
        self._innov_frame.setVisible(False)
        self._set_status("")
        self._plot_panel.clear()
        # Réinitialiser et désactiver l'onglet p(y_{n+1} | …)
        self._pred_y_panel.clear()
        self._right_tabs.setTabEnabled(1, False)
        self._right_tabs.setCurrentIndex(0)
        self.statusBar().showMessage("Reset — ready.", 4000)

    def _cancel_active_workers(self) -> None:
        """Disconnect signals and request interruption on every running worker.

        Disconnecting first ensures that any `finished` / `error` / `progress`
        signal already queued on the event loop will be discarded by Qt
        instead of running our handlers and corrupting freshly-reset state.
        """
        for attr in ("_worker", "_filter_worker"):
            w = getattr(self, attr, None)
            if w is None:
                continue
            try:
                w.finished.disconnect()
            except TypeError, RuntimeError:
                pass
            try:
                w.error.disconnect()
            except TypeError, RuntimeError:
                pass
            if hasattr(w, "progress"):
                try:
                    w.progress.disconnect()
                except TypeError, RuntimeError:
                    pass
            if w.isRunning():
                w.requestInterruption()
                w.quit()
                # Don't block the UI: the worker checks isInterruptionRequested()
                # at fixed checkpoints and aborts silently. We let it die alone.
            setattr(self, attr, None)

    def _on_simulate(self) -> None:
        self._on_simulate_single()

    def _on_simulate_single(self) -> None:
        params = self._build_gss_params()
        if params is None:
            return

        N = self._n_spin.value()
        seed = self._parse_seed()

        self._state.begin_simulation(params, self._params_signature())
        self._refresh_filter_button_drift_indicator()
        self._btn_simulate.setEnabled(False)
        self._btn_save.setEnabled(False)
        self._btn_filter.setEnabled(False)
        self._btn_innov_hist.setEnabled(False)
        self._sync_menu_actions()
        self._mse_frame.setVisible(False)
        self._innov_frame.setVisible(False)
        self._plot_panel.clear_filter_overlay()
        self._set_status("")
        self._op_t0 = time.perf_counter()
        self.statusBar().showMessage(f"Simulating  N = {N}…")

        self._wait_dlg = _WaitDialog(f"Simulating  N = {N}…", on_cancel=self._on_reset, parent=self)
        self._wait_dlg.show()

        self._worker = _SimWorker(params, N, seed, parent=self)
        self._worker.finished.connect(self._on_sim_finished)
        self._worker.error.connect(self._on_sim_error)
        self._worker.start()

    def _on_sim_finished(
        self,
        ns: list[int],
        rs: list[int],
        xs: np.ndarray,
        ys: np.ndarray,
    ) -> None:
        # Drop signals that arrived after a Reset / new operation: the worker
        # was disconnected & nulled out, so any leftover queued signal is stale.
        if self.sender() is not self._worker:
            return
        if self._wait_dlg:
            self._wait_dlg.accept()
            self._wait_dlg = None

        seed = self._parse_seed()
        self._state.store_data(ns, rs, xs, ys, seed)
        self._plot_panel.update_plots(ns, rs, xs, ys, self._K)

        self._btn_save.setEnabled(True)
        self._btn_filter.setEnabled(True)
        self._btn_export_plots.setEnabled(True)
        self._btn_regime_diag.setEnabled(True)  # B7/B8: available after simulate
        self._sync_menu_actions()
        self._refresh_simulate_button()

        elapsed = time.perf_counter() - getattr(self, "_op_t0", time.perf_counter())
        self.statusBar().showMessage(
            f"Simulation complete — N = {len(ns)} steps in {elapsed:.2f}s.", 6000
        )

        # Auto-filter chaining
        if self._auto_filter_check.isChecked() and self._state.can_filter():
            self._on_filter()

    def _on_sim_error(self, msg: str) -> None:
        if self.sender() is not self._worker:
            return
        if self._wait_dlg:
            self._wait_dlg.reject()
            self._wait_dlg = None

        self._set_status(f"Error: {msg}", error=True)
        self.statusBar().showMessage(f"Simulation error: {msg}", 8000)
        self._refresh_simulate_button()

    def _h5_exact_blockers(self, params) -> list[str]:
        """Reasons the selected mode would be inexact for ``params``.

        Returns the list of NGH-MSM/CNS violations (from ``validate_ngh_msm``)
        when the filter-mode combo is set to ``"h5_exact"`` and ``params`` is
        not a valid NGH-MSM; otherwise an empty list. Pure logic (no UI), so it
        is unit-testable independently of the warning dialog.
        """
        if params is None or self._mode_combo.currentData() != "h5_exact":
            return []
        from prg.utils.h5_constraint import validate_ngh_msm

        return validate_ngh_msm(params)

    def _on_filter(self) -> None:
        if not self._state.can_filter():
            return

        # CNS guard: 'exactIMM (H5 required)' is exact only on a valid NGH-MSM.
        # If the user picked it for a model that violates (H5) / the structural
        # CNS, surface the issues (the filter would otherwise only emit a
        # RuntimeWarning the GUI swallows) and let them proceed or cancel.
        blockers = self._h5_exact_blockers(self._state.params)
        if blockers:
            from PyQt6.QtWidgets import QMessageBox

            detail = "\n".join(f"  • {m}" for m in blockers)
            resp = QMessageBox.warning(
                self,
                "Model is not a valid NGH-MSM",
                "The 'exactIMM (H5 required)' filter is exact only for a model that "
                "satisfies (H5) and the structural conditions. The captured model "
                f"violates:\n\n{detail}\n\nThe filter will be biased. Switch to "
                "'Approximate IMM', or enforce the AB constraint on every regime.\n\n"
                "Run the exact filter anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if resp != QMessageBox.StandardButton.Yes:
                return

        _, _, _, ys, _ = self._state.data

        # Warn the user if the parameters displayed on screen have drifted
        # from those captured at Simulate (or Load CSV) — the filter uses
        # the captured ones, not whatever is in the widgets right now.
        live_sig = self._params_signature()
        params_drifted = (
            live_sig is not None
            and self._state.params_signature is not None
            and live_sig != self._state.params_signature
        )

        self._btn_filter.setEnabled(False)
        self._sync_menu_actions()
        self._mse_frame.setVisible(False)
        self._set_status("")
        self._op_t0 = time.perf_counter()
        msg = f"Filtering  N = {len(ys)}…"
        if params_drifted:
            msg += "  ⚠ using parameters captured at last Simulate (GUI values differ)"
        self.statusBar().showMessage(msg)

        self._wait_dlg = _WaitDialog("Filtering…", on_cancel=self._on_reset, parent=self)
        self._wait_dlg.show()

        self._filter_worker = _FilterWorker(
            self._state.params,
            ys,
            joseph=self._joseph_check.isChecked(),
            mode=self._mode_combo.currentData(),
            parent=self,
        )
        self._filter_worker.finished.connect(self._on_filter_finished)  # type: ignore[arg-type]
        self._filter_worker.error.connect(self._on_filter_error)
        # D8: wire progress → wait dialog progress bar
        if self._wait_dlg is not None:
            self._filter_worker.progress.connect(
                lambda n, tot, dlg=self._wait_dlg: dlg.set_progress(n, tot)
            )
        self._filter_worker.start()

    def _on_filter_finished(
        self,
        E_xs: np.ndarray,  # (N, q)
        Var_xs: np.ndarray,  # (N, q)
        pis: np.ndarray,  # (N, K)
        innovations: np.ndarray,  # (N, s)
        log_lik_total: float,
    ) -> None:
        if self.sender() is not self._filter_worker:
            return
        if not self._state.has_data():
            return  # state cleared while filter was running
        if self._wait_dlg:
            self._wait_dlg.accept()
            self._wait_dlg = None

        ns, rs, xs, ys, _ = self._state.data
        N = len(ns)

        # ── Store + plot overlays ─────────────────────────────────────
        self._state.store_innovations(innovations)
        self._state.store_filter_results(E_xs, Var_xs, pis, log_lik_total)  # D5
        self._plot_panel.add_filter_overlay(ns, E_xs, Var_xs)
        self._plot_panel.add_pi_overlay(ns, pis, self._K)
        self._plot_panel.update_innovations(ns, innovations)

        # ── Filter quality frame (C5: extracted helper) ───────────────
        self._apply_filter_quality_frame(ns, xs, E_xs, log_lik_total)

        self._btn_filter.setEnabled(True)
        self._btn_innov_hist.setEnabled(True)
        self._sync_menu_actions()

        # ── Predicted-Y tab ──────────────────────────────────────────
        if self._filter_worker is not None and hasattr(self._filter_worker, "cond_moments"):
            cm = self._filter_worker.cond_moments
            _, _, _, ys, _ = self._state.data
            self._pred_y_panel.set_data(
                cm["mu_Y_jk"],
                cm["M_t"],
                cm["Gamma"],
                cm["mu_Y"],
                ys,
                M_simple=cm.get("M_simple"),
                Gamma2=cm.get("Gamma2"),
                b_Y=cm.get("b_Y"),
            )
            self._right_tabs.setTabEnabled(1, True)

        elapsed = time.perf_counter() - getattr(self, "_op_t0", time.perf_counter())
        self.statusBar().showMessage(
            f"Filter complete — N = {N}, log L = {log_lik_total:.4g}  ({elapsed:.2f}s).",
            8000,
        )

        # ── Innovation diagnostics (C5: extracted helper) ─────────────
        _mix_w = None
        _Gamma_cm = None
        _muY_jk = None
        if self._filter_worker is not None and hasattr(self._filter_worker, "cond_moments"):
            cm_diag = self._filter_worker.cond_moments
            if all(k in cm_diag for k in ("mix_w", "Gamma", "mu_Y_jk")):
                _mix_w = cm_diag["mix_w"]
                _Gamma_cm = cm_diag["Gamma"]
                _muY_jk = cm_diag["mu_Y_jk"]

        self._apply_innovation_diagnostics(innovations, _mix_w, _Gamma_cm, _muY_jk)

    # ── C5 sub-methods (called from _on_filter_finished + session restore) ──

    def _apply_filter_quality_frame(
        self,
        ns: list,
        xs: np.ndarray | None,
        E_xs: np.ndarray,
        log_lik_total: float,
    ) -> None:
        """Refresh the Filter quality frame (log L, MSE, RMSE, per-component)."""
        N = len(ns)
        mean_ll = log_lik_total / N if N > 0 else float("nan")
        self._loglik_label.setText(f"log L = {log_lik_total:.4g}   (mean = {mean_ll:.4g})")

        if xs is not None:
            err = xs - E_xs
            mse_per_comp = np.mean(err**2, axis=0)
            mse_global = float(mse_per_comp.mean())
            rmse_global = float(np.sqrt(mse_global))
            self._mse_global_label.setText(f"MSE  = {mse_global:.5g}")
            self._rmse_label.setText(f"RMSE = {rmse_global:.5g}")
            sig_std = float(xs.std()) if xs.std() > 0 else 1.0
            ratio = rmse_global / sig_std
            if ratio < 0.20:
                bg, fg, border = "#d4edda", "#155724", "#c3e6cb"
                quality_icon = "✓"
            elif ratio < 0.50:
                bg, fg, border = "#fff3cd", "#856404", "#ffc107"
                quality_icon = "~"
            else:
                bg, fg, border = "#f8d7da", "#721c24", "#f5c6cb"
                quality_icon = "✗"
            title_text = f"Filter quality  {quality_icon}  (RMSE/σ = {ratio:.2f})"
        else:
            self._mse_global_label.setText("")
            self._rmse_label.setText("")
            bg, fg, border = "#eef2f7", "#333333", "#c8d0d8"
            title_text = "Filter quality  (log L only)"

        self._mse_title.setText(title_text)
        self._mse_frame.setStyleSheet(
            f"QFrame {{ background-color: {bg}; border: 1px solid {border}; border-radius: 4px; }}"
        )
        title_style = f"font-weight: bold; font-size: 11px; color: {fg};"
        value_style = f"font-size: 10px; color: {fg};"
        self._mse_title.setStyleSheet(title_style)
        self._loglik_label.setStyleSheet(value_style)
        self._mse_global_label.setStyleSheet(value_style)
        self._rmse_label.setStyleSheet(value_style)

        # Per-component rows (rebuild if q changed)
        mse_vbox = self._mse_frame.layout()
        for lbl in self._mse_comp_labels:
            mse_vbox.removeWidget(lbl)
            lbl.deleteLater()
        self._mse_comp_labels.clear()
        if xs is not None and self._q > 1:
            for i, v in enumerate(np.mean((xs - E_xs) ** 2, axis=0)):
                lbl = QLabel(f"  MSE(X^{i}) = {v:.5g}")
                lbl.setStyleSheet(f"font-size: 9px; color: {fg};")
                mse_vbox.addWidget(lbl)
                self._mse_comp_labels.append(lbl)

        self._mse_frame.setVisible(True)

    def _apply_innovation_diagnostics(
        self,
        innovations: np.ndarray,
        mix_w: np.ndarray | None = None,
        Gamma_cm: object = None,
        muY_jk: object = None,
    ) -> None:
        """Update Ljung-Box + shape badge pills in the innovation diagnostics frame.

        Standardises innovations (D1 / A12) before the shape tests:
          h5_exact  → weighted mixture covariance from cond_moments
          imm_general → sample covariance (fallback)
        Applies Bonferroni correction over 2·s simultaneous tests (D2).
        """
        try:
            innov_std = _standardise_innovations(innovations, mix_w, Gamma_cm, muY_jk)
        except Exception:  # noqa: BLE001
            innov_std = innovations
        std_mode = "h5 S" if mix_w is not None else "sample S"

        n_tests = max(1, 2 * self._s)
        alpha_fam = _INNOV_ALPHA_FAM  # C3: named constant
        alpha_lb = alpha_fam / n_tests
        thresh_lb_ok = 2.0 * alpha_lb
        thresh_lb_warn = alpha_lb
        bonf_note = (
            f"Bonferroni-corrected threshold: α_per = {alpha_lb:.4g} "
            f"(family-wise α={alpha_fam}, {n_tests} tests)."
        )

        for i in range(self._s):
            _, p_lb, h_lb = _ljung_box(innovations[:, i])
            if p_lb > thresh_lb_ok:
                style, icon, verdict = _PILL_OK, "✓", "white"
            elif p_lb > thresh_lb_warn:
                style, icon, verdict = _PILL_WARN, "~", "border"
            else:
                style, icon, verdict = _PILL_ERR, "✗", "autocor."
            self._innov_lb_badges[i].setText(f"{icon} {verdict}  p={p_lb:.3f}")
            self._innov_lb_badges[i].setStyleSheet(style)
            self._innov_lb_badges[i].setToolTip(
                f"Ljung-Box: Q stat with h = {h_lb} lags.\n"
                f"p = {p_lb:.4g}   (OK if p > {thresh_lb_ok:.4g}).\n{bonf_note}"
            )
            S, K, JB, p_jb = _shape_diagnostics(innov_std[:, i])
            if abs(S) < 0.25 and abs(K) < 0.50:
                style, icon = _PILL_OK, "✓"
            elif abs(S) < 0.50 and abs(K) < 1.00:
                style, icon = _PILL_WARN, "~"
            else:
                style, icon = _PILL_ERR, "✗"
            self._innov_jb_badges[i].setText(f"{icon}  S={S:+.2f}  K={K:+.2f}")
            self._innov_jb_badges[i].setStyleSheet(style)
            self._innov_jb_badges[i].setToolTip(
                f"Standardised innovation ν̃  ({std_mode})\n"
                f"Skewness  S = {S:+.4f}   (target ≈ 0)\n"
                f"Excess kurtosis  K = {K:+.4f}   (target ≈ 0 for Gaussian)\n"
                f"Jarque-Bera  JB = {JB:.3f}   p = {p_jb:.4g}\n"
                "Standardisation removes the scale-mixing effect of regime\n"
                "switching, so S ≈ 0 and K ≈ 0 are achievable for a\n"
                f"well-calibrated filter.\n{bonf_note}"
            )
        self._innov_frame.setVisible(True)

    def _on_filter_error(self, msg: str) -> None:
        if self.sender() is not self._filter_worker:
            return
        if self._wait_dlg:
            self._wait_dlg.reject()
            self._wait_dlg = None

        self._set_status(f"Filter error: {msg}", error=True)
        self.statusBar().showMessage(f"Filter error: {msg}", 8000)
        self._btn_filter.setEnabled(True)
        self._sync_menu_actions()

    def _on_export_model(self) -> None:
        """Open a save-file dialog and write a ready-to-use Python model file."""
        # A14: guard against exporting with invalid parameters
        if not (self._param_panel.is_valid() and self._p_widget.is_valid()):
            QMessageBox.warning(
                self,
                "Invalid parameters",
                "One or more parameters are currently invalid.\n"
                "Fix all highlighted cells before exporting the model.",
            )
            return
        default_name = f"model_gss_K{self._K}_q{self._q}_s{self._s}_custom.py"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Python model",
            str(pathlib.Path("prg/models") / default_name),
            "Python files (*.py);;All files (*)",
        )
        if not path:
            return  # user cancelled

        code = self._generate_model_code(pathlib.Path(path).stem)
        try:
            pathlib.Path(path).write_text(code, encoding="utf-8")
            QMessageBox.information(
                self,
                "Model exported",
                f"File saved:\n{pathlib.Path(path).resolve()}",
            )
        except OSError as exc:
            QMessageBox.critical(self, "Write error", str(exc))

    def _on_save(self) -> None:
        if not self._state.has_data():
            return
        ns, rs, xs, ys, seed_used = self._state.data

        output_dir = pathlib.Path("data/simulated")
        output_dir.mkdir(parents=True, exist_ok=True)

        N = len(ns)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        seed_str = str(seed_used) if seed_used is not None else "rnd"
        filename = f"simulated_gui_N{N}_seed{seed_str}_{ts}.csv"
        filepath = output_dir / filename

        q, s = self._q, self._s
        header = ["n", "r"] + [f"x_{i}" for i in range(q)] + [f"y_{i}" for i in range(s)]

        with filepath.open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(header)
            for n, r, x_row, y_row in zip(ns, rs, xs, ys):
                writer.writerow([n, r] + list(x_row) + list(y_row))

        QMessageBox.information(
            self,
            "CSV saved",
            f"File saved:\n{filepath.resolve()}",
        )
        self._push_recent_csv(str(filepath))  # D9

    def _on_preset_selected(self, index: int) -> None:
        """Load a preset model; restart the window if K/q/s differ."""
        if index == 0:  # placeholder item
            return
        entry = self._presets[index - 1]
        try:
            model = entry.load()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Preset error", str(exc))
            self._preset_combo.setCurrentIndex(0)
            return

        if entry.K == self._K and entry.q == self._q and entry.s == self._s:
            # Same dimensions — load in place and reset plots
            self._load_model(model)
            self._on_reset()
        else:
            answer = QMessageBox.question(
                self,
                "Dimension change",
                f"This model requires K={entry.K}, q={entry.q}, s={entry.s}.\n"
                f"The window will be recreated. Continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if answer == QMessageBox.StandardButton.Yes:
                self.restart_with_model.emit(model)
            else:
                self._preset_combo.setCurrentIndex(0)

    def _on_export_plots(self) -> None:
        """Save the current plot panel figure to a PNG or PDF file."""
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export plots",
            str(pathlib.Path("data/simulated") / "plots.pdf"),
            "PDF (*.pdf);;PNG (*.png);;SVG (*.svg);;All files (*)",
        )
        if not path:
            return
        try:
            self._plot_panel.save_figure(path)
            QMessageBox.information(
                self,
                "Export plots",
                f"Figure saved:\n{pathlib.Path(path).resolve()}",
            )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Export error", str(exc))

    def _on_load_data(self) -> None:
        """Open a CSV file and display it without running a simulation."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load simulation data",
            str(pathlib.Path("data/simulated")),
            "CSV files (*.csv);;All files (*)",
        )
        if not path:
            return

        try:
            with open(path, newline="", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                rows = list(reader)

            if not rows:
                QMessageBox.warning(self, "Load error", "File is empty.")
                return

            headers = list(rows[0].keys())

            # --- Validate required columns ---
            required = ["n", "r"] + [f"y_{i}" for i in range(self._s)]
            missing = [c for c in required if c not in headers]
            if missing:
                QMessageBox.warning(
                    self,
                    "Load error",
                    f"Missing column(s): {missing}\n"
                    f"Expected at least: n, r, y_0 … y_{self._s - 1}\n"
                    f"Found: {headers}",
                )
                return

            # --- Parse ---
            ns = [int(float(row["n"])) for row in rows]
            rs = [int(float(row["r"])) for row in rows]
            ys = np.array([[float(row[f"y_{i}"]) for i in range(self._s)] for row in rows])

            x_cols = [f"x_{i}" for i in range(self._q)]
            has_x = all(c in headers for c in x_cols)
            xs = (
                np.array([[float(row[f"x_{i}"]) for i in range(self._q)] for row in rows])
                if has_x
                else None
            )

        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Load error", str(exc))
            return

        # --- Update state ---
        live_params = self._build_gss_params()  # uses current GUI params
        self._state.load_external(
            ns,
            rs,
            xs,
            ys,
            params=live_params,
            signature=self._params_signature(),
        )
        self._refresh_filter_button_drift_indicator()

        self._plot_panel.clear_filter_overlay()
        self._plot_panel.update_plots(ns, rs, xs, ys, self._K)

        self._btn_filter.setEnabled(self._state.can_filter())
        self._btn_save.setEnabled(False)  # nothing new generated
        self._btn_export_plots.setEnabled(True)
        self._btn_regime_diag.setEnabled(True)  # B7/B8
        self._sync_menu_actions()
        self._mse_frame.setVisible(False)

        info = f"Loaded {len(ns)} steps from '{pathlib.Path(path).name}'"
        if not has_x:
            info += "  (no ground-truth X)"
        self._set_status(info)
        self.statusBar().showMessage(info, 6000)
        self._push_recent_csv(path)  # D9

    # ------------------------------------------------------------------
    # D5 / B10 — Save / Load complete session
    # ------------------------------------------------------------------

    def _on_save_session(self) -> None:
        """Save params + sim data + filter results to a .exactIMM file."""
        import io

        default_dir = pathlib.Path("data/sessions")
        default_dir.mkdir(parents=True, exist_ok=True)
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save session",
            str(default_dir / "session.exactIMM"),
            "exactIMM sessions (*.exactIMM);;All files (*)",
        )
        if not path:
            return
        if not path.endswith(".exactIMM"):
            path += ".exactIMM"

        # Collect GUI parameters
        F_list = self._param_panel.get_F_list()
        SW_list = self._param_panel.get_Sigma_W_list()
        mu_list = self._param_panel.get_mu_z0_list()
        b_list = self._param_panel.get_b_list()
        P = self._p_widget.get_matrix()
        if F_list is None or SW_list is None:
            QMessageBox.warning(
                self,
                "Save session",
                "Cannot save: one or more parameter matrices are invalid.",
            )
            return

        arrays: dict = {
            "_K": np.array(self._K),
            "_q": np.array(self._q),
            "_s": np.array(self._s),
            "_filter_mode": np.array(self._mode_combo.currentData() or "imm_general", dtype=object),
            "_joseph": np.array(self._joseph_check.isChecked()),
            "_P": P,
        }
        for k in range(self._K):
            q, s = self._q, self._s
            arrays[f"_F_{k}"] = F_list[k]
            arrays[f"_SW_{k}"] = SW_list[k]
            arrays[f"_mu_{k}"] = mu_list[k] if mu_list else np.zeros((q + s, 1))
            arrays[f"_b_{k}"] = b_list[k] if b_list else np.zeros((q + s, 1))

        has_data = self._state.has_data()
        arrays["_has_data"] = np.array(has_data)
        if has_data:
            ns, rs, xs, ys, seed_used = self._state.data
            has_xs = xs is not None
            arrays["_ns"] = np.array(ns, dtype=np.int32)
            arrays["_rs"] = np.array(rs, dtype=np.int32)
            arrays["_ys"] = ys
            arrays["_has_xs"] = np.array(has_xs)
            arrays["_xs"] = xs if has_xs else np.zeros(0, dtype=np.float64)
            arrays["_seed"] = np.array(seed_used if seed_used is not None else -1, dtype=np.int64)

        has_filt = self._state.has_filter() and self._state.filter_E_xs is not None
        arrays["_has_filter"] = np.array(has_filt)
        if has_filt:
            arrays["_innovations"] = self._state.innovations
            arrays["_E_xs"] = self._state.filter_E_xs
            arrays["_Var_xs"] = self._state.filter_Var_xs
            arrays["_pis"] = self._state.filter_pis
            arrays["_log_lik"] = np.array(self._state.filter_log_lik or 0.0)

        try:
            buf = io.BytesIO()
            np.savez_compressed(buf, **arrays)
            buf.seek(0)
            pathlib.Path(path).write_bytes(buf.read())
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Save session error", str(exc))
            return

        self._set_status(f"Session saved: {pathlib.Path(path).name}")
        self.statusBar().showMessage(f"Session saved to {pathlib.Path(path).resolve()}", 6000)
        self._push_recent_session(path)  # D9

    def _on_load_session(self) -> None:
        """Restore a .exactIMM session (params + data + filter)."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load session",
            str(pathlib.Path("data/sessions")),
            "exactIMM sessions (*.exactIMM *.npz);;All files (*)",
        )
        if not path:
            return
        self._load_session_from(path)

    def _restore_session_from_npz(self, npz, path: str) -> None:
        """Core session-restore logic — shared by dialog and recent-files paths."""
        K = int(npz["_K"])
        q = int(npz["_q"])
        s = int(npz["_s"])

        if K != self._K or q != self._q or s != self._s:
            QMessageBox.warning(
                self,
                "Dimension mismatch",
                f"Session has K={K}, q={q}, s={s} but the current window "
                f"has K={self._K}, q={self._q}, s={self._s}.\n\n"
                "Close this window and open one with the matching dimensions "
                "(use the Preset combo or launch the app with the correct model), "
                "then load the session again.",
            )
            return

        # --- 1. Restore parameters ---
        self._param_panel.blockSignals(True)
        self._p_widget.blockSignals(True)
        try:
            P = npz["_P"]
            F_list = [npz[f"_F_{k}"] for k in range(K)]
            SW_list = [npz[f"_SW_{k}"] for k in range(K)]
            mu_list = [npz[f"_mu_{k}"] for k in range(K)]
            b_list = [npz[f"_b_{k}"] for k in range(K)]

            A_list = [f[:q, :q] for f in F_list]
            B_list = [f[:q, q:] for f in F_list]
            C_list = [f[q:, :q] for f in F_list]
            D_list = [f[q:, q:] for f in F_list]
            SU_list = [sw[:q, :q] for sw in SW_list]
            Dl_list = [sw[:q, q:] for sw in SW_list]
            SV_list = [sw[q:, q:] for sw in SW_list]

            fm = FMatrix(K, q, s, A_list, B_list, C_list, D_list)
            nc = GSSNoiseCovariance(K, q, s, SU_list, Dl_list, SV_list)
            for k in range(K):
                self._param_panel.set_state_params(k, fm.F(k), nc.Sigma_W(k), mu_list[k], b_list[k])

            # Re-apply any active AB constraint to the newly loaded
            # parameter values.  The ParamPanel signals are still blocked here,
            # so constraint_toggled cannot propagate to _on_reset.
            self._param_panel.reapply_active_constraints()

            self._P = P
            self._p_widget.set_matrix(self._P)
            self._update_stationary_display()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(
                self, "Load session error", f"Failed to restore parameters:\n{exc}"
            )
            return
        finally:
            self._param_panel.blockSignals(False)
            self._p_widget.blockSignals(False)
        self._refresh_simulate_button()

        # Restore filter mode + Joseph flag
        try:
            fm_str = str(npz["_filter_mode"])
            idx = self._mode_combo.findData(fm_str)
            if idx >= 0:
                self._mode_combo.setCurrentIndex(idx)
            self._joseph_check.setChecked(bool(npz["_joseph"]))
        except Exception:  # noqa: BLE001
            pass

        # --- 2. Clear and restore simulation data ---
        self._state.reset()
        self._plot_panel.clear()
        self._mse_frame.setVisible(False)
        self._innov_frame.setVisible(False)
        self._btn_filter.setEnabled(False)
        self._btn_save.setEnabled(False)
        self._btn_innov_hist.setEnabled(False)
        self._right_tabs.setTabEnabled(1, False)

        has_data = bool(npz["_has_data"]) if "_has_data" in npz else False
        ns: list = []
        xs_arr: np.ndarray | None = None
        ys_arr: np.ndarray = np.zeros((0, s))
        if has_data:
            try:
                ns = npz["_ns"].tolist()
                rs = npz["_rs"].tolist()
                ys_arr = npz["_ys"]
                has_xs = bool(npz["_has_xs"])
                xs_arr = npz["_xs"] if has_xs else None
                seed_val = int(npz["_seed"])
                seed_val = seed_val if seed_val >= 0 else None

                live_params = self._build_gss_params()
                self._state.store_data(ns, rs, xs_arr, ys_arr, seed_val)
                self._state.params = live_params
                self._state.params_signature = self._params_signature()

                self._plot_panel.update_plots(ns, rs, xs_arr, ys_arr, K)
                self._btn_filter.setEnabled(True)
                self._btn_export_plots.setEnabled(True)
                self._btn_regime_diag.setEnabled(True)  # B7/B8
            except Exception as exc:  # noqa: BLE001
                QMessageBox.warning(
                    self, "Load session", f"Could not restore simulation data:\n{exc}"
                )

        # --- 3. Restore filter results ---
        has_filt = bool(npz["_has_filter"]) if "_has_filter" in npz else False
        if has_filt and has_data and "_E_xs" in npz:
            try:
                E_xs = npz["_E_xs"]
                Var_xs = npz["_Var_xs"]
                pis = npz["_pis"]
                innovations = npz["_innovations"]
                log_lik_total = float(npz["_log_lik"])

                self._state.store_innovations(innovations)
                self._state.store_filter_results(E_xs, Var_xs, pis, log_lik_total)

                self._plot_panel.add_filter_overlay(ns, E_xs, Var_xs)
                self._plot_panel.add_pi_overlay(ns, pis, K)
                self._plot_panel.update_innovations(ns, innovations)

                # C5: use shared helpers (sample-covariance fallback — no cond_moments)
                self._apply_filter_quality_frame(ns, xs_arr, E_xs, log_lik_total)
                self._apply_innovation_diagnostics(innovations)

                self._btn_filter.setEnabled(True)
                self._btn_innov_hist.setEnabled(True)
                # regime diag already enabled from data restore above

            except Exception as exc:  # noqa: BLE001
                QMessageBox.warning(
                    self, "Load session", f"Could not restore filter results:\n{exc}"
                )

        self._sync_menu_actions()
        self._refresh_filter_button_drift_indicator()
        name = pathlib.Path(path).name
        self._set_status(f"Session loaded: {name}")
        self.statusBar().showMessage(f"Session loaded from {name}", 6000)
        self._push_recent_session(path)  # D9: update recent-sessions menu

    # ------------------------------------------------------------------
    # Menu bar, status bar, settings
    # ------------------------------------------------------------------

    def _build_menu_bar(self) -> None:
        """Create the application menu bar with keyboard shortcuts."""
        mb = self.menuBar()

        # ── File menu ────────────────────────────────────────────────
        file_menu = mb.addMenu("&File")

        self._act_save = QAction("Save &CSV", self)
        self._act_save.setShortcut(QKeySequence("Ctrl+S"))
        self._act_save.setStatusTip("Save current simulation as CSV")
        self._act_save.setEnabled(False)
        self._act_save.triggered.connect(self._on_save)
        file_menu.addAction(self._act_save)

        self._act_load = QAction("&Load CSV…", self)
        self._act_load.setShortcut(QKeySequence("Ctrl+O"))
        self._act_load.setStatusTip("Load a previously saved simulation CSV")
        self._act_load.triggered.connect(self._on_load_data)
        file_menu.addAction(self._act_load)

        # D9/B4: recent CSV submenu (populated dynamically)
        self._recent_csv_menu = file_menu.addMenu("Recent &CSV")
        self._rebuild_recent_csv_menu()

        file_menu.addSeparator()

        # D5/B10: session persistence
        self._act_save_session = QAction("Save &session…", self)
        self._act_save_session.setShortcut(QKeySequence("Ctrl+Shift+S"))
        self._act_save_session.setStatusTip(
            "Save complete session (params + data + filter) to a .exactIMM file"
        )
        self._act_save_session.triggered.connect(self._on_save_session)
        file_menu.addAction(self._act_save_session)

        self._act_load_session = QAction("L&oad session…", self)
        self._act_load_session.setShortcut(QKeySequence("Ctrl+Shift+O"))
        self._act_load_session.setStatusTip("Restore a previously saved .exactIMM session")
        self._act_load_session.triggered.connect(self._on_load_session)
        file_menu.addAction(self._act_load_session)

        # D9/B4: recent sessions submenu
        self._recent_session_menu = file_menu.addMenu("Recent &sessions")
        self._rebuild_recent_session_menu()

        file_menu.addSeparator()

        self._act_export = QAction("Export &model…", self)
        self._act_export.setShortcut(QKeySequence("Ctrl+E"))
        self._act_export.setStatusTip("Export current parameters as a Python model file")
        self._act_export.setEnabled(False)
        self._act_export.triggered.connect(self._on_export_model)
        file_menu.addAction(self._act_export)

        self._act_export_plots = QAction("Export &plots…", self)
        self._act_export_plots.setShortcut(QKeySequence("Ctrl+Shift+E"))
        self._act_export_plots.setStatusTip("Save current plot panel as PDF/PNG/SVG")
        self._act_export_plots.setEnabled(False)
        self._act_export_plots.triggered.connect(self._on_export_plots)
        file_menu.addAction(self._act_export_plots)

        file_menu.addSeparator()

        act_quit = QAction("&Quit", self)
        act_quit.setShortcut(QKeySequence.StandardKey.Quit)
        act_quit.setStatusTip("Quit the application")
        act_quit.triggered.connect(self.close)
        file_menu.addAction(act_quit)

        # ── Simulation menu ──────────────────────────────────────────
        sim_menu = mb.addMenu("&Simulation")

        self._act_simulate = QAction("Si&mulate", self)
        self._act_simulate.setShortcut(QKeySequence("Ctrl+R"))
        self._act_simulate.setStatusTip("Run simulation (single or Monte-Carlo)")
        self._act_simulate.triggered.connect(self._on_simulate)
        sim_menu.addAction(self._act_simulate)

        self._act_filter = QAction("&Filter", self)
        self._act_filter.setShortcut(QKeySequence("Ctrl+F"))
        self._act_filter.setStatusTip("Run the optimal filter on the current simulation")
        self._act_filter.setEnabled(False)
        self._act_filter.triggered.connect(self._on_filter)
        sim_menu.addAction(self._act_filter)

        sim_menu.addSeparator()

        self._act_reset = QAction("&Reset", self)
        self._act_reset.setShortcut(QKeySequence("Ctrl+Shift+R"))
        self._act_reset.setStatusTip("Clear all results and reset the plots")
        self._act_reset.triggered.connect(self._on_reset)
        sim_menu.addAction(self._act_reset)

        # ── View menu ────────────────────────────────────────────────
        view_menu = mb.addMenu("&View")

        self._act_innov_hist = QAction("&Innovation histograms…", self)
        self._act_innov_hist.setShortcut(QKeySequence("Ctrl+I"))
        self._act_innov_hist.setStatusTip("Show histogram of each innovation component")
        self._act_innov_hist.setEnabled(False)
        self._act_innov_hist.triggered.connect(self._on_innov_hist)
        view_menu.addAction(self._act_innov_hist)

        # B7/B8: regime diagnostics
        self._act_regime_diag = QAction("&Regime diagnostics…", self)
        self._act_regime_diag.setShortcut(QKeySequence("Ctrl+D"))
        self._act_regime_diag.setStatusTip("Confusion matrix and regime-duration histograms")
        self._act_regime_diag.setEnabled(False)
        self._act_regime_diag.triggered.connect(self._on_regime_diag)
        view_menu.addAction(self._act_regime_diag)

    # ------------------------------------------------------------------
    # D9 / B4 — Recent files helpers
    # ------------------------------------------------------------------

    _MAX_RECENT = 5

    def _push_recent_csv(self, path: str) -> None:
        """Prepend *path* to the recent-CSV list and persist to QSettings."""
        s = self._settings()
        recent: list[str] = s.value("recent_csv", [], type=list) or []
        path = str(pathlib.Path(path).resolve())
        if path in recent:
            recent.remove(path)
        recent.insert(0, path)
        s.setValue("recent_csv", recent[: self._MAX_RECENT])
        if hasattr(self, "_recent_csv_menu"):
            self._rebuild_recent_csv_menu()

    def _push_recent_session(self, path: str) -> None:
        """Prepend *path* to the recent-sessions list and persist to QSettings."""
        s = self._settings()
        recent: list[str] = s.value("recent_sessions", [], type=list) or []
        path = str(pathlib.Path(path).resolve())
        if path in recent:
            recent.remove(path)
        recent.insert(0, path)
        s.setValue("recent_sessions", recent[: self._MAX_RECENT])
        if hasattr(self, "_recent_session_menu"):
            self._rebuild_recent_session_menu()

    def _rebuild_recent_csv_menu(self) -> None:
        """Repopulate the Recent CSV submenu from QSettings."""
        menu = self._recent_csv_menu
        menu.clear()
        s = self._settings()
        recent: list[str] = s.value("recent_csv", [], type=list) or []
        recent = [p for p in recent if pathlib.Path(p).exists()]
        if not recent:
            act = menu.addAction("(no recent files)")
            act.setEnabled(False)
            return
        for path in recent:
            lbl = pathlib.Path(path).name
            act = menu.addAction(lbl)
            act.setStatusTip(path)
            act.triggered.connect(lambda checked, p=path: self._load_csv_from(p))
        menu.addSeparator()
        clr = menu.addAction("Clear recent CSV")
        clr.triggered.connect(self._clear_recent_csv)

    def _rebuild_recent_session_menu(self) -> None:
        """Repopulate the Recent sessions submenu from QSettings."""
        menu = self._recent_session_menu
        menu.clear()
        s = self._settings()
        recent: list[str] = s.value("recent_sessions", [], type=list) or []
        recent = [p for p in recent if pathlib.Path(p).exists()]
        if not recent:
            act = menu.addAction("(no recent sessions)")
            act.setEnabled(False)
            return
        for path in recent:
            lbl = pathlib.Path(path).name
            act = menu.addAction(lbl)
            act.setStatusTip(path)
            act.triggered.connect(lambda checked, p=path: self._load_session_from(p))
        menu.addSeparator()
        clr = menu.addAction("Clear recent sessions")
        clr.triggered.connect(self._clear_recent_sessions)

    def _clear_recent_csv(self) -> None:
        self._settings().remove("recent_csv")
        self._rebuild_recent_csv_menu()

    def _clear_recent_sessions(self) -> None:
        self._settings().remove("recent_sessions")
        self._rebuild_recent_session_menu()

    def _load_csv_from(self, path: str) -> None:
        """Load a CSV directly (called from Recent CSV menu)."""
        # Temporarily patch QFileDialog to return the known path
        # instead of showing the dialog.  We call _on_load_data's internals.
        try:
            with open(path, newline="", encoding="utf-8") as fh:
                import csv as _csv_mod

                reader = _csv_mod.DictReader(fh)
                rows = list(reader)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Load error", str(exc))
            return

        if not rows:
            QMessageBox.warning(self, "Load error", "File is empty.")
            return

        headers = list(rows[0].keys())
        required = ["n", "r"] + [f"y_{i}" for i in range(self._s)]
        missing = [c for c in required if c not in headers]
        if missing:
            QMessageBox.warning(
                self,
                "Load error",
                f"Missing column(s): {missing}\n"
                f"Expected at least: n, r, y_0 … y_{self._s - 1}\n"
                f"Found: {headers}",
            )
            return

        try:
            ns = [int(float(row["n"])) for row in rows]
            rs = [int(float(row["r"])) for row in rows]
            ys = np.array([[float(row[f"y_{i}"]) for i in range(self._s)] for row in rows])
            x_cols = [f"x_{i}" for i in range(self._q)]
            has_x = all(c in headers for c in x_cols)
            xs = (
                np.array([[float(row[f"x_{i}"]) for i in range(self._q)] for row in rows])
                if has_x
                else None
            )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Load error", str(exc))
            return

        live_params = self._build_gss_params()
        self._state.load_external(
            ns, rs, xs, ys, params=live_params, signature=self._params_signature()
        )
        self._refresh_filter_button_drift_indicator()
        self._plot_panel.clear_filter_overlay()
        self._plot_panel.update_plots(ns, rs, xs, ys, self._K)
        self._btn_filter.setEnabled(self._state.can_filter())
        self._btn_save.setEnabled(False)
        self._btn_export_plots.setEnabled(True)
        self._btn_regime_diag.setEnabled(True)
        self._sync_menu_actions()
        self._mse_frame.setVisible(False)
        info = f"Loaded {len(ns)} steps from '{pathlib.Path(path).name}'"
        if not has_x:
            info += "  (no ground-truth X)"
        self._set_status(info)
        self.statusBar().showMessage(info, 6000)
        self._push_recent_csv(path)  # refresh menu with this path on top

    def _load_session_from(self, path: str) -> None:
        """Load a session directly (called from Recent sessions menu)."""
        # Delegate to _on_load_session but bypass the file dialog.
        # We temporarily override the dialog call by calling the shared body.
        try:
            npz = np.load(path, allow_pickle=True)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Load session error", f"Cannot read session file:\n{exc}")
            return
        self._restore_session_from_npz(npz, path)

    def _build_status_bar(self) -> None:
        """Permanent status bar: ephemeral messages + session summary."""
        sb = self.statusBar()
        sb.setSizeGripEnabled(False)

        # Permanent right-side session summary label
        self._sb_session_lbl = QLabel("")
        self._sb_session_lbl.setStyleSheet("font-size: 10px; color: #333333;")
        sb.addPermanentWidget(self._sb_session_lbl)

        # Wire signals that change session parameters to refresh the summary
        self._n_spin.valueChanged.connect(self._refresh_session_summary)
        self._seed_edit.textChanged.connect(self._refresh_session_summary)
        self._auto_filter_check.toggled.connect(self._refresh_session_summary)
        self._joseph_check.toggled.connect(self._refresh_session_summary)
        self._mode_combo.currentIndexChanged.connect(lambda _: self._refresh_session_summary())

    def _refresh_session_summary(self) -> None:
        """Update the right-hand permanent label in the status bar."""
        if not hasattr(self, "_sb_session_lbl"):
            return
        seed = self._seed_edit.text().strip() or "random"
        auto = "auto-filter" if self._auto_filter_check.isChecked() else ""
        mode = self._mode_combo.currentData()
        mode_short = "H5-exact" if mode == "h5_exact" else "IMM-general"
        parts = [
            f"K={self._K}·q={self._q}·s={self._s}",
            f"N={self._n_spin.value()}",
            f"seed={seed}",
            f"filter={mode_short}",
        ]
        # Joseph flag is only meaningful in h5_exact mode
        if mode == "h5_exact":
            joseph = "Joseph" if self._joseph_check.isChecked() else "short"
            parts.append(f"cov={joseph}")
        if auto:
            parts.append(auto)
        self._sb_session_lbl.setText("  |  ".join(parts))

    # -- Settings persistence via QSettings --------------------------------

    def _settings(self) -> QSettings:
        return QSettings("exactIMM", "Simulator")

    def _load_settings(self) -> None:
        s = self._settings()
        # N is intentionally not restored: every launch starts with N=1000
        # so a previous large-N session can't make the next launch sluggish.
        seed = s.value("seed", self._seed_edit.text())
        if seed is not None:
            self._seed_edit.setText(str(seed))
        self._auto_filter_check.setChecked(s.value("auto_filter", False, type=bool))
        self._joseph_check.setChecked(s.value("joseph", False, type=bool))
        saved_mode = s.value("filter_mode", "h5_exact")
        idx = self._mode_combo.findData(saved_mode)
        if idx >= 0:
            self._mode_combo.setCurrentIndex(idx)

        geom = s.value("geometry")
        if geom is not None:
            try:
                self.restoreGeometry(geom)
            except Exception:  # noqa: BLE001
                pass
        split = s.value("splitter")
        if split is not None:
            try:
                self._splitter.restoreState(split)
            except Exception:  # noqa: BLE001
                pass

    def _save_settings(self) -> None:
        s = self._settings()
        # N intentionally not saved — see _load_settings.
        s.setValue("seed", self._seed_edit.text())
        s.setValue("auto_filter", self._auto_filter_check.isChecked())
        s.setValue("joseph", self._joseph_check.isChecked())
        s.setValue("filter_mode", self._mode_combo.currentData())
        s.setValue("geometry", self.saveGeometry())
        s.setValue("splitter", self._splitter.saveState())
        # Clean up legacy keys from previous versions so they can't resurface
        s.remove("N")
        s.remove("mc_on")
        s.remove("M")

    def closeEvent(self, event) -> None:  # noqa: N802
        self._cancel_active_workers()
        self._save_settings()
        super().closeEvent(event)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _update_stationary_display(self) -> None:
        """Recompute and display the stationary distribution π* of P."""
        P = self._p_widget.get_matrix()
        if P is None:
            self._stationary_label.setText("")
            return
        pi = _stationary_dist(P)
        if pi is None:
            self._stationary_label.setText("π* = ?")
            return
        parts = "   ".join(f"π<sub>{k}</sub> = {pi[k]:.3f}" for k in range(len(pi)))
        self._stationary_label.setText(f"π* :  {parts}")

    def _on_regime_diag(self) -> None:
        """Open regime diagnostics dialog (B7/B8): confusion matrix + duration hist."""
        if not self._state.has_data():
            return
        ns, rs, xs, ys, _ = self._state.data
        pis = self._state.filter_pis  # None if filter not run
        dlg = _RegimeDiagDialog(
            K=self._K,
            rs=rs,
            P=self._P,
            pis=pis,
            parent=self,
        )
        dlg.show()

    def _on_innov_hist(self) -> None:
        """Open innovation histogram dialog."""
        if not self._state.has_filter():
            return
        mix_params = None
        mu_Y_pr = None  # [K] ndarray (s,1) — stationary regime means
        S_YY_pr = None  # [K] ndarray (s,s) — stationary regime Y-covariances
        ys_pr = None  # (N, s) observations (for per-regime residuals)

        if self._filter_worker is not None and hasattr(self._filter_worker, "cond_moments"):
            cm = self._filter_worker.cond_moments
            if all(k in cm for k in ("mix_w", "Gamma", "mu_Y_jk")):
                mix_params = {
                    "w": cm["mix_w"],  # (K,K)
                    "Gamma": cm["Gamma"],  # [K][K] (s,s)
                    "mu_Y_jk": cm["mu_Y_jk"],  # [K][K] (s,1)
                }
            # Per-regime residuals: always available after stationary prior commit
            mu_Y_pr = cm.get("mu_Y")  # [K] ndarray (s,1)
            S_YY_pr = cm.get("S_YY")  # [K] ndarray (s,s)

        if self._state.data is not None:
            _, _, _, ys_pr, _ = self._state.data  # (N, s)

        dlg = _InnovHistDialog(
            self._state.innovations,
            mix_params=mix_params,
            pis=self._state.filter_pis,  # D11: per-regime tab
            mu_Y=mu_Y_pr,
            S_YY=S_YY_pr,
            ys=ys_pr,
            parent=self,
        )
        dlg.show()

    def _refresh_simulate_button(self) -> None:
        valid = self._param_panel.is_valid() and self._p_widget.is_valid()
        self._btn_simulate.setEnabled(valid)
        self._btn_export.setEnabled(valid)
        if valid:
            self._btn_simulate.setStyleSheet("")
            self._set_status("")
        else:
            self._btn_simulate.setStyleSheet("border: 2px solid #cc0000; color: #cc0000;")
            self._set_status("Invalid parameter(s) — fix before simulating.", error=True)
        self._sync_menu_actions()

    def _sync_menu_actions(self) -> None:
        """Mirror every QPushButton's enabled state into its matching QAction."""
        if not hasattr(self, "_act_simulate"):
            return  # menu bar not built yet
        self._act_simulate.setEnabled(self._btn_simulate.isEnabled())
        self._act_filter.setEnabled(self._btn_filter.isEnabled())
        self._act_save.setEnabled(self._btn_save.isEnabled())
        self._act_export.setEnabled(self._btn_export.isEnabled())
        self._act_export_plots.setEnabled(self._btn_export_plots.isEnabled())
        self._act_innov_hist.setEnabled(self._btn_innov_hist.isEnabled())
        self._act_regime_diag.setEnabled(self._btn_regime_diag.isEnabled())
        # D5: session actions always available (save checks validity inline)
        self._act_save_session.setEnabled(True)
        self._act_load_session.setEnabled(True)

    def _set_status(self, msg: str, *, error: bool = False) -> None:
        """Update the left-panel status label with appropriate styling (A10).

        Parameters
        ----------
        msg   : message text (empty string clears the label).
        error : if True, use red error styling; if False, use neutral grey.
        """
        self._status_bar.setText(msg)
        color = "#cc0000" if error else "#444444"
        self._status_bar.setStyleSheet(f"font-size: 10px; color: {color};")

    def _parse_seed(self) -> int | None:
        text = self._seed_edit.text().strip()
        if not text:
            return None
        try:
            return int(text)
        except ValueError:
            return None

    def _params_signature(self) -> tuple | None:
        """Compact byte-level signature of the GUI parameters.

        Two simulations done with identical widget values produce the same
        signature; any edit invalidates it. Used to warn the user when the
        parameters displayed on screen no longer match those captured at
        the last Simulate (and thus used by Filter).

        Returns None if any widget is currently invalid.
        """
        F = self._param_panel.get_F_list()
        S = self._param_panel.get_Sigma_W_list()
        mu = self._param_panel.get_mu_z0_list()
        b = self._param_panel.get_b_list()
        P = self._p_widget.get_matrix()
        if any(x is None for x in (F, S, mu, b, P)):
            return None
        return (
            tuple(np.ascontiguousarray(m).tobytes() for m in F),
            tuple(np.ascontiguousarray(m).tobytes() for m in S),
            tuple(np.ascontiguousarray(m).tobytes() for m in mu),
            tuple(np.ascontiguousarray(m).tobytes() for m in b),
            np.ascontiguousarray(P).tobytes(),
        )

    def _build_gss_params(self) -> GSSParams | None:
        """Collect GUI values and build a GSSParams object."""
        K, q, s = self._K, self._q, self._s

        F_list = self._param_panel.get_F_list()
        Sigma_W_list = self._param_panel.get_Sigma_W_list()
        mu_z0_list = self._param_panel.get_mu_z0_list()
        b_list = self._param_panel.get_b_list()
        P = self._p_widget.get_matrix()
        if (
            F_list is None
            or Sigma_W_list is None
            or mu_z0_list is None
            or b_list is None
            or P is None
        ):
            self._set_status("Invalid parameter(s).", error=True)
            return None

        # Decompose Sigma_W(k) into blocks Sigma_U, Delta, Sigma_V
        Sigma_U_list = [sw[:q, :q] for sw in Sigma_W_list]
        Delta_list = [sw[:q, q:] for sw in Sigma_W_list]
        Sigma_V_list = [sw[q:, q:] for sw in Sigma_W_list]

        # Decompose F(k) into blocks A, B, C, D
        A_list = [f[:q, :q] for f in F_list]
        B_list = [f[:q, q:] for f in F_list]
        C_list = [f[q:, :q] for f in F_list]
        D_list = [f[q:, q:] for f in F_list]

        try:
            f_matrix = FMatrix(K, q, s, A_list, B_list, C_list, D_list)
            noise_cov = GSSNoiseCovariance(K, q, s, Sigma_U_list, Delta_list, Sigma_V_list)

            Sigma_z0_list = [np.eye(q + s) for _ in range(K)]

            params = GSSParams(
                K=K,
                q=q,
                s=s,
                P=P,
                f_matrix=f_matrix,
                noise_cov=noise_cov,
                pi0=None,  # stationary
                mu_z0_list=mu_z0_list,  # from GUI
                Sigma_z0_list=Sigma_z0_list,
                b_list=b_list,  # from GUI
            )
        except Exception as exc:  # noqa: BLE001
            self._set_status(f"Parameter error: {exc}", error=True)
            return None

        return params

    def _generate_model_code(self, file_stem: str) -> str:
        """
        Generate a complete Python model file from the current GUI parameters.

        Parameters
        ----------
        file_stem : str
            Stem of the target file (e.g. ``'model_gss_K2_q1_s1_custom'``).
            Used to derive the class name.
        """
        K, q, s = self._K, self._q, self._s

        # Class name: CamelCase from file stem
        class_name = "".join(w.capitalize() for w in file_stem.split("_"))

        # Collect parameters from GUI
        F_list = self._param_panel.get_F_list()
        Sigma_W_list = self._param_panel.get_Sigma_W_list()
        mu_z0_list = self._param_panel.get_mu_z0_list()
        b_list = self._param_panel.get_b_list()
        P = self._p_widget.get_matrix()

        # Decompose blocks
        A_list = [f[:q, :q] for f in F_list]
        B_list = [f[:q, q:] for f in F_list]
        C_list = [f[q:, :q] for f in F_list]
        D_list = [f[q:, q:] for f in F_list]
        Sigma_U_list = [sw[:q, :q] for sw in Sigma_W_list]
        Delta_list = [sw[:q, q:] for sw in Sigma_W_list]
        Sigma_V_list = [sw[q:, q:] for sw in Sigma_W_list]

        def _fmt_arr(arr: np.ndarray) -> str:
            """Format a 2-D numpy array as a compact np.array(...) literal.

            All rows after the first are aligned under the opening ``[``.
            e.g.  np.array([[0.8, 0.1],
                             [0.0, 0.7]])
            """
            prefix = "np.array(["
            align = " " * len(prefix)  # align continuation rows
            rows = []
            for r in range(arr.shape[0]):
                vals = ", ".join(f"{v:.8g}" for v in arr[r])
                rows.append(f"[{vals}]")
            inner = (",\n" + align).join(rows)
            return f"{prefix}{inner}])"

        def _fmt_list(arrays, field_indent: int = 4) -> str:
            """Format a list of arrays with each item on its own line.

            field_indent : spaces before each item (matches class body indent).
            """
            pad = " " * (field_indent + 1)  # one extra to align past '['
            items = [_fmt_arr(a) for a in arrays]
            if len(items) == 1:
                return f"[{items[0]}]"
            joined = (",\n" + pad).join(items)
            return f"[{joined}]"

        lines: list[str] = []
        lines += [
            "#!/usr/bin/env python3",
            "# -*- coding: utf-8 -*-",
            '"""',
            f"prg/models/{file_stem}.py",
            f"{'=' * (len('prg/models/') + len(file_stem) + 3)}",
            f"GSS model: K={K} states, q={q} (hidden), s={s} (observed).",
            "",
            "Generated by exactIMM GUI.",
            '"""',
            "",
            "from __future__ import annotations",
            "",
            "import numpy as np",
            "",
            "from prg.models.base_gss_model import BaseGSSModel",
            "",
            f'__all__ = ["{class_name}"]',
            "",
            "",
            f"class {class_name}(BaseGSSModel):",
            f'    """GSS model exported from the GUI (K={K}, q={q}, s={s})."""',
            "",
            f"    K: int = {K}",
            f"    q: int = {q}",
            f"    s: int = {s}",
            "",
            "    # --- Markov chain ---",
            f"    P: np.ndarray = {_fmt_arr(P)}",
            "",
            "    # --- Dynamics: F(k) = [[A_k, B_k], [C_k, D_k]] ---",
            f"    A_list: list[np.ndarray] = {_fmt_list(A_list)}",
            f"    B_list: list[np.ndarray] = {_fmt_list(B_list)}",
            f"    C_list: list[np.ndarray] = {_fmt_list(C_list)}",
            f"    D_list: list[np.ndarray] = {_fmt_list(D_list)}",
            "",
            "    # --- Noise covariances: Sigma_W(k) = [[Sigma_U, Delta], [Delta^T, Sigma_V]] ---",
            f"    Sigma_U_list: list[np.ndarray] = {_fmt_list(Sigma_U_list)}",
            f"    Delta_list:   list[np.ndarray] = {_fmt_list(Delta_list)}",
            f"    Sigma_V_list: list[np.ndarray] = {_fmt_list(Sigma_V_list)}",
            "",
            "    # --- Initial conditions ---",
            "    pi0: np.ndarray | None = None   # None → stationary distribution",
            "",
            f"    mu_z0_list: list[np.ndarray] = {_fmt_list(mu_z0_list)}",
            f"    Sigma_z0_list: list[np.ndarray] = {_fmt_list([np.eye(q + s)] * K)}",
            f"    b_list: list[np.ndarray] = {_fmt_list(b_list)}",
            "",
            "    # ------------------------------------------------------------------",
            "",
            "    def get_params(self) -> dict:",
            "        return {",
            '            "K": self.K, "q": self.q, "s": self.s, "P": self.P,',
            '            "A_list": self.A_list, "B_list": self.B_list,',
            '            "C_list": self.C_list, "D_list": self.D_list,',
            '            "Sigma_U_list": self.Sigma_U_list,',
            '            "Delta_list": self.Delta_list,',
            '            "Sigma_V_list": self.Sigma_V_list,',
            '            "pi0": self.pi0,',
            '            "mu_z0_list": self.mu_z0_list,',
            '            "Sigma_z0_list": self.Sigma_z0_list,',
            '            "b_list": self.b_list,',
            "        }",
            "",
        ]
        return "\n".join(lines)

    def _load_model(self, model) -> None:
        """Pre-fill tables from a BaseGSSModel instance.

        Signals on ``_param_panel`` and ``_p_widget`` are blocked during the
        bulk update so that validity / value signals don't fire repeatedly for
        each individual state — preventing the double-refresh of the Simulate
        button and the flicker of intermediate validity states (A11 / A16).
        Callers must invoke ``_refresh_simulate_button()`` after this method.
        """
        # A11 / A16: block validity/value signals during bulk loading
        self._param_panel.blockSignals(True)
        self._p_widget.blockSignals(True)
        try:
            p = model.get_params()
            K, q, s = p["K"], p["q"], p["s"]

            # Build block noise cov to get full Σ_W
            nc = GSSNoiseCovariance(K, q, s, p["Sigma_U_list"], p["Delta_list"], p["Sigma_V_list"])
            fm = FMatrix(K, q, s, p["A_list"], p["B_list"], p["C_list"], p["D_list"])

            mu_list = p.get("mu_z0_list")
            b_list = p.get("b_list")
            for k in range(K):
                mu = np.asarray(mu_list[k]) if mu_list is not None else None
                b = np.asarray(b_list[k]) if b_list is not None else None
                self._param_panel.set_state_params(k, fm.F(k), nc.Sigma_W(k), mu, b)

            if p.get("P") is not None:
                self._P = np.asarray(p["P"])
                self._p_widget.set_matrix(self._P)
                self._update_stationary_display()
        finally:
            self._param_panel.blockSignals(False)
            self._p_widget.blockSignals(False)
