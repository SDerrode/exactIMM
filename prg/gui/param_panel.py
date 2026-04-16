#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prg/gui/param_panel.py
======================
ParamPanel — QTabWidget with one tab per Markov state k.

Each tab (_StateTab) exposes:
  - F(k)     : MatrixTableWidget (no SPD check)
  - Σ_W(k)   : MatrixTableWidget (SPD check enabled)

Constraint checkbox (4.7)
--------------------------
Each tab has an optional checkbox "Calculer B(k) automatiquement".
When checked, the B block of F(k) is recomputed in real-time from the
constraint (4.7) — i.e. B is the unique solution of:

    (Σ_V − P M⁻¹ R) Bᵀ = P M⁻¹ (Q Aᵀ + Δᵀ) − Δᵀ A

with  P = Δᵀ Cᵀ + Σ_V Dᵀ,  Q = C Σ_U + D Δᵀ,
      R = C Δ + D Σ_V,       M = Q Cᵀ + R Dᵀ + Σ_V.

The B cells are then rendered read-only (light-blue tint).

ParamPanel aggregates validity across all tabs and propagates a
validity_changed signal.
"""

import numpy as np
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTabWidget, QScrollArea,
    QCheckBox,
)

from prg.gui.matrix_widget import MatrixTableWidget, VectorWidget
from prg.utils.h5_constraint import compute_B_from_h5


# ---------------------------------------------------------------------------
# _StateTab
# ---------------------------------------------------------------------------

class _StateTab(QWidget):
    """One tab: F(k), Σ_W(k), μ_z0(k), b_X(k), b_Y(k) side by side,
    plus an optional H5-constraint checkbox that auto-computes B(k)."""

    validity_changed = pyqtSignal(bool)

    def __init__(self, k: int, q: int, s: int, parent=None):
        super().__init__(parent)
        self._k = k
        self._q = q
        self._s = s
        self._updating_B = False   # re-entrancy guard for _recompute_B

        # ── Main layout: checkbox row on top, matrix widgets below ──────
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # -- Constraint checkbox row --
        # Styled in B-block green (#d5f5e3 / #1a7a3a) to recall the B column
        chk_row = QHBoxLayout()
        chk_row.setSpacing(12)
        self._constraint_check = QCheckBox(f"Constraint on F({k})")
        self._constraint_check.setStyleSheet(
            "QCheckBox { color: #1a7a3a; font-weight: bold; font-size: 11px; }"
            "QCheckBox::indicator:checked   { border: 2px solid #1a7a3a;"
            "                                 background-color: #27ae60; }"
            "QCheckBox::indicator:unchecked { border: 2px solid #1a7a3a; }"
        )
        chk_row.addWidget(self._constraint_check)
        chk_row.addStretch()
        layout.addLayout(chk_row)

        # -- Matrix/vector widgets row --
        widgets_row = QHBoxLayout()
        widgets_row.setSpacing(16)

        # Default F(k): 0.5 * I (stable system)
        self._f_widget = MatrixTableWidget(
            q, s,
            is_covariance=False,
            title=f"F({k})",
            default_value=0.0,
        )
        self._f_widget.set_matrix(np.eye(q + s) * 0.5)

        # Default Σ_W(k): 0.1 * I (SPD)
        self._sigma_widget = MatrixTableWidget(
            q, s,
            is_covariance=True,
            title=f"Σ_W({k})",
            default_value=0.1,
        )

        # Default μ_z0(k): zero vector (q+s components)
        self._mu_widget = VectorWidget(
            q + s,
            title=f"μ_z0({k})",
            default_value=0.0,
        )

        # Drift bias b(k) split into X part (q) and Y part (s)
        self._bx_widget = VectorWidget(
            q,
            title=f"b_X({k})",
            default_value=0.0,
        )
        self._by_widget = VectorWidget(
            s,
            title=f"b_Y({k})",
            default_value=0.0,
        )

        widgets_row.addWidget(self._f_widget)
        widgets_row.addWidget(self._sigma_widget)
        widgets_row.addWidget(self._mu_widget)
        widgets_row.addWidget(self._bx_widget)
        widgets_row.addWidget(self._by_widget)
        layout.addLayout(widgets_row)

        # ── Signal connections ──────────────────────────────────────────
        self._f_widget.validity_changed.connect(self._on_child_validity)
        self._sigma_widget.validity_changed.connect(self._on_child_validity)
        self._mu_widget.validity_changed.connect(self._on_child_validity)
        self._bx_widget.validity_changed.connect(self._on_child_validity)
        self._by_widget.validity_changed.connect(self._on_child_validity)

        # Real-time B recomputation on any value change
        self._f_widget.value_changed.connect(self._recompute_B)
        self._sigma_widget.value_changed.connect(self._recompute_B)

        # Toggle checkbox
        self._constraint_check.toggled.connect(self._on_constraint_toggled)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def get_F(self) -> np.ndarray | None:
        return self._f_widget.get_matrix()

    def get_Sigma_W(self) -> np.ndarray | None:
        return self._sigma_widget.get_matrix()

    def get_mu_z0(self) -> np.ndarray | None:
        return self._mu_widget.get_vector()

    def get_b(self) -> np.ndarray | None:
        """Return full bias vector (q+s, 1) = [b_X; b_Y], or None if invalid."""
        bx = self._bx_widget.get_vector()   # (q, 1)
        by = self._by_widget.get_vector()   # (s, 1)
        if bx is None or by is None:
            return None
        return np.vstack([bx, by])

    def set_F(self, mat: np.ndarray) -> None:
        self._f_widget.set_matrix(mat)

    def set_Sigma_W(self, mat: np.ndarray) -> None:
        self._sigma_widget.set_matrix(mat)

    def set_mu_z0(self, vec: np.ndarray) -> None:
        self._mu_widget.set_vector(vec)

    def set_b(self, vec: np.ndarray) -> None:
        """Accept full (q+s, 1) vector and split into b_X / b_Y."""
        flat = np.asarray(vec).ravel()
        self._bx_widget.set_vector(flat[:self._q])
        self._by_widget.set_vector(flat[self._q:])

    def is_valid(self) -> bool:
        return (
            self._f_widget.is_valid()
            and self._sigma_widget.is_valid()
            and self._mu_widget.is_valid()
            and self._bx_widget.is_valid()
            and self._by_widget.is_valid()
        )

    # ------------------------------------------------------------------
    # Constraint (4.7) — auto-compute B
    # ------------------------------------------------------------------

    def _on_constraint_toggled(self, checked: bool) -> None:
        """Enable / disable the H5 constraint mode."""
        if checked:
            self._recompute_B()
        else:
            # Restore B block to fully editable and hide status label
            q, s = self._q, self._s
            self._f_widget.set_block_editable(0, q, q, q + s, True)
            self._f_widget.set_constraint_status("")

    def _recompute_B(self) -> None:
        """Solve for B(k) in real-time and overwrite it in F(k)."""
        if not self._constraint_check.isChecked() or self._updating_B:
            return
        F  = self._f_widget.get_matrix()
        Sw = self._sigma_widget.get_matrix()
        if F is None or Sw is None:
            return

        B_new = self._compute_B_from_constraint(F, Sw)

        if B_new is None:
            self._f_widget.set_constraint_status(
                "✗  B — singular system",
                "color: #cc0000; font-size: 10px;",
            )
            return

        q, s = self._q, self._s
        new_F = F.copy()
        new_F[:q, q:] = B_new

        self._updating_B = True
        self._f_widget.set_matrix(new_F)
        # Re-apply computed tint on the B block (set_matrix resets all cells)
        self._f_widget.set_block_editable(0, q, q, q + s, False)
        self._updating_B = False

        self._f_widget.set_constraint_status(
            "✓  B satisfies constraint (4.7)",
            "color: #1a7a3a; font-size: 10px;",
        )

    def _compute_B_from_constraint(
        self, F: np.ndarray, Sw: np.ndarray
    ) -> np.ndarray | None:
        """Return B (q×s) from the H5 constraint, or None if the system is singular."""
        q = self._q
        try:
            return compute_B_from_h5(
                A=F[:q, :q], C=F[q:, :q], D=F[q:, q:],
                SU=Sw[:q, :q], Dt=Sw[:q, q:], SV=Sw[q:, q:],
            )
        except ValueError:
            return None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _on_child_validity(self, _: bool) -> None:
        self.validity_changed.emit(self.is_valid())


# ---------------------------------------------------------------------------
# ParamPanel
# ---------------------------------------------------------------------------

class ParamPanel(QWidget):
    """
    QTabWidget with K tabs (one per Markov state).

    Signals
    -------
    validity_changed(bool)
        Emitted whenever the overall validity changes.
    """

    validity_changed = pyqtSignal(bool)

    def __init__(self, K: int, q: int, s: int, parent=None):
        super().__init__(parent)
        self._K = K
        self._q = q
        self._s = s

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        info = QLabel(
            f"K = {K} states,  q = {q} (hidden),  s = {s} (observed)"
        )
        info.setStyleSheet("font-size: 11px; padding: 2px 4px;")
        layout.addWidget(info)

        self._tabs = QTabWidget()
        layout.addWidget(self._tabs)

        self._state_tabs: list[_StateTab] = []
        for k in range(K):
            tab = _StateTab(k, q, s)
            tab.validity_changed.connect(self._on_tab_validity)

            scroll = QScrollArea()
            scroll.setWidget(tab)
            scroll.setWidgetResizable(True)
            self._tabs.addTab(scroll, f"State {k}")
            self._state_tabs.append(tab)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def get_F_list(self) -> list[np.ndarray] | None:
        """Return list of K full F(k) matrices, or None if any invalid."""
        result = []
        for tab in self._state_tabs:
            mat = tab.get_F()
            if mat is None:
                return None
            result.append(mat)
        return result

    def get_Sigma_W_list(self) -> list[np.ndarray] | None:
        """Return list of K Σ_W(k) matrices, or None if any invalid."""
        result = []
        for tab in self._state_tabs:
            mat = tab.get_Sigma_W()
            if mat is None:
                return None
            result.append(mat)
        return result

    def get_mu_z0_list(self) -> list[np.ndarray] | None:
        """Return list of K μ_z0(k) column vectors (q+s,1), or None if any invalid."""
        result = []
        for tab in self._state_tabs:
            vec = tab.get_mu_z0()
            if vec is None:
                return None
            result.append(vec)
        return result

    def get_b_list(self) -> list[np.ndarray] | None:
        """Return list of K b(k) drift bias vectors (q+s,1), or None if any invalid."""
        result = []
        for tab in self._state_tabs:
            vec = tab.get_b()
            if vec is None:
                return None
            result.append(vec)
        return result

    def set_state_params(
        self,
        k: int,
        F: np.ndarray,
        Sigma_W: np.ndarray,
        mu_z0: np.ndarray | None = None,
        b: np.ndarray | None = None,
    ) -> None:
        """Load pre-built matrices/vector into tab k."""
        self._state_tabs[k].set_F(F)
        self._state_tabs[k].set_Sigma_W(Sigma_W)
        if mu_z0 is not None:
            self._state_tabs[k].set_mu_z0(mu_z0)
        if b is not None:
            self._state_tabs[k].set_b(b)

    def is_valid(self) -> bool:
        return all(tab.is_valid() for tab in self._state_tabs)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _on_tab_validity(self, _: bool) -> None:
        self.validity_changed.emit(self.is_valid())
