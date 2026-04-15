#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prg/gui/param_panel.py
======================
ParamPanel — QTabWidget with one tab per Markov state k.

Each tab (_StateTab) exposes:
  - F(k)     : MatrixTableWidget (no SPD check)
  - Σ_W(k)   : MatrixTableWidget (SPD check enabled)

ParamPanel aggregates validity across all tabs and propagates a
validity_changed signal.
"""

import numpy as np
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTabWidget, QScrollArea,
)

from prg.gui.matrix_widget import MatrixTableWidget, VectorWidget


# ---------------------------------------------------------------------------
# _StateTab
# ---------------------------------------------------------------------------

class _StateTab(QWidget):
    """One tab: F(k) and Σ_W(k) side by side."""

    validity_changed = pyqtSignal(bool)

    def __init__(self, k: int, q: int, s: int, parent=None):
        super().__init__(parent)
        self._k = k
        self._q = q
        self._s = s

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(16)

        # Default F(k): identity (stable system)
        self._f_widget = MatrixTableWidget(
            q, s,
            is_covariance=False,
            title=f"F({k})",
            default_value=0.0,
        )
        # Set default to 0.5 * I so the simulator stays bounded
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

        layout.addWidget(self._f_widget)
        layout.addWidget(self._sigma_widget)
        layout.addWidget(self._mu_widget)
        layout.addWidget(self._bx_widget)
        layout.addWidget(self._by_widget)

        self._f_widget.validity_changed.connect(self._on_child_validity)
        self._sigma_widget.validity_changed.connect(self._on_child_validity)
        self._mu_widget.validity_changed.connect(self._on_child_validity)
        self._bx_widget.validity_changed.connect(self._on_child_validity)
        self._by_widget.validity_changed.connect(self._on_child_validity)

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
