#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prg/gui/matrix_widget.py
========================
MatrixTableWidget — editable QTableWidget for a (q+s) × (q+s) matrix with:
  - Per-cell float validation (red background on parse error)
  - Optional SPD validation for covariance matrices
  - validity_changed signal
"""

import numpy as np
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QColor, QBrush
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QTableWidget, QTableWidgetItem,
)

from prg.utils.matrix_checks import CovarianceMatrix


_COLOUR_BAD = QColor("#ff8888")  # invalid cell


# ---------------------------------------------------------------------------
# MatrixTableWidget
# ---------------------------------------------------------------------------

class MatrixTableWidget(QWidget):
    """
    Editable (q+s) × (q+s) table.

    Parameters
    ----------
    q, s:
        Dimensions of the hidden / observed parts.
    is_covariance:
        If True, validate the full matrix for SPD and show a status label.
    title:
        Optional label displayed above the table.
    default_value:
        Float used to fill every cell on construction.
    """

    validity_changed = pyqtSignal(bool)

    def __init__(
        self,
        q: int,
        s: int,
        *,
        is_covariance: bool = False,
        title: str = "",
        default_value: float = 0.0,
        parent=None,
    ):
        super().__init__(parent)
        self._q = q
        self._s = s
        self._n = q + s
        self._is_covariance = is_covariance
        self._building = False          # guard against recursive itemChanged
        self._valid = True

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        if title:
            lbl = QLabel(title)
            lbl.setStyleSheet("font-weight: bold;")
            layout.addWidget(lbl)

        self._table = QTableWidget(self._n, self._n)
        self._table.horizontalHeader().setVisible(False)
        self._table.verticalHeader().setVisible(False)
        self._table.setMinimumSize(self._n * 60, self._n * 32)
        layout.addWidget(self._table)

        if is_covariance:
            self._status_label = QLabel()
            layout.addWidget(self._status_label)
        else:
            self._status_label = None

        self._populate(default_value)
        self._table.itemChanged.connect(self._on_item_changed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_matrix(self) -> np.ndarray | None:
        """Return the current (n×n) numpy array, or None if any cell is invalid."""
        mat = np.zeros((self._n, self._n))
        for r in range(self._n):
            for c in range(self._n):
                item = self._table.item(r, c)
                if item is None:
                    return None
                try:
                    mat[r, c] = float(item.text())
                except ValueError:
                    return None
        return mat

    def set_matrix(self, mat: np.ndarray) -> None:
        """Fill the table from a numpy array (silent, no signal spam)."""
        self._building = True
        try:
            for r in range(self._n):
                for c in range(self._n):
                    item = QTableWidgetItem(f"{mat[r, c]:.6g}")
                    item.setTextAlignment(
                        Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter
                    )
                    item.setBackground(QBrush(QColor("white")))
                    self._table.setItem(r, c, item)
        finally:
            self._building = False
        self._validate_all()

    def is_valid(self) -> bool:
        return self._valid

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _populate(self, default_value: float) -> None:
        """Initial fill: identity×default_value for covariance, else default_value."""
        if self._is_covariance:
            mat = np.eye(self._n) * default_value
        else:
            mat = np.full((self._n, self._n), default_value)
        self.set_matrix(mat)

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        if self._building:
            return
        self._validate_cell(item)
        self._validate_all()

    def _validate_cell(self, item: QTableWidgetItem) -> bool:
        """Colour the cell red if the text is not a valid float. Return success."""
        r, c = item.row(), item.column()
        try:
            float(item.text())
            item.setBackground(QBrush(QColor("white")))
            return True
        except ValueError:
            item.setBackground(QBrush(_COLOUR_BAD))
            return False

    def _validate_all(self) -> None:
        """Re-check all cells; if covariance, also check SPD. Emit if changed."""
        # Check all cells parseable
        all_float = True
        for r in range(self._n):
            for c in range(self._n):
                item = self._table.item(r, c)
                if item is None or not self._validate_cell(item):
                    all_float = False

        if not all_float:
            self._set_valid(False, "")
            return

        if not self._is_covariance:
            self._set_valid(True, "")
            return

        # SPD check
        mat = self.get_matrix()
        if mat is None:
            self._set_valid(False, "")
            return

        report = CovarianceMatrix(mat).check()
        if report.is_valid:
            self._set_valid(True, "Définie positive")
        else:
            reason = next(
                (r.message for r in report.results if r.status.name == "FAIL"),
                "Non définie positive",
            )
            self._set_valid(False, reason)

    def _set_valid(self, valid: bool, reason: str) -> None:
        if self._status_label is not None:
            if valid:
                self._status_label.setText("✓ Définie positive")
                self._status_label.setStyleSheet("color: #007700; font-size: 10px;")
            elif reason:
                self._status_label.setText(f"✗ {reason}")
                self._status_label.setStyleSheet("color: #cc0000; font-size: 10px;")
            else:
                self._status_label.setText("✗ Valeur(s) invalide(s)")
                self._status_label.setStyleSheet("color: #cc0000; font-size: 10px;")

        if valid != self._valid:
            self._valid = valid
            self.validity_changed.emit(valid)
        else:
            self._valid = valid
