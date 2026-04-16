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
    value_changed    = pyqtSignal()   # fired on every cell edit (valid or not)

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
        self._table.setMinimumSize(self._n * 46, self._n * 26)
        self._table.setMaximumWidth(self._n * 58)
        for col in range(self._n):
            self._table.setColumnWidth(col, 52)
        for row in range(self._n):
            self._table.setRowHeight(row, 24)
        layout.addWidget(self._table)

        if is_covariance:
            self._status_label = QLabel()
            layout.addWidget(self._status_label)
        else:
            self._status_label = None

        # Constraint feedback label (shown below the table when B is auto-computed)
        self._constraint_label = QLabel("")
        self._constraint_label.setStyleSheet("font-size: 10px;")
        self._constraint_label.setVisible(False)
        layout.addWidget(self._constraint_label)

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
                    item.setBackground(QBrush(QColor(self._cell_bg(r, c))))
                    item.setForeground(QBrush(QColor("black")))
                    self._table.setItem(r, c, item)
        finally:
            self._building = False
        self._validate_all()

    def is_valid(self) -> bool:
        return self._valid

    def set_block_editable(
        self,
        row_start: int, row_end: int,
        col_start: int, col_end: int,
        editable: bool,
    ) -> None:
        """Make a rectangular block of cells editable (True) or read-only (False).

        Read-only cells receive a saturated version of their block colour to
        signal that the value is auto-computed.
        """
        for r in range(row_start, row_end):
            for c in range(col_start, col_end):
                item = self._table.item(r, c)
                if item is None:
                    continue
                flags = item.flags()
                if editable:
                    item.setFlags(flags | Qt.ItemFlag.ItemIsEditable)
                    item.setBackground(QBrush(QColor(self._cell_bg(r, c))))
                else:
                    item.setFlags(flags & ~Qt.ItemFlag.ItemIsEditable)
                    item.setBackground(QBrush(QColor(self._cell_computed_bg(r, c))))

    def set_constraint_status(self, text: str, style: str = "") -> None:
        """Show or hide the constraint feedback label below the table.

        Pass an empty string to hide it.
        """
        if text:
            self._constraint_label.setText(text)
            if style:
                self._constraint_label.setStyleSheet(style)
            self._constraint_label.setVisible(True)
        else:
            self._constraint_label.setVisible(False)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    #  F(k) block layout:          Σ_W(k) block layout:
    #  ┌──────────┬──────────┐     ┌──────────┬──────────┐
    #  │  A  blue │  B green │     │  Σ_U blue│  Δ green │
    #  ├──────────┼──────────┤     ├──────────┼──────────┤
    #  │  C  yell │  D  pink │     │  Δᵀ yell │  Σ_V pink│
    #  └──────────┴──────────┘     └──────────┴──────────┘
    _BLOCK_BG: dict[tuple[bool, bool], str] = {
        (True,  True):  "#d6eaf8",   # top-left  — blue
        (True,  False): "#d5f5e3",   # top-right — green  (B / Δ)
        (False, True):  "#fef9e7",   # bot-left  — yellow (C / Δᵀ)
        (False, False): "#fde8e8",   # bot-right — pink   (D / Σ_V)
    }
    # Saturated version: cell is auto-computed (read-only)
    _BLOCK_COMPUTED_BG: dict[tuple[bool, bool], str] = {
        (True,  True):  "#aed6f1",   # blue  — computed
        (True,  False): "#a9dfbf",   # green — computed B
        (False, True):  "#f9e79f",   # yellow — computed
        (False, False): "#f1948a",   # pink  — computed
    }

    def _cell_bg(self, r: int, c: int) -> str:
        """Normal background colour for cell (r, c)."""
        if self._is_covariance:
            return "white"
        return self._BLOCK_BG[(r < self._q, c < self._q)]

    def _cell_computed_bg(self, r: int, c: int) -> str:
        """Saturated background colour for a locked / auto-computed cell."""
        if self._is_covariance:
            return "#ddeeff"
        return self._BLOCK_COMPUTED_BG[(r < self._q, c < self._q)]

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
        self.value_changed.emit()

    def _validate_cell(self, item: QTableWidgetItem) -> bool:
        """Colour the cell red if the text is not a valid float. Return success."""
        r, c = item.row(), item.column()
        try:
            float(item.text())
            # Restore block colour only when not locked (editable flag present)
            if item.flags() & Qt.ItemFlag.ItemIsEditable:
                item.setBackground(QBrush(QColor(self._cell_bg(r, c))))
            item.setForeground(QBrush(QColor("black")))
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
            self._set_valid(True, "Positive definite")
        else:
            reason = next(
                (c.message for c in report.checks if c.status.name == "FAIL"),
                "Not positive definite",
            )
            self._set_valid(False, reason)

    def _set_valid(self, valid: bool, reason: str) -> None:
        if self._status_label is not None:
            if valid:
                self._status_label.setText("✓ Positive definite")
                self._status_label.setStyleSheet("color: #007700; font-size: 10px;")
            elif reason:
                self._status_label.setText(f"✗ {reason}")
                self._status_label.setStyleSheet("color: #cc0000; font-size: 10px;")
            else:
                self._status_label.setText("✗ Invalid value(s)")
                self._status_label.setStyleSheet("color: #cc0000; font-size: 10px;")

        if valid != self._valid:
            self._valid = valid
            self.validity_changed.emit(valid)
        else:
            self._valid = valid


# ---------------------------------------------------------------------------
# StochasticMatrixWidget
# ---------------------------------------------------------------------------

class StochasticMatrixWidget(QWidget):
    """
    Editable K × K row-stochastic matrix.

    Validation
    ----------
    - Every cell must be a non-negative float.
    - Every row must sum to 1 (tolerance 1 e-6).

    A status label below the table reports the first offending row.
    """

    validity_changed = pyqtSignal(bool)

    def __init__(
        self,
        K: int,
        *,
        title: str = "P",
        parent=None,
    ):
        super().__init__(parent)
        self._K = K
        self._building = False
        self._valid = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        if title:
            lbl = QLabel(title)
            lbl.setStyleSheet("font-weight: bold;")
            layout.addWidget(lbl)

        self._table = QTableWidget(K, K)
        self._table.horizontalHeader().setVisible(False)
        self._table.verticalHeader().setVisible(False)
        self._table.setMinimumSize(K * 46, K * 26)
        self._table.setMaximumSize(K * 64, K * 26 + 4)   # hauteur calée sur les lignes
        for col in range(K):
            self._table.setColumnWidth(col, 56)
        for row in range(K):
            self._table.setRowHeight(row, 24)
        layout.addWidget(self._table)

        self._status_label = QLabel()
        self._status_label.setStyleSheet("font-size: 10px;")
        self._status_label.setFixedHeight(16)
        layout.addWidget(self._status_label)

        # Ne pas s'étirer verticalement au-delà du contenu
        from PyQt6.QtWidgets import QSizePolicy as _SP
        self.setSizePolicy(_SP.Policy.Preferred, _SP.Policy.Fixed)

        # Default: uniform 1/K
        self.set_matrix(np.full((K, K), 1.0 / K))
        self._table.itemChanged.connect(self._on_item_changed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_matrix(self) -> np.ndarray | None:
        """Return (K, K) array or None if invalid."""
        mat = np.zeros((self._K, self._K))
        for r in range(self._K):
            for c in range(self._K):
                item = self._table.item(r, c)
                if item is None:
                    return None
                try:
                    mat[r, c] = float(item.text())
                except ValueError:
                    return None
        return mat

    def set_matrix(self, mat: np.ndarray) -> None:
        """Fill the table from a (K, K) array."""
        self._building = True
        try:
            for r in range(self._K):
                for c in range(self._K):
                    val = float(mat[r, c]) if r < mat.shape[0] and c < mat.shape[1] else 0.0
                    item = QTableWidgetItem(f"{val:.6g}")
                    item.setTextAlignment(
                        Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter
                    )
                    item.setBackground(QBrush(QColor("white")))
                    item.setForeground(QBrush(QColor("black")))
                    self._table.setItem(r, c, item)
        finally:
            self._building = False
        self._validate_all()

    def is_valid(self) -> bool:
        return self._valid

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        if self._building:
            return
        self._validate_cell(item)
        self._validate_all()

    def _validate_cell(self, item: QTableWidgetItem) -> bool:
        try:
            v = float(item.text())
            ok = v >= 0.0
        except ValueError:
            ok = False
        item.setBackground(QBrush(QColor("white") if ok else _COLOUR_BAD))
        item.setForeground(QBrush(QColor("black")))
        return ok

    def _validate_all(self) -> None:
        # 1) all cells parseable and ≥ 0
        for r in range(self._K):
            for c in range(self._K):
                item = self._table.item(r, c)
                if item is None or not self._validate_cell(item):
                    self._set_valid(False, "Invalid or negative value(s)")
                    return

        mat = self.get_matrix()
        if mat is None:
            self._set_valid(False, "")
            return

        # 2) row sums ≈ 1
        for r in range(self._K):
            row_sum = mat[r].sum()
            if abs(row_sum - 1.0) > 1e-6:
                self._set_valid(False, f"Row {r}: sum = {row_sum:.4g} ≠ 1")
                return

        self._set_valid(True, "")

    def _set_valid(self, valid: bool, reason: str) -> None:
        if valid:
            self._status_label.setText("✓ Row-stochastic")
            self._status_label.setStyleSheet("color: #007700; font-size: 10px;")
        else:
            self._status_label.setText(f"✗ {reason}" if reason else "✗ Not row-stochastic")
            self._status_label.setStyleSheet("color: #cc0000; font-size: 10px;")
        if valid != self._valid:
            self._valid = valid
            self.validity_changed.emit(valid)
        else:
            self._valid = valid


# ---------------------------------------------------------------------------
# VectorWidget
# ---------------------------------------------------------------------------

class VectorWidget(QWidget):
    """
    Editable column vector of length n.

    Displays an n × 1 table with float validation (red cell on error).

    Parameters
    ----------
    n : int
        Length of the vector (number of rows).
    title : str
        Optional bold label displayed above.
    default_value : float
        Initial value for every component (default 0.0).
    """

    validity_changed = pyqtSignal(bool)

    def __init__(
        self,
        n: int,
        *,
        title: str = "",
        default_value: float = 0.0,
        parent=None,
    ):
        super().__init__(parent)
        self._n = n
        self._building = False
        self._valid = True

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        if title:
            lbl = QLabel(title)
            lbl.setStyleSheet("font-weight: bold;")
            layout.addWidget(lbl)

        self._table = QTableWidget(n, 1)
        self._table.horizontalHeader().setVisible(False)
        self._table.verticalHeader().setVisible(False)
        self._table.setMinimumSize(56, n * 26)
        self._table.setMaximumWidth(70)
        self._table.setColumnWidth(0, 60)
        for row in range(n):
            self._table.setRowHeight(row, 24)
        layout.addWidget(self._table)

        self._populate(default_value)
        self._table.itemChanged.connect(self._on_item_changed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_vector(self) -> np.ndarray | None:
        """Return shape (n, 1) array, or None if any cell is invalid."""
        vec = np.zeros((self._n, 1))
        for r in range(self._n):
            item = self._table.item(r, 0)
            if item is None:
                return None
            try:
                vec[r, 0] = float(item.text())
            except ValueError:
                return None
        return vec

    def set_vector(self, vec: np.ndarray) -> None:
        """Fill from an array of shape (n,) or (n, 1)."""
        flat = np.asarray(vec).ravel()
        self._building = True
        try:
            for r in range(self._n):
                val = flat[r] if r < len(flat) else 0.0
                item = QTableWidgetItem(f"{val:.6g}")
                item.setTextAlignment(
                    Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter
                )
                item.setBackground(QBrush(QColor("white")))
                item.setForeground(QBrush(QColor("black")))
                self._table.setItem(r, 0, item)
        finally:
            self._building = False
        self._validate_all()

    def is_valid(self) -> bool:
        return self._valid

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _populate(self, default_value: float) -> None:
        self.set_vector(np.full(self._n, default_value))

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        if self._building:
            return
        self._validate_cell(item)
        self._validate_all()

    def _validate_cell(self, item: QTableWidgetItem) -> bool:
        try:
            float(item.text())
            item.setBackground(QBrush(QColor("white")))
            item.setForeground(QBrush(QColor("black")))
            return True
        except ValueError:
            item.setBackground(QBrush(_COLOUR_BAD))
            return False

    def _validate_all(self) -> None:
        all_float = all(
            self._validate_cell(self._table.item(r, 0))
            for r in range(self._n)
            if self._table.item(r, 0) is not None
        )
        valid = all_float
        if valid != self._valid:
            self._valid = valid
            self.validity_changed.emit(valid)
        else:
            self._valid = valid
