#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prg/gui/param_panel.py
======================
ParamPanel — QTabWidget with one tab per Markov state k.

Each tab (_StateTab) exposes:
  - F(k)     : MatrixTableWidget (no SPD check)
  - Σ_W(k)   : MatrixTableWidget (SPD check enabled)

Constraint checkboxes (eq. 4.8 / 4.20)
----------------------------------------
The constraint is a single equation in A, B, C, D, Σ_U, Δ, Σ_V.
It can be solved for exactly one of {A, B, C, Σ_U} at a time — those four
are therefore mutually exclusive.

Δ = 0 is an independent constraint (it zeros the off-diagonal block of
Σ_W) and can coexist with any of {A, B, C, Σ_U}.  When Δ changes state,
the active H5 constraint is automatically re-evaluated with the new Δ.

Valid combinations:
  □ A alone  □ B alone  □ C alone  □ Σ_U alone
  □ Δ=0 alone  □ Δ=0 + A  □ Δ=0 + B  □ Δ=0 + C  □ Δ=0 + Σ_U

  □ Constraint on A(k)   — A determined by B, C, D, Σ_U, Δ, Σ_V  (eq. 4.8)
  □ Constraint on B(k)   — B determined by A, C, D, Σ_U, Δ, Σ_V  (eq. 4.8)
  □ Constraint on C(k)   — C determined by A, B, D, Σ_U, Δ, Σ_V  (eq. 4.20, iterative)
  □ Constraint on Σ_U(k) — Σ_U determined by A, B, C, D, Δ, Σ_V  (eq. 4.8)
  □ Δ = 0(k)             — off-diagonal block of Σ_W forced to zero (independent)

Stability indicators
--------------------
Two colour-coded badges are always visible in the checkbox row and update
in real-time as the user edits F(k):
  • ρ(A(k)) — spectral radius of the hidden-to-hidden block A
  • ρ(D(k)) — spectral radius of the observation-to-observation block D

Colour coding:  green ρ < 0.90 ✓  |  amber 0.90 ≤ ρ < 1.00 ~  |  red ρ ≥ 1.00 ✗

ParamPanel aggregates validity across all tabs and propagates a
validity_changed signal.
"""

import numpy as np
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTabWidget, QScrollArea,
    QCheckBox, QPushButton,
)

from prg.gui.matrix_widget import MatrixTableWidget, VectorWidget
from prg.utils.h5_constraint import (
    compute_A_from_h5, compute_B_from_h5, compute_SU_from_h5, compute_C_from_h5,
)


# ---------------------------------------------------------------------------
# _StateTab
# ---------------------------------------------------------------------------

class _StateTab(QWidget):
    """One tab: F(k), Σ_W(k), μ_z0(k), b_X(k), b_Y(k) side by side,
    plus four mutually exclusive constraint checkboxes and a stability badge."""

    validity_changed  = pyqtSignal(bool)
    value_changed     = pyqtSignal()    # emitted whenever any cell changes
    constraint_toggled = pyqtSignal()   # emitted when any checkbox is checked

    # Checkbox colour palette  (text-color, checked-fill, border)
    _CHK_STYLES = {
        "A":  ("QCheckBox { color: #155399; font-weight: bold; font-size: 11px; }"
               "QCheckBox::indicator:checked   { border: 2px solid #155399;"
               "                                 background-color: #2980b9; }"
               "QCheckBox::indicator:unchecked { border: 2px solid #155399; }"),
        "B":  ("QCheckBox { color: #1a7a3a; font-weight: bold; font-size: 11px; }"
               "QCheckBox::indicator:checked   { border: 2px solid #1a7a3a;"
               "                                 background-color: #27ae60; }"
               "QCheckBox::indicator:unchecked { border: 2px solid #1a7a3a; }"),
        "SU": ("QCheckBox { color: #6c3483; font-weight: bold; font-size: 11px; }"
               "QCheckBox::indicator:checked   { border: 2px solid #6c3483;"
               "                                 background-color: #8e44ad; }"
               "QCheckBox::indicator:unchecked { border: 2px solid #6c3483; }"),
        "C":  ("QCheckBox { color: #a04000; font-weight: bold; font-size: 11px; }"
               "QCheckBox::indicator:checked   { border: 2px solid #a04000;"
               "                                 background-color: #e67e22; }"
               "QCheckBox::indicator:unchecked { border: 2px solid #a04000; }"),
        "delta": ("QCheckBox { color: #0e6655; font-weight: bold; font-size: 11px; }"
                  "QCheckBox::indicator:checked   { border: 2px solid #0e6655;"
                  "                                 background-color: #1abc9c; }"
                  "QCheckBox::indicator:unchecked { border: 2px solid #0e6655; }"),
    }

    def __init__(self, k: int, q: int, s: int, parent=None):
        super().__init__(parent)
        self._k = k
        self._q = q
        self._s = s
        self._updating = False              # re-entrancy guard

        self._saved_A:     np.ndarray | None = None  # for H5 constraint on A
        self._saved_B:     np.ndarray | None = None  # for H5 constraint on B
        self._saved_C:     np.ndarray | None = None  # for H5 constraint on C
        self._saved_SU:    np.ndarray | None = None  # for H5 constraint on Σ_U
        self._saved_Delta: np.ndarray | None = None  # for Δ = 0 constraint

        # ── Main layout: checkbox row on top, matrix widgets below ──────
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # -- Constraint checkbox row (mutually exclusive) + stability badge --
        chk_row = QHBoxLayout()
        chk_row.setSpacing(16)

        self._constraint_A_check     = QCheckBox(f"Constraint on A({k})")
        self._constraint_B_check     = QCheckBox(f"Constraint on B({k})")
        self._constraint_C_check     = QCheckBox(f"Constraint on C({k})")
        self._constraint_SU_check    = QCheckBox(f"Constraint on Σ_U({k})")
        self._constraint_delta_check = QCheckBox("Δ = 0")

        for key, chk in [("A",     self._constraint_A_check),
                          ("B",     self._constraint_B_check),
                          ("C",     self._constraint_C_check),
                          ("SU",    self._constraint_SU_check),
                          ("delta", self._constraint_delta_check)]:
            chk.setStyleSheet(self._CHK_STYLES[key])
            chk_row.addWidget(chk)

        chk_row.addStretch()
        btn_rand = QPushButton("🎲 Randomize")
        btn_rand.setFixedHeight(24)
        btn_rand.setToolTip(
            "Fill F(k) and Σ_W(k) with random stable parameters."
        )
        btn_rand.clicked.connect(self._randomize)
        chk_row.addWidget(btn_rand)
        layout.addLayout(chk_row)

        # -- Matrix/vector widgets row --
        widgets_row = QHBoxLayout()
        widgets_row.setSpacing(16)

        self._f_widget = MatrixTableWidget(
            q, s, is_covariance=False,
            title=f"<i>F</i>({k})", default_value=0.0,
        )
        self._f_widget.set_matrix(np.eye(q + s) * 0.5)

        # Stability badges live in a sub-column directly below _f_widget
        self._stab_F_badge = QLabel()
        self._stab_F_badge.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self._stab_F_badge.setFixedHeight(16)
        self._stab_A_badge = QLabel()
        self._stab_A_badge.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self._stab_A_badge.setFixedHeight(16)
        self._stab_D_badge = QLabel()
        self._stab_D_badge.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self._stab_D_badge.setFixedHeight(16)

        f_col = QVBoxLayout()
        f_col.setSpacing(2)
        f_col.setContentsMargins(0, 0, 0, 0)
        f_col.addWidget(self._f_widget)
        f_col.addWidget(self._stab_F_badge)
        f_col.addWidget(self._stab_A_badge)
        f_col.addWidget(self._stab_D_badge)
        f_col.addStretch()
        widgets_row.addLayout(f_col)

        self._sigma_widget = MatrixTableWidget(
            q, s, is_covariance=True,
            title=f"Σ<sub>W</sub>({k})", default_value=0.1,
        )

        self._mu_widget = VectorWidget(
            q + s, title=f"μ<sub>z₀</sub>({k})", default_value=0.0,
        )
        self._bx_widget = VectorWidget(
            q, title=f"<i>b</i><sub>X</sub>({k})", default_value=0.0,
        )
        self._by_widget = VectorWidget(
            s, title=f"<i>b</i><sub>Y</sub>({k})", default_value=0.0,
        )

        for w in (self._sigma_widget,
                  self._mu_widget, self._bx_widget, self._by_widget):
            _col = QVBoxLayout()
            _col.setContentsMargins(0, 0, 0, 0)
            _col.setSpacing(2)
            _col.addWidget(w)
            _col.addStretch()
            widgets_row.addLayout(_col)
        widgets_row.addStretch()          # prevent horizontal expansion
        layout.addLayout(widgets_row)
        layout.addStretch()               # push everything to the top

        # ── Signal connections ──────────────────────────────────────────
        for w in (self._f_widget, self._sigma_widget,
                  self._mu_widget, self._bx_widget, self._by_widget):
            w.validity_changed.connect(self._on_child_validity)
            w.value_changed.connect(self.value_changed)   # forward upward

        self._f_widget.value_changed.connect(self._on_value_changed)
        self._sigma_widget.value_changed.connect(self._on_value_changed)

        self._constraint_A_check.toggled.connect(self._on_A_toggled)
        self._constraint_B_check.toggled.connect(self._on_B_toggled)
        self._constraint_C_check.toggled.connect(self._on_C_toggled)
        self._constraint_SU_check.toggled.connect(self._on_SU_toggled)
        self._constraint_delta_check.toggled.connect(self._on_delta_toggled)

        # Initialise the badges with the default matrix
        self._update_stability_badges()

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
        bx = self._bx_widget.get_vector()
        by = self._by_widget.get_vector()
        if bx is None or by is None:
            return None
        return np.vstack([bx, by])

    def set_F(self, mat: np.ndarray) -> None:
        self._f_widget.set_matrix(mat)
        self._update_stability_badges()

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
    # Constraint toggle handlers
    # ------------------------------------------------------------------

    def _on_A_toggled(self, checked: bool) -> None:
        if checked:
            self._uncheck_h5_others(self._constraint_A_check)
            F = self._f_widget.get_matrix()
            if F is not None:
                self._saved_A = F[:self._q, :self._q].copy()
            self._recompute_A()
            self.constraint_toggled.emit()
        else:
            self._restore_A()

    def _on_B_toggled(self, checked: bool) -> None:
        if checked:
            self._uncheck_h5_others(self._constraint_B_check)
            F = self._f_widget.get_matrix()
            if F is not None:
                self._saved_B = F[:self._q, self._q:].copy()
            self._recompute_B()
            self.constraint_toggled.emit()
        else:
            self._restore_B()

    def _on_C_toggled(self, checked: bool) -> None:
        if checked:
            self._uncheck_h5_others(self._constraint_C_check)
            F = self._f_widget.get_matrix()
            if F is not None:
                self._saved_C = F[self._q:, :self._q].copy()
            self._recompute_C()
            self.constraint_toggled.emit()
        else:
            self._restore_C()

    def _on_SU_toggled(self, checked: bool) -> None:
        if checked:
            self._uncheck_h5_others(self._constraint_SU_check)
            Sw = self._sigma_widget.get_matrix()
            if Sw is not None:
                self._saved_SU = Sw[:self._q, :self._q].copy()
            self._recompute_SU()
            self.constraint_toggled.emit()
        else:
            self._restore_SU()

    def _on_delta_toggled(self, checked: bool) -> None:
        # Δ=0 is independent — does NOT uncheck {A, B, Σ_U}
        if checked:
            Sw = self._sigma_widget.get_matrix()
            if Sw is not None:
                self._saved_Delta = Sw[:self._q, self._q:].copy()
            self._recompute_delta()
            self.constraint_toggled.emit()
        else:
            self._restore_delta()

    # ------------------------------------------------------------------
    # Recompute methods (fire on every value_changed)
    # ------------------------------------------------------------------

    def _on_value_changed(self) -> None:
        """Dispatch to the active constraint recomputation; always refresh badges."""
        if self._constraint_A_check.isChecked():
            self._recompute_A()
        elif self._constraint_B_check.isChecked():
            self._recompute_B()
        elif self._constraint_C_check.isChecked():
            self._recompute_C()
        elif self._constraint_SU_check.isChecked():
            self._recompute_SU()
        # Δ = 0 locks the off-diagonal, so no re-projection needed on value_changed.
        self._update_stability_badges()

    def _recompute_A(self) -> None:
        if not self._constraint_A_check.isChecked() or self._updating:
            return
        F, Sw = self._f_widget.get_matrix(), self._sigma_widget.get_matrix()
        if F is None or Sw is None:
            return

        A_new = self._call_constraint(compute_A_from_h5, dict(
            B=F[:self._q, self._q:], C=F[self._q:, :self._q], D=F[self._q:, self._q:],
            SU=Sw[:self._q, :self._q], Dt=Sw[:self._q, self._q:], SV=Sw[self._q:, self._q:],
        ))
        if A_new is None:
            self._f_widget.set_constraint_status(
                "✗  A — singular system", "color: #cc0000; font-size: 10px;")
            return

        q = self._q
        new_F = F.copy()
        new_F[:q, :q] = A_new
        self._updating = True
        self._f_widget.set_matrix(new_F)
        self._f_widget.set_block_editable(0, q, 0, q, False)
        self._updating = False
        self._f_widget.set_constraint_status(
            "✓  A satisfies constraint (4.8)", "color: #155399; font-size: 10px;")
        self._update_stability_badges()

    def _recompute_B(self) -> None:
        if not self._constraint_B_check.isChecked() or self._updating:
            return
        F, Sw = self._f_widget.get_matrix(), self._sigma_widget.get_matrix()
        if F is None or Sw is None:
            return

        B_new = self._call_constraint(compute_B_from_h5, dict(
            A=F[:self._q, :self._q], C=F[self._q:, :self._q], D=F[self._q:, self._q:],
            SU=Sw[:self._q, :self._q], Dt=Sw[:self._q, self._q:], SV=Sw[self._q:, self._q:],
        ))
        if B_new is None:
            self._f_widget.set_constraint_status(
                "✗  B — singular system", "color: #cc0000; font-size: 10px;")
            return

        q, s = self._q, self._s
        new_F = F.copy()
        new_F[:q, q:] = B_new
        self._updating = True
        self._f_widget.set_matrix(new_F)
        self._f_widget.set_block_editable(0, q, q, q + s, False)
        self._updating = False
        self._f_widget.set_constraint_status(
            "✓  B satisfies constraint (4.8)", "color: #1a7a3a; font-size: 10px;")
        self._update_stability_badges()

    def _recompute_SU(self) -> None:
        if not self._constraint_SU_check.isChecked() or self._updating:
            return
        F, Sw = self._f_widget.get_matrix(), self._sigma_widget.get_matrix()
        if F is None or Sw is None:
            return

        SU_new = self._call_constraint(compute_SU_from_h5, dict(
            A=F[:self._q, :self._q], B=F[:self._q, self._q:],
            C=F[self._q:, :self._q], D=F[self._q:, self._q:],
            Dt=Sw[:self._q, self._q:], SV=Sw[self._q:, self._q:],
        ))
        if SU_new is None:
            self._sigma_widget.set_constraint_status(
                "✗  Σ_U — singular system", "color: #cc0000; font-size: 10px;")
            return

        q = self._q
        new_Sw = Sw.copy()
        new_Sw[:q, :q] = SU_new
        self._updating = True
        self._sigma_widget.set_matrix(new_Sw)
        self._sigma_widget.set_block_editable(0, q, 0, q, False)
        self._updating = False
        self._sigma_widget.set_constraint_status(
            "✓  Σ_U satisfies constraint (4.8)", "color: #6c3483; font-size: 10px;")

    def _recompute_C(self) -> None:
        if not self._constraint_C_check.isChecked() or self._updating:
            return
        F, Sw = self._f_widget.get_matrix(), self._sigma_widget.get_matrix()
        if F is None or Sw is None:
            return

        q, s = self._q, self._s
        # Use the current C block as warm start (preserves continuity during editing)
        C_init = F[q:, :q].copy()

        C_new = self._call_constraint(compute_C_from_h5, dict(
            A=F[:q, :q], B=F[:q, q:], D=F[q:, q:],
            SU=Sw[:q, :q], Dt=Sw[:q, q:], SV=Sw[q:, q:],
            C_init=C_init,
        ))
        if C_new is None:
            self._f_widget.set_constraint_status(
                "✗  C — non-convergence / singular", "color: #cc0000; font-size: 10px;")
            return

        new_F = F.copy()
        new_F[q:, :q] = C_new
        self._updating = True
        self._f_widget.set_matrix(new_F)
        self._f_widget.set_block_editable(q, q + s, 0, q, False)
        self._updating = False
        self._f_widget.set_constraint_status(
            "✓  C satisfies constraint (4.20) [iter]",
            "color: #a04000; font-size: 10px;")
        self._update_stability_badges()

    def _recompute_delta(self) -> None:
        """Set Δ(k) = 0 and lock both off-diagonal blocks of Σ_W.

        If an H5 constraint (A, B or Σ_U) is also active, it is re-evaluated
        immediately so that it uses the updated Δ = 0.
        """
        if not self._constraint_delta_check.isChecked() or self._updating:
            return
        Sw = self._sigma_widget.get_matrix()
        if Sw is None:
            return

        q, s = self._q, self._s
        new_Sw = Sw.copy()
        new_Sw[:q, q:] = 0.0   # Δ  = 0
        new_Sw[q:, :q] = 0.0   # Δᵀ = 0
        self._updating = True
        self._sigma_widget.set_matrix(new_Sw)
        # Lock both off-diagonal blocks (top-right and bottom-left)
        self._sigma_widget.set_block_editable(0, q, q, q + s, False)
        self._sigma_widget.set_block_editable(q, q + s, 0, q, False)
        self._updating = False
        self._sigma_widget.set_constraint_status("")
        # Re-trigger the active H5 constraint so it uses Δ = 0
        self._retrigger_h5()

    # ------------------------------------------------------------------
    # Restore methods (called when a checkbox is unchecked)
    # ------------------------------------------------------------------

    def _restore_A(self) -> None:
        q = self._q
        self._f_widget.set_block_editable(0, q, 0, q, True)
        self._f_widget.set_constraint_status("")
        if self._saved_A is not None:
            F = self._f_widget.get_matrix()
            if F is not None:
                restored = F.copy()
                restored[:q, :q] = self._saved_A
                self._updating = True
                self._f_widget.set_matrix(restored)
                self._updating = False
            self._saved_A = None
        self._update_stability_badges()

    def _restore_B(self) -> None:
        q, s = self._q, self._s
        self._f_widget.set_block_editable(0, q, q, q + s, True)
        self._f_widget.set_constraint_status("")
        if self._saved_B is not None:
            F = self._f_widget.get_matrix()
            if F is not None:
                restored = F.copy()
                restored[:q, q:] = self._saved_B
                self._updating = True
                self._f_widget.set_matrix(restored)
                self._updating = False
            self._saved_B = None

    def _restore_SU(self) -> None:
        q = self._q
        self._sigma_widget.set_block_editable(0, q, 0, q, True)
        self._sigma_widget.set_constraint_status("")
        if self._saved_SU is not None:
            Sw = self._sigma_widget.get_matrix()
            if Sw is not None:
                restored = Sw.copy()
                restored[:q, :q] = self._saved_SU
                self._updating = True
                self._sigma_widget.set_matrix(restored)
                self._updating = False
            self._saved_SU = None

    def _restore_C(self) -> None:
        q, s = self._q, self._s
        self._f_widget.set_block_editable(q, q + s, 0, q, True)
        self._f_widget.set_constraint_status("")
        if self._saved_C is not None:
            F = self._f_widget.get_matrix()
            if F is not None:
                restored = F.copy()
                restored[q:, :q] = self._saved_C
                self._updating = True
                self._f_widget.set_matrix(restored)
                self._updating = False
            self._saved_C = None
        self._update_stability_badges()

    def _restore_delta(self) -> None:
        q, s = self._q, self._s
        # Unlock both off-diagonal blocks
        self._sigma_widget.set_block_editable(0, q, q, q + s, True)
        self._sigma_widget.set_block_editable(q, q + s, 0, q, True)
        self._sigma_widget.set_constraint_status("")
        if self._saved_Delta is not None:
            Sw = self._sigma_widget.get_matrix()
            if Sw is not None:
                restored = Sw.copy()
                restored[:q, q:] = self._saved_Delta
                restored[q:, :q] = self._saved_Delta.T
                self._updating = True
                self._sigma_widget.set_matrix(restored)
                self._updating = False
            self._saved_Delta = None
        # Re-trigger the active H5 constraint so it uses the restored Δ
        self._retrigger_h5()

    # ------------------------------------------------------------------
    # Stability badges  (read-only display, no correction)
    # ------------------------------------------------------------------

    def _update_stability_badges(self) -> None:
        """Recompute ρ(F(k)), ρ(A(k)) and ρ(D(k)); refresh all three badges."""
        F = self._f_widget.get_matrix()
        n = self._q + self._s
        self._set_badge(self._stab_F_badge, "ρ(F)", F, 0, n, 0, n)
        self._set_badge(self._stab_A_badge, "ρ(A)", F, 0, self._q, 0, self._q)
        self._set_badge(self._stab_D_badge, "ρ(D)", F,
                        self._q, self._q + self._s,
                        self._q, self._q + self._s)

    @staticmethod
    def _set_badge(
        badge: QLabel,
        label: str,
        F: np.ndarray | None,
        r0: int, r1: int,
        c0: int, c1: int,
    ) -> None:
        """Extract block F[r0:r1, c0:c1], compute its spectral radius, style *badge*."""
        _GREY = ("font-size: 10px; padding: 2px 8px; border-radius: 3px;"
                 "background: #e9ecef; color: #6c757d; border: 1px solid #adb5bd;")
        if F is None:
            badge.setText(f"{label} = ?")
            badge.setStyleSheet(_GREY)
            return
        try:
            rho = float(np.max(np.abs(np.linalg.eigvals(F[r0:r1, c0:c1]))))
        except np.linalg.LinAlgError:
            badge.setText(f"{label} = ?")
            badge.setStyleSheet(_GREY)
            return

        if rho < 0.90:
            bg, fg, border = "#d4edda", "#155724", "#c3e6cb"
            icon = "✓"
        elif rho < 1.00:
            bg, fg, border = "#fff3cd", "#856404", "#ffc107"
            icon = "~"
        else:
            bg, fg, border = "#f8d7da", "#721c24", "#f5c6cb"
            icon = "✗"

        badge.setText(f"{label} = {rho:.4f} {icon}")
        badge.setStyleSheet(
            f"font-size: 10px; padding: 2px 8px; border-radius: 3px;"
            f"background: {bg}; color: {fg}; border: 1px solid {border};"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _randomize(self) -> None:
        """Fill F(k) and Σ_W(k) with random valid parameters."""
        rng = np.random.default_rng()
        n = self._q + self._s

        # Random stable F: scale so ρ(F) ∈ [0.70, 0.90]
        F_raw = rng.uniform(-1.0, 1.0, (n, n))
        rho = float(np.max(np.abs(np.linalg.eigvals(F_raw))))
        target = float(rng.uniform(0.70, 0.90))
        F_rand = F_raw * (target / max(rho, 1e-10))

        # Random SPD Σ_W: Wishart-like L @ L.T + ε·I, diagonal ≈ 0.1–0.5
        L  = rng.uniform(-0.5, 0.5, (n, n))
        Sw = L @ L.T + 0.05 * np.eye(n)
        Sw = Sw / (np.sqrt(np.diag(Sw).mean()) / 0.3)

        self._f_widget.set_matrix(F_rand)
        self._sigma_widget.set_matrix(Sw)
        self._update_stability_badges()
        self.constraint_toggled.emit()   # reset plots in main window

    def _uncheck_h5_others(self, keep: QCheckBox) -> None:
        """Silently uncheck the other H5 checkboxes ({A, B, C, Σ_U}) except *keep*.

        Δ=0 is intentionally NOT touched — it can coexist with any H5 constraint.
        """
        for chk, restore in [
            (self._constraint_A_check,  self._restore_A),
            (self._constraint_B_check,  self._restore_B),
            (self._constraint_C_check,  self._restore_C),
            (self._constraint_SU_check, self._restore_SU),
        ]:
            if chk is not keep and chk.isChecked():
                chk.blockSignals(True)
                chk.setChecked(False)
                chk.blockSignals(False)
                restore()

    def _retrigger_h5(self) -> None:
        """Re-evaluate the active H5 constraint (A, B, C or Σ_U), if any."""
        if self._constraint_A_check.isChecked():
            self._recompute_A()
        elif self._constraint_B_check.isChecked():
            self._recompute_B()
        elif self._constraint_C_check.isChecked():
            self._recompute_C()
        elif self._constraint_SU_check.isChecked():
            self._recompute_SU()

    @staticmethod
    def _call_constraint(fn, kwargs: dict) -> np.ndarray | None:
        """Call a constraint function and return None on ValueError."""
        try:
            return fn(**kwargs)
        except ValueError:
            return None

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

    validity_changed   = pyqtSignal(bool)
    value_changed      = pyqtSignal()   # forwarded from any _StateTab cell edit
    constraint_toggled = pyqtSignal()   # forwarded from any _StateTab

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
            tab.value_changed.connect(self.value_changed)   # forward upward
            tab.constraint_toggled.connect(self.constraint_toggled)

            scroll = QScrollArea()
            scroll.setWidget(tab)
            scroll.setWidgetResizable(True)
            scroll.setStyleSheet(
                "QScrollArea { background-color: palette(window); border: none; }"
                "QScrollArea > QWidget > QWidget { background-color: palette(window); }"
            )
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
