#!/usr/bin/env python3
"""
prg/gui/param_panel.py
======================
ParamPanel — QTabWidget with one tab per Markov state k.

Each tab (_StateTab) exposes:
  - F(k)     : MatrixTableWidget (no SPD check)
  - Σ_W(k)   : MatrixTableWidget (SPD check enabled)
  - μ_z0(k), b_X(k), b_Y(k) : VectorWidgets

(H5)-compatible AB constraint
----------------------------------
A single checkbox per regime enforces the closed-form (H5)-compatible
"AB constraint" parametrisation::

    A(k) = Δ(k) Σ_V(k)⁻¹ C(k),
    B(k) = Δ(k) Σ_V(k)⁻¹ D(k).

When checked, the A and B blocks of F(k) are read-only and are recomputed
on every edit of (C(k), D(k), Δ(k), Σ_V(k)). When unchecked, the previous
(saved) values of A and B are restored and the blocks become editable again.

The constraint is unchecked by default on every newly built / loaded tab.

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
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from prg.gui.matrix_widget import MatrixTableWidget, VectorWidget
from prg.utils.h5_constraint import compute_AB, compute_h5_residual

# Tolerance for the live (H5) residual badge — kept in sync with
# prg.filter.gss_filter.H5_TOL so the GUI shows a green badge in
# exactly the regime where mode='h5_exact' would not warn.
_H5_BADGE_TOL = 1e-6

# Pill style for constraint error messages (explicit background → readable on any theme)
_CONSTRAINT_ERR_STYLE = (
    "font-size: 10px; padding: 1px 6px; border-radius: 3px;"
    "background: #fff8f8; color: #c0392b; border: 1px solid #f5c6cb;"
)

# Style of the checked AB constraint status badge
_CONSTRAINT_OK_STYLE = "color: #155399; font-size: 10px;"


# ---------------------------------------------------------------------------
# _StateTab
# ---------------------------------------------------------------------------


class _StateTab(QWidget):
    """One tab: F(k), Σ_W(k), μ_z0(k), b_X(k), b_Y(k) side by side,
    plus a (H5)-compatible AB constraint checkbox and stability badges."""

    validity_changed = pyqtSignal(bool)
    value_changed = pyqtSignal()  # emitted whenever any cell changes
    constraint_toggled = pyqtSignal()  # emitted when the AB-constraint checkbox is toggled

    # Checkbox colour palette  (text-color, checked-fill, border)
    _CHK_STYLE_AB = (
        "QCheckBox { color: #155399; font-weight: bold; font-size: 11px; }"
        "QCheckBox::indicator:checked   { border: 2px solid #155399;"
        "                                 background-color: #2980b9; }"
        "QCheckBox::indicator:unchecked { border: 2px solid #155399; }"
    )

    def __init__(self, k: int, q: int, s: int, parent=None):
        super().__init__(parent)
        self._k = k
        self._q = q
        self._s = s
        self._updating = False  # re-entrancy guard

        # Saved A, B blocks of F(k) — restored when the constraint is unchecked
        self._saved_A: np.ndarray | None = None
        self._saved_B: np.ndarray | None = None

        # ── Main layout: checkbox row on top, matrix widgets below ──────
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # -- Constraint checkbox row (single AB-constraint box) + stability badges --
        chk_row = QHBoxLayout()
        chk_row.setSpacing(16)

        self._constraint_AB_check = QCheckBox(f"AB constraint on (A({k}), B({k}))")
        self._constraint_AB_check.setToolTip(
            "Force A(k) = Δ(k) Σ_V(k)⁻¹ C(k) and B(k) = Δ(k) Σ_V(k)⁻¹ D(k).\n"
            "This closed form is sufficient for (H5); when checked, the A and\n"
            "B blocks of F(k) become read-only and are recomputed live as you\n"
            "edit C, D, Δ or Σ_V. Other (H5)-compatible (A, B) may exist when\n"
            "K·s < q+s; the live (H5) badge is the source of truth."
        )
        self._constraint_AB_check.setStyleSheet(self._CHK_STYLE_AB)
        chk_row.addWidget(self._constraint_AB_check)

        chk_row.addStretch()
        layout.addLayout(chk_row)

        # -- Matrix/vector widgets row --
        widgets_row = QHBoxLayout()
        widgets_row.setSpacing(16)

        self._f_widget = MatrixTableWidget(
            q,
            s,
            is_covariance=False,
            title=f"<i>F</i>({k})",
            default_value=0.0,
        )
        self._f_widget.set_matrix(np.eye(q + s) * 0.5)

        # Stability badges live in a sub-column directly below _f_widget.
        # The (H5) residual badge — green when ‖F‖_F < _H5_BADGE_TOL,
        # amber otherwise — lives beside them so the user sees at a glance
        # whether the current model is (H5)-compatible.
        self._stab_F_badge = QLabel()
        self._stab_F_badge.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self._stab_F_badge.setFixedHeight(16)
        self._stab_A_badge = QLabel()
        self._stab_A_badge.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self._stab_A_badge.setFixedHeight(16)
        self._stab_D_badge = QLabel()
        self._stab_D_badge.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self._stab_D_badge.setFixedHeight(16)
        self._h5_badge = QLabel()
        self._h5_badge.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self._h5_badge.setFixedHeight(16)

        f_col = QVBoxLayout()
        f_col.setSpacing(2)
        f_col.setContentsMargins(0, 0, 0, 0)
        f_col.addWidget(self._f_widget)
        f_col.addWidget(self._stab_F_badge)
        f_col.addWidget(self._stab_A_badge)
        f_col.addWidget(self._stab_D_badge)
        f_col.addWidget(self._h5_badge)
        f_col.addStretch()
        widgets_row.addLayout(f_col)

        self._sigma_widget = MatrixTableWidget(
            q,
            s,
            is_covariance=True,
            title=f"Σ<sub>W</sub>({k})",
            default_value=0.1,
        )

        self._mu_widget = VectorWidget(
            q + s,
            title=f"μ<sub>z₀</sub>({k})",
            default_value=0.0,
        )
        self._bx_widget = VectorWidget(
            q,
            title=f"<i>b</i><sub>X</sub>({k})",
            default_value=0.0,
        )
        self._by_widget = VectorWidget(
            s,
            title=f"<i>b</i><sub>Y</sub>({k})",
            default_value=0.0,
        )

        for w in (self._sigma_widget, self._mu_widget, self._bx_widget, self._by_widget):
            _col = QVBoxLayout()
            _col.setContentsMargins(0, 0, 0, 0)
            _col.setSpacing(2)
            _col.addWidget(w)
            _col.addStretch()
            widgets_row.addLayout(_col)
        widgets_row.addStretch()  # prevent horizontal expansion
        layout.addLayout(widgets_row)
        layout.addStretch()  # push everything to the top

        # ── Signal connections ──────────────────────────────────────────
        for w in (
            self._f_widget,
            self._sigma_widget,
            self._mu_widget,
            self._bx_widget,
            self._by_widget,
        ):
            w.validity_changed.connect(self._on_child_validity)
            w.value_changed.connect(self.value_changed)  # forward upward

        self._f_widget.value_changed.connect(self._on_value_changed)
        self._sigma_widget.value_changed.connect(self._on_value_changed)

        self._constraint_AB_check.toggled.connect(self._on_AB_toggled)

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
        self._bx_widget.set_vector(flat[: self._q])
        self._by_widget.set_vector(flat[self._q :])

    def is_valid(self) -> bool:
        return (
            self._f_widget.is_valid()
            and self._sigma_widget.is_valid()
            and self._mu_widget.is_valid()
            and self._bx_widget.is_valid()
            and self._by_widget.is_valid()
        )

    # ------------------------------------------------------------------
    # Constraint state — public accessors
    # ------------------------------------------------------------------

    def is_AB_constraint_active(self) -> bool:
        """Return True iff the (A, B) constraint is currently active."""
        return self._constraint_AB_check.isChecked()

    def apply_constraint(self, active: bool) -> None:
        """Programmatically set the AB constraint flag.

        Mirrors the user clicking the checkbox so that all the recompute
        and constraint_toggled logic fires normally.
        """
        if self._constraint_AB_check.isChecked() != active:
            self._constraint_AB_check.setChecked(active)

    # ------------------------------------------------------------------
    # Constraint toggle handlers
    # ------------------------------------------------------------------

    def _on_AB_toggled(self, checked: bool) -> None:
        if checked:
            F = self._f_widget.get_matrix()
            if F is not None:
                q, s = self._q, self._s
                self._saved_A = F[:q, :q].copy()
                self._saved_B = F[:q, q:].copy()
            self._recompute_AB()
            self.constraint_toggled.emit()
        else:
            self._restore_AB()
            self.constraint_toggled.emit()

    # ------------------------------------------------------------------
    # Recompute and restore methods
    # ------------------------------------------------------------------

    def _on_value_changed(self) -> None:
        """Re-run the AB-constraint projection on every value edit; refresh badges."""
        if self._constraint_AB_check.isChecked():
            self._recompute_AB()
        self._update_stability_badges()

    def _recompute_AB(self) -> None:
        """Replace A(k), B(k) blocks of F(k) by the AB-constraint closed form."""
        if not self._constraint_AB_check.isChecked() or self._updating:
            return
        F, Sw = self._f_widget.get_matrix(), self._sigma_widget.get_matrix()
        if F is None or Sw is None:
            return

        q, s = self._q, self._s
        try:
            A_new, B_new = compute_AB(
                C=F[q:, :q],
                D=F[q:, q:],
                Dt=Sw[:q, q:],
                SV=Sw[q:, q:],
            )
        except ValueError:
            self._f_widget.set_constraint_status(
                "✗  AB constraint — Σ_V singular", _CONSTRAINT_ERR_STYLE
            )
            return

        new_F = F.copy()
        new_F[:q, :q] = A_new
        new_F[:q, q:] = B_new
        self._updating = True
        self._f_widget.set_matrix(new_F)
        # Lock both A and B blocks (top-left q×q and top-right q×s)
        self._f_widget.set_block_editable(0, q, 0, q, False)
        self._f_widget.set_block_editable(0, q, q, q + s, False)
        self._updating = False
        self._f_widget.set_constraint_status(
            "✓  A, B satisfy the AB constraint (A = Δ Σ_V⁻¹ C, B = Δ Σ_V⁻¹ D)",
            _CONSTRAINT_OK_STYLE,
        )
        self._update_stability_badges()

    def _restore_AB(self) -> None:
        """Unlock A and B blocks; restore their saved values."""
        q, s = self._q, self._s
        self._f_widget.set_block_editable(0, q, 0, q, True)
        self._f_widget.set_block_editable(0, q, q, q + s, True)
        self._f_widget.set_constraint_status("")
        if self._saved_A is not None and self._saved_B is not None:
            F = self._f_widget.get_matrix()
            if F is not None:
                restored = F.copy()
                restored[:q, :q] = self._saved_A
                restored[:q, q:] = self._saved_B
                self._updating = True
                self._f_widget.set_matrix(restored)
                self._updating = False
        self._saved_A = None
        self._saved_B = None
        self._update_stability_badges()

    # ------------------------------------------------------------------
    # Stability badges  (read-only display, no correction)
    # ------------------------------------------------------------------

    def _update_stability_badges(self) -> None:
        """Recompute ρ(F), ρ(A), ρ(D) and the (H5) residual; refresh all four badges."""
        F = self._f_widget.get_matrix()
        n = self._q + self._s
        self._set_badge(self._stab_F_badge, "ρ(F)", F, 0, n, 0, n)
        self._set_badge(self._stab_A_badge, "ρ(A)", F, 0, self._q, 0, self._q)
        self._set_badge(
            self._stab_D_badge, "ρ(D)", F, self._q, self._q + self._s, self._q, self._q + self._s
        )
        self._update_h5_badge()

    def _update_h5_badge(self) -> None:
        """Compute the (H5) Frobenius residual and style ``_h5_badge``.

        Green ✓ when ‖F‖_F < ``_H5_BADGE_TOL`` (model is (H5)-compatible
        and ``mode='h5_exact'`` is safe), amber ⚠ otherwise (filter would
        emit a warning), grey ? when the residual cannot be evaluated
        (singular M, missing matrix data).
        """
        _GREY = (
            "font-size: 10px; padding: 2px 8px; border-radius: 3px;"
            "background: #e9ecef; color: #6c757d; border: 1px solid #adb5bd;"
        )
        F = self._f_widget.get_matrix()
        Sw = self._sigma_widget.get_matrix()
        if F is None or Sw is None:
            self._h5_badge.setText("(H5) = ?")
            self._h5_badge.setStyleSheet(_GREY)
            return

        q, s = self._q, self._s
        try:
            res = compute_h5_residual(
                A=F[:q, :q],
                B=F[:q, q:],
                C=F[q:, :q],
                D=F[q:, q:],
                SU=Sw[:q, :q],
                Dt=Sw[:q, q:],
                SV=Sw[q:, q:],
            )
            res_norm = float(np.linalg.norm(res, "fro"))
        except np.linalg.LinAlgError, ValueError:
            self._h5_badge.setText("(H5) = ?")
            self._h5_badge.setStyleSheet(_GREY)
            return

        if res_norm < _H5_BADGE_TOL:
            bg, fg, border = "#d4edda", "#155724", "#c3e6cb"
            icon = "✓"
        else:
            bg, fg, border = "#fff3cd", "#856404", "#ffc107"
            icon = "⚠"
        self._h5_badge.setText(f"(H5) ‖F‖ = {res_norm:.2e} {icon}")
        self._h5_badge.setStyleSheet(
            f"font-size: 10px; padding: 2px 8px; border-radius: 3px;"
            f"background: {bg}; color: {fg}; border: 1px solid {border};"
        )

    @staticmethod
    def _set_badge(
        badge: QLabel,
        label: str,
        F: np.ndarray | None,
        r0: int,
        r1: int,
        c0: int,
        c1: int,
    ) -> None:
        """Extract block F[r0:r1, c0:c1], compute its spectral radius, style *badge*."""
        _GREY = (
            "font-size: 10px; padding: 2px 8px; border-radius: 3px;"
            "background: #e9ecef; color: #6c757d; border: 1px solid #adb5bd;"
        )
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
        L = rng.uniform(-0.5, 0.5, (n, n))
        Sw = L @ L.T + 0.05 * np.eye(n)
        Sw = Sw / (np.sqrt(np.diag(Sw).mean()) / 0.3)

        self._f_widget.set_matrix(F_rand)
        self._sigma_widget.set_matrix(Sw)
        self._update_stability_badges()
        self.constraint_toggled.emit()  # reset plots in main window

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
    value_changed = pyqtSignal()  # forwarded from any _StateTab cell edit
    constraint_toggled = pyqtSignal()  # forwarded from any _StateTab

    def __init__(self, K: int, q: int, s: int, parent=None):
        super().__init__(parent)
        self._K = K
        self._q = q
        self._s = s

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        info = QLabel(f"K = {K} states,  q = {q} (hidden),  s = {s} (observed)")
        info.setStyleSheet("font-size: 11px; padding: 2px 4px;")
        layout.addWidget(info)

        self._tabs = QTabWidget()
        # Reserve enough vertical room for one full _StateTab content (checkbox
        # row + matrix widgets + status labels) so the State-k tabs are always
        # usable without scrolling, even on small windows. The natural height
        # of a _StateTab scales roughly with (q+s); 26 px per cell, plus
        # ~140 px of constant overhead (checkbox row, badges, margins).
        n = q + s
        min_tab_h = 26 * n + 140  # ≈ 192 for n=2, ≈ 244 for n=4
        self._tabs.setMinimumHeight(min_tab_h)
        layout.addWidget(self._tabs)

        self._state_tabs: list[_StateTab] = []
        for k in range(K):
            tab = _StateTab(k, q, s)
            tab.validity_changed.connect(self._on_tab_validity)
            tab.value_changed.connect(self.value_changed)  # forward upward
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

        # Corner widget: [🎲] and (K>1) [Apply AB → all] in the tab-bar top-right.
        corner = QWidget()
        c_lay = QHBoxLayout(corner)
        c_lay.setContentsMargins(0, 0, 4, 0)
        c_lay.setSpacing(4)

        btn_rand_corner = QPushButton("🎲")
        btn_rand_corner.setFixedSize(26, 22)
        btn_rand_corner.setToolTip(
            "Randomize F(k) and Σ_W(k) for the current state\nwith random stable parameters."
        )
        btn_rand_corner.clicked.connect(
            lambda: self._state_tabs[self._tabs.currentIndex()]._randomize()
        )
        c_lay.addWidget(btn_rand_corner)

        if K > 1:
            self._btn_apply_AB_all = QPushButton("Apply AB → all")
            self._btn_apply_AB_all.setFixedHeight(22)
            self._btn_apply_AB_all.setToolTip(
                "Copy the AB constraint state from the currently visible\n"
                "state tab to ALL other states.  Each target tab recomputes\n"
                "its (A, B) blocks from its own (C, D, Δ, Σ_V)."
            )
            self._btn_apply_AB_all.clicked.connect(self._on_apply_AB_all)
            c_lay.addWidget(self._btn_apply_AB_all)

        self._tabs.setCornerWidget(corner, Qt.Corner.TopRightCorner)

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

    def reapply_active_constraints(self) -> None:
        """Re-evaluate active AB constraints on each state tab.

        Call this after loading external parameter values (e.g. session
        restore) to ensure that any checked constraint boxes are re-projected
        onto the new matrix values.  Does **not** emit ``constraint_toggled``,
        so no simulation reset is triggered.
        """
        for tab in self._state_tabs:
            if tab.is_AB_constraint_active():
                tab._recompute_AB()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _on_tab_validity(self, _: bool) -> None:
        self.validity_changed.emit(self.is_valid())

    def _on_apply_AB_all(self) -> None:
        """Copy the AB constraint state from the current tab to all others."""
        src_k = self._tabs.currentIndex()
        if src_k < 0 or src_k >= self._K:
            return
        src = self._state_tabs[src_k]
        active = src.is_AB_constraint_active()
        if not active:
            from PyQt6.QtWidgets import QMessageBox

            QMessageBox.information(
                self,
                "Apply AB constraint to all states",
                "The AB constraint is not active on the current tab.\n"
                "Check it first, then click this button.",
            )
            return
        for k, tab in enumerate(self._state_tabs):
            if k != src_k:
                tab.apply_constraint(active)
