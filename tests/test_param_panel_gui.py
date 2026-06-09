#!/usr/bin/env python3
"""
tests/test_param_panel_gui.py
=============================
End-to-end GUI tests for prg/gui/param_panel.py using pytest-qt.

Coverage
--------
- AB-constraint checkbox: default state, toggle on/off, value-change recompute
- (H5) badge: green when AB is active, amber when not
- "Apply AB → all" propagation across regime tabs
"""

from __future__ import annotations

import os

# Force a headless Qt platform before any PyQt6 import.  The fixture
# qtbot from pytest-qt creates a QApplication that uses the platform
# selected at import time.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

from prg.gui.param_panel import ParamPanel
from prg.utils.h5_constraint import compute_AB, compute_h5_residual


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _build_panel(qtbot, K: int = 2, q: int = 2, s: int = 2) -> ParamPanel:
    """Construct a ParamPanel and register it with qtbot."""
    panel = ParamPanel(K=K, q=q, s=s)
    qtbot.addWidget(panel)
    return panel


def _seed_model(tab, q: int, s: int) -> tuple[np.ndarray, np.ndarray]:
    """Set deterministic, valid F(k) and Σ_W(k) on *tab*.  Return (F, Σ_W)."""
    F = np.array(
        [
            [0.7, 0.1, 0.42, 0.13],
            [0.0, 0.6, 0.07, 0.31],
            [0.3, 0.1, 0.5, 0.0],
            [0.1, 0.2, 0.0, 0.4],
        ]
    )
    SW = np.array(
        [
            [0.10, 0.00, 0.05, 0.00],
            [0.00, 0.10, 0.00, 0.05],
            [0.05, 0.00, 0.10, 0.00],
            [0.00, 0.05, 0.00, 0.10],
        ]
    )
    assert F.shape == (q + s, q + s) and SW.shape == (q + s, q + s)
    tab.set_F(F)
    tab.set_Sigma_W(SW)
    return F, SW


# ---------------------------------------------------------------------------
# Default state
# ---------------------------------------------------------------------------
def test_AB_default_unchecked(qtbot):
    """On a freshly built panel, the AB checkbox is unchecked on every tab."""
    panel = _build_panel(qtbot, K=3, q=2, s=2)
    for tab in panel._state_tabs:
        assert tab.is_AB_constraint_active() is False
        assert tab._constraint_AB_check.isChecked() is False


# ---------------------------------------------------------------------------
# Toggle behaviour
# ---------------------------------------------------------------------------
def test_toggle_AB_locks_blocks_and_writes_closed_form(qtbot):
    """Checking the AB box overwrites A, B with Δ Σ_V⁻¹ C, Δ Σ_V⁻¹ D."""
    q, s = 2, 2
    panel = _build_panel(qtbot, K=2, q=q, s=s)
    tab = panel._state_tabs[0]
    F0, SW = _seed_model(tab, q, s)

    tab._constraint_AB_check.setChecked(True)

    F = tab.get_F()
    A_expected, B_expected = compute_AB(C=F0[q:, :q], D=F0[q:, q:], Dt=SW[:q, q:], SV=SW[q:, q:])
    np.testing.assert_allclose(F[:q, :q], A_expected, atol=1e-12)
    np.testing.assert_allclose(F[:q, q:], B_expected, atol=1e-12)


def test_toggle_AB_off_restores_previous_AB(qtbot):
    """Unchecking the AB box restores the A, B blocks captured at toggle-on."""
    q, s = 2, 2
    panel = _build_panel(qtbot, K=2, q=q, s=s)
    tab = panel._state_tabs[0]
    F0, _ = _seed_model(tab, q, s)
    A0 = F0[:q, :q].copy()
    B0 = F0[:q, q:].copy()

    tab._constraint_AB_check.setChecked(True)
    tab._constraint_AB_check.setChecked(False)

    F = tab.get_F()
    np.testing.assert_allclose(F[:q, :q], A0, atol=1e-12)
    np.testing.assert_allclose(F[:q, q:], B0, atol=1e-12)


# ---------------------------------------------------------------------------
# Live recompute on input edits
# ---------------------------------------------------------------------------
def test_value_change_triggers_recompute(qtbot):
    """While AB is active, editing C, D, Δ, or Σ_V re-derives A and B.

    ``MatrixTableWidget.set_matrix`` is silent (it does not emit
    ``value_changed``); the recompute is normally triggered by the user
    typing into a cell. To exercise the same code path here we edit a
    cell via ``QTableWidgetItem.setText``, which fires Qt's
    ``itemChanged`` signal and ultimately ``value_changed``.
    """
    q, s = 2, 2
    panel = _build_panel(qtbot, K=2, q=q, s=s)
    tab = panel._state_tabs[0]
    _seed_model(tab, q, s)
    tab._constraint_AB_check.setChecked(True)
    F_before = tab.get_F().copy()

    # Edit C(0, 0): bump it by 0.5.  Position in the F table is row q, col 0.
    item = tab._f_widget._table.item(q, 0)
    item.setText(f"{F_before[q, 0] + 0.5:.6g}")

    F_after = tab.get_F()
    assert not np.allclose(F_after[:q, :q], F_before[:q, :q]), (
        "A block should have changed after editing C"
    )

    # And the new (A, B) must match the closed form for the *new* C.
    A_expected, B_expected = compute_AB(
        C=F_after[q:, :q],
        D=F_after[q:, q:],
        Dt=tab.get_Sigma_W()[:q, q:],
        SV=tab.get_Sigma_W()[q:, q:],
    )
    np.testing.assert_allclose(F_after[:q, :q], A_expected, atol=1e-12)
    np.testing.assert_allclose(F_after[:q, q:], B_expected, atol=1e-12)


# ---------------------------------------------------------------------------
# (H5) residual badge
# ---------------------------------------------------------------------------
def test_h5_badge_green_when_AB_active(qtbot):
    """The (H5) badge displays the ✓ glyph when the AB constraint is active."""
    q, s = 2, 2
    panel = _build_panel(qtbot, K=2, q=q, s=s)
    tab = panel._state_tabs[0]
    _seed_model(tab, q, s)
    tab._constraint_AB_check.setChecked(True)
    text = tab._h5_badge.text()
    assert "✓" in text, f"expected ✓ in badge, got {text!r}"


def test_h5_badge_amber_for_generic_model(qtbot):
    """The (H5) badge displays the ⚠ glyph for a model that does not satisfy (H5)."""
    q, s = 2, 2
    panel = _build_panel(qtbot, K=2, q=q, s=s)
    tab = panel._state_tabs[0]
    F, SW = _seed_model(tab, q, s)

    # Sanity: the seeded model is generically non-(H5)-compatible.
    res = compute_h5_residual(
        A=F[:q, :q],
        B=F[:q, q:],
        C=F[q:, :q],
        D=F[q:, q:],
        SU=SW[:q, :q],
        Dt=SW[:q, q:],
        SV=SW[q:, q:],
    )
    assert np.linalg.norm(res, "fro") > 1e-6, "test seed should yield a non-(H5)-compatible model"

    text = tab._h5_badge.text()
    assert "⚠" in text, f"expected ⚠ in badge, got {text!r}"


# ---------------------------------------------------------------------------
# Apply AB → all
# ---------------------------------------------------------------------------
def test_apply_AB_all_propagates_across_tabs(qtbot):
    """Triggering the corner button copies the AB-constraint state to every tab."""
    q, s = 2, 2
    panel = _build_panel(qtbot, K=3, q=q, s=s)
    for tab in panel._state_tabs:
        _seed_model(tab, q, s)

    panel._tabs.setCurrentIndex(0)
    panel._state_tabs[0]._constraint_AB_check.setChecked(True)

    # Other tabs are still unchecked at this point.
    for tab in panel._state_tabs[1:]:
        assert tab.is_AB_constraint_active() is False

    panel._on_apply_AB_all()

    # Now every tab has the AB constraint active.
    for tab in panel._state_tabs:
        assert tab.is_AB_constraint_active() is True
