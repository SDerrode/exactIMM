#!/usr/bin/env python3
"""
prg/models/presets.py
=====================
Central registry of named preset models for the GUI combobox.

Each PresetEntry stores enough metadata to display the model in the
dropdown and to instantiate it on demand (lazy import).
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from prg.models.base_gss_model import BaseGSSModel

__all__ = ["PresetEntry", "PRESETS"]


@dataclass(frozen=True)
class PresetEntry:
    label: str  # shown in the combobox
    tooltip: str  # shown on hover
    module_name: str  # e.g. "model_gss_K2_q1_s1"
    class_name: str  # e.g. "ModelGssK2Q1S1"
    K: int
    q: int
    s: int
    p: int = 0  # exogenous-input ("consigne") dimension (0 = autonomous)

    def load(self) -> BaseGSSModel:
        """Instantiate and return the model (lazy import)."""
        mod = importlib.import_module(f"prg.models.{self.module_name}")
        return getattr(mod, self.class_name)()


# ---------------------------------------------------------------------------
# Preset list  (displayed in this order in the combobox)
# ---------------------------------------------------------------------------

PRESETS: list[PresetEntry] = [
    PresetEntry(
        label="K=2, q=1, s=1 — Reference",
        tooltip="Reference model: 2 regimes, 1 hidden variable, 1 observed.",
        module_name="model_gss_K2_q1_s1",
        class_name="ModelGssK2Q1S1",
        K=2,
        q=1,
        s=1,
    ),
    PresetEntry(
        label="K=2, q=1, s=2 — Slow / Fast",
        tooltip="2 regimes (slow / fast), 1 hidden variable, 2 observed.",
        module_name="model_gss_K2_q1_s2",
        class_name="ModelGss_K2_q1_s2",
        K=2,
        q=1,
        s=2,
    ),
    # NOTE: the K=2, q=2, s=1 model is not offered as a preset. Its raw blocks are
    # not AB-constrained, so it is not a valid NGH-MSM (the AB residual is
    # nonzero). Note that s < q (rank-deficient C) is NOT itself a defect: an
    # AB-constrained s < q model IS a valid NGH-MSM (the full-column-rank condition
    # was an over-restriction). The file model_gss_K2_q2_s1.py is kept as a test
    # fixture for the non-AB / sub-determined (K·s < q+s) case.
    PresetEntry(
        label="K=2, q=2, s=2 — Stable / Active",
        tooltip="2 regimes (stable / active), 2 hidden variables, 2 observed.",
        module_name="model_gss_K2_q2_s2",
        class_name="ModelGss_K2_q2_s2",
        K=2,
        q=2,
        s=2,
    ),
    PresetEntry(
        label="K=3, q=1, s=1 — Calm / Medium / Turbulent",
        tooltip="3 regimes (calm / medium / turbulent), 1 hidden variable, 1 observed.",
        module_name="model_gss_K3_q1_s1",
        class_name="ModelGss_K3_q1_s1",
        K=3,
        q=1,
        s=1,
    ),
    PresetEntry(
        label="K=2, q=1, s=1 — Consigne (exogenous input)",
        tooltip="NGH-MSM with a 1-D exogenous input (consigne); the hidden-state "
        "gain G^X flips sign across regimes.",
        module_name="model_gss_K2_q1_s1_consigne",
        class_name="ModelGssK2Q1S1Consigne",
        K=2,
        q=1,
        s=1,
        p=1,
    ),
]
