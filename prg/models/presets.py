#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
    label:       str   # shown in the combobox
    tooltip:     str   # shown on hover
    module_name: str   # e.g. "model_gss_K2_q1_s1"
    class_name:  str   # e.g. "ModelGssK2Q1S1"
    K: int
    q: int
    s: int

    def load(self) -> "BaseGSSModel":
        """Instantiate and return the model (lazy import)."""
        mod = importlib.import_module(f"prg.models.{self.module_name}")
        return getattr(mod, self.class_name)()


# ---------------------------------------------------------------------------
# Preset list  (displayed in this order in the combobox)
# ---------------------------------------------------------------------------

PRESETS: list[PresetEntry] = [
    PresetEntry(
        label   = "K=2, q=1, s=1 — Référence",
        tooltip = "Modèle de référence : 2 régimes, 1 variable cachée, 1 observée.",
        module_name = "model_gss_K2_q1_s1",
        class_name  = "ModelGssK2Q1S1",
        K=2, q=1, s=1,
    ),
    PresetEntry(
        label   = "K=2, q=1, s=2 — Lent / Rapide",
        tooltip = "2 régimes (lent / rapide), 1 variable cachée, 2 observées.",
        module_name = "model_gss_K2_q1_s2",
        class_name  = "ModelGss_K2_q1_s2",
        K=2, q=1, s=2,
    ),
    PresetEntry(
        label   = "K=2, q=2, s=1 — Lent / Rapide",
        tooltip = "2 régimes (lent / rapide), 2 variables cachées, 1 observée.",
        module_name = "model_gss_K2_q2_s1",
        class_name  = "ModelGss_K2_q2_s1",
        K=2, q=2, s=1,
    ),
    PresetEntry(
        label   = "K=2, q=2, s=2 — Stable / Actif",
        tooltip = "2 régimes (stable / actif), 2 variables cachées, 2 observées.",
        module_name = "model_gss_K2_q2_s2",
        class_name  = "ModelGss_K2_q2_s2",
        K=2, q=2, s=2,
    ),
    PresetEntry(
        label   = "K=3, q=1, s=1 — Calme / Moyen / Agité",
        tooltip = "3 régimes (calme / moyen / agité), 1 variable cachée, 1 observée.",
        module_name = "model_gss_K3_q1_s1",
        class_name  = "ModelGss_K3_q1_s1",
        K=3, q=1, s=1,
    ),
]
