#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prg/models/model_gss_K2_q2_s2.py
=================================
GSS model: K=2 états, q=2 (dim cachée), s=2 (dim observée).

Interprétation
--------------
  k=0  "régime stable"  — dynamique lente, couplages faibles
  k=1  "régime actif"   — dynamique plus rapide, bruit plus élevé

Matrice de transition (diagonale dominante — peu de transitions)
----------------------------------------------------------------
  P = [[0.95, 0.05],
       [0.05, 0.95]]

A_k, B_k sont dérivés de (C, D, Δ, Σ_V) via la contrainte AB
(A = Δ Σ_V⁻¹ C, B = Δ Σ_V⁻¹ D) dans get_params() — (H5) par construction.
Pour ce préréglage, C_k est de rang colonne plein et D_k inversible (ni l'un ni
l'autre n'est requis par la CNS, qui n'exige que Σ_V ≻ 0) ; Σ_W(k) SPD :
  k=0 : max|λ(F)| = 0.608   min λ(Σ_W) ≈ 0.12
  k=1 : max|λ(F)| = 0.481   min λ(Σ_W) ≈ 0.30

Conditions initiales : pi0=None (stationnaire), Z_0 ~ N(0, I_4).
"""

from __future__ import annotations

import numpy as np

from prg.models.base_gss_model import BaseGSSModel

__all__ = ["ModelGss_K2_q2_s2"]


class ModelGss_K2_q2_s2(BaseGSSModel):
    """Two-state GSS: 2-D hidden state X, 2-D observation Y."""

    K: int = 2
    q: int = 2
    s: int = 2

    # --- Markov chain ---
    P: np.ndarray = np.array([[0.95, 0.05],
                               [0.05, 0.95]])

    # --- Dynamics: F(k) = [[A_k, B_k], [C_k, D_k]] ---
    #   A_k : (2,2)   B_k : (2,2)   C_k : (2,2)   D_k : (2,2)
    # A_k, B_k are derived from (C, D, Δ, Σ_V) in get_params() via the AB
    # constraint (A = Δ Σ_V⁻¹ C, B = Δ Σ_V⁻¹ D) — (H5) holds by construction.
    # C_k happens to be full column rank here (not required by the CNS — only
    # Σ_V ≻ 0 is); both blocks below are.
    C_list: list[np.ndarray] = [
        np.array([[0.08, 0.06],   # k=0 : faible couplage X→Y
                  [0.04, 0.10]]),
        np.array([[0.08, 0.06],   # k=1
                  [0.05, 0.10]]),
    ]
    D_list: list[np.ndarray] = [
        np.array([[0.55, 0.03],   # k=0 : λ ≈ 0.57 / 0.48
                  [0.03, 0.50]]),
        np.array([[0.42, 0.04],   # k=1 : λ ≈ 0.46 / 0.36
                  [0.04, 0.40]]),
    ]

    # --- Bruit : Σ_W(k) = [[Σ_U, Δ], [Δᵀ, Σ_V]] ---
    #   Σ_U : (2,2) SPD   Δ : (2,2)   Σ_V : (2,2) SPD
    Sigma_U_list: list[np.ndarray] = [
        np.array([[0.30, 0.05],   # k=0
                  [0.05, 0.25]]),
        np.array([[0.60, -0.08],  # k=1
                  [-0.08, 0.50]]),
    ]
    Delta_list: list[np.ndarray] = [
        np.array([[0.04, 0.03],   # k=0 : couplage croisé faible
                  [0.03, 0.05]]),
        np.array([[0.05, 0.04],   # k=1
                  [0.03, 0.06]]),
    ]
    Sigma_V_list: list[np.ndarray] = [
        np.array([[0.20, 0.04],   # k=0
                  [0.04, 0.15]]),
        np.array([[0.40, 0.06],   # k=1
                  [0.06, 0.35]]),
    ]

    # --- Conditions initiales ---
    pi0: np.ndarray | None = None   # None → distribution stationnaire

    mu_z0_list: list[np.ndarray] = [
        np.zeros((4, 1)),   # k=0
        np.zeros((4, 1)),   # k=1
    ]
    Sigma_z0_list: list[np.ndarray] = [
        np.eye(4),           # k=0
        np.eye(4),           # k=1
    ]

    # --- Drift bias b(k) — zero by default ---
    b_list: list[np.ndarray] = [
        np.zeros((4, 1)),   # k=0
        np.zeros((4, 1)),   # k=1
    ]

    # ------------------------------------------------------------------

    def get_params(self) -> dict:
        A_list, B_list = self._ab_constraint(
            self.C_list, self.D_list, self.Delta_list, self.Sigma_V_list
        )
        return {
            "K": self.K, "q": self.q, "s": self.s, "P": self.P,
            "A_list": A_list, "B_list": B_list,
            "C_list": self.C_list, "D_list": self.D_list,
            "Sigma_U_list": self.Sigma_U_list,
            "Delta_list": self.Delta_list,
            "Sigma_V_list": self.Sigma_V_list,
            "pi0": self.pi0,
            "mu_z0_list": self.mu_z0_list,
            "Sigma_z0_list": self.Sigma_z0_list,
            "b_list": self.b_list,
        }
