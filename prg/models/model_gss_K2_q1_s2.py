#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prg/models/model_gss_K2_q1_s2.py
=================================
GSS model: K=2 états, q=1 (dim cachée), s=2 (dim observée).

Interprétation
--------------
  k=0  "régime lent"   — dynamique douce, faible bruit
  k=1  "régime rapide" — dynamique plus agitée, bruit plus fort

Matrice de transition (diagonale dominante — peu de transitions)
----------------------------------------------------------------
  P = [[0.95, 0.05],
       [0.05, 0.95]]

A_k, B_k sont dérivés de (C, D, Δ, Σ_V) via la contrainte AB
(A = Δ Σ_V⁻¹ C, B = Δ Σ_V⁻¹ D) dans get_params(), de sorte que le modèle
vérifie (H5) par construction. La stabilité (|λ(F(k))| < 1) est gouvernée par D
(contractif) ; Σ_W(k) est SPD :
  k=0 : max|λ(F)| = 0.618   min λ(Σ_W) ≈ 0.17
  k=1 : max|λ(F)| = 0.449   min λ(Σ_W) ≈ 0.31

Conditions initiales : pi0=None (stationnaire), Z_0 ~ N(0, I_3).
"""

from __future__ import annotations

import numpy as np

from prg.models.base_gss_model import BaseGSSModel

__all__ = ["ModelGss_K2_q1_s2"]


class ModelGss_K2_q1_s2(BaseGSSModel):
    """Two-state GSS: scalar hidden state X, 2-D observation Y."""

    K: int = 2
    q: int = 1
    s: int = 2

    # --- Markov chain ---
    P: np.ndarray = np.array([[0.95, 0.05],
                               [0.05, 0.95]])

    # --- Dynamics: F(k) = [[A_k, B_k], [C_k, D_k]] ---
    #   A_k : (1,1)   B_k : (1,2)   C_k : (2,1)   D_k : (2,2)
    # A_k, B_k are derived from (C, D, Δ, Σ_V) in get_params() via the AB
    # constraint (A = Δ Σ_V⁻¹ C, B = Δ Σ_V⁻¹ D) — (H5) holds by construction.
    C_list: list[np.ndarray] = [
        np.array([[0.20],                # k=0 : X observe les deux Y
                  [0.15]]),
        np.array([[0.15],                # k=1
                  [0.12]]),
    ]
    D_list: list[np.ndarray] = [
        np.array([[0.55, 0.04],          # k=0 : λ ≈ 0.59 / 0.46
                  [0.04, 0.50]]),
        np.array([[0.40, 0.04],          # k=1 : λ ≈ 0.44 / 0.31
                  [0.04, 0.35]]),
    ]

    # --- Bruit : Σ_W(k) = [[Σ_U, Δ], [Δᵀ, Σ_V]] ---
    #   Σ_U : (1,1)   Δ : (1,2)   Σ_V : (2,2) SPD
    Sigma_U_list: list[np.ndarray] = [
        np.array([[0.20]]),              # k=0 : faible bruit sur X
        np.array([[0.55]]),              # k=1 : fort bruit sur X
    ]
    Delta_list: list[np.ndarray] = [
        np.array([[0.04, 0.03]]),        # k=0 : couplage croisé faible
        np.array([[0.05, 0.04]]),        # k=1
    ]
    Sigma_V_list: list[np.ndarray] = [
        np.array([[0.25, 0.04],          # k=0
                  [0.04, 0.20]]),
        np.array([[0.40, 0.06],          # k=1
                  [0.06, 0.35]]),
    ]

    # --- Conditions initiales ---
    pi0: np.ndarray | None = None   # None → distribution stationnaire

    mu_z0_list: list[np.ndarray] = [
        np.zeros((3, 1)),   # k=0
        np.zeros((3, 1)),   # k=1
    ]
    Sigma_z0_list: list[np.ndarray] = [
        np.eye(3),           # k=0
        np.eye(3),           # k=1
    ]

    # --- Drift bias b(k) — zero by default ---
    b_list: list[np.ndarray] = [
        np.zeros((3, 1)),   # k=0
        np.zeros((3, 1)),   # k=1
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
