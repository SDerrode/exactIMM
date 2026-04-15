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

Stabilité (|λ(F(k))| < 1) et SPD de Σ_W(k) vérifiées numériquement :
  k=0 : max|λ(F)| = 0.782   min λ(Σ_W) = 0.120
  k=1 : max|λ(F)| = 0.581   min λ(Σ_W) = 0.300

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
    A_list: list[np.ndarray] = [
        np.array([[0.75, 0.05],   # k=0 : λ = 0.75 / 0.70
                  [0.00, 0.70]]),
        np.array([[0.50, 0.05],   # k=1 : λ = 0.50 / 0.45
                  [0.00, 0.45]]),
    ]
    B_list: list[np.ndarray] = [
        np.array([[0.04, 0.02],   # k=0 : faible couplage Y→X
                  [0.02, 0.03]]),
        np.array([[0.05, 0.03],   # k=1
                  [0.03, 0.04]]),
    ]
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
        return {
            "K": self.K, "q": self.q, "s": self.s, "P": self.P,
            "A_list": self.A_list, "B_list": self.B_list,
            "C_list": self.C_list, "D_list": self.D_list,
            "Sigma_U_list": self.Sigma_U_list,
            "Delta_list": self.Delta_list,
            "Sigma_V_list": self.Sigma_V_list,
            "pi0": self.pi0,
            "mu_z0_list": self.mu_z0_list,
            "Sigma_z0_list": self.Sigma_z0_list,
            "b_list": self.b_list,
        }
