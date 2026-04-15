#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prg/models/model_gss_K2_q2_s1.py
=================================
GSS model: K=2 états, q=2 (dim cachée), s=1 (dim observée).

Interprétation
--------------
  k=0  "régime lent"   — dynamique douce, faible bruit
  k=1  "régime rapide" — dynamique plus agitée, bruit plus fort

Matrice de transition (diagonale dominante — peu de transitions)
----------------------------------------------------------------
  P = [[0.95, 0.05],
       [0.05, 0.95]]

Stabilité (|λ(F(k))| < 1) et SPD de Σ_W(k) vérifiées numériquement :
  k=0 : max|λ(F)| = 0.786   min λ(Σ_W) = 0.160
  k=1 : max|λ(F)| = 0.561   min λ(Σ_W) = 0.317

Conditions initiales : pi0=None (stationnaire), Z_0 ~ N(0, I_3).
"""

from __future__ import annotations

import numpy as np

from prg.models.base_gss_model import BaseGSSModel

__all__ = ["ModelGss_K2_q2_s1"]


class ModelGss_K2_q2_s1(BaseGSSModel):
    """Two-state GSS: 2-D hidden state X, scalar observation Y."""

    K: int = 2
    q: int = 2
    s: int = 1

    # --- Markov chain ---
    P: np.ndarray = np.array([[0.95, 0.05],
                               [0.05, 0.95]])

    # --- Dynamics: F(k) = [[A_k, B_k], [C_k, D_k]] ---
    #   A_k : (2,2)   B_k : (2,1)   C_k : (1,2)   D_k : (1,1)
    A_list: list[np.ndarray] = [
        np.array([[0.75, 0.05],   # k=0 : triangulaire sup., λ = 0.75 / 0.70
                  [0.00, 0.70]]),
        np.array([[0.50, 0.05],   # k=1 : triangulaire sup., λ = 0.50 / 0.45
                  [0.00, 0.45]]),
    ]
    B_list: list[np.ndarray] = [
        np.array([[0.04],         # k=0
                  [0.03]]),
        np.array([[0.05],         # k=1
                  [0.04]]),
    ]
    C_list: list[np.ndarray] = [
        np.array([[0.10, 0.08]]),  # k=0
        np.array([[0.08, 0.06]]),  # k=1
    ]
    D_list: list[np.ndarray] = [
        np.array([[0.60]]),        # k=0
        np.array([[0.45]]),        # k=1
    ]

    # --- Bruit : Σ_W(k) = [[Σ_U, Δ], [Δᵀ, Σ_V]] ---
    #   Σ_U : (2,2) SPD   Δ : (2,1)   Σ_V : (1,1)
    Sigma_U_list: list[np.ndarray] = [
        np.array([[0.30, 0.05],   # k=0
                  [0.05, 0.25]]),
        np.array([[0.60, -0.08],  # k=1
                  [-0.08, 0.50]]),
    ]
    Delta_list: list[np.ndarray] = [
        np.array([[0.04],         # k=0
                  [0.06]]),
        np.array([[0.06],         # k=1
                  [0.04]]),
    ]
    Sigma_V_list: list[np.ndarray] = [
        np.array([[0.20]]),        # k=0
        np.array([[0.35]]),        # k=1
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
