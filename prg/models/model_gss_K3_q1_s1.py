#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prg/models/model_gss_K3_q1_s1.py
=================================
GSS model: K=3 états, q=1 (dim cachée), s=1 (dim observée).

Interprétation
--------------
  k=0  "régime calme"  — dynamique lente, très faible bruit
  k=1  "régime moyen"  — dynamique intermédiaire, bruit modéré
  k=2  "régime agité"  — dynamique rapide, bruit élevé

Matrice de transition (diagonale très dominante — rares transitions)
--------------------------------------------------------------------
  P = [[0.92, 0.04, 0.04],
       [0.04, 0.92, 0.04],
       [0.04, 0.04, 0.92]]

  Distribution stationnaire : π ≈ [1/3, 1/3, 1/3]

A_k, B_k sont dérivés de (C, D, Δ, Σ_V) via la contrainte AB
(A = Δ Σ_V⁻¹ C, B = Δ Σ_V⁻¹ D) dans get_params() — (H5) par construction.
La stabilité (|λ(F(k))| < 1) est gouvernée par D (contractif) ; Σ_W(k) SPD :
  k=0 : max|λ(F)| = 0.895   min λ(Σ_W) ≈ 0.07
  k=1 : max|λ(F)| = 0.760   min λ(Σ_W) ≈ 0.22
  k=2 : max|λ(F)| = 0.560   min λ(Σ_W) ≈ 0.68

Conditions initiales : pi0=None (stationnaire), Z_0 ~ N(0, I_2).
"""

from __future__ import annotations

import numpy as np

from prg.models.base_gss_model import BaseGSSModel

__all__ = ["ModelGss_K3_q1_s1"]


class ModelGss_K3_q1_s1(BaseGSSModel):
    """Three-state GSS: scalar hidden state X, scalar observation Y."""

    K: int = 3
    q: int = 1
    s: int = 1

    # --- Markov chain ---
    P: np.ndarray = np.array([[0.92, 0.04, 0.04],
                               [0.04, 0.92, 0.04],
                               [0.04, 0.04, 0.92]])

    # --- Dynamics: F(k) = [[A_k, B_k], [C_k, D_k]] (all 1×1 scalars) ---
    # A_k, B_k are derived from (C, D, Δ, Σ_V) in get_params() via the AB
    # constraint (A = Δ Σ_V⁻¹ C, B = Δ Σ_V⁻¹ D) — (H5) holds by construction.
    C_list: list[np.ndarray] = [
        np.array([[0.15]]),   # k=0
        np.array([[0.30]]),   # k=1
        np.array([[0.10]]),   # k=2
    ]
    D_list: list[np.ndarray] = [
        np.array([[0.85]]),   # k=0 : observation très persistante
        np.array([[0.70]]),   # k=1
        np.array([[0.55]]),   # k=2
    ]

    # --- Bruit : Σ_W(k) = [[Σ_U, Δ], [Δᵀ, Σ_V]] (tous scalaires) ---
    Sigma_U_list: list[np.ndarray] = [
        np.array([[0.10]]),   # k=0 : faible bruit sur X
        np.array([[0.30]]),   # k=1
        np.array([[0.90]]),   # k=2 : fort bruit sur X
    ]
    Delta_list: list[np.ndarray] = [
        np.array([[0.03]]),   # k=0
        np.array([[0.05]]),   # k=1
        np.array([[0.07]]),   # k=2
    ]
    Sigma_V_list: list[np.ndarray] = [
        np.array([[0.10]]),   # k=0 : faible bruit sur Y
        np.array([[0.25]]),   # k=1
        np.array([[0.70]]),   # k=2 : fort bruit sur Y
    ]

    # --- Conditions initiales ---
    pi0: np.ndarray | None = None   # None → distribution stationnaire

    mu_z0_list: list[np.ndarray] = [
        np.zeros((2, 1)),   # k=0
        np.zeros((2, 1)),   # k=1
        np.zeros((2, 1)),   # k=2
    ]
    Sigma_z0_list: list[np.ndarray] = [
        np.eye(2),           # k=0
        np.eye(2),           # k=1
        np.eye(2),           # k=2
    ]

    # --- Drift bias b(k) — zero by default ---
    b_list: list[np.ndarray] = [
        np.zeros((2, 1)),   # k=0
        np.zeros((2, 1)),   # k=1
        np.zeros((2, 1)),   # k=2
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
