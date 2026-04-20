#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prg/models/model_em_K2_q1_s1.py
===============================
GSS model: K=2 states, q=1 (hidden), s=1 (observed).

Estimated by Baum-Welch EM from (X,Y) data — R hidden.  (semi-supervised EM, K=2, log L=-2874.4626, 8 iters)
Source CSV : /Users/MacBook_Derrode/Documents/Recherche/Wojciech/FofGss/fofgss/data/simulated/simulated_model_gss_K2_q1_s1_N3000_seed42.csv
Estimated  : 2026-04-19 17:33:20
Constraint : none
Delta=0    : no
"""

from __future__ import annotations

import numpy as np

from prg.models.base_gss_model import BaseGSSModel

__all__ = ["ModelEmK2Q1S1"]


class ModelEmK2Q1S1(BaseGSSModel):
    """GSS model estimated from simulated_model_gss_K2_q1_s1_N3000_seed42.csv (K=2, q=1, s=1)."""

    K: int = 2
    q: int = 1
    s: int = 1

    # --- Markov chain ---
    P: np.ndarray = np.array([[0.9631959852, 0.03680401482],
          [0.01709014523, 0.9829098548]])

    # --- Dynamics: F(k) = [[A_k, B_k], [C_k, D_k]] ---
    A_list: list[np.ndarray] = [np.array([[0.7804031498]]),
     np.array([[0.4829083184]])]
    B_list: list[np.ndarray] = [np.array([[0.1206032618]]),
     np.array([[0.3140060654]])]
    C_list: list[np.ndarray] = [np.array([[0.1644050145]]),
     np.array([[0.1169821083]])]
    D_list: list[np.ndarray] = [np.array([[0.7365261328]]),
     np.array([[0.5861557568]])]

    # --- Noise covariances: Σ_W(k) = [[Σ_U, Δ], [Δᵀ, Σ_V]] ---
    Sigma_U_list: list[np.ndarray] = [np.array([[0.09987158294]]),
     np.array([[0.193392007]])]
    Delta_list:   list[np.ndarray] = [np.array([[0.05236822333]]),
     np.array([[0.01835878659]])]
    Sigma_V_list: list[np.ndarray] = [np.array([[0.1006790317]]),
     np.array([[0.1511342988]])]

    # --- Drift bias ---
    b_list: list[np.ndarray] = [np.array([[0.9565453408],
          [1.898830456]]),
     np.array([[-2.053855096],
          [-0.9510631988]])]

    # --- Initial conditions ---
    pi0: np.ndarray | None = None   # None → stationary distribution

    mu_z0_list:    list[np.ndarray] = [np.array([[8.217135251],
          [10.6574249]]),
     np.array([[-5.765379535],
          [-3.535506487]])]
    Sigma_z0_list: list[np.ndarray] = [np.array([[27.17171196, 27.22349288],
          [27.22349288, 27.4358202]]),
     np.array([[5.170909117, 4.234561512],
          [4.234561512, 3.759943964]])]

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
