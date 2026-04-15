#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prg/models/model_gss_K2_q1_s1.py
=================================
Minimal GSS model with K=2 switching states, q=1 (dim X), s=1 (dim Y).

This model is primarily intended for unit tests and as a template
for building more complex models.

Model structure (all sub-blocks are 1x1 scalars)
-------------------------------------------------
Two regimes: k=0 "slow" and k=1 "fast".

State k=0:  F(0) = [[0.8,  0.1],    Sigma_W(0) = [[0.10, 0.05],
                    [0.2,  0.7]]                   [0.05, 0.10]]

State k=1:  F(1) = [[0.5,  0.3],    Sigma_W(1) = [[0.20, 0.02],
                    [0.1,  0.6]]                   [0.02, 0.15]]

Markov chain:  P = [[0.9, 0.1],   stationary pi ≈ [0.667, 0.333]
                    [0.2, 0.8]]

Initial conditions: pi0 = None (stationary), Z_0 ~ N(0, I_2) for both k.
"""

from __future__ import annotations

import numpy as np

from prg.models.base_gss_model import BaseGSSModel

__all__ = ["ModelGss_K2_q1_s1"]


class ModelGss_K2_q1_s1(BaseGSSModel):
    """
    Two-state GSS model with scalar X and scalar Y.

    All 2×2 sub-blocks are 1×1 (scalar wrapped in an array).
    Sigma_W(k) is verified to be positive definite:
      - Sigma_W(0): det = 0.1*0.1 - 0.05^2 = 0.0075 > 0  ✓
      - Sigma_W(1): det = 0.2*0.15 - 0.02^2 = 0.0296 > 0  ✓
    """

    K: int = 2
    q: int = 1
    s: int = 1

    # --- Markov chain ---
    P: np.ndarray = np.array([[0.97, 0.03], [0.02, 0.98]])

    # --- Dynamics: F(k) = [[A_k, B_k], [C_k, D_k]] ---
    A_list: list[np.ndarray] = [np.array([[0.8]]), np.array([[0.5]])]
    B_list: list[np.ndarray] = [np.array([[0.1]]), np.array([[0.3]])]
    C_list: list[np.ndarray] = [np.array([[0.2]]), np.array([[0.1]])]
    D_list: list[np.ndarray] = [np.array([[0.7]]), np.array([[0.6]])]

    # --- Noise covariances: Sigma_W(k) = [[Sigma_U, Delta], [Delta^T, Sigma_V]] ---
    Sigma_U_list: list[np.ndarray] = [np.array([[0.10]]), np.array([[0.20]])]
    Delta_list: list[np.ndarray] = [np.array([[0.05]]), np.array([[0.02]])]
    Sigma_V_list: list[np.ndarray] = [np.array([[0.10]]), np.array([[0.15]])]

    # --- Initial conditions ---
    pi0: np.ndarray | None = None  # None → stationary distribution

    mu_z0_list: list[np.ndarray] = [
        np.zeros((2, 1)),  # k=0
        np.zeros((2, 1)),  # k=1
    ]
    Sigma_z0_list: list[np.ndarray] = [
        np.eye(2),  # k=0
        np.eye(2),  # k=1
    ]

    # --- Drift bias b(k) — zero by default ---
    b_list: list[np.ndarray] = [
        np.zeros((2, 1)),  # k=0
        np.zeros((2, 1)),  # k=1
    ]

    # ------------------------------------------------------------------

    def get_params(self) -> dict:
        """Return a dict of all parameters for GSSParams.from_model()."""
        return {
            "K": self.K,
            "q": self.q,
            "s": self.s,
            "P": self.P,
            "A_list": self.A_list,
            "B_list": self.B_list,
            "C_list": self.C_list,
            "D_list": self.D_list,
            "Sigma_U_list": self.Sigma_U_list,
            "Delta_list": self.Delta_list,
            "Sigma_V_list": self.Sigma_V_list,
            "pi0": self.pi0,
            "mu_z0_list": self.mu_z0_list,
            "Sigma_z0_list": self.Sigma_z0_list,
            "b_list": self.b_list,
        }
