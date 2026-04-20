#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prg/models/model_gss_K2_q1_s1_contrast.py
=========================================
GSS model designed to produce *visually contrasted* X^0 and Y^0 signals.

Design rationale
----------------
We want the hidden component X and the observation Y to look obviously
different in shape and amplitude — different operating points, different
dynamics, different noise levels. The trick is:

  * Decouple the cross-terms (B ≈ 0, C ≈ 0): X drives itself, Y drives
    itself.
  * Choose the diagonal terms so that the two coordinates have very
    different memory: smooth slow drift for one, fast oscillation /
    noisy regime for the other.
  * Choose biases so that the regime fixed points are well separated
    in (X, Y) — and *swapped* between the two regimes (X is large in
    regime 0 / small in regime 1, Y does the opposite).
  * Choose noise variances so that one component is smooth while the
    other is jittery.
  * Sticky transition matrix so each regime stays long enough to be
    visually identifiable.

Per regime, the dynamics read
    X_{n+1} = A X_n + B Y_n + b_X + U
    Y_{n+1} = C X_n + D Y_n + b_Y + V
with W = (U, V)^T ~ N(0, [[Σ_U, Δ], [Δ, Σ_V]]).

  Regime 0  ("X high & smooth, Y around 0 & jittery")
    A = +0.95   B ≈ 0   →  X is a slow drift, fixed point = +12
    D = -0.70   C ≈ 0   →  Y oscillates around 0, fast
    Σ_U = 0.05  (smooth)        Σ_V = 1.20  (very noisy)

  Regime 1  ("X around -6 & noisy, Y high & smooth")
    A = -0.60   B ≈ 0   →  X oscillates, fixed point = -6
    D = +0.92   C ≈ 0   →  Y is a slow drift, fixed point = +10
    Σ_U = 0.80  (noisy)         Σ_V = 0.04  (smooth)

  Sticky chain: P = [[0.985, 0.015], [0.015, 0.985]]
  → mean run length ~ 67 steps, perfect to see each regime several times
    in a length-1000 simulation.
"""

from __future__ import annotations

import numpy as np

from prg.models.base_gss_model import BaseGSSModel

__all__ = ["ModelGssK2Q1S1Contrast"]


class ModelGssK2Q1S1Contrast(BaseGSSModel):
    """Strongly contrasted K=2, q=1, s=1 GSS model."""

    K: int = 2
    q: int = 1
    s: int = 1

    # --- Markov chain (sticky) -----------------------------------------
    P: np.ndarray = np.array([[0.985, 0.015],
                              [0.015, 0.985]])

    # --- Dynamics: F(k) = [[A, B], [C, D]] -----------------------------
    # Regime 0:  A = +0.95 (slow drift), D = -0.70 (oscillation)
    # Regime 1:  A = -0.60 (oscillation), D = +0.92 (slow drift)
    A_list: list[np.ndarray] = [np.array([[ 0.95]]),
                                np.array([[-0.60]])]
    B_list: list[np.ndarray] = [np.array([[ 0.00]]),
                                np.array([[ 0.00]])]
    C_list: list[np.ndarray] = [np.array([[ 0.00]]),
                                np.array([[ 0.00]])]
    D_list: list[np.ndarray] = [np.array([[-0.70]]),
                                np.array([[ 0.92]])]

    # --- Noise covariances ---------------------------------------------
    # Regime 0: smooth X (Σ_U small), jittery Y (Σ_V large)
    # Regime 1: jittery X (Σ_U large), smooth Y (Σ_V small)
    Sigma_U_list: list[np.ndarray] = [np.array([[0.05]]),
                                      np.array([[0.80]])]
    Delta_list:   list[np.ndarray] = [np.array([[0.00]]),
                                      np.array([[0.00]])]
    Sigma_V_list: list[np.ndarray] = [np.array([[1.20]]),
                                      np.array([[0.04]])]

    # --- Biases (drive the fixed points) -------------------------------
    # Fixed point in regime k satisfies  Z* = (I - F_k)^{-1} b_k.
    # With B = C = 0 and F diagonal we get  X* = b_X / (1 - A),  Y* = b_Y / (1 - D).
    #
    # Regime 0:  X* = +12 → b_X = (1 - 0.95) * 12  = +0.60
    #            Y* =  0  → b_Y = 0
    # Regime 1:  X* = -6  → b_X = (1 - (-0.60)) * (-6) = -9.60
    #            Y* = +10 → b_Y = (1 - 0.92) * 10 = +0.80
    b_list: list[np.ndarray] = [np.array([[ 0.60],
                                          [ 0.00]]),
                                np.array([[-9.60],
                                          [ 0.80]])]

    # --- Initial conditions: start at the regime fixed point -----------
    pi0: np.ndarray | None = None   # → stationary distribution

    mu_z0_list: list[np.ndarray] = [np.array([[12.0],
                                              [ 0.0]]),
                                    np.array([[-6.0],
                                              [10.0]])]
    Sigma_z0_list: list[np.ndarray] = [np.array([[1.0, 0.0],
                                                 [0.0, 1.0]]),
                                       np.array([[1.0, 0.0],
                                                 [0.0, 1.0]])]

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
