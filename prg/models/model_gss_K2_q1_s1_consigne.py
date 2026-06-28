#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prg/models/model_gss_K2_q1_s1_consigne.py
=========================================
NGH-MSM demo *with an exogenous input* ("consigne"): K=2, q=1, s=1, p=1.

Same NGH-MSM blocks as ``model_gss_K2_q1_s1`` (A, B derived from the AB
constraint, so (H5) holds by construction), plus a regime-dependent input gain
G(k) = [G^X(k); G^Y(k)] whose hidden-state component **flips sign across
regimes** — so the known input u_n drives X in opposite directions depending on
the active regime, and the exact filter's ``N_k u_{n-1}`` read-out genuinely
matters.

Example
-------
    python -m prg.simulate --model model_gss_K2_q1_s1_consigne -N 500 \
        --seed 0 --input "sin(80)"
"""

from __future__ import annotations

import numpy as np

from prg.models.base_gss_model import BaseGSSModel

__all__ = ["ModelGssK2Q1S1Consigne"]


class ModelGssK2Q1S1Consigne(BaseGSSModel):
    """NGH-MSM with a 1-D exogenous input (K=2, q=1, s=1, p=1)."""

    K: int = 2
    q: int = 1
    s: int = 1

    # --- Markov chain ---
    P: np.ndarray = np.array([[0.97, 0.03],
                              [0.02, 0.98]])

    # --- Dynamics (free blocks; A, B derived via the AB constraint) ---
    C_list: list[np.ndarray] = [np.array([[0.2]]), np.array([[0.1]])]
    D_list: list[np.ndarray] = [np.array([[0.7]]), np.array([[0.6]])]

    # --- Noise covariances ---
    Sigma_U_list: list[np.ndarray] = [np.array([[0.1]]), np.array([[0.2]])]
    Delta_list:   list[np.ndarray] = [np.array([[0.05]]), np.array([[0.02]])]
    Sigma_V_list: list[np.ndarray] = [np.array([[0.1]]), np.array([[0.15]])]

    # --- Initial conditions ---
    pi0: np.ndarray | None = None
    mu_z0_list: list[np.ndarray] = [np.zeros((2, 1)), np.zeros((2, 1))]
    Sigma_z0_list: list[np.ndarray] = [np.eye(2), np.eye(2)]
    b_list: list[np.ndarray] = [np.zeros((2, 1)), np.zeros((2, 1))]

    # --- Exogenous-input gain G(k) = [G^X(k); G^Y(k)], shape (q+s, p) = (2, 1) ---
    # The hidden-state gain G^X flips sign across regimes.
    G_list: list[np.ndarray] = [np.array([[1.0], [0.3]]),
                                np.array([[-1.0], [0.2]])]

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
            "G_list": self.G_list,
        }
