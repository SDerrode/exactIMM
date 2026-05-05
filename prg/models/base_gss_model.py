#!/usr/bin/env python3
"""
prg/models/base_gss_model.py
============================
Abstract base class for all GSS (Gaussian Switching System) models.

A concrete model is defined by subclassing BaseGSSModel and providing
the class-level attributes listed below.  The ``get_params()`` method
returns a plain dict that GSSParams can consume.

Convention (mirrors awesomepkf's BaseModelLinear pattern)
---------------------------------------------------------
  - MODEL_NAME is derived automatically from the class name
    (lower-case first letter, e.g. ``ModelGSS_K2_q1_s1`` →
    ``"modelGSS_K2_q1_s1"``).
  - All matrix lists are indexed by state k = 0, ..., K-1.
  - Column vectors use shape (dim, 1).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

__all__ = ["BaseGSSModel"]


class BaseGSSModel(ABC):
    """
    Abstract base class for GSS models.

    Subclasses must define the following **class attributes**:

    Scalars
    -------
    K : int  — number of switching states (R in {0, ..., K-1}), >= 2
    q : int  — dimension of X, >= 1
    s : int  — dimension of Y, >= 1

    Markov chain
    ------------
    P : ndarray (K, K)  — row-stochastic transition matrix

    Dynamics — one list of K arrays per sub-block
    ----------------------------------------------
    A_list : list[(q, q)]
    B_list : list[(q, s)]
    C_list : list[(s, q)]
    D_list : list[(s, s)]

    Noise covariances — one list of K arrays per sub-block
    -------------------------------------------------------
    Sigma_U_list : list[(q, q)]  — cov of noise on X
    Delta_list   : list[(q, s)]  — cross-covariance
    Sigma_V_list : list[(s, s)]  — cov of noise on Y

    Drift bias (optional — defaults to zero)
    -----------------------------------------
    b_list : list[(q+s, 1)]  — regime-dependent bias b(k) in eq (7bis).
                               If absent or None, b(k) = 0 for all k.

    Initial conditions
    ------------------
    pi0           : ndarray (K,) or None
                    Initial distribution of R_0.
                    ``None`` means "use the stationary distribution of P".
    mu_z0_list    : list of K ndarrays, each shape (q+s, 1)
                    Initial mean of Z_0 given R_0 = k.
    Sigma_z0_list : list of K ndarrays, each shape (q+s, q+s)
                    Initial covariance of Z_0 given R_0 = k.
    """

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        # Derive model_name from the class name (lower-case first letter)
        n = cls.__name__
        cls.MODEL_NAME: str = n[0].lower() + n[1:]

    @property
    def model_name(self) -> str:
        """String identifier derived from the class name."""
        return self.__class__.MODEL_NAME  # type: ignore[attr-defined]

    @abstractmethod
    def get_params(self) -> dict:
        """
        Return a dict with all model parameters, ready for GSSParams.

        Keys
        ----
        K, q, s, P,
        A_list, B_list, C_list, D_list,
        Sigma_U_list, Delta_list, Sigma_V_list,
        pi0, mu_z0_list, Sigma_z0_list
        """
        ...

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__}"
            f"(K={getattr(self, 'K', '?')}, "
            f"q={getattr(self, 'q', '?')}, "
            f"s={getattr(self, 's', '?')})>"
        )
