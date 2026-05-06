#!/usr/bin/env python3
"""
prg/experiments/models_paper.py
================================
Reference GSS models used in the simulation study (§6 of the paper).

Models
------
M1  K=2, q=1, s=1  — canonical scalar case (Table §6.1)
M2  K=2, q=2, s=2  — multivariate cross-coupled case (Table 1 in §6.1)
M3  K=3, q=1, s=1  — three-regime scalar case (§6.1)

All models satisfy (H5) exactly: A(k) and B(k) are computed from
C, D, Δ, Σ_V via the closed-form (H5)-compatible AB constraint
A = Δ Σ_V⁻¹ C, B = Δ Σ_V⁻¹ D.  An assertion at module construction
time verifies the H5 residual < 1e-8 for every regime.

Usage
-----
    from prg.experiments.models_paper import get_params_M1, get_params_M2, get_params_M3
    from prg.classes.GSSParams import GSSParams

    params = GSSParams.from_dict(get_params_M1())
"""

from __future__ import annotations

import numpy as np

from prg.utils.h5_constraint import compute_AB, compute_h5_residual

__all__ = [
    "get_params_M1",
    "get_params_M2",
    "get_params_M3",
    "MODEL_NAMES",
]

MODEL_NAMES = ("M1", "M2", "M3")

_H5_ASSERT_TOL = 1e-8  # tighter than filter's H5_TOL=1e-6 for ground-truth models


def _check_h5(name: str, k: int, A, B, C, D, SU, Dt, SV) -> None:
    """Raise AssertionError if H5 residual is too large."""
    res = compute_h5_residual(A, B, C, D, SU, Dt, SV)
    Z = Dt.T @ A.T + SV @ B.T  # LHS of (H5): Δᵀ Aᵀ + Σ_V Bᵀ
    scale = max(float(np.linalg.norm(Z, "fro")), 1.0)
    rel = float(np.linalg.norm(res, "fro")) / scale
    assert rel < _H5_ASSERT_TOL, (
        f"Model {name} regime k={k}: H5 residual {rel:.2e} >= {_H5_ASSERT_TOL:.0e}"
    )


# ---------------------------------------------------------------------------
# M1  —  K=2, q=1, s=1
# ---------------------------------------------------------------------------


def get_params_M1() -> dict:
    """
    Return a parameter dict for model M1 (K=2, q=s=1).

    Transition matrix
        P = [[0.97, 0.03], [0.02, 0.98]]

    Per-regime parameters (regime index 0 = regime 1 in the paper)
        C:   [0.2]   / [0.1]
        D:   [0.7]   / [0.6]
        Σ_U: [0.10]  / [0.20]
        Δ:   [0.05]  / [0.02]
        Σ_V: [0.10]  / [0.15]
        A, B: computed from the (H5)-compatible AB constraint
              A = Δ Σ_V⁻¹ C,   B = Δ Σ_V⁻¹ D.
        b:   [0.10, 0.05]^T  /  [-0.05, 0.02]^T
    """
    K, q, s = 2, 1, 1

    P = np.array([[0.97, 0.03], [0.02, 0.98]])

    C_raw = [np.array([[0.2]]), np.array([[0.1]])]
    D_raw = [np.array([[0.7]]), np.array([[0.6]])]
    SU = [np.array([[0.10]]), np.array([[0.20]])]
    Dt = [np.array([[0.05]]), np.array([[0.02]])]
    SV = [np.array([[0.10]]), np.array([[0.15]])]

    A_list, B_list = [], []
    for k in range(K):
        A_k, B_k = compute_AB(C_raw[k], D_raw[k], Dt[k], SV[k])
        _check_h5("M1", k, A_k, B_k, C_raw[k], D_raw[k], SU[k], Dt[k], SV[k])
        A_list.append(A_k)
        B_list.append(B_k)

    dim_z = q + s
    b_list = [
        np.array([[0.10], [0.05]]),
        np.array([[-0.05], [0.02]]),
    ]
    mu_z0 = [np.zeros((dim_z, 1)) for _ in range(K)]
    Sig_z0 = [np.eye(dim_z) for _ in range(K)]

    return dict(
        K=K,
        q=q,
        s=s,
        P=P,
        A_list=A_list,
        B_list=B_list,
        C_list=C_raw,
        D_list=D_raw,
        Sigma_U_list=SU,
        Delta_list=Dt,
        Sigma_V_list=SV,
        b_list=b_list,
        pi0=None,
        mu_z0_list=mu_z0,
        Sigma_z0_list=Sig_z0,
    )


# ---------------------------------------------------------------------------
# M2  —  K=2, q=2, s=2
# ---------------------------------------------------------------------------


def get_params_M2() -> dict:
    """
    Return a parameter dict for model M2 (K=2, q=s=2).

    Full 2×2 matrices per regime; A, B computed from the (H5)-compatible
    AB constraint A = Δ Σ_V⁻¹ C, B = Δ Σ_V⁻¹ D.
    No bias (b_r = 0) to isolate the cross-coupling effect.
    See Table 1 in §6.1 of the paper.
    """
    K, q, s = 2, 2, 2

    P = np.array([[0.97, 0.03], [0.02, 0.98]])

    C_raw = [
        np.array([[0.30, 0.10], [0.05, 0.20]]),
        np.array([[0.20, 0.05], [0.10, 0.15]]),
    ]
    D_raw = [
        np.array([[0.60, 0.05], [0.00, 0.55]]),
        np.array([[0.50, 0.10], [0.05, 0.45]]),
    ]
    SU = [
        np.array([[0.10, 0.03], [0.03, 0.12]]),
        np.array([[0.15, 0.04], [0.04, 0.18]]),
    ]
    Dt = [
        np.array([[0.03, 0.01], [0.01, 0.04]]),
        np.array([[0.02, 0.01], [0.00, 0.03]]),
    ]
    SV = [
        np.array([[0.08, 0.02], [0.02, 0.10]]),
        np.array([[0.12, 0.03], [0.03, 0.14]]),
    ]

    A_list, B_list = [], []
    for k in range(K):
        A_k, B_k = compute_AB(C_raw[k], D_raw[k], Dt[k], SV[k])
        _check_h5("M2", k, A_k, B_k, C_raw[k], D_raw[k], SU[k], Dt[k], SV[k])
        A_list.append(A_k)
        B_list.append(B_k)

    dim_z = q + s
    b_list = [np.zeros((dim_z, 1)) for _ in range(K)]
    mu_z0 = [np.zeros((dim_z, 1)) for _ in range(K)]
    Sig_z0 = [np.eye(dim_z) for _ in range(K)]

    return dict(
        K=K,
        q=q,
        s=s,
        P=P,
        A_list=A_list,
        B_list=B_list,
        C_list=C_raw,
        D_list=D_raw,
        Sigma_U_list=SU,
        Delta_list=Dt,
        Sigma_V_list=SV,
        b_list=b_list,
        pi0=None,
        mu_z0_list=mu_z0,
        Sigma_z0_list=Sig_z0,
    )


# ---------------------------------------------------------------------------
# M3  —  K=3, q=1, s=1
# ---------------------------------------------------------------------------


def get_params_M3() -> dict:
    """
    Return a parameter dict for model M3 (K=3, q=s=1).

    Three regimes: two persistent (0 and 2) and one transient (1).
    Non-zero bias on regimes 0 and 2.
    A, B computed from the AB constraint A = Δ Σ_V⁻¹ C, B = Δ Σ_V⁻¹ D.
    """
    K, q, s = 3, 1, 1

    P = np.array([[0.95, 0.04, 0.01], [0.10, 0.80, 0.10], [0.02, 0.03, 0.95]])

    C_vals = [0.25, 0.40, 0.10]
    D_vals = [0.65, 0.50, 0.75]
    SU_vals = [0.08, 0.25, 0.12]
    Dt_vals = [0.04, 0.02, 0.06]
    SV_vals = [0.09, 0.18, 0.11]

    C_raw = [np.array([[v]]) for v in C_vals]
    D_raw = [np.array([[v]]) for v in D_vals]
    SU = [np.array([[v]]) for v in SU_vals]
    Dt = [np.array([[v]]) for v in Dt_vals]
    SV = [np.array([[v]]) for v in SV_vals]

    A_list, B_list = [], []
    for k in range(K):
        A_k, B_k = compute_AB(C_raw[k], D_raw[k], Dt[k], SV[k])
        _check_h5("M3", k, A_k, B_k, C_raw[k], D_raw[k], SU[k], Dt[k], SV[k])
        A_list.append(A_k)
        B_list.append(B_k)

    dim_z = q + s
    b_list = [
        np.array([[0.08], [0.03]]),
        np.zeros((dim_z, 1)),
        np.array([[-0.06], [-0.02]]),
    ]
    mu_z0 = [np.zeros((dim_z, 1)) for _ in range(K)]
    Sig_z0 = [np.eye(dim_z) for _ in range(K)]

    return dict(
        K=K,
        q=q,
        s=s,
        P=P,
        A_list=A_list,
        B_list=B_list,
        C_list=C_raw,
        D_list=D_raw,
        Sigma_U_list=SU,
        Delta_list=Dt,
        Sigma_V_list=SV,
        b_list=b_list,
        pi0=None,
        mu_z0_list=mu_z0,
        Sigma_z0_list=Sig_z0,
    )


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_BUILDERS = {"M1": get_params_M1, "M2": get_params_M2, "M3": get_params_M3}


def get_params(model_name: str) -> dict:
    """Return the parameter dict for *model_name* in {'M1', 'M2', 'M3'}."""
    if model_name not in _BUILDERS:
        raise ValueError(f"Unknown model {model_name!r}. Choose from {list(_BUILDERS)}.")
    return _BUILDERS[model_name]()
