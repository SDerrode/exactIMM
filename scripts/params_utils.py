#!/usr/bin/env python3
"""
scripts/params_utils.py
=======================
Utilities to convert the output dict of :func:`prg.learning.supervised.fit_supervised`
(or the EM dict of :func:`prg.learning.semi_supervised.fit_semi_supervised`) into a
validated :class:`prg.classes.GSSParams.GSSParams` ready to be fed into
:class:`prg.filter.gss_filter.GSSFilter` or any baseline filter.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from prg.classes.FMatrix import FMatrix
from prg.classes.GSSParams import GSSParams
from prg.classes.NoiseCovariance import GSSNoiseCovariance


def params_from_dict(d: dict[str, Any]) -> GSSParams:
    """
    Build a :class:`GSSParams` from a dict as returned by
    :func:`fit_supervised` or :func:`fit_semi_supervised`.

    Required keys:
        K, q, s, P, A_list, B_list, C_list, D_list,
        Sigma_U_list, Delta_list, Sigma_V_list,
        mu_z0_list, Sigma_z0_list

    Optional keys:
        pi0, b_list
    """
    required = (
        "K",
        "q",
        "s",
        "P",
        "A_list",
        "B_list",
        "C_list",
        "D_list",
        "Sigma_U_list",
        "Delta_list",
        "Sigma_V_list",
        "mu_z0_list",
        "Sigma_z0_list",
    )
    missing = [k for k in required if k not in d]
    if missing:
        raise KeyError(f"Dict is missing keys: {missing}")

    f_matrix = FMatrix(
        K=d["K"],
        q=d["q"],
        s=d["s"],
        A_list=d["A_list"],
        B_list=d["B_list"],
        C_list=d["C_list"],
        D_list=d["D_list"],
    )
    noise_cov = GSSNoiseCovariance(
        K=d["K"],
        q=d["q"],
        s=d["s"],
        Sigma_U_list=d["Sigma_U_list"],
        Delta_list=d["Delta_list"],
        Sigma_V_list=d["Sigma_V_list"],
    )
    return GSSParams(
        K=d["K"],
        q=d["q"],
        s=d["s"],
        P=np.asarray(d["P"], dtype=float),
        f_matrix=f_matrix,
        noise_cov=noise_cov,
        pi0=d.get("pi0"),
        mu_z0_list=d["mu_z0_list"],
        Sigma_z0_list=d["Sigma_z0_list"],
        b_list=d.get("b_list"),
    )
