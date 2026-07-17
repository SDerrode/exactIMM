#!/usr/bin/env python3
"""
tests/test_filter_modes.py
==========================
Equivalence of the two GSSFilter modes under AB, across dimensions.

Regression guard for the gpb2 pair-variance bug: on any
AB-constrained model the time-varying ``gpb2`` recursion and the
constant-gain ``ngh_kf`` recursion must agree to machine precision. The
original suite only exercised the scalar case K=2, q=1, s=1, which let the bug
through; here we also cover K>2 and q, s > 1.
"""

from __future__ import annotations

import numpy as np
import pytest

from prg.classes.FMatrix import FMatrix
from prg.classes.GSSParams import GSSParams
from prg.classes.GSSSimulator import GSSSimulator
from prg.classes.NoiseCovariance import GSSNoiseCovariance
from prg.experiments.models_paper import get_params
from prg.experiments.reference_filters import with_stationary_init
from prg.experiments.run_simulations import _params_from_dict
from prg.filter.gss_filter import GSSFilter
from prg.utils.ab_constraint import ab_residual_max, apply_AB_constraint


def _random_ab_model(K: int, q: int, s: int, seed: int, scale: float = 0.3) -> GSSParams:
    """Build a *stable*, AB-constrained GSSParams for arbitrary (K, q, s).

    Retries with incremented seeds until the spectral radius of every F(k) is
    < 0.95, so simulated trajectories stay bounded.
    """
    dz = q + s
    for attempt in range(64):
        rng = np.random.default_rng(seed + attempt)
        P = rng.random((K, K)) + 0.5
        P /= P.sum(axis=1, keepdims=True)
        f = FMatrix(
            K=K,
            q=q,
            s=s,
            A_list=[np.zeros((q, q)) for _ in range(K)],
            B_list=[np.zeros((q, s)) for _ in range(K)],
            C_list=[scale * rng.standard_normal((s, q)) for _ in range(K)],
            D_list=[scale * rng.standard_normal((s, s)) for _ in range(K)],
        )
        SU, Dt, SV = [], [], []
        for _ in range(K):
            M = rng.standard_normal((dz, dz))
            J = M @ M.T + dz * np.eye(dz)  # SPD joint noise covariance
            SU.append(J[:q, :q].copy())
            Dt.append(J[:q, q:].copy())
            SV.append(J[q:, q:].copy())
        nc = GSSNoiseCovariance(K=K, q=q, s=s, Sigma_U_list=SU, Delta_list=Dt, Sigma_V_list=SV)
        params = apply_AB_constraint(
            GSSParams(
                K=K,
                q=q,
                s=s,
                P=P,
                f_matrix=f,
                noise_cov=nc,
                pi0=None,
                mu_z0_list=[np.zeros((dz, 1)) for _ in range(K)],
                Sigma_z0_list=[np.eye(dz) for _ in range(K)],
            )
        )
        rho = max(float(np.max(np.abs(np.linalg.eigvals(params.f_matrix.F(k))))) for k in range(K))
        if rho < 0.95:
            return params
    raise RuntimeError(f"could not build a stable AB model for (K={K}, q={q}, s={s})")


def _run(params: GSSParams, mode: str, n: int = 300, seed: int = 2):
    filt = GSSFilter(params, mode=mode)
    sim = GSSSimulator(params, N=n, seed=seed)
    pis, exs, ll = [], [], 0.0
    for tup in sim:
        r = filt.step(tup[-1])
        pis.append(np.asarray(r.pi).ravel())
        exs.append(np.asarray(r.E_x).ravel())
        ll += float(r.log_lik)
    return np.array(pis), np.array(exs), ll


def _assert_modes_agree(params: GSSParams) -> None:
    """Both modes must agree on an AB model *at stationarity*.

    The stationary initialisation is not incidental, it is ``ngh_kf``'s
    precondition: that mode reads the state off pre-computed per-regime moments,
    which are the true ones only once the chain sits at its stationary law.
    ``gpb2`` instead starts from the model's stated p(Z_0 | r_0). On a model
    initialised away from stationarity the two therefore disagree — by ~1e-1 on
    M1--M3 — and that is not a bug: ``gpb2`` is the one honouring the stated
    prior. (Before v0.10 this held silently, because the superseded
    ``imm_general`` recursion also started from the stationary moments.)
    """
    params = with_stationary_init(params)
    pe, xe, le = _run(params, "ngh_kf")
    pg, xg, lg = _run(params, "gpb2")
    assert np.all(np.isfinite(pe)) and np.all(np.isfinite(pg))
    assert np.abs(pe - pg).max() < 1e-8, f"max|Δπ| = {np.abs(pe - pg).max():.2e}"
    assert np.abs(xe - xg).max() < 1e-8, f"max|ΔE_x| = {np.abs(xe - xg).max():.2e}"
    assert abs(le - lg) < 1e-6 * max(1.0, abs(le)), f"Δlog_lik = {abs(le - lg):.2e}"


class TestModeEquivalenceUnderAB:
    @pytest.mark.parametrize("name", ["M1", "M2", "M3"])
    def test_paper_models(self, name):
        """ngh_kf ≡ gpb2 on the AB-constrained paper models (incl. M2: q=s=2)."""
        _assert_modes_agree(_params_from_dict(get_params(name)))

    @pytest.mark.parametrize("K,q,s", [(3, 2, 2), (3, 2, 1), (4, 1, 1), (2, 3, 2)])
    def test_generated_models(self, K, q, s):
        """ngh_kf ≡ gpb2 on freshly built AB models with K>2 and/or q,s>1."""
        params = _random_ab_model(K, q, s, seed=1)
        h5max, _ = ab_residual_max(params)
        assert h5max < 1e-8, f"built model is not AB-constrained: {h5max:.2e}"
        _assert_modes_agree(params)
