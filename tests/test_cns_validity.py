#!/usr/bin/env python3
"""
tests/test_cns_validity.py
==========================
Guard tests for the corrected NGH-MSM condition (Proposition 2) and the
closed-form regime-conditional moments.

Coverage
--------
- Every GUI preset is a *valid* NGH-MSM (AB constraint + s ≥ q, rank(C) = q,
  D invertible, Σ_V ≻ 0, Γ ⪰ 0).  This would fail on the pre-fix presets,
  which all violated the AB constraint — it is the regression guard for P0.
- The retired s < q model is correctly rejected.
- ``NoiseCovariance.M`` / ``Gamma`` match the closed forms, and under AB
  ``A = M C``, ``B = M D``.
- ``validate_ngh_msm`` / ``GSSParams.check_ngh_msm`` flag each violated
  condition (AB residual, singular D, s < q).
"""

from __future__ import annotations

import numpy as np
import pytest

from prg.classes.FMatrix import FMatrix
from prg.classes.GSSParams import GSSParams, NGHMSMParams
from prg.classes.GSSSimulator import GSSSimulator
from prg.classes.NoiseCovariance import GSSNoiseCovariance
from prg.filter.gss_filter import GSSFilter
from prg.models.model_gss_K2_q1_s1 import ModelGssK2Q1S1
from prg.models.model_gss_K2_q2_s1 import ModelGss_K2_q2_s1
from prg.models.presets import PRESETS
from prg.utils.exceptions import ParamError
from prg.utils.h5_constraint import (
    apply_AB_constraint,
    compute_AB,
    is_ngh_msm,
    validate_ngh_msm,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _params(A, B, C, D, SU, Dt, SV, *, K=2, q=1, s=1):
    """Build a GSSParams from explicit block lists (zero mean, Z0 ~ N(0, I))."""
    dim = q + s
    fm = FMatrix(K=K, q=q, s=s, A_list=A, B_list=B, C_list=C, D_list=D)
    nc = GSSNoiseCovariance(K=K, q=q, s=s, Sigma_U_list=SU, Delta_list=Dt, Sigma_V_list=SV)
    return GSSParams(
        K=K,
        q=q,
        s=s,
        P=np.full((K, K), 1.0 / K),
        f_matrix=fm,
        noise_cov=nc,
        pi0=None,
        mu_z0_list=[np.zeros((dim, 1)) for _ in range(K)],
        Sigma_z0_list=[np.eye(dim) for _ in range(K)],
    )


def _free_blocks():
    """Free blocks (C, D, Σ_U, Δ, Σ_V) of a small valid NGH-MSM (K=2, q=1, s=1)."""
    C = [np.array([[0.2]]), np.array([[0.1]])]
    D = [np.array([[0.7]]), np.array([[0.6]])]
    SU = [np.array([[0.1]]), np.array([[0.2]])]
    Dt = [np.array([[0.05]]), np.array([[0.02]])]
    SV = [np.array([[0.1]]), np.array([[0.15]])]
    return C, D, SU, Dt, SV


def _valid_ab_params():
    """A small, valid NGH-MSM model (K=2, q=1, s=1), built via the deriving factory."""
    C, D, SU, Dt, SV = _free_blocks()
    return NGHMSMParams.from_free_blocks(
        K=2,
        q=1,
        s=1,
        P=np.full((2, 2), 0.5),
        C_list=C,
        D_list=D,
        Sigma_U_list=SU,
        Delta_list=Dt,
        Sigma_V_list=SV,
        pi0=None,
        mu_z0_list=[np.zeros((2, 1)) for _ in range(2)],
        Sigma_z0_list=[np.eye(2) for _ in range(2)],
    )


# ---------------------------------------------------------------------------
# Presets are valid NGH-MSM (regression guard for P0)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("entry", PRESETS, ids=lambda e: e.module_name)
def test_preset_is_valid_ngh_msm(entry):
    params = GSSParams.from_model(entry.load())
    issues = validate_ngh_msm(params)
    assert issues == [], f"{entry.module_name} violates the CNS:\n  - " + "\n  - ".join(issues)


@pytest.mark.parametrize("entry", PRESETS, ids=lambda e: e.module_name)
def test_preset_satisfies_s_ge_q(entry):
    assert entry.s >= entry.q, f"{entry.module_name} has s={entry.s} < q={entry.q}"


def test_retired_s_lt_q_model_is_rejected():
    params = GSSParams.from_model(ModelGss_K2_q2_s1())
    issues = validate_ngh_msm(params)
    assert any("s = 1 < q = 2" in m for m in issues)
    assert any("rank(C)" in m for m in issues)
    assert not is_ngh_msm(params)


# ---------------------------------------------------------------------------
# Closed-form regime moments M_k, Γ_k
# ---------------------------------------------------------------------------
def test_M_Gamma_closed_form_and_AB_identities():
    params = GSSParams.from_model(ModelGssK2Q1S1())
    nc = params.noise_cov
    fm = params.f_matrix
    for k in range(params.K):
        SV = nc.Sigma_V(k)
        Dt = nc.Delta(k)
        SU = nc.Sigma_U(k)
        SV_inv = np.linalg.inv(SV)
        # M_k = Δ_k Σ_V_k⁻¹  ;  Γ_k = Σ_U_k − Δ_k Σ_V_k⁻¹ Δ_k^T
        np.testing.assert_allclose(nc.M(k), Dt @ SV_inv, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(nc.Gamma(k), SU - Dt @ SV_inv @ Dt.T, rtol=1e-10, atol=1e-12)
        # Γ_k symmetric and PSD (Schur complement of an SPD Σ_W)
        np.testing.assert_allclose(nc.Gamma(k), nc.Gamma(k).T, atol=1e-12)
        assert np.linalg.eigvalsh(nc.Gamma(k))[0] > -1e-12
        # Under AB: A_k = M_k C_k, B_k = M_k D_k
        np.testing.assert_allclose(fm.A(k), nc.M(k) @ fm.C(k), rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(fm.B(k), nc.M(k) @ fm.D(k), rtol=1e-10, atol=1e-12)


# ---------------------------------------------------------------------------
# validate_ngh_msm flags each violated condition
# ---------------------------------------------------------------------------
def test_valid_model_has_no_issues():
    params = _valid_ab_params()
    assert validate_ngh_msm(params) == []
    assert is_ngh_msm(params)
    assert params.check_ngh_msm() == []  # raise_on_fail=True must not raise


def test_non_ab_model_is_flagged():
    p = _valid_ab_params()
    # Perturb A(0) off the AB manifold (stays stable: 0.x + 0.3 < 1).
    A = [p.f_matrix.A(k).copy() for k in range(p.K)]
    A[0] = A[0] + 0.3
    broken = _params(
        A,
        [p.f_matrix.B(k) for k in range(p.K)],
        [p.f_matrix.C(k) for k in range(p.K)],
        [p.f_matrix.D(k) for k in range(p.K)],
        [p.noise_cov.Sigma_U(k) for k in range(p.K)],
        [p.noise_cov.Delta(k) for k in range(p.K)],
        [p.noise_cov.Sigma_V(k) for k in range(p.K)],
    )
    issues = validate_ngh_msm(broken)
    assert any("AB / (H5) constraint violated" in m for m in issues)


def test_singular_D_is_flagged():
    # Valid AB blocks but D(0) singular → D-invertibility violated.
    C = [np.array([[0.2]]), np.array([[0.1]])]
    D = [np.array([[0.0]]), np.array([[0.6]])]  # D(0) singular
    SU = [np.array([[0.1]]), np.array([[0.2]])]
    Dt = [np.array([[0.05]]), np.array([[0.02]])]
    SV = [np.array([[0.1]]), np.array([[0.15]])]
    A, B = [], []
    for k in range(2):
        Ak, Bk = compute_AB(C[k], D[k], Dt[k], SV[k])
        A.append(Ak)
        B.append(Bk)
    params = _params(A, B, C, D, SU, Dt, SV)
    issues = validate_ngh_msm(params)
    assert any("D is singular" in m for m in issues)


def test_check_ngh_msm_raises_on_invalid():
    params = GSSParams.from_model(ModelGss_K2_q2_s1())
    with pytest.raises(ParamError, match="not a valid NGH-MSM"):
        params.check_ngh_msm(raise_on_fail=True)
    # raise_on_fail=False returns the issue list instead
    assert params.check_ngh_msm(raise_on_fail=False)


# ---------------------------------------------------------------------------
# NGHMSMParams: the validated-NGH-MSM type
# ---------------------------------------------------------------------------
class TestNGHMSMParams:
    def test_from_free_blocks_is_validated_subtype(self):
        p = _valid_ab_params()
        assert isinstance(p, NGHMSMParams)
        assert isinstance(p, GSSParams)  # Liskov: substitutes for the base type
        assert validate_ngh_msm(p) == []
        assert "NGHMSMParams" in repr(p)

    def test_from_model_derives_and_validates(self):
        p = NGHMSMParams.from_model(ModelGssK2Q1S1())
        assert isinstance(p, NGHMSMParams)
        # A, B were derived from the free blocks: A = M C, B = M D
        for k in range(p.K):
            np.testing.assert_allclose(p.f_matrix.A(k), p.M(k) @ p.f_matrix.C(k), atol=1e-12)
            np.testing.assert_allclose(p.f_matrix.B(k), p.M(k) @ p.f_matrix.D(k), atol=1e-12)

    def test_M_Gamma_delegate_to_noise_cov(self):
        p = _valid_ab_params()
        for k in range(p.K):
            np.testing.assert_array_equal(p.M(k), p.noise_cov.M(k))
            np.testing.assert_array_equal(p.Gamma(k), p.noise_cov.Gamma(k))

    def test_from_free_blocks_rejects_s_lt_q(self):
        d = ModelGss_K2_q2_s1().get_params()  # s=1 < q=2
        with pytest.raises(ParamError, match="s = 1 < q = 2"):
            NGHMSMParams.from_free_blocks(
                K=d["K"],
                q=d["q"],
                s=d["s"],
                P=d["P"],
                C_list=d["C_list"],
                D_list=d["D_list"],
                Sigma_U_list=d["Sigma_U_list"],
                Delta_list=d["Delta_list"],
                Sigma_V_list=d["Sigma_V_list"],
                pi0=d["pi0"],
                mu_z0_list=d["mu_z0_list"],
                Sigma_z0_list=d["Sigma_z0_list"],
                b_list=d.get("b_list"),
            )

    def test_from_free_blocks_rejects_singular_D(self):
        # compute_AB succeeds with D=0 (B=0); __init__ validation catches it.
        C, D, SU, Dt, SV = _free_blocks()
        D = [np.array([[0.0]]), D[1]]  # D(0) singular
        with pytest.raises(ParamError, match="D is singular"):
            NGHMSMParams.from_free_blocks(
                K=2,
                q=1,
                s=1,
                P=np.full((2, 2), 0.5),
                C_list=C,
                D_list=D,
                Sigma_U_list=SU,
                Delta_list=Dt,
                Sigma_V_list=SV,
                pi0=None,
                mu_z0_list=[np.zeros((2, 1))] * 2,
                Sigma_z0_list=[np.eye(2)] * 2,
            )

    def test_init_rejects_off_manifold_AB(self):
        # An assembled FMatrix with hand-set A, B off the AB manifold is rejected.
        p = _valid_ab_params()
        fm = FMatrix(
            K=2,
            q=1,
            s=1,
            A_list=[np.array([[0.8]]), np.array([[0.5]])],  # not Δ Σ_V⁻¹ C
            B_list=[np.array([[0.1]]), np.array([[0.3]])],
            C_list=[p.f_matrix.C(k) for k in range(2)],
            D_list=[p.f_matrix.D(k) for k in range(2)],
        )
        with pytest.raises(ParamError, match="AB / \\(H5\\) constraint violated"):
            NGHMSMParams(
                K=2,
                q=1,
                s=1,
                P=p.P,
                f_matrix=fm,
                noise_cov=p.noise_cov,
                pi0=p.pi0,
                mu_z0_list=[p.mu_z0(k) for k in range(2)],
                Sigma_z0_list=[p.Sigma_z0(k) for k in range(2)],
            )

    def test_apply_AB_constraint_returns_nghmsm_on_valid(self):
        base = GSSParams.from_model(ModelGssK2Q1S1())
        out = apply_AB_constraint(base)
        assert isinstance(out, NGHMSMParams)

    def test_apply_AB_constraint_falls_back_on_degenerate(self):
        # s<q model: AB is computable but the structural CNS fails → base, no raise.
        base = GSSParams.from_model(ModelGss_K2_q2_s1())
        out = apply_AB_constraint(base)
        assert isinstance(out, GSSParams)
        assert not isinstance(out, NGHMSMParams)


# ---------------------------------------------------------------------------
# Closed-form filter (Prop. 4 is redundant) and the regime-conditional law
# ---------------------------------------------------------------------------
class TestClosedFormFilter:
    def test_h5_exact_gain_equals_M_and_Gamma(self):
        """Under AB, the h5_exact gain/posterior are exactly M_k and Γ_k."""
        p = NGHMSMParams.from_model(ModelGssK2Q1S1())
        filt = GSSFilter(p, mode="h5_exact")
        for k in range(p.K):
            np.testing.assert_allclose(filt._K_gain[k], p.M(k), atol=1e-12)
            np.testing.assert_allclose(filt._P_post[k], p.Gamma(k), atol=1e-12)

    def test_derived_gain_also_converges_to_closed_form(self):
        """The general fixed-point derivation (base GSSParams) reaches the same
        M_k / Γ_k — i.e. the Riccati/Prop.-4 machinery is redundant under AB."""
        ref = NGHMSMParams.from_model(ModelGssK2Q1S1())
        base = GSSParams.from_model(ModelGssK2Q1S1())  # AB-valid, but base type
        filt = GSSFilter(base, mode="h5_exact")
        for k in range(base.K):
            np.testing.assert_allclose(filt._K_gain[k], ref.M(k), atol=1e-7)
            np.testing.assert_allclose(filt._P_post[k], ref.Gamma(k), atol=1e-7)

    def test_default_mode_dispatches_on_type(self):
        nghmsm = NGHMSMParams.from_model(ModelGssK2Q1S1())
        base = GSSParams.from_model(ModelGssK2Q1S1())
        assert GSSFilter(nghmsm).mode == "h5_exact"
        assert GSSFilter(base).mode == "imm_general"

    def test_simulator_matches_regime_conditional_law(self):
        """Empirically X_n | (r_n=k, y_n) = M_k y_n + ξ, Cov(ξ)=Γ_k (slaving)."""
        p = NGHMSMParams.from_model(ModelGssK2Q1S1())
        xs = {k: [] for k in range(p.K)}
        ys = {k: [] for k in range(p.K)}
        for n, r, x, y in GSSSimulator(p, N=40000, seed=0):
            if n >= 2:  # slaving holds on transitions, not the arbitrary init
                xs[r].append(x.ravel())
                ys[r].append(y.ravel())
        for k in range(p.K):
            X = np.array(xs[k])  # (n_k, q)
            Y = np.array(ys[k])  # (n_k, s)
            # OLS through the origin (zero-mean): slope = (X^T Y)(Y^T Y)^-1 ≈ M_k
            slope = (X.T @ Y) @ np.linalg.inv(Y.T @ Y)
            np.testing.assert_allclose(slope, p.M(k), atol=0.03)
            resid = X - Y @ slope.T
            cov = resid.T @ resid / len(resid)
            np.testing.assert_allclose(cov, p.Gamma(k), atol=0.03)
