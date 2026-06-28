#!/usr/bin/env python3
"""
tests/test_consigne.py
======================
Exogenous-input ("consigne") integration: GSSParams gains, simulator injection,
exact-filter read-out, gold-standard exactness vs the brute-force Kᴺ filter,
reference filters, the shared input-signal helper, and backward compatibility.
"""

from __future__ import annotations

import numpy as np
import pytest

from prg.classes.GSSParams import NGHMSMParams
from prg.classes.GSSSimulator import GSSSimulator
from prg.experiments.reference_filters import (
    exact_mixture_filter,
    imm_filter,
    oracle_filter,
    single_kalman_filter,
    with_stationary_init,
)
from prg.filter.gss_filter import GSSFilter
from prg.models.model_gss_K2_q1_s1_consigne import ModelGssK2Q1S1Consigne
from prg.models.model_gss_K2_q2_s2 import ModelGss_K2_q2_s2
from prg.utils.exceptions import ParamError
from prg.utils.input_signal import INPUT_GENERATORS, make_input


def _consigne_params(seed=0, scale=1.3, p=2):
    """An NGH-MSM (K=2, q=2, s=2) augmented with a random p-dim input gain."""
    d = ModelGss_K2_q2_s2().get_params()
    K, q, s = d["K"], d["q"], d["s"]
    rng = np.random.default_rng(seed)
    G_list = [rng.standard_normal((q + s, p)) * scale for _ in range(K)]

    class _M:
        def get_params(self):
            dd = dict(d)
            dd["G_list"] = G_list
            return dd

    return NGHMSMParams.from_model(_M()), G_list


# ---------------------------------------------------------------------------
# GSSParams: gains and read-out
# ---------------------------------------------------------------------------
def test_params_autonomous_backward_compat():
    p0 = NGHMSMParams.from_model(ModelGss_K2_q2_s2())
    assert p0.p == 0
    assert p0.G(0).shape == (p0.dim_z, 0)
    assert p0.N(0).shape == (p0.q, 0)


def test_params_N_equals_GX_minus_M_GY():
    pc, G_list = _consigne_params()
    assert pc.p == 2
    q = pc.q
    for k in range(pc.K):
        assert np.allclose(pc.G(k), G_list[k])
        GX, GY = G_list[k][:q, :], G_list[k][q:, :]
        assert np.allclose(pc.N(k), GX - pc.noise_cov.M(k) @ GY)
        assert pc.N(k).shape == (q, pc.p)


def test_params_rejects_wrong_G_shape():
    d = ModelGss_K2_q2_s2().get_params()
    bad = [np.zeros((d["q"] + d["s"] + 1, 2)), np.zeros((d["q"] + d["s"], 2))]

    class _B:
        def get_params(self):
            dd = dict(d)
            dd["G_list"] = bad
            return dd

    with pytest.raises(ParamError):
        NGHMSMParams.from_model(_B())


# ---------------------------------------------------------------------------
# Simulator: exact injection + backward compatibility
# ---------------------------------------------------------------------------
def test_simulator_injects_deterministic_input():
    pc, _ = _consigne_params(seed=1)
    K, q, s, p = pc.K, pc.q, pc.s, pc.p
    N = 60
    rng = np.random.default_rng(2)
    u = rng.standard_normal((N, p))

    def collect(uu):
        R, Z = [], []
        for n, r, x, y in GSSSimulator(pc, N=N, seed=7, u=uu):
            R.append(r)
            Z.append(np.vstack([x, y]).ravel())
        return np.array(R), np.array(Z)

    R_u, Z_u = collect(u)
    R_0, Z_0 = collect(np.zeros((N, p)))
    # u does not consume RNG -> identical regimes/noise
    assert np.array_equal(R_u, R_0)
    # the difference is exactly the deterministic propagation d_n = F d_{n-1} + G u_{n-1}
    d = np.zeros((q + s, 1))
    D = [d.ravel().copy()]
    for n in range(1, N):
        d = pc.f_matrix.F(R_u[n]) @ d + pc.G(R_u[n]) @ u[n - 1].reshape(p, 1)
        D.append(d.ravel().copy())
    assert np.max(np.abs((Z_u - Z_0) - np.array(D))) < 1e-10


def test_simulator_u_none_matches_autonomous():
    pc, _ = _consigne_params(seed=1)
    auto = NGHMSMParams.from_model(ModelGss_K2_q2_s2())
    g = lambda params, u: np.array(
        [np.vstack([x, y]).ravel() for _n, _r, x, y in GSSSimulator(params, N=40, seed=7, u=u)]
    )
    assert np.allclose(g(pc, None), g(auto, None))


def test_simulator_csv_columns():
    pc, _ = _consigne_params()
    import pathlib
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        auto = NGHMSMParams.from_model(ModelGss_K2_q2_s2())
        pa = GSSSimulator(auto, N=10, seed=0).run(output_dir=td, model_name="a")
        u = np.zeros((10, pc.p))
        pcsv = GSSSimulator(pc, N=10, seed=0, u=u).run(output_dir=td, model_name="c")
        assert "u_0" not in pathlib.Path(pa).read_text().splitlines()[0]
        assert pathlib.Path(pcsv).read_text().splitlines()[0].endswith("u_0,u_1")


# ---------------------------------------------------------------------------
# Filter: gold-standard exactness vs the brute-force Kᴺ
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("with_input", [False, True])
def test_h5_exact_equals_brute_force_kn(with_input):
    pc, _ = _consigne_params(seed=5, scale=1.2)
    pc = with_stationary_init(pc)  # align init law to stationary (h5 assumes it)
    N = 11
    rng = np.random.default_rng(9)
    u = rng.standard_normal((N, pc.p)) if with_input else None
    Y = np.array(
        [y.ravel() for _n, _r, _x, y in GSSSimulator(pc, N=N, seed=9, u=u)]
    )
    filt = GSSFilter(pc, mode="h5_exact")
    Ex_h5 = np.array([filt.step(Y[n], u=(u[n] if with_input else None)).E_x.ravel() for n in range(N)])
    Ex_kn, _, _ = exact_mixture_filter(pc, Y, us=u)
    assert np.max(np.abs(Ex_h5 - Ex_kn)) < 1e-9


def test_filter_u_none_is_autonomous():
    pc, _ = _consigne_params(seed=5)
    N = 50
    Y = np.array([y.ravel() for _n, _r, _x, y in GSSSimulator(pc, N=N, seed=3)])
    f1 = GSSFilter(pc, mode="h5_exact")
    f2 = GSSFilter(pc, mode="h5_exact")
    a = np.array([f1.step(Y[n]).E_x.ravel() for n in range(N)])
    b = np.array([f2.step(Y[n], u=np.zeros(pc.p)).E_x.ravel() for n in range(N)])
    assert np.allclose(a, b)


@pytest.mark.parametrize("mode", ["h5_exact", "imm_general"])
def test_consigne_improves_estimate(mode):
    pc, _ = _consigne_params(seed=3, scale=1.5)
    N = 600
    t = np.linspace(0, 20, N)
    u = np.stack([np.sin(t), np.sign(np.sin(0.7 * t))], axis=1)
    sim = GSSSimulator(pc, N=N, seed=11, u=u)
    X = np.array([x.ravel() for _n, _r, x, _y in sim])
    Y = np.array([y.ravel() for _n, _r, _x, y in GSSSimulator(pc, N=N, seed=11, u=u)])

    def run(pass_u):
        f = GSSFilter(pc, mode=mode)
        return np.array([f.step(Y[n], u=(u[n] if pass_u else None)).E_x.ravel() for n in range(N)])

    mse_u = np.mean((X[1:] - run(True)[1:]) ** 2)
    mse_no = np.mean((X[1:] - run(False)[1:]) ** 2)
    assert mse_u < mse_no


# ---------------------------------------------------------------------------
# Reference filters + with_stationary_init
# ---------------------------------------------------------------------------
def test_with_stationary_init_preserves_consigne():
    pc, _ = _consigne_params()
    ps = with_stationary_init(pc)
    assert ps.p == pc.p
    for k in range(pc.K):
        assert np.allclose(ps.G(k), pc.G(k))


def test_reference_filters_accept_input():
    pc, _ = _consigne_params(seed=5)
    pc = with_stationary_init(pc)
    N = 40
    rng = np.random.default_rng(1)
    u = rng.standard_normal((N, pc.p))
    R, Y = [], []
    for _n, r, _x, y in GSSSimulator(pc, N=N, seed=2, u=u):
        R.append(r)
        Y.append(y.ravel())
    R, Y = np.array(R), np.array(Y)
    assert single_kalman_filter(pc, Y, us=u)[0].shape == (N, pc.q)
    assert oracle_filter(pc, R, Y, us=u)[0].shape == (N, pc.q)
    assert imm_filter(pc, Y, us=u)[0].shape == (N, pc.q)


# ---------------------------------------------------------------------------
# Demo consigne model + input-signal helper
# ---------------------------------------------------------------------------
def test_demo_consigne_model_is_valid_nghmsm():
    pc = NGHMSMParams.from_model(ModelGssK2Q1S1Consigne())
    assert pc.p == 1 and pc.is_ngh_msm()


@pytest.mark.parametrize("spec", list(INPUT_GENERATORS) + ["step(4)", "sin(6)", "const(1.0,-0.5)"])
def test_make_input_shapes(spec):
    p = 2 if spec.startswith("const") else 1
    u = make_input(spec, 12, p, seed=0)
    assert u.shape == (12, p)


def test_make_input_step_and_none():
    assert np.allclose(make_input(None, 8, 2), 0.0)
    u = make_input("step(3)", 8, 1)
    assert np.allclose(u[:3], 0.0) and np.allclose(u[3:], 1.0)
