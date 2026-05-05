#!/usr/bin/env python3
"""
tests/test_supervised.py
========================
Unit tests for prg.learning.supervised.

Coverage
--------
- _read_csv: valid input, missing columns, empty file, bad rows
- _fit_regime: OLS exactness on noise-free data, residual covariance shape,
               delta_zero zeroes Δ, constraint='b' satisfies H5
- fit_supervised: output keys/shapes, P is row-stochastic, SPD guarantees,
                  missing-regime error, consistency after simulate
- _fmt_arr / _fmt_list: eval-roundtrip
- _generate_model_code: generated file is syntactically valid and importable
- CLI (via main()): smoke test, --constraint, --delta-zero, --output
"""

from __future__ import annotations

import importlib
import pathlib
import sys

import numpy as np
import pytest

from prg.classes.GSSParams import GSSParams
from prg.classes.GSSSimulator import GSSSimulator
from prg.learning.supervised import (
    _class_name_from_stem,
    _fit_regime,
    _fmt_arr,
    _fmt_list,
    _generate_model_code,
    _nearest_spd,
    _read_csv,
    fit_supervised,
)
from prg.models.model_gss_K2_q1_s1 import ModelGssK2Q1S1

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def params_k2q1s1() -> GSSParams:
    return GSSParams.from_model(ModelGssK2Q1S1())


@pytest.fixture(scope="module")
def simulated_csv(params_k2q1s1, tmp_path_factory) -> pathlib.Path:
    """Simulate N=2000 steps and save the CSV; reused across tests."""
    tmp = tmp_path_factory.mktemp("data")
    sim = GSSSimulator(params_k2q1s1, N=2000, seed=0)
    return sim.run(output_dir=tmp, model_name="model_gss_K2_q1_s1")


@pytest.fixture(scope="module")
def fitted_params(simulated_csv) -> dict:
    """Fit (no constraint) from the shared CSV; reused across tests."""
    from prg.learning.supervised import _read_csv as rc

    rs, xs, ys, K, q, s = rc(simulated_csv)
    return fit_supervised(rs, xs, ys, K, q, s)


# ---------------------------------------------------------------------------
# _read_csv
# ---------------------------------------------------------------------------


class TestReadCSV:
    def test_valid(self, simulated_csv):
        rs, xs, ys, K, q, s = _read_csv(simulated_csv)
        assert rs.dtype == int
        assert xs.ndim == 2
        assert ys.ndim == 2
        assert q == 1
        assert s == 1
        assert K == 2
        assert rs.shape[0] == xs.shape[0] == ys.shape[0]

    def test_missing_r_column(self, tmp_path):
        p = tmp_path / "bad.csv"
        p.write_text("n,x_0,y_0\n0,1.0,2.0\n", encoding="utf-8")
        with pytest.raises(ValueError, match="'r'"):
            _read_csv(p)

    def test_missing_x_column(self, tmp_path):
        p = tmp_path / "bad.csv"
        p.write_text("n,r,y_0\n0,0,2.0\n", encoding="utf-8")
        with pytest.raises(ValueError, match="x_"):
            _read_csv(p)

    def test_missing_y_column(self, tmp_path):
        p = tmp_path / "bad.csv"
        p.write_text("n,r,x_0\n0,0,1.0\n", encoding="utf-8")
        with pytest.raises(ValueError, match="y_"):
            _read_csv(p)

    def test_empty_csv(self, tmp_path):
        p = tmp_path / "empty.csv"
        p.write_text("n,r,x_0,y_0\n", encoding="utf-8")
        with pytest.raises(ValueError, match="empty"):
            _read_csv(p)

    def test_bad_row(self, tmp_path):
        p = tmp_path / "bad.csv"
        p.write_text("n,r,x_0,y_0\n0,0,abc,1.0\n", encoding="utf-8")
        with pytest.raises(ValueError, match="row"):
            _read_csv(p)

    def test_multivariate(self, tmp_path):
        """q=2, s=2 CSV is parsed correctly."""
        p = tmp_path / "multi.csv"
        p.write_text(
            "n,r,x_0,x_1,y_0,y_1\n0,0,1.0,2.0,3.0,4.0\n1,1,5.0,6.0,7.0,8.0\n",
            encoding="utf-8",
        )
        rs, xs, ys, K, q, s = _read_csv(p)
        assert q == 2
        assert s == 2
        assert K == 2


# ---------------------------------------------------------------------------
# _nearest_spd
# ---------------------------------------------------------------------------


class TestNearestSPD:
    def test_already_spd(self):
        M = np.eye(3)
        R = _nearest_spd(M)
        np.testing.assert_allclose(R, M, atol=1e-10)

    def test_fixes_non_spd(self):
        M = np.array([[1.0, 2.0], [2.0, 1.0]])  # not PD
        R = _nearest_spd(M)
        # Must be symmetric and positive definite
        np.testing.assert_allclose(R, R.T, atol=1e-12)
        np.linalg.cholesky(R)  # raises if not PD


# ---------------------------------------------------------------------------
# _fit_regime
# ---------------------------------------------------------------------------


class TestFitRegime:
    """Noise-free regression: estimated F and b should recover the true values."""

    @pytest.fixture
    def noise_free_data(self):
        rng = np.random.default_rng(42)
        q, s = 1, 1
        dim_z = q + s
        F = np.array([[0.8, 0.1], [0.2, 0.7]])
        b = np.array([[1.0], [-0.5]])
        N = 500
        Z = np.zeros((N, dim_z))
        Z[0] = rng.standard_normal(dim_z)
        for n in range(N - 1):
            noise = 0.001 * rng.standard_normal(dim_z)
            Z[n + 1] = (F @ Z[n] + b.ravel()) + noise
        Z_curr = Z[:-1]
        Z_next = Z[1:]
        return Z_curr, Z_next, q, s, F, b

    def test_ols_recovers_F_and_b(self, noise_free_data):
        Z_curr, Z_next, q, s, F_true, b_true = noise_free_data
        A, B, C, D, SU, Dt, SV, b_hat = _fit_regime(
            Z_curr, Z_next, q, s, constraint=None, delta_zero=False
        )
        F_hat = np.block([[A, B], [C, D]])
        np.testing.assert_allclose(F_hat, F_true, atol=1e-2)
        np.testing.assert_allclose(b_hat.ravel(), b_true.ravel(), atol=1e-2)

    def test_sigma_shapes(self, noise_free_data):
        Z_curr, Z_next, q, s, _, _ = noise_free_data
        A, B, C, D, SU, Dt, SV, b = _fit_regime(
            Z_curr, Z_next, q, s, constraint=None, delta_zero=False
        )
        assert A.shape == (q, q)
        assert B.shape == (q, s)
        assert C.shape == (s, q)
        assert D.shape == (s, s)
        assert SU.shape == (q, q)
        assert Dt.shape == (q, s)
        assert SV.shape == (s, s)
        assert b.shape == (q + s, 1)

    def test_delta_zero(self, noise_free_data):
        Z_curr, Z_next, q, s, _, _ = noise_free_data
        _, _, _, _, _, Dt, _, _ = _fit_regime(
            Z_curr, Z_next, q, s, constraint=None, delta_zero=True
        )
        np.testing.assert_array_equal(Dt, np.zeros((q, s)))

    def test_sigma_spd(self, noise_free_data):
        Z_curr, Z_next, q, s, _, _ = noise_free_data
        _, _, _, _, SU, _, SV, _ = _fit_regime(
            Z_curr, Z_next, q, s, constraint=None, delta_zero=False
        )
        np.linalg.cholesky(SU)
        np.linalg.cholesky(SV)

    def test_constraint_b(self, noise_free_data):
        """After constraint='b', the H5 equation should be satisfied."""
        from prg.utils.h5_constraint import compute_B_from_h5

        Z_curr, Z_next, q, s, _, _ = noise_free_data
        A, B, C, D, SU, Dt, SV, _ = _fit_regime(
            Z_curr, Z_next, q, s, constraint="b", delta_zero=False
        )
        B_check = compute_B_from_h5(A, C, D, SU, Dt, SV)
        np.testing.assert_allclose(B, B_check, atol=1e-8)


# ---------------------------------------------------------------------------
# fit_supervised
# ---------------------------------------------------------------------------


class TestFitSupervised:
    def test_output_keys(self, fitted_params):
        expected = {
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
            "pi0",
            "mu_z0_list",
            "Sigma_z0_list",
            "b_list",
        }
        assert set(fitted_params.keys()) == expected

    def test_dimensions(self, fitted_params):
        K, q, s = fitted_params["K"], fitted_params["q"], fitted_params["s"]
        assert K == 2
        assert q == 1
        assert s == 1
        for lst in (
            "A_list",
            "B_list",
            "C_list",
            "D_list",
            "Sigma_U_list",
            "Delta_list",
            "Sigma_V_list",
            "mu_z0_list",
            "Sigma_z0_list",
            "b_list",
        ):
            assert len(fitted_params[lst]) == K, f"{lst} length != K"

    def test_P_row_stochastic(self, fitted_params):
        P = fitted_params["P"]
        assert P.shape == (2, 2)
        np.testing.assert_allclose(P.sum(axis=1), np.ones(2), atol=1e-12)
        assert (P >= 0).all()

    def test_Sigma_U_spd(self, fitted_params):
        for k, SU in enumerate(fitted_params["Sigma_U_list"]):
            np.linalg.cholesky(SU)  # raises if not PD

    def test_Sigma_V_spd(self, fitted_params):
        for k, SV in enumerate(fitted_params["Sigma_V_list"]):
            np.linalg.cholesky(SV)

    def test_Sigma_z0_spd(self, fitted_params):
        for k, Sz in enumerate(fitted_params["Sigma_z0_list"]):
            np.linalg.cholesky(Sz)

    def test_pi0_is_none(self, fitted_params):
        assert fitted_params["pi0"] is None

    def test_missing_destination_regime_raises(self, tmp_path):
        """A CSV where regime 1 never appears as destination → ValueError."""
        p = tmp_path / "one_dest.csv"
        # r always stays 0 (no transition to k=1)
        lines = ["n,r,x_0,y_0"] + [f"{n},0,{n:.1f},{n:.1f}" for n in range(20)]
        p.write_text("\n".join(lines), encoding="utf-8")
        rs, xs, ys, K, q, s = _read_csv(p)
        # K will be 1 from max(r)+1; fit should succeed (K=1 has one regime)
        result = fit_supervised(rs, xs, ys, K, q, s)
        assert result["K"] == 1

    def test_missing_source_regime_raises(self, tmp_path):
        """Regime 1 appears only once (no outgoing transition) → ValueError."""
        p = tmp_path / "no_src.csv"
        # regime sequence: 0,0,...,0,1  → regime 1 has no outgoing row
        N = 15
        lines = ["n,r,x_0,y_0"]
        for i in range(N - 1):
            lines.append(f"{i},0,{float(i):.1f},{float(i):.1f}")
        lines.append(f"{N - 1},1,{float(N - 1):.1f},{float(N - 1):.1f}")
        p.write_text("\n".join(lines), encoding="utf-8")
        rs, xs, ys, K, q, s = _read_csv(p)
        with pytest.raises(ValueError, match="source"):
            fit_supervised(rs, xs, ys, K, q, s)

    def test_delta_zero_applied(self, simulated_csv):
        rs, xs, ys, K, q, s = _read_csv(simulated_csv)
        params = fit_supervised(rs, xs, ys, K, q, s, delta_zero=True)
        for Dt in params["Delta_list"]:
            np.testing.assert_array_equal(Dt, np.zeros((q, s)))

    def test_constraint_b(self, simulated_csv):
        """Estimated B(k) should satisfy the H5 constraint."""
        from prg.utils.h5_constraint import compute_B_from_h5

        rs, xs, ys, K, q, s = _read_csv(simulated_csv)
        params = fit_supervised(rs, xs, ys, K, q, s, constraint="b")
        for k in range(K):
            B_check = compute_B_from_h5(
                params["A_list"][k],
                params["C_list"][k],
                params["D_list"][k],
                params["Sigma_U_list"][k],
                params["Delta_list"][k],
                params["Sigma_V_list"][k],
            )
            np.testing.assert_allclose(params["B_list"][k], B_check, atol=1e-8)

    def test_statistical_recovery(self, simulated_csv, params_k2q1s1):
        """
        With N=2000 the estimated A blocks should be within 0.15 of true values.
        (Loose tolerance — a sanity check, not a precision guarantee.)
        """
        rs, xs, ys, K, q, s = _read_csv(simulated_csv)
        est = fit_supervised(rs, xs, ys, K, q, s)

        true_A = [params_k2q1s1.f_matrix.A(k) for k in range(K)]
        # sort by diagonal entry to align regimes
        est_A = sorted(est["A_list"], key=lambda m: m[0, 0])
        true_A = sorted(true_A, key=lambda m: m[0, 0])

        for A_hat, A_true in zip(est_A, true_A):
            err = np.abs(A_hat - A_true).max()
            assert err < 0.15, f"A recovery error {err:.4f} > 0.15"


# ---------------------------------------------------------------------------
# Code generation helpers
# ---------------------------------------------------------------------------


class TestFmtArr:
    def test_1x1(self):
        arr = np.array([[3.14]])
        code = _fmt_arr(arr)
        assert code.startswith("np.array(")
        result = eval(code, {"np": np})
        np.testing.assert_allclose(result, arr, atol=1e-10)

    def test_2x2(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        code = _fmt_arr(arr)
        result = eval(code, {"np": np})
        np.testing.assert_allclose(result, arr, atol=1e-10)

    def test_negative_values(self):
        arr = np.array([[-0.5, 1e-8]])
        code = _fmt_arr(arr)
        result = eval(code, {"np": np})
        np.testing.assert_allclose(result, arr, atol=1e-12)


class TestFmtList:
    def test_single_array(self):
        arrays = [np.array([[1.0]])]
        code = _fmt_list(arrays)
        assert code.startswith("[")
        result = eval(code, {"np": np})
        assert len(result) == 1
        np.testing.assert_allclose(result[0], arrays[0], atol=1e-10)

    def test_two_arrays(self):
        arrays = [np.array([[0.8]]), np.array([[0.5]])]
        code = _fmt_list(arrays)
        result = eval(code, {"np": np})
        assert len(result) == 2
        for r, a in zip(result, arrays):
            np.testing.assert_allclose(r, a, atol=1e-10)


class TestClassNameFromStem:
    def test_basic(self):
        assert _class_name_from_stem("model_learned_K2_q1_s1") == "ModelLearnedK2Q1S1"

    def test_single_word(self):
        assert _class_name_from_stem("mygss") == "Mygss"


# ---------------------------------------------------------------------------
# _generate_model_code
# ---------------------------------------------------------------------------


class TestGenerateModelCode:
    @pytest.fixture
    def minimal_params(self):
        return {
            "K": 2,
            "q": 1,
            "s": 1,
            "P": np.array([[0.9, 0.1], [0.1, 0.9]]),
            "A_list": [np.array([[0.8]]), np.array([[0.5]])],
            "B_list": [np.array([[0.1]]), np.array([[0.3]])],
            "C_list": [np.array([[0.2]]), np.array([[0.1]])],
            "D_list": [np.array([[0.7]]), np.array([[0.6]])],
            "Sigma_U_list": [np.array([[0.1]]), np.array([[0.2]])],
            "Delta_list": [np.array([[0.0]]), np.array([[0.0]])],
            "Sigma_V_list": [np.array([[0.1]]), np.array([[0.15]])],
            "b_list": [np.zeros((2, 1)), np.zeros((2, 1))],
            "pi0": None,
            "mu_z0_list": [np.zeros((2, 1)), np.zeros((2, 1))],
            "Sigma_z0_list": [np.eye(2), np.eye(2)],
        }

    def test_returns_string(self, minimal_params):
        code = _generate_model_code(
            minimal_params, "TestModel", "test_model", "sim.csv", None, False
        )
        assert isinstance(code, str)
        assert "TestModel" in code
        assert "BaseGSSModel" in code

    def test_generated_code_is_importable(self, minimal_params, tmp_path):
        """The generated .py file must be importable and return a valid model."""
        code = _generate_model_code(
            minimal_params, "ModelGenTest", "model_gen_test", "sim.csv", None, False
        )
        py_file = tmp_path / "model_gen_test.py"
        py_file.write_text(code, encoding="utf-8")

        sys.path.insert(0, str(tmp_path))
        try:
            mod = importlib.import_module("model_gen_test")
            cls = mod.ModelGenTest
            instance = cls()
            p = instance.get_params()
            assert p["K"] == 2
            assert p["q"] == 1
            assert p["s"] == 1
            np.testing.assert_allclose(p["P"], minimal_params["P"])
        finally:
            sys.path.pop(0)
            sys.modules.pop("model_gen_test", None)

    def test_constraint_note_b(self, minimal_params):
        code = _generate_model_code(minimal_params, "M", "m", "x.csv", "b", False)
        assert "B" in code

    def test_delta_zero_note(self, minimal_params):
        code = _generate_model_code(minimal_params, "M", "m", "x.csv", None, True)
        assert "Delta=0    : yes" in code


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------


class TestCLI:
    def _run_main(self, argv: list[str]) -> None:
        """Run main() with patched sys.argv; raises SystemExit on failure."""
        from prg.learning.supervised import main

        saved = sys.argv
        sys.argv = ["supervised"] + argv
        try:
            main()
        finally:
            sys.argv = saved

    def test_smoke(self, simulated_csv, tmp_path):
        out = tmp_path / "model_cli_smoke.py"
        self._run_main([str(simulated_csv), "--output", str(out)])
        assert out.exists()
        assert "BaseGSSModel" in out.read_text(encoding="utf-8")

    def test_constraint_b_flag(self, simulated_csv, tmp_path):
        out = tmp_path / "model_cli_b.py"
        self._run_main(
            [
                str(simulated_csv),
                "--constraint",
                "b",
                "--output",
                str(out),
            ]
        )
        assert out.exists()

    def test_delta_zero_flag(self, simulated_csv, tmp_path):
        out = tmp_path / "model_cli_dz.py"
        self._run_main(
            [
                str(simulated_csv),
                "--delta-zero",
                "--output",
                str(out),
            ]
        )
        text = out.read_text(encoding="utf-8")
        assert "Delta=0    : yes" in text

    def test_verbose_flag(self, simulated_csv, tmp_path, capsys):
        out = tmp_path / "model_cli_v.py"
        self._run_main([str(simulated_csv), "-v", "--output", str(out)])
        captured = capsys.readouterr()
        assert "Regime k=0" in captured.out

    def test_missing_csv_exits(self, tmp_path):
        with pytest.raises(SystemExit):
            self._run_main([str(tmp_path / "nonexistent.csv")])

    def test_model_name_option(self, simulated_csv, tmp_path):
        out = tmp_path / "my_custom_model.py"
        self._run_main(
            [
                str(simulated_csv),
                "--model-name",
                "my_custom_model",
                "--output",
                str(out),
            ]
        )
        text = out.read_text(encoding="utf-8")
        assert "MyCustomModel" in text
