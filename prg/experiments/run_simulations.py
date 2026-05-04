#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prg/experiments/run_simulations.py
===================================
Monte-Carlo simulation study for the paper §6.

Protocol
--------
For each model (M1, M2, M3) × sequence length N ∈ {500, 2000, 5000}
× filter mode ("h5_exact", "imm_general") × seed in range(N_RUNS):

    1. Simulate N steps from the true model using GSSSimulator.
    2. Run GSSFilter on the simulated observations (step-by-step).
    3. Compute RMSE, ANEES, Ljung-Box p-value, Jarque-Bera p-value,
       total log-likelihood, BIC, and CPU time.
    4. Append a result row to the output DataFrame.

Results are written to ``data/experiments/mc_results.csv``.

Usage
-----
    # From the repo root, with the virtual environment active:
    python -m prg.experiments.run_simulations

    # Or import and call directly:
    from prg.experiments.run_simulations import run_all
    df = run_all(n_runs=100, N_list=(500, 2000, 5000), output_dir="data/experiments")
"""

from __future__ import annotations

import logging
import pathlib
import sys
import time
from typing import Sequence

import numpy as np
import pandas as pd

from prg.classes.FMatrix import FMatrix
from prg.classes.GSSParams import GSSParams
from prg.classes.GSSSimulator import GSSSimulator
from prg.classes.NoiseCovariance import GSSNoiseCovariance
from prg.experiments.metrics import (
    compute_bic,
    compute_jarque_bera,
    compute_ljung_box,
    compute_nees,
    compute_rmse,
)
from prg.experiments.models_paper import MODEL_NAMES, get_params
from prg.filter.gss_filter import GSSFilter

__all__ = ["run_all", "run_one_trial"]

logger = logging.getLogger("fofgss.experiments")

# ---------------------------------------------------------------------------
# Default protocol parameters (overridable via run_all arguments)
# ---------------------------------------------------------------------------

DEFAULT_N_RUNS  = 100
DEFAULT_N_LIST  = (500, 2_000, 5_000)
DEFAULT_MODES   = ("h5_exact", "imm_general")
DEFAULT_OUT_DIR = pathlib.Path("data") / "experiments"

# ---------------------------------------------------------------------------
# Helper: build GSSParams from a parameter dict
# ---------------------------------------------------------------------------

def _params_from_dict(d: dict) -> GSSParams:
    """
    Build a :class:`GSSParams` from the dict returned by ``get_params_Mx()``.

    Mirrors the logic of :meth:`GSSParams.from_model` without requiring a
    full ``BaseGSSModel`` object.
    """
    f_matrix  = FMatrix(
        K=d["K"], q=d["q"], s=d["s"],
        A_list=d["A_list"], B_list=d["B_list"],
        C_list=d["C_list"], D_list=d["D_list"],
    )
    noise_cov = GSSNoiseCovariance(
        K=d["K"], q=d["q"], s=d["s"],
        Sigma_U_list=d["Sigma_U_list"],
        Delta_list=d["Delta_list"],
        Sigma_V_list=d["Sigma_V_list"],
    )
    return GSSParams(
        K=d["K"], q=d["q"], s=d["s"],
        P=d["P"],
        f_matrix=f_matrix,
        noise_cov=noise_cov,
        pi0=d.get("pi0", None),
        mu_z0_list=d["mu_z0_list"],
        Sigma_z0_list=d["Sigma_z0_list"],
        b_list=d.get("b_list", None),
    )


# ---------------------------------------------------------------------------
# Single-trial runner
# ---------------------------------------------------------------------------

def run_one_trial(
    model_name:  str,
    N:           int,
    seed:        int,
    filter_mode: str,
    lb_lags:     int = 20,
) -> dict:
    """
    Execute one Monte-Carlo trial and return a metrics dict.

    Parameters
    ----------
    model_name : str
        One of "M1", "M2", "M3".
    N : int
        Sequence length.
    seed : int
        Random seed for :class:`GSSSimulator`.
    filter_mode : str
        ``"h5_exact"`` or ``"imm_general"``.
    lb_lags : int, default 20
        Number of lags for the Ljung–Box test.

    Returns
    -------
    dict with keys:
        model, N, seed, mode,
        rmse, nees, lb_pval, jb_pval,
        log_lik, bic, cpu_s
    """
    # --- Build model and filter -------------------------------------------
    param_dict = get_params(model_name)
    params     = _params_from_dict(param_dict)
    K, q, s    = params.K, params.q, params.s

    # Suppress H5 warnings: models_paper.py already asserts H5 residual < 1e-8
    import warnings
    filt = GSSFilter(params, mode=filter_mode)

    # --- Simulate ---------------------------------------------------------
    sim = GSSSimulator(params, N=N, seed=seed)

    x_true_list: list[np.ndarray] = []
    x_est_list:  list[np.ndarray] = []
    var_x_list:  list[np.ndarray] = []
    innov_list:  list[np.ndarray] = []
    log_lik_total = 0.0

    t0 = time.perf_counter()

    for _, _r, x, y in sim:
        result = filt.step(y)

        x_true_list.append(x.ravel())                       # (q,)
        x_est_list .append(result.E_x.ravel())              # (q,)
        var_x_list .append(result.Var_x)                    # (q, q)
        innov_list .append(result.innovation.ravel())        # (s,)
        log_lik_total += result.log_lik

    cpu_s = time.perf_counter() - t0

    # --- Stack arrays -----------------------------------------------------
    x_true  = np.array(x_true_list)    # (N, q)
    x_est   = np.array(x_est_list)     # (N, q)
    var_x   = np.array(var_x_list)     # (N, q, q)
    innov   = np.array(innov_list)      # (N, s)
    errors  = x_true - x_est           # (N, q)

    # --- Metrics ----------------------------------------------------------
    rmse   = compute_rmse(x_true, x_est)
    nees   = compute_nees(errors, var_x)
    lb     = compute_ljung_box(innov, lags=lb_lags)
    jb     = compute_jarque_bera(innov)
    bic    = compute_bic(log_lik_total, N, K, q, s)

    return {
        "model":   model_name,
        "N":       N,
        "seed":    seed,
        "mode":    filter_mode,
        "rmse":    rmse,
        "nees":    nees,
        "lb_pval": lb,
        "jb_pval": jb,
        "log_lik": log_lik_total,
        "bic":     bic,
        "cpu_s":   cpu_s,
    }


# ---------------------------------------------------------------------------
# Full Monte-Carlo runner
# ---------------------------------------------------------------------------

def run_all(
    models:     Sequence[str]       = MODEL_NAMES,
    N_list:     Sequence[int]       = DEFAULT_N_LIST,
    modes:      Sequence[str]       = DEFAULT_MODES,
    n_runs:     int                 = DEFAULT_N_RUNS,
    lb_lags:    int                 = 20,
    output_dir: str | pathlib.Path  = DEFAULT_OUT_DIR,
    verbose:    bool                = True,
) -> pd.DataFrame:
    """
    Run the full Monte-Carlo simulation study and save results to CSV.

    Parameters
    ----------
    models : sequence of str, default ("M1", "M2", "M3")
        Models to evaluate.
    N_list : sequence of int, default (500, 2000, 5000)
        Sequence lengths.
    modes : sequence of str, default ("h5_exact", "imm_general")
        Filter modes.
    n_runs : int, default 100
        Number of Monte-Carlo runs per configuration (seeds 0..n_runs-1).
    lb_lags : int, default 20
        Number of lags for Ljung–Box test.
    output_dir : path-like, default "data/experiments"
        Directory where ``mc_results.csv`` is written.
    verbose : bool, default True
        Print progress to stdout.

    Returns
    -------
    pd.DataFrame
        One row per (model, N, mode, seed) trial.
    """
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "mc_results.csv"

    total = len(models) * len(N_list) * len(modes) * n_runs
    if verbose:
        print(
            f"Monte-Carlo study: {len(models)} models × "
            f"{len(N_list)} N values × {len(modes)} modes × {n_runs} runs "
            f"= {total} trials"
        )

    rows: list[dict] = []
    done  = 0
    t_all = time.perf_counter()

    for model_name in models:
        for N in N_list:
            for mode in modes:
                for seed in range(n_runs):
                    try:
                        row = run_one_trial(
                            model_name=model_name,
                            N=N,
                            seed=seed,
                            filter_mode=mode,
                            lb_lags=lb_lags,
                        )
                    except Exception as exc:
                        logger.error(
                            "Trial failed: model=%s N=%d mode=%s seed=%d — %s",
                            model_name, N, mode, seed, exc,
                        )
                        row = {
                            "model": model_name, "N": N,
                            "seed": seed,        "mode": mode,
                            "rmse": float("nan"), "nees": float("nan"),
                            "lb_pval": float("nan"), "jb_pval": float("nan"),
                            "log_lik": float("nan"), "bic": float("nan"),
                            "cpu_s":   float("nan"),
                        }

                    rows.append(row)
                    done += 1

                    if verbose and (done % max(1, total // 50) == 0 or done == total):
                        elapsed = time.perf_counter() - t_all
                        eta = elapsed / done * (total - done)
                        print(
                            f"  [{done:>{len(str(total))}}/{total}] "
                            f"model={model_name} N={N:>5} mode={mode:>12} "
                            f"seed={seed:>3}  "
                            f"elapsed={elapsed:6.1f}s  ETA={eta:6.1f}s",
                            flush=True,
                        )

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)

    elapsed_total = time.perf_counter() - t_all
    if verbose:
        print(f"\nDone in {elapsed_total:.1f}s.  Results saved to: {out_path}")
        print(_summary_table(df))

    return df


# ---------------------------------------------------------------------------
# Quick summary of results
# ---------------------------------------------------------------------------

def _summary_table(df: pd.DataFrame) -> str:
    """Return a compact text summary of mean metrics per (model, N, mode)."""
    lines = [
        f"\n{'Model':>5}  {'N':>5}  {'Mode':>12}  "
        f"{'RMSE':>8}  {'ANEES':>7}  {'LB p':>7}  {'JB p':>7}  {'CPU(s)':>7}",
        "-" * 65,
    ]
    for (model, N, mode), grp in df.groupby(["model", "N", "mode"]):
        lines.append(
            f"{model:>5}  {N:>5}  {mode:>12}  "
            f"{grp['rmse'].mean():8.4f}  "
            f"{grp['nees'].mean():7.3f}  "
            f"{grp['lb_pval'].mean():7.3f}  "
            f"{grp['jb_pval'].mean():7.3f}  "
            f"{grp['cpu_s'].mean():7.3f}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Command-line entry point
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None):
    import argparse
    parser = argparse.ArgumentParser(
        description="Run the Monte-Carlo simulation study (paper §6).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--models", nargs="+", default=list(MODEL_NAMES),
        choices=list(MODEL_NAMES),
        help="Models to evaluate.",
    )
    parser.add_argument(
        "--N-list", nargs="+", type=int, default=list(DEFAULT_N_LIST),
        dest="N_list",
        help="Sequence lengths.",
    )
    parser.add_argument(
        "--modes", nargs="+", default=list(DEFAULT_MODES),
        choices=["h5_exact", "imm_general"],
        help="Filter modes.",
    )
    parser.add_argument(
        "--n-runs", type=int, default=DEFAULT_N_RUNS,
        help="Number of MC runs (seeds 0..n_runs-1).",
    )
    parser.add_argument(
        "--lb-lags", type=int, default=20,
        help="Number of lags for Ljung-Box test.",
    )
    parser.add_argument(
        "--output-dir", default=str(DEFAULT_OUT_DIR),
        help="Output directory for mc_results.csv.",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress output.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    args = _parse_args()
    run_all(
        models=args.models,
        N_list=args.N_list,
        modes=args.modes,
        n_runs=args.n_runs,
        lb_lags=args.lb_lags,
        output_dir=args.output_dir,
        verbose=not args.quiet,
    )
