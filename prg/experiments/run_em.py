#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prg/experiments/run_em.py
==========================
Monte-Carlo evaluation of the semi-supervised Baum-Welch EM estimator
(paper §6.4).

Two EM variants are compared
------------------------------
PH  (post-hoc projection, default)
    Standard Baum-Welch EM; the H5 constraint (τ=B) is enforced once
    as a post-hoc projection on the converged parameters of the best run.
    Log-likelihood is monotonically non-decreasing during EM.

GEM (Generalized EM)
    H5 constraint (τ=B) is applied at *every* M-step.  Log-likelihood
    monotonicity is no longer guaranteed, but the constraint is satisfied
    throughout optimisation.

Protocol (§6.4)
---------------
For each N ∈ {500, 2000, 5000} × seed ∈ range(N_RUNS):
  1. Simulate N steps from M1 (regime sequence hidden).
  2. Run PH and GEM, each with n_inits=10 restarts.
  3. Record:
       - best log-likelihood over restarts
       - per-restart log-likelihoods  (→ basin selection rate)
       - number of EM iterations for the best run
       - relative Frobenius error on F (permutation-aligned)
       - filter RMSE with estimated parameters
       - filter RMSE with true parameters (oracle)

Results are saved to ``data/experiments/em_results.csv``.
The log-likelihood history of the best run (for Fig. 3) is saved
separately in ``data/experiments/em_ll_history.csv``.

Usage
-----
    python -m prg.experiments.run_em
    python -m prg.experiments.run_em --n-runs 10 --N-list 500 2000
"""

from __future__ import annotations

import argparse
import itertools
import logging
import pathlib
import time
from typing import Sequence

import numpy as np
import pandas as pd

from prg.classes.GSSSimulator import GSSSimulator
from prg.experiments.metrics import compute_rmse
from prg.experiments.models_paper import get_params
from prg.experiments.run_simulations import _params_from_dict
from prg.experiments.run_supervised import _filter_rmse
from prg.filter.gss_filter import GSSFilter
from prg.learning.semi_supervised import fit_semi_supervised

__all__ = ["run_em_all", "run_em_trial"]

logger = logging.getLogger("exactIMM.experiments.em")

# ---------------------------------------------------------------------------
# Default protocol parameters
# ---------------------------------------------------------------------------

DEFAULT_MODEL   = "M1"
DEFAULT_N_LIST  = (500, 2_000, 5_000)
DEFAULT_N_RUNS  = 100
DEFAULT_N_INITS = 10          # EM restarts per trial
DEFAULT_MAX_ITER = 100
DEFAULT_TOL      = 1e-5
DEFAULT_OUT_DIR  = pathlib.Path("data") / "experiments"

# Tolerance for "same basin": restart log-lik within BASIN_TOL × |best LL|
BASIN_TOL = 0.01


# ---------------------------------------------------------------------------
# Permutation-aligned parameter error
# ---------------------------------------------------------------------------

def _align_permutation(est: dict, true_d: dict, K: int, q: int, s: int) -> int:
    """
    Find the regime permutation that minimises total Frobenius distance
    between estimated and true F matrices.

    Returns the index of the best permutation in itertools.permutations(range(K)).
    """
    best_perm  = list(range(K))
    best_dist  = np.inf

    for perm in itertools.permutations(range(K)):
        dist = 0.0
        for k_est, k_true in enumerate(perm):
            F_est  = np.block([[est["A_list"][k_est],  est["B_list"][k_est]],
                                [est["C_list"][k_est],  est["D_list"][k_est]]])
            F_true = np.block([[true_d["A_list"][k_true], true_d["B_list"][k_true]],
                                [true_d["C_list"][k_true], true_d["D_list"][k_true]]])
            dist += np.linalg.norm(F_est - F_true, "fro")
        if dist < best_dist:
            best_dist = dist
            best_perm = list(perm)

    return best_perm


def _rel_err_F_aligned(
    est: dict,
    true_d: dict,
    K: int, q: int, s: int,
) -> float:
    """
    Relative Frobenius error on F after best-permutation alignment.

    Returns mean over regimes of  ‖F_est[k] − F_true[perm[k]]‖_F / ‖F_true[perm[k]]‖_F.
    """
    perm = _align_permutation(est, true_d, K, q, s)

    errs = []
    for k_est, k_true in enumerate(perm):
        F_est  = np.block([[est["A_list"][k_est],  est["B_list"][k_est]],
                            [est["C_list"][k_est],  est["D_list"][k_est]]])
        F_true = np.block([[true_d["A_list"][k_true], true_d["B_list"][k_true]],
                            [true_d["C_list"][k_true], true_d["D_list"][k_true]]])
        denom = np.linalg.norm(F_true, "fro")
        errs.append(float(np.linalg.norm(F_est - F_true, "fro") / denom)
                    if denom > 1e-14 else float("nan"))

    return float(np.nanmean(errs))


# ---------------------------------------------------------------------------
# Single-trial runner
# ---------------------------------------------------------------------------

def run_em_trial(
    model_name:  str,
    N:           int,
    seed:        int,
    n_inits:     int = DEFAULT_N_INITS,
    max_iter:    int = DEFAULT_MAX_ITER,
    tol:         float = DEFAULT_TOL,
) -> tuple[list[dict], list[dict]]:
    """
    Run one semi-supervised EM trial (PH and GEM) and return metrics.

    Parameters
    ----------
    model_name : str
    N          : sequence length
    seed       : random seed for the simulator AND for EM restarts
    n_inits    : number of EM restarts per variant
    max_iter, tol : EM convergence parameters

    Returns
    -------
    rows : list of dict (one per EM variant: PH and GEM)
        Keys: model, N, seed, variant,
              best_log_lik, basin_rate, best_n_iter, best_converged,
              rel_err_F, rmse_estimated, rmse_oracle,
              all_log_liks (semicolon-separated string)
    history_rows : list of dict (one per iteration of best run, for Fig. 3)
        Keys: model, N, seed, variant, iteration, log_lik
    """
    # ------------------------------------------------------------------
    # Build true params
    # ------------------------------------------------------------------
    true_d = get_params(model_name)
    true_p = _params_from_dict(true_d)
    K, q, s = true_p.K, true_p.q, true_p.s

    # ------------------------------------------------------------------
    # Simulate (regime hidden)
    # ------------------------------------------------------------------
    sim = GSSSimulator(true_p, N=N, seed=seed)
    xs_list, ys_list = [], []
    for _, _r, x, y in sim:
        xs_list.append(x.ravel())
        ys_list.append(y.ravel())

    xs = np.array(xs_list)    # (N, q)
    ys = np.array(ys_list)    # (N, s)

    # ------------------------------------------------------------------
    # Oracle filter RMSE (true params)
    # ------------------------------------------------------------------
    oracle_rmse = _filter_rmse(true_p, ys, xs)

    # ------------------------------------------------------------------
    # EM variants
    # ------------------------------------------------------------------
    em_variants = [
        ("PH",  False),   # post-hoc projection
        ("GEM", True),    # constraint at every M-step
    ]

    rows: list[dict] = []
    history_rows: list[dict] = []

    for variant_name, each_iter in em_variants:
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                est, info = fit_semi_supervised(
                    xs, ys, K,
                    constraint="b",               # τ=B for both PH and GEM
                    delta_zero=False,
                    constraint_each_iter=each_iter,
                    n_inits=n_inits,
                    max_iter=max_iter,
                    tol=tol,
                    seed=seed,                    # reproducible restarts
                    verbose=False,
                )
        except (RuntimeError, Exception) as exc:
            logger.warning(
                "EM failed: model=%s N=%d seed=%d variant=%s — %s",
                model_name, N, seed, variant_name, exc,
            )
            rows.append({
                "model": model_name, "N": N, "seed": seed,
                "variant": variant_name,
                "best_log_lik": float("nan"),
                "basin_rate": float("nan"),
                "best_n_iter": 0,
                "best_converged": False,
                "rel_err_F": float("nan"),
                "rmse_estimated": float("nan"),
                "rmse_oracle": oracle_rmse,
                "all_log_liks": "",
            })
            continue

        # --- Basin selection rate --------------------------------------
        all_lls = info["all_log_liks"]
        best_ll = info["best_log_lik"]
        if len(all_lls) > 0 and np.isfinite(best_ll):
            threshold = best_ll - BASIN_TOL * abs(best_ll)
            basin_rate = float(np.mean([ll >= threshold for ll in all_lls]))
        else:
            basin_rate = float("nan")

        # --- Parameter error (permutation-aligned) ----------------------
        try:
            rel_F = _rel_err_F_aligned(est, true_d, K, q, s)
        except Exception:
            rel_F = float("nan")

        # --- Filter with estimated params --------------------------------
        try:
            _all_mats = (
                est["A_list"] + est["B_list"]
                + est["C_list"] + est["D_list"]
            )
            _valid = all(
                np.isfinite(M).all() and np.linalg.norm(M, "fro") < 1e4
                for M in _all_mats
            )
            if _valid:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    est_p  = _params_from_dict(est)
                    rmse_e = _filter_rmse(est_p, ys, xs)
                    if rmse_e > 1e6:
                        rmse_e = float("nan")
            else:
                rmse_e = float("nan")
        except Exception:
            rmse_e = float("nan")

        # --- Store main row -------------------------------------------
        rows.append({
            "model":         model_name,
            "N":             N,
            "seed":          seed,
            "variant":       variant_name,
            "best_log_lik":  best_ll,
            "basin_rate":    basin_rate,
            "best_n_iter":   info["best_n_iter"],
            "best_converged": info["best_converged"],
            "rel_err_F":     rel_F,
            "rmse_estimated": rmse_e,
            "rmse_oracle":   oracle_rmse,
            "all_log_liks":  ";".join(f"{ll:.4f}" for ll in all_lls),
        })

        # --- LL history rows (for Fig. 3) ------------------------------
        for it, ll in enumerate(info["log_lik_history"]):
            history_rows.append({
                "model":   model_name,
                "N":       N,
                "seed":    seed,
                "variant": variant_name,
                "iteration": it,
                "log_lik": ll,
            })

    return rows, history_rows


# ---------------------------------------------------------------------------
# Full Monte-Carlo runner
# ---------------------------------------------------------------------------

def run_em_all(
    models:      Sequence[str]       = (DEFAULT_MODEL,),
    N_list:      Sequence[int]       = DEFAULT_N_LIST,
    n_runs:      int                 = DEFAULT_N_RUNS,
    n_inits:     int                 = DEFAULT_N_INITS,
    max_iter:    int                 = DEFAULT_MAX_ITER,
    tol:         float               = DEFAULT_TOL,
    output_dir:  str | pathlib.Path  = DEFAULT_OUT_DIR,
    verbose:     bool                = True,
) -> pd.DataFrame:
    """
    Full semi-supervised EM Monte-Carlo study.

    Writes two CSV files:
      - ``em_results.csv``      one row per (model, N, seed, variant)
      - ``em_ll_history.csv``   LL-per-iteration for best run, all trials

    Returns
    -------
    pd.DataFrame of em_results.
    """
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_main    = output_dir / "em_results.csv"
    out_history = output_dir / "em_ll_history.csv"

    total = len(models) * len(N_list) * n_runs
    if verbose:
        print(
            f"Semi-supervised EM MC: {len(models)} model(s) × "
            f"{len(N_list)} N values × {n_runs} runs × 2 variants "
            f"= {total * 2} rows  (n_inits={n_inits})"
        )

    all_rows:    list[dict] = []
    all_history: list[dict] = []
    done = 0
    t0   = time.perf_counter()

    for model_name in models:
        for N in N_list:
            for seed in range(n_runs):
                try:
                    rows, hist = run_em_trial(
                        model_name=model_name,
                        N=N, seed=seed,
                        n_inits=n_inits,
                        max_iter=max_iter,
                        tol=tol,
                    )
                    all_rows.extend(rows)
                    all_history.extend(hist)
                except Exception as exc:
                    logger.error(
                        "Trial failed: model=%s N=%d seed=%d — %s",
                        model_name, N, seed, exc,
                    )
                    for v in ("PH", "GEM"):
                        all_rows.append({
                            "model": model_name, "N": N,
                            "seed": seed, "variant": v,
                            "best_log_lik": float("nan"),
                            "basin_rate": float("nan"),
                            "best_n_iter": 0,
                            "best_converged": False,
                            "rel_err_F": float("nan"),
                            "rmse_estimated": float("nan"),
                            "rmse_oracle": float("nan"),
                            "all_log_liks": "",
                        })

                done += 1
                if verbose and (done % max(1, total // 20) == 0
                                or done == total):
                    elapsed = time.perf_counter() - t0
                    eta = elapsed / done * (total - done)
                    print(
                        f"  [{done:>{len(str(total))}}/{total}] "
                        f"model={model_name} N={N:>5} seed={seed:>3}  "
                        f"elapsed={elapsed:6.1f}s  ETA={eta:6.1f}s",
                        flush=True,
                    )

    df         = pd.DataFrame(all_rows)
    df_history = pd.DataFrame(all_history)

    df.to_csv(out_main, index=False)
    df_history.to_csv(out_history, index=False)

    elapsed_total = time.perf_counter() - t0
    if verbose:
        print(f"\nDone in {elapsed_total:.1f}s.")
        print(f"Results  → {out_main}")
        print(f"LL hist  → {out_history}")
        _print_em_summary(df)

    return df


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _print_em_summary(df: pd.DataFrame) -> None:
    """Compact mean metrics per (model, N, variant)."""
    print("\n" + "=" * 75)
    print("Semi-supervised EM summary (mean over runs)")
    print("=" * 75)
    print(f"{'Model':>5}  {'N':>5}  {'Var':>3}  "
          f"{'best LL':>10}  {'basin%':>7}  {'n_iter':>6}  "
          f"{'rel F':>7}  {'RMSE est':>9}  {'RMSE ora':>9}")
    print("-" * 75)
    for (model, N, var), g in df.groupby(["model", "N", "variant"]):
        print(
            f"{model:>5}  {N:>5}  {var:>3}  "
            f"{g['best_log_lik'].mean():10.2f}  "
            f"{g['basin_rate'].mean()*100:7.1f}  "
            f"{g['best_n_iter'].mean():6.1f}  "
            f"{g['rel_err_F'].mean():7.4f}  "
            f"{g['rmse_estimated'].mean():9.4f}  "
            f"{g['rmse_oracle'].mean():9.4f}"
        )
    print("=" * 75)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run semi-supervised EM MC study (paper §6.4).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--models", nargs="+", default=[DEFAULT_MODEL],
                        choices=["M1", "M2", "M3"])
    parser.add_argument("--N-list", nargs="+", type=int,
                        default=list(DEFAULT_N_LIST), dest="N_list")
    parser.add_argument("--n-runs",   type=int, default=DEFAULT_N_RUNS)
    parser.add_argument("--n-inits",  type=int, default=DEFAULT_N_INITS)
    parser.add_argument("--max-iter", type=int, default=DEFAULT_MAX_ITER)
    parser.add_argument("--tol",      type=float, default=DEFAULT_TOL)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    args = _parse_args()
    run_em_all(
        models=args.models,
        N_list=args.N_list,
        n_runs=args.n_runs,
        n_inits=args.n_inits,
        max_iter=args.max_iter,
        tol=args.tol,
        output_dir=args.output_dir,
        verbose=not args.quiet,
    )
