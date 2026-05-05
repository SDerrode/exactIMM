#!/usr/bin/env python3
"""
prg/experiments/run_supervised.py
==================================
Monte-Carlo evaluation of the supervised OLS estimator (paper §6.3).

Protocol (model M1, 100 runs)
------------------------------
For each N_train ∈ {200, 500, 1000, 2000} × seed ∈ range(100):

  1. Simulate N_train steps from the true M1 model; the full
     (r_n, x_n, y_n) sequence is observed.
  2. Apply :func:`prg.learning.supervised.fit_supervised` with four
     H5 projection choices τ ∈ {None, 'b', 'a', 'su'}.
  3. For each projection:
       - relative Frobenius error on the full F matrix and on b
         (averaged over regimes)
       - H5 residual (max over regimes of the relative residual
         ‖Δᵀ A + Σ_V Bᵀ − P M⁻¹ W‖_F / max(‖Z‖_F, 1))
  4. For each projection: build estimated GSSParams → run h5_exact
     filter on the simulated observations → compute RMSE.
  5. Also compute the oracle filter RMSE (true parameters).

Results are written to ``data/experiments/supervised_results.csv``.

Usage
-----
    python -m prg.experiments.run_supervised
    python -m prg.experiments.run_supervised --n-runs 10 --N-list 200 500
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import time
from collections.abc import Sequence

import numpy as np
import pandas as pd

from prg.classes.GSSParams import GSSParams
from prg.classes.GSSSimulator import GSSSimulator
from prg.experiments.metrics import compute_rmse
from prg.experiments.models_paper import get_params
from prg.experiments.run_simulations import _params_from_dict
from prg.filter.gss_filter import GSSFilter
from prg.learning.supervised import fit_supervised
from prg.utils.h5_constraint import compute_h5_residual

__all__ = ["run_supervised_all", "run_supervised_trial"]

logger = logging.getLogger("exactIMM.experiments.supervised")

# ---------------------------------------------------------------------------
# Protocol constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "M1"  # only M1 for §6.3
DEFAULT_N_LIST = (200, 500, 1_000, 2_000)
DEFAULT_N_RUNS = 100
DEFAULT_PROJS = (None, "b", "a", "su")  # τ ∈ {none, B, A, Σ_U}
DEFAULT_OUT_DIR = pathlib.Path("data") / "experiments"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_params(d: dict) -> GSSParams:
    """Build GSSParams from a parameter dict (same as run_simulations)."""
    return _params_from_dict(d)


def _rel_frob_F(
    A_est: np.ndarray,
    B_est: np.ndarray,
    C_est: np.ndarray,
    D_est: np.ndarray,
    A_true: np.ndarray,
    B_true: np.ndarray,
    C_true: np.ndarray,
    D_true: np.ndarray,
    q: int,
    s: int,
) -> float:
    """
    Relative Frobenius error of the full transition matrix F.

    ‖F_est − F_true‖_F / ‖F_true‖_F
    with F = [[A, B], [C, D]]  (dimension (q+s) × (q+s)).
    """
    F_est = np.block([[A_est, B_est], [C_est, D_est]])
    F_true = np.block([[A_true, B_true], [C_true, D_true]])
    denom = np.linalg.norm(F_true, "fro")
    if denom < 1e-14:
        return float("nan")
    return float(np.linalg.norm(F_est - F_true, "fro") / denom)


def _rel_frob(M_est: np.ndarray, M_true: np.ndarray) -> float:
    """Relative Frobenius error of a single matrix."""
    denom = np.linalg.norm(M_true, "fro")
    if denom < 1e-14:
        return float("nan")
    return float(np.linalg.norm(M_est - M_true, "fro") / denom)


def _max_h5_residual(est: dict) -> float:
    """
    Max relative H5 residual over all regimes of an estimated model.

    Uses the same definition as GSSFilter._check_h5 and
    prg/experiments/models_paper.py::_check_h5.
    """
    K, q = est["K"], est["q"]
    max_rel = 0.0
    for k in range(K):
        A = est["A_list"][k]
        B = est["B_list"][k]
        C = est["C_list"][k]
        D = est["D_list"][k]
        SU = est["Sigma_U_list"][k]
        Dt = est["Delta_list"][k]
        SV = est["Sigma_V_list"][k]
        try:
            F = compute_h5_residual(A, B, C, D, SU, Dt, SV)
        except np.linalg.LinAlgError:
            return float("nan")
        Z = Dt.T @ A + SV @ B.T
        scale = max(float(np.linalg.norm(Z, "fro")), 1.0)
        rel = float(np.linalg.norm(F, "fro")) / scale
        max_rel = max(max_rel, rel)
    return max_rel


def _filter_rmse(
    params: GSSParams,
    ys: np.ndarray,
    xs: np.ndarray,
) -> float:
    """
    Run h5_exact filter with *params* on observations *ys* and return RMSE.

    Parameters
    ----------
    params : GSSParams
    ys : ndarray (N, s)   observations
    xs : ndarray (N, q)   true hidden states (for RMSE computation)

    Returns
    -------
    float  RMSE normalised by q.
    """
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        filt = GSSFilter(params, mode="h5_exact")

    N, s = ys.shape
    q = xs.shape[1]
    x_est = np.empty_like(xs)

    for n in range(N):
        y = ys[n].reshape(-1, 1)
        res = filt.step(y)
        x_est[n] = res.E_x.ravel()

    return compute_rmse(xs, x_est)


# ---------------------------------------------------------------------------
# Single-trial runner
# ---------------------------------------------------------------------------


def run_supervised_trial(
    model_name: str,
    N: int,
    seed: int,
    projections: Sequence[str | None] = DEFAULT_PROJS,
) -> list[dict]:
    """
    Run one supervised estimation trial and return one dict per projection.

    Parameters
    ----------
    model_name : str
        Reference model (typically "M1").
    N : int
        Training sequence length.
    seed : int
        Random seed for the simulator.
    projections : sequence
        H5 projection choices passed as ``constraint`` to fit_supervised.
        Elements are None, 'b', 'a', or 'su'.

    Returns
    -------
    list of dict, one per projection, with keys:
        model, N, seed, projection,
        rel_err_F,  rel_err_b,
        h5_residual,
        rmse_estimated, rmse_oracle
    """
    # ------------------------------------------------------------------
    # Build true params
    # ------------------------------------------------------------------
    true_d = get_params(model_name)
    true_p = _build_params(true_d)
    K, q, s = true_p.K, true_p.q, true_p.s

    # True F and b per regime
    true_F = [
        np.block(
            [[true_d["A_list"][k], true_d["B_list"][k]], [true_d["C_list"][k], true_d["D_list"][k]]]
        )
        for k in range(K)
    ]
    true_b = true_d["b_list"]  # list of (q+s, 1)

    # ------------------------------------------------------------------
    # Simulate
    # ------------------------------------------------------------------
    sim = GSSSimulator(true_p, N=N, seed=seed)
    rs_list, xs_list, ys_list = [], [], []
    for _, r, x, y in sim:
        rs_list.append(r)
        xs_list.append(x.ravel())
        ys_list.append(y.ravel())

    rs = np.array(rs_list, dtype=int)  # (N,)
    xs = np.array(xs_list)  # (N, q)
    ys = np.array(ys_list)  # (N, s)

    # ------------------------------------------------------------------
    # Oracle filter RMSE (true params, h5_exact)
    # ------------------------------------------------------------------
    oracle_rmse = _filter_rmse(true_p, ys, xs)

    # ------------------------------------------------------------------
    # Per-projection estimation
    # ------------------------------------------------------------------
    rows: list[dict] = []

    for proj in projections:
        proj_label = proj if proj is not None else "none"
        try:
            est = fit_supervised(
                rs,
                xs,
                ys,
                K,
                q,
                s,
                constraint=proj,
                delta_zero=False,
                verbose=False,
            )
        except (ValueError, np.linalg.LinAlgError) as exc:
            logger.warning(
                "supervised fit failed: model=%s N=%d seed=%d proj=%s — %s",
                model_name,
                N,
                seed,
                proj_label,
                exc,
            )
            rows.append(
                {
                    "model": model_name,
                    "N": N,
                    "seed": seed,
                    "projection": proj_label,
                    "rel_err_F": float("nan"),
                    "rel_err_b": float("nan"),
                    "h5_residual": float("nan"),
                    "rmse_estimated": float("nan"),
                    "rmse_oracle": oracle_rmse,
                }
            )
            continue

        # --- Parameter errors (averaged over regimes) ------------------
        rel_F_vals, rel_b_vals = [], []
        for k in range(K):
            rel_F_vals.append(
                _rel_frob_F(
                    est["A_list"][k],
                    est["B_list"][k],
                    est["C_list"][k],
                    est["D_list"][k],
                    true_d["A_list"][k],
                    true_d["B_list"][k],
                    true_d["C_list"][k],
                    true_d["D_list"][k],
                    q,
                    s,
                )
            )
            rel_b_vals.append(_rel_frob(est["b_list"][k], true_b[k]))

        rel_err_F = float(np.nanmean(rel_F_vals))
        rel_err_b = float(np.nanmean(rel_b_vals))
        h5_res = _max_h5_residual(est)

        # --- Filter with estimated params (h5_exact) -------------------
        # Guard: if any matrix is non-finite or has huge norm (ill-conditioned
        # projection), skip the filter and report nan.
        _all_mats = est["A_list"] + est["B_list"] + est["C_list"] + est["D_list"]
        _valid = all(np.isfinite(M).all() and np.linalg.norm(M, "fro") < 1e4 for M in _all_mats)
        if not _valid:
            logger.warning(
                "estimated params contain inf/huge values: model=%s N=%d "
                "seed=%d proj=%s — skipping filter",
                model_name,
                N,
                seed,
                proj_label,
            )
            rmse_e = float("nan")
        else:
            try:
                est_p = _build_params(est)
                rmse_e = _filter_rmse(est_p, ys, xs)
                # Physical sanity: RMSE > 1e6 → diverged filter
                if rmse_e > 1e6:
                    rmse_e = float("nan")
            except Exception as exc:
                logger.warning(
                    "filter with estimated params failed: model=%s N=%d seed=%d proj=%s — %s",
                    model_name,
                    N,
                    seed,
                    proj_label,
                    exc,
                )
                rmse_e = float("nan")

        rows.append(
            {
                "model": model_name,
                "N": N,
                "seed": seed,
                "projection": proj_label,
                "rel_err_F": rel_err_F,
                "rel_err_b": rel_err_b,
                "h5_residual": h5_res,
                "rmse_estimated": rmse_e,
                "rmse_oracle": oracle_rmse,
            }
        )

    return rows


# ---------------------------------------------------------------------------
# Full Monte-Carlo runner
# ---------------------------------------------------------------------------


def run_supervised_all(
    models: Sequence[str] = (DEFAULT_MODEL,),
    N_list: Sequence[int] = DEFAULT_N_LIST,
    projections: Sequence[str | None] = DEFAULT_PROJS,
    n_runs: int = DEFAULT_N_RUNS,
    output_dir: str | pathlib.Path = DEFAULT_OUT_DIR,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run the full supervised estimation Monte-Carlo study and save results.

    Parameters
    ----------
    models : sequence of str
        Models to evaluate (default: ("M1",)).
    N_list : sequence of int
        Training sequence lengths.
    projections : sequence of str or None
        H5 projection choices (None=free OLS, 'b', 'a', 'su').
    n_runs : int
        Number of MC runs (seeds 0..n_runs-1).
    output_dir : path-like
        Directory for ``supervised_results.csv``.
    verbose : bool

    Returns
    -------
    pd.DataFrame
        One row per (model, N, seed, projection).
    """
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "supervised_results.csv"

    total_trials = len(models) * len(N_list) * n_runs
    total_rows = total_trials * len(projections)
    if verbose:
        print(
            f"Supervised MC: {len(models)} model(s) × "
            f"{len(N_list)} N values × {n_runs} runs "
            f"× {len(projections)} projections = {total_rows} rows"
        )

    all_rows: list[dict] = []
    done = 0
    t0 = time.perf_counter()

    for model_name in models:
        for N in N_list:
            for seed in range(n_runs):
                try:
                    rows = run_supervised_trial(
                        model_name=model_name,
                        N=N,
                        seed=seed,
                        projections=projections,
                    )
                    all_rows.extend(rows)
                except Exception as exc:
                    logger.error(
                        "Trial failed: model=%s N=%d seed=%d — %s",
                        model_name,
                        N,
                        seed,
                        exc,
                    )
                    for proj in projections:
                        all_rows.append(
                            {
                                "model": model_name,
                                "N": N,
                                "seed": seed,
                                "projection": proj if proj else "none",
                                "rel_err_F": float("nan"),
                                "rel_err_b": float("nan"),
                                "h5_residual": float("nan"),
                                "rmse_estimated": float("nan"),
                                "rmse_oracle": float("nan"),
                            }
                        )

                done += 1
                if verbose and (done % max(1, total_trials // 40) == 0 or done == total_trials):
                    elapsed = time.perf_counter() - t0
                    eta = elapsed / done * (total_trials - done)
                    print(
                        f"  [{done:>{len(str(total_trials))}}/{total_trials}] "
                        f"model={model_name} N={N:>5} seed={seed:>3}  "
                        f"elapsed={elapsed:6.1f}s  ETA={eta:6.1f}s",
                        flush=True,
                    )

    df = pd.DataFrame(all_rows)
    df.to_csv(out_path, index=False)

    elapsed_total = time.perf_counter() - t0
    if verbose:
        print(f"\nDone in {elapsed_total:.1f}s.  Results → {out_path}")
        _print_supervised_summary(df)

    return df


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def _print_supervised_summary(df: pd.DataFrame) -> None:
    """Print mean metrics per (model, N, projection)."""
    print("\n" + "=" * 72)
    print("Supervised estimation summary (mean over runs)")
    print("=" * 72)
    print(
        f"{'Model':>5}  {'N':>5}  {'Proj':>4}  "
        f"{'rel F err':>10}  {'rel b err':>10}  "
        f"{'H5 resid':>10}  {'RMSE est':>9}  {'RMSE ora':>9}"
    )
    print("-" * 72)
    for (model, N, proj), g in df.groupby(["model", "N", "projection"]):
        print(
            f"{model:>5}  {N:>5}  {proj:>4}  "
            f"{g['rel_err_F'].mean():10.4f}  "
            f"{g['rel_err_b'].mean():10.4f}  "
            f"{g['h5_residual'].mean():10.2e}  "
            f"{g['rmse_estimated'].mean():9.4f}  "
            f"{g['rmse_oracle'].mean():9.4f}"
        )
    print("=" * 72)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run supervised estimation MC study (paper §6.3).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=[DEFAULT_MODEL],
        choices=["M1", "M2", "M3"],
    )
    parser.add_argument(
        "--N-list",
        nargs="+",
        type=int,
        default=list(DEFAULT_N_LIST),
        dest="N_list",
    )
    parser.add_argument(
        "--projections",
        nargs="+",
        default=["none", "b", "a", "su"],
        choices=["none", "b", "a", "su"],
        help="H5 projection choices (none = free OLS).",
    )
    parser.add_argument("--n-runs", type=int, default=DEFAULT_N_RUNS)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    args = _parse_args()
    # Convert "none" string back to Python None
    projs = [None if p == "none" else p for p in args.projections]
    run_supervised_all(
        models=args.models,
        N_list=args.N_list,
        projections=projs,
        n_runs=args.n_runs,
        output_dir=args.output_dir,
        verbose=not args.quiet,
    )
