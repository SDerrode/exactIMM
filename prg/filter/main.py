#!/usr/bin/env python3
"""
prg/filter/main.py
==================
CLI entry point for the GSS fast optimal filter.

Usage — from an existing simulation CSV
----------------------------------------
    python -m prg.filter.main --csv data/simulated/sim.csv \\
        --model model_gss_K2_q1_s1 --output filter_results.csv

Usage — simulate and filter in one step
-----------------------------------------
    python -m prg.filter.main --model model_gss_K2_q1_s1 -N 1000 --seed 42
    python -m prg.filter.main --model model_gss_K2_q1_s1 -N 500  --no-save -v 2

Options
-------
    --model      str    Model name in prg/models/ (always required)
    --csv        str    Path to an existing simulation CSV (optional)
    -N           int    Steps to simulate if --csv is not given (required w/o --csv)
    --seed       int    Random seed for simulation (ignored with --csv)
    --output     str    Output CSV filename for filter results (auto if omitted)
    --no-save           Do not write any CSV
    --log-level  str    DEBUG | INFO | WARNING | ERROR
    -v, --verbose int   0=silent  1=normal  2=full debug
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import logging
import pathlib
import sys
import tomllib
from datetime import datetime

from prg.classes.GSSParams import GSSParams
from prg.filter.gss_filter import GSSFilter
from prg.models.base_gss_model import BaseGSSModel
from prg.utils.ab_constraint import apply_AB_constraint
from prg.utils.exceptions import GSSError

# ---------------------------------------------------------------------------
_CONFIG_FILENAME = "config.toml"
_DEFAULT_LOG_LEVEL = "INFO"


# ---------------------------------------------------------------------------
# Helpers (shared with prg/simulate.py)
# ---------------------------------------------------------------------------


def _project_root() -> pathlib.Path:
    here = pathlib.Path(__file__).resolve().parent
    for candidate in [here.parent.parent, here.parent.parent.parent]:
        if (candidate / _CONFIG_FILENAME).exists():
            return candidate
    cwd = pathlib.Path.cwd()
    return cwd if (cwd / _CONFIG_FILENAME).exists() else cwd


def _load_config(root: pathlib.Path) -> dict:
    cfg_path = root / _CONFIG_FILENAME
    if not cfg_path.exists():
        return {}
    with cfg_path.open("rb") as fh:
        return tomllib.load(fh)


def _setup_logging(
    log_level: str,
    logs_dir: pathlib.Path,
    tag: str,
    verbose: int,
) -> None:
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"filter_{tag}_{ts}.log"
    numeric = getattr(logging, log_level.upper(), logging.INFO)
    fmt = logging.Formatter(
        fmt="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    root_logger = logging.getLogger("exactIMM")
    root_logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    root_logger.addHandler(fh)

    if verbose > 0:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(numeric if verbose == 1 else logging.DEBUG)
        ch.setFormatter(fmt)
        root_logger.addHandler(ch)

    root_logger.info("Log file: %s", log_path)


def _load_model(model_name: str) -> BaseGSSModel:
    module_path = f"prg.models.{model_name}"
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        logging.getLogger("exactIMM").error("Cannot import model '%s': %s", module_path, exc)
        sys.exit(1)
    for _, obj in inspect.getmembers(module, inspect.isclass):
        if issubclass(obj, BaseGSSModel) and obj is not BaseGSSModel:
            return obj()
    logging.getLogger("exactIMM").error("No BaseGSSModel subclass found in '%s'.", module_path)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m prg.filter.main",
        description="Run the GSS fast optimal filter.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        required=True,
        metavar="MODEL",
        help="Model name in prg/models/ (e.g. model_gss_K2_q1_s1).",
    )
    parser.add_argument(
        "--csv",
        default=None,
        metavar="PATH",
        help="Path to an existing simulation CSV.  "
        "If omitted, a new simulation is run (-N required).",
    )
    parser.add_argument(
        "-N",
        dest="N",
        type=int,
        default=None,
        metavar="N",
        help="Number of time steps to simulate (required without --csv).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        metavar="SEED",
        help="RNG seed for simulation (ignored when --csv is supplied).",
    )
    parser.add_argument(
        "--output",
        default=None,
        metavar="FILENAME",
        help="Output CSV for filter results (auto-generated if omitted).",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        metavar="LEVEL",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not write any CSV (dry run).",
    )
    parser.add_argument(
        "--constraint",
        action="store_true",
        help="Enforce the AB constraint: recompute "
        "A(k) = Δ(k) Σ_V(k)⁻¹ C(k) and B(k) = Δ(k) Σ_V(k)⁻¹ D(k) for every "
        "regime before filtering.",
    )
    parser.add_argument(
        "--mode",
        default=None,
        choices=["gpb2", "ngh_kf"],
        help="Filter recursion variant. Omit to dispatch on the model: a model "
        "satisfying AB — e.g. after --constraint — uses 'ngh_kf' (exact "
        "fast filter), any other model uses 'gpb2' (order-2 approximation).",
    )
    parser.add_argument(
        "--input",
        default=None,
        metavar="SPEC",
        help="Exogenous input ('consigne') for the simulate+filter path (-N): a "
        "generator (zeros|ones|gaussian[(std)]|step[(n0)]|ramp[(slope)]|"
        "sin[(period)]|square[(period)]), const(a,b,...), or a CSV path. "
        "Requires the model to define G_list. (With --csv, the file's u_* "
        "columns are used automatically.)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        type=int,
        default=1,
        choices=[0, 1, 2],
        metavar="LEVEL",
        help="Console verbosity: 0=silent  1=normal  2=debug.",
    )
    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # Validate: need -N when --csv is not given
    if args.csv is None and args.N is None:
        parser.error("Either --csv PATH or -N N is required.")

    root = _project_root()
    cfg = _load_config(root)

    log_level = args.log_level or cfg.get("general", {}).get("log_level", _DEFAULT_LOG_LEVEL)
    paths_cfg = cfg.get("paths", {})
    logs_dir = root / paths_cfg.get("logs", "logs")
    data_dir = root / paths_cfg.get("data_simulated", "data/simulated")

    tag = f"{args.model}_N{args.N or 'csv'}_seed{args.seed or 'random'}"
    _setup_logging(log_level, logs_dir, tag, args.verbose)
    log = logging.getLogger("exactIMM.filter.main")

    # --- Load model and build params ---
    log.info("Loading model '%s' …", args.model)
    model = _load_model(args.model)
    try:
        params = GSSParams.from_model(model)
    except GSSError as exc:
        log.error("Parameter error: %s", exc)
        sys.exit(1)

    if args.verbose >= 2:
        params.summary()

    # --- Apply AB constraint to A, B (optional) ---
    if args.constraint:
        log.info("--constraint: applying AB constraint to A(k), B(k) …")
        try:
            params = apply_AB_constraint(params, logger=log)
        except ValueError as exc:
            log.error("AB constraint failed: %s", exc)
            sys.exit(1)
        log.info("AB constraint applied — A(k), B(k) updated for all %d regimes.", params.K)
        if args.verbose >= 2:
            log.info("Updated parameters:")
            params.summary()

    log.info("GSSParams OK: K=%d  q=%d  s=%d", params.K, params.q, params.s)

    # --- Build filter ---
    # mode=None lets GSSFilter dispatch on the params type (NGHMSMParams →
    # ngh_kf, base GSSParams → gpb2); --mode overrides explicitly.
    filt = GSSFilter(params, mode=args.mode)
    log.info("GSSFilter ready: %s", filt)

    if args.no_save:
        # ---- Dry run ----
        if args.csv:
            import pandas as pd

            df_in = pd.read_csv(pathlib.Path(args.csv))
            y_cols = [f"y_{i}" for i in range(params.s)]
            for _, row in df_in.iterrows():
                y = row[y_cols].to_numpy(dtype=float)
                filt.step(y)
        else:
            from prg.classes.GSSSimulator import GSSSimulator

            sim = GSSSimulator(params, N=args.N, seed=args.seed)
            for _, _r, _x, y in sim:
                filt.step(y)
        log.info("Dry run complete (%d steps).", filt.n)
        return

    # ---- Run and save ----
    if args.csv:
        # Filter from existing CSV
        log.info("Filtering from CSV: %s", args.csv)
        df = filt.run_csv(args.csv)
        out_path = _resolve_output(args.output, data_dir, args.model, None, None)
    else:
        # Simulate + filter jointly
        if args.input is not None and params.p == 0:
            log.error(
                "--input given but model '%s' has no input gain (params.p == 0).",
                args.model,
            )
            sys.exit(1)
        log.info("Simulating %d steps (seed=%s) + filtering …", args.N, args.seed)
        sim_path, df = filt.run(
            N=args.N,
            seed=args.seed,
            output_dir=data_dir,
            model_name=args.model,
            u=args.input,
        )
        if sim_path:
            log.info("Simulation CSV: %s", sim_path)
            if args.verbose > 0:
                print(f"Simulation saved: {sim_path}")
        out_path = _resolve_output(args.output, data_dir, args.model, args.N, args.seed)

    df.to_csv(out_path, index=False)
    log.info("Filter results saved: %s", out_path)
    if args.verbose > 0:
        print(f"Filter results saved: {out_path}")
        rmse = (df["sq_err"] ** 0.5).mean() if "sq_err" in df.columns else float("nan")
        print(f"Mean ||error||: {rmse:.4f}  (average over {len(df)} steps)")


def _resolve_output(
    explicit: str | None,
    data_dir: pathlib.Path,
    model_name: str,
    N: int | None,
    seed: int | None,
) -> pathlib.Path:
    if explicit:
        p = pathlib.Path(explicit)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
    data_dir.mkdir(parents=True, exist_ok=True)
    seed_str = str(seed) if seed is not None else "random"
    n_str = str(N) if N is not None else "csv"
    from datetime import datetime

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return data_dir / f"filter_{model_name}_N{n_str}_seed{seed_str}_{ts}.csv"


if __name__ == "__main__":
    main()
