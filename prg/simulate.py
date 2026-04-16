#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prg/simulate.py
===============
CLI entry point for the GSS simulator.

Usage
-----
    python -m prg.simulate --model <model_name> -N <int> [OPTIONS]

Options
-------
    --model      str    Name of the model module in prg/models/ (required)
    -N           int    Number of time steps n=0..N-1 (required)
    --seed       int    Random seed (default: None = non-deterministic)
    --output     str    Output CSV filename (default: auto-generated)
    --log-level  str    Override log level: DEBUG|INFO|WARNING|ERROR
    --no-save           Do not write the CSV (dry run)
    -v, --verbose int   Console verbosity: 0=silent, 1=normal, 2=full summary

Example
-------
    python -m prg.simulate --model model_gss_K2_q1_s1 -N 1000 --seed 42 -v 2
    python -m prg.simulate --model model_gss_K2_q1_s1 -N 5000 --no-save --log-level DEBUG
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
from prg.classes.GSSSimulator import GSSSimulator
from prg.models.base_gss_model import BaseGSSModel
from prg.utils.exceptions import GSSError
from prg.utils.h5_constraint import apply_h5_constraint

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CONFIG_FILENAME = "config.toml"
_DEFAULT_LOG_LEVEL = "INFO"


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def _load_config(root: pathlib.Path) -> dict:
    """
    Load config.toml from the project root.

    Falls back to an empty dict with defaults if the file is not found.
    """
    cfg_path = root / _CONFIG_FILENAME
    if not cfg_path.exists():
        logging.getLogger("fofgss").warning(
            "config.toml not found at '%s'. Using built-in defaults.", cfg_path
        )
        return {}
    with cfg_path.open("rb") as fh:
        return tomllib.load(fh)


def _project_root() -> pathlib.Path:
    """
    Return the project root directory.

    Strategy: walk up from this file until we find config.toml,
    or fall back to cwd.
    """
    here = pathlib.Path(__file__).resolve().parent
    for candidate in [here.parent, here.parent.parent]:
        if (candidate / _CONFIG_FILENAME).exists():
            return candidate
    cwd = pathlib.Path.cwd()
    if (cwd / _CONFIG_FILENAME).exists():
        return cwd
    return cwd


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------


def _setup_logging(
    log_level: str,
    logs_dir: pathlib.Path,
    model_name: str,
    N: int,
    seed: int | None,
    verbose: int,
) -> None:
    """Configure the root 'fofgss' logger with console + file handlers."""
    logs_dir.mkdir(parents=True, exist_ok=True)

    seed_str = str(seed) if seed is not None else "random"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"simulate_{model_name}_N{N}_seed{seed_str}_{timestamp}.log"
    log_path = logs_dir / log_filename

    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    fmt = logging.Formatter(
        fmt="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    root_logger = logging.getLogger("fofgss")
    root_logger.setLevel(logging.DEBUG)   # capture everything; handlers filter

    # --- File handler (always DEBUG) ---
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    root_logger.addHandler(fh)

    # --- Console handler (level from config / CLI) ---
    if verbose > 0:
        console_level = numeric_level if verbose == 1 else logging.DEBUG
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(console_level)
        ch.setFormatter(fmt)
        root_logger.addHandler(ch)

    root_logger.info("Log file: %s", log_path)


# ---------------------------------------------------------------------------
# Dynamic model loading
# ---------------------------------------------------------------------------


def _load_model(model_name: str) -> BaseGSSModel:
    """
    Import ``prg.models.<model_name>`` and return an instance of the
    first BaseGSSModel subclass found in that module.

    Raises
    ------
    SystemExit
        If the module cannot be imported or no subclass is found.
    """
    module_path = f"prg.models.{model_name}"
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        logging.getLogger("fofgss").error(
            "Cannot import model '%s': %s", module_path, exc
        )
        sys.exit(1)

    for _, obj in inspect.getmembers(module, inspect.isclass):
        if issubclass(obj, BaseGSSModel) and obj is not BaseGSSModel:
            return obj()

    logging.getLogger("fofgss").error(
        "No BaseGSSModel subclass found in '%s'.", module_path
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m prg.simulate",
        description="Simulate data from a GSS model (equation 7).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", required=True, metavar="MODEL",
        help="Name of the model module in prg/models/ "
             "(e.g. model_gss_K2_q1_s1).",
    )
    parser.add_argument(
        "-N", dest="N", required=True, type=int, metavar="N",
        help="Number of time steps to simulate (n = 0 … N-1).",
    )
    parser.add_argument(
        "--seed", type=int, default=None, metavar="SEED",
        help="Random seed for reproducibility. "
             "Omit for a non-deterministic run.",
    )
    parser.add_argument(
        "--output", default=None, metavar="FILENAME",
        help="Output CSV filename (auto-generated if omitted).",
    )
    parser.add_argument(
        "--log-level", default=None, metavar="LEVEL",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Override the log level from config.toml.",
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Do not write the CSV (dry run for testing).",
    )
    parser.add_argument(
        "--constraint", action="store_true",
        help="Enforce the H5 constraint (eq. 4.8): recompute every B(k) "
             "from A(k), C(k), D(k), Σ_U(k), Δ(k), Σ_V(k) before simulating.",
    )
    parser.add_argument(
        "-v", "--verbose", type=int, default=1, choices=[0, 1, 2],
        metavar="LEVEL",
        help="Console verbosity: 0=silent, 1=normal, 2=full debug.",
    )
    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # --- Project root and config ---
    root = _project_root()
    cfg = _load_config(root)

    # --- Resolve log level (CLI > config > default) ---
    log_level = (
        args.log_level
        or cfg.get("general", {}).get("log_level", _DEFAULT_LOG_LEVEL)
    )

    # --- Resolve output paths from config ---
    paths_cfg = cfg.get("paths", {})
    logs_dir = root / paths_cfg.get("logs", "logs")
    data_simulated_dir = root / paths_cfg.get("data_simulated", "data/simulated")

    # --- Logging ---
    _setup_logging(
        log_level=log_level,
        logs_dir=logs_dir,
        model_name=args.model,
        N=args.N,
        seed=args.seed,
        verbose=args.verbose,
    )
    log = logging.getLogger("fofgss.simulate")

    # --- Load model ---
    log.info("Loading model '%s' …", args.model)
    model = _load_model(args.model)
    log.info("Model loaded: %s", model)

    # --- Build parameters ---
    log.info("Building GSSParams …")
    try:
        params = GSSParams.from_model(model)
    except GSSError as exc:
        log.error("Parameter error: %s", exc)
        sys.exit(1)

    if args.verbose >= 2:
        params.summary()

    # --- Apply H5 constraint on B (optional) ---
    if args.constraint:
        log.info("--constraint: applying H5 constraint to recompute B(k) …")
        try:
            params = apply_h5_constraint(params, logger=log)
        except ValueError as exc:
            log.error("H5 constraint failed: %s", exc)
            sys.exit(1)
        log.info("H5 constraint applied — B(k) updated for all %d regimes.", params.K)
        if args.verbose >= 2:
            log.info("Updated parameters:")
            params.summary()

    log.info(
        "Parameters OK: K=%d  q=%d  s=%d  dim_z=%d",
        params.K, params.q, params.s, params.dim_z,
    )

    # --- Simulate ---
    sim = GSSSimulator(params, N=args.N, seed=args.seed)
    log.info("GSSSimulator ready: %s", sim)

    if args.no_save:
        log.info("--no-save: running without writing CSV.")
        for _step in sim:
            pass
        log.info("Dry run complete (%d steps).", args.N)
    else:
        # Handle custom output filename
        out_dir = data_simulated_dir
        if args.output is not None:
            out_path = pathlib.Path(args.output)
            if out_path.parent != pathlib.Path("."):
                out_dir = out_path.parent
        try:
            filepath = sim.run(output_dir=out_dir, model_name=args.model)
            log.info("CSV saved: %s", filepath)
            if args.verbose > 0:
                print(f"Saved: {filepath}")
        except GSSError as exc:
            log.error("Simulation error: %s", exc)
            sys.exit(1)
        except OSError as exc:
            log.error("I/O error writing CSV: %s", exc)
            sys.exit(1)


if __name__ == "__main__":
    main()
