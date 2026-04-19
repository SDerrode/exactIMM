#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prg/gui/main.py
===============
CLI entry point for the GSS simulator GUI.

Usage
-----
  python -m prg.gui.main
  python -m prg.gui.main --model model_gss_K2_q1_s1
  python -m prg.gui.main -K 3 -q 2 -s 1
"""

import argparse
import importlib
import inspect
import sys

import numpy as np
from PyQt6.QtCore import QSettings
from PyQt6.QtWidgets import QApplication

from prg.gui.main_window import GSSMainWindow
from prg.models.base_gss_model import BaseGSSModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_model(model_name: str) -> BaseGSSModel:
    """Dynamically import and instantiate a model from prg/models/."""
    module = importlib.import_module(f"prg.models.{model_name}")
    for _, obj in inspect.getmembers(module, inspect.isclass):
        if issubclass(obj, BaseGSSModel) and obj is not BaseGSSModel:
            return obj()
    raise ImportError(
        f"No BaseGSSModel subclass found in prg.models.{model_name}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="GSS Simulator — interactive PyQt6 GUI",
    )
    parser.add_argument(
        "--model", "-m",
        metavar="NAME",
        help="Model name in prg/models/ (e.g. model_gss_K2_q1_s1). "
             "Overrides -K/-q/-s when provided.",
    )
    parser.add_argument("-K", type=int, default=2, help="Number of Markov states (default 2)")
    parser.add_argument("-q", type=int, default=1, help="Hidden dimension (default 1)")
    parser.add_argument("-s", type=int, default=1, help="Observed dimension (default 1)")
    args = parser.parse_args()

    model = None
    K, q, s = args.K, args.q, args.s
    P = None

    if args.model:
        try:
            model = _load_model(args.model)
            p = model.get_params()
            K, q, s = p["K"], p["q"], p["s"]
            P = np.asarray(p["P"]) if p.get("P") is not None else None
        except Exception as exc:  # noqa: BLE001
            print(f"[ERROR] Could not load model '{args.model}': {exc}", file=sys.stderr)
            sys.exit(1)

    app = QApplication(sys.argv)
    app.setApplicationName("FofGss Simulator")

    # Restart loop: recreate the window when a preset with new K/q/s is selected.
    while True:
        win = GSSMainWindow(K=K, q=q, s=s, P=P, model=model)
        # Only apply default size if no saved geometry (QSettings) exists
        if QSettings("FofGss", "Simulator").value("geometry") is None:
            win.resize(1100, 700)

        # Container for the next model (mutable so the lambda can write to it).
        _next: list = []

        def _on_restart(m, w=win, buf=_next) -> None:
            buf.append(m)
            w.close()

        win.restart_with_model.connect(_on_restart)
        win.show()
        app.exec()          # blocks until the window is closed

        if not _next:       # normal exit — no restart requested
            break

        # Prepare next iteration
        model = _next[0]
        p_dict = model.get_params()
        K, q, s = p_dict["K"], p_dict["q"], p_dict["s"]
        P = np.asarray(p_dict["P"]) if p_dict.get("P") is not None else None

    sys.exit(0)


if __name__ == "__main__":
    main()
