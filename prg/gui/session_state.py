#!/usr/bin/env python3
"""
prg/gui/session_state.py
========================
``_SessionState`` — the GUI's single source of truth for everything
produced by Simulate / Filter / Monte-Carlo / Load. Extracted verbatim
from ``prg/gui/main_window.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class _SessionState:
    """Holds everything the UI may need to display about the current session.

    Two independent slots:

    * data        — produced by Simulate (single) or Load CSV
    * innovations — produced by Filter, only meaningful with `data` AND `params`

    `params` and `params_signature` are captured at the moment Simulate /
    Load runs, NOT live from the widgets. The Filter step uses them as-is
    even if the widgets have since been edited (drift is signalled in the
    UI but not silently corrected).
    """

    data: tuple | None = None  # (ns, rs, xs, ys, seed_used)
    params: object | None = None  # GSSParams (avoids circular import)
    params_signature: tuple | None = None  # bytes signature of GUI at capture
    u: np.ndarray | None = None  # exogenous input (N, p) used by Simulate / Filter
    innovations: np.ndarray | None = None  # (N, s)
    # D5: filter arrays kept for session persistence
    filter_E_xs: np.ndarray | None = None  # (N, q)
    filter_Var_xs: np.ndarray | None = None  # (N, q)
    filter_pis: np.ndarray | None = None  # (N, K)
    filter_log_lik: float | None = None

    # ----- Predicates --------------------------------------------------

    def has_data(self) -> bool:
        return self.data is not None

    def has_filter(self) -> bool:
        return self.innovations is not None

    def can_filter(self) -> bool:
        return self.has_data() and self.params is not None

    # ----- Atomic mutations -------------------------------------------

    def _clear_filter(self) -> None:
        """Drop every filter-produced array (innovations + D5 results).

        Must run whenever the data or params change, otherwise a stale
        ``filter_pis`` / ``filter_E_xs`` from a previous run would be paired
        with new ground-truth regimes (e.g. in the regime-diagnostics
        confusion matrix), silently producing wrong figures.
        """
        self.innovations = None
        self.filter_E_xs = None
        self.filter_Var_xs = None
        self.filter_pis = None
        self.filter_log_lik = None

    def reset(self) -> None:
        """Forget everything — called from Reset button."""
        self.data = None
        self.params = None
        self.params_signature = None
        self.u = None
        self._clear_filter()

    def begin_simulation(self, params: object, signature: tuple | None) -> None:
        """About to launch a new Simulate: capture params, drop stale results."""
        self.params = params
        self.params_signature = signature
        self._clear_filter()  # previous filter results no longer match the new data

    def store_data(self, ns, rs, xs, ys, seed, u: np.ndarray | None = None) -> None:
        self.data = (ns, rs, xs, ys, seed)
        self.u = u

    def store_innovations(self, innov: np.ndarray) -> None:
        self.innovations = innov

    def store_filter_results(  # D5
        self,
        E_xs: np.ndarray,
        Var_xs: np.ndarray,
        pis: np.ndarray,
        log_lik_total: float,
    ) -> None:
        self.filter_E_xs = E_xs
        self.filter_Var_xs = Var_xs
        self.filter_pis = pis
        self.filter_log_lik = log_lik_total

    def load_external(self, ns, rs, xs, ys, params: object, signature: tuple | None) -> None:
        """User loaded an external CSV: store it with the live GUI params."""
        self.data = (ns, rs, xs, ys, None)
        self.params = params
        self.params_signature = signature
        self.u = None  # loaded data carries no input sequence (recomputed on filter)
        self._clear_filter()  # any previous filter results belong to the old data
