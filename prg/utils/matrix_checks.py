#!/usr/bin/env python3
"""
prg/utils/matrix_checks.py
==========================
Diagnostic tools for matrices used in the GSS model.

Two checkers are provided:
    CovarianceMatrix  — symmetric positive definite (SPD)
    StochasticMatrix  — row-stochastic (Markov transition)

Each exposes:
    check()   → DiagnosticReport  (structured result with per-check status)
    summary() → prints the report to the console

Pattern mirrors awesomepkf's MatrixDiagnostics for consistency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np

__all__ = [
    "CovarianceMatrix",
    "StochasticMatrix",
    "DiagnosticReport",
    "Status",
]

# ---------------------------------------------------------------------------
# Tolerances
# ---------------------------------------------------------------------------

_EPS_SYM = 1e-10  # max |M - M^T| to consider M symmetric
_EPS_STOCH = 1e-10  # max |row_sum - 1| to consider P row-stochastic


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------


class Status(Enum):
    OK = auto()
    WARNING = auto()
    FAIL = auto()

    def __str__(self) -> str:
        return {
            Status.OK: "OK",
            Status.WARNING: "WARNING",
            Status.FAIL: "FAIL",
        }[self]


# ---------------------------------------------------------------------------
# Individual check result
# ---------------------------------------------------------------------------


@dataclass
class CheckResult:
    name: str
    status: Status
    value: float | None
    threshold: float | None
    message: str

    def __str__(self) -> str:
        thr = f"  (threshold: {self.threshold})" if self.threshold is not None else ""
        val = f"  [value: {self.value:.6g}]" if self.value is not None else ""
        return f"  [{str(self.status):<7}]  {self.name}{val}{thr}\n             -> {self.message}"


# ---------------------------------------------------------------------------
# Full diagnostic report
# ---------------------------------------------------------------------------


@dataclass
class DiagnosticReport:
    matrix_type: str
    shape: tuple
    dtype: str
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def overall_status(self) -> Status:
        if any(c.status == Status.FAIL for c in self.checks):
            return Status.FAIL
        if any(c.status == Status.WARNING for c in self.checks):
            return Status.WARNING
        return Status.OK

    @property
    def is_ok(self) -> bool:
        """True only if all checks passed with no warnings."""
        return self.overall_status == Status.OK

    @property
    def is_valid(self) -> bool:
        """True if no check failed (warnings tolerated)."""
        return self.overall_status != Status.FAIL

    def __str__(self) -> str:
        lines = [
            f"--- {self.matrix_type} Diagnostic ---",
            f"Shape : {self.shape}   dtype : {self.dtype}",
            f"Overall: {self.overall_status}",
            "",
        ]
        for c in self.checks:
            lines.append(str(c))
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# CovarianceMatrix
# ---------------------------------------------------------------------------


class CovarianceMatrix:
    """
    Diagnostic for a symmetric positive definite (SPD) matrix.

    Checks (in order, short-circuiting on FAIL)
    --------------------------------------------
    1. Square         — M must be a 2D square array
    2. Finite         — no NaN or Inf
    3. Symmetric      — max |M - M^T| < _EPS_SYM
    4. PositiveDefinite — Cholesky factorisation must succeed

    Parameters
    ----------
    M : array-like
        The matrix to diagnose.  Converted to float64 internally.

    Examples
    --------
    >>> import numpy as np
    >>> report = CovarianceMatrix(np.eye(3)).check()
    >>> report.is_valid
    True
    >>> report = CovarianceMatrix(-np.eye(2)).check()
    >>> report.is_valid
    False
    """

    def __init__(self, M: np.ndarray) -> None:
        self._M = np.asarray(M, dtype=float)

    def check(self) -> DiagnosticReport:
        M = self._M
        report = DiagnosticReport(
            matrix_type="CovarianceMatrix",
            shape=tuple(M.shape),
            dtype=str(M.dtype),
        )

        # 1. Square
        if M.ndim != 2 or M.shape[0] != M.shape[1]:
            report.checks.append(
                CheckResult(
                    "Square",
                    Status.FAIL,
                    None,
                    None,
                    f"Expected a square 2D array; got shape {M.shape}.",
                )
            )
            return report
        report.checks.append(CheckResult("Square", Status.OK, None, None, "OK."))

        # 2. Finite
        if not np.all(np.isfinite(M)):
            report.checks.append(
                CheckResult(
                    "Finite",
                    Status.FAIL,
                    None,
                    None,
                    "Matrix contains NaN or Inf.",
                )
            )
            return report
        report.checks.append(CheckResult("Finite", Status.OK, None, None, "OK."))

        # 3. Symmetric
        sym_err = float(np.max(np.abs(M - M.T)))
        sym_ok = sym_err < _EPS_SYM
        report.checks.append(
            CheckResult(
                "Symmetric",
                Status.OK if sym_ok else Status.FAIL,
                sym_err,
                _EPS_SYM,
                "OK." if sym_ok else f"Max asymmetry = {sym_err:.3e}.",
            )
        )
        if not sym_ok:
            return report

        # 4. Positive definite (Cholesky)
        try:
            np.linalg.cholesky(M)
            report.checks.append(
                CheckResult(
                    "PositiveDefinite",
                    Status.OK,
                    None,
                    None,
                    "Cholesky factorisation succeeded.",
                )
            )
        except np.linalg.LinAlgError:
            min_eig = float(np.linalg.eigvalsh(M).min())
            report.checks.append(
                CheckResult(
                    "PositiveDefinite",
                    Status.FAIL,
                    min_eig,
                    0.0,
                    f"Cholesky failed.  Min eigenvalue = {min_eig:.3e}.",
                )
            )

        return report

    def summary(self) -> None:
        """Print the diagnostic report to the console."""
        print(self.check())


# ---------------------------------------------------------------------------
# StochasticMatrix
# ---------------------------------------------------------------------------


class StochasticMatrix:
    """
    Diagnostic for a row-stochastic matrix (Markov transition matrix).

    Checks (in order, short-circuiting on FAIL)
    --------------------------------------------
    1. Square      — P must be a 2D square array
    2. Finite      — no NaN or Inf
    3. NonNegative — all entries >= 0
    4. RowSumsToOne — max |row_sum - 1| < _EPS_STOCH

    Parameters
    ----------
    P : array-like
        The matrix to diagnose.  Converted to float64 internally.

    Examples
    --------
    >>> import numpy as np
    >>> P = np.array([[0.9, 0.1], [0.2, 0.8]])
    >>> StochasticMatrix(P).check().is_valid
    True
    >>> P_bad = np.array([[0.9, 0.2], [0.2, 0.8]])   # rows do not sum to 1
    >>> StochasticMatrix(P_bad).check().is_valid
    False
    """

    def __init__(self, P: np.ndarray) -> None:
        self._P = np.asarray(P, dtype=float)

    def check(self) -> DiagnosticReport:
        P = self._P
        report = DiagnosticReport(
            matrix_type="StochasticMatrix",
            shape=tuple(P.shape),
            dtype=str(P.dtype),
        )

        # 1. Square
        if P.ndim != 2 or P.shape[0] != P.shape[1]:
            report.checks.append(
                CheckResult(
                    "Square",
                    Status.FAIL,
                    None,
                    None,
                    f"Expected a square 2D array; got shape {P.shape}.",
                )
            )
            return report
        report.checks.append(CheckResult("Square", Status.OK, None, None, "OK."))

        # 2. Finite
        if not np.all(np.isfinite(P)):
            report.checks.append(
                CheckResult(
                    "Finite",
                    Status.FAIL,
                    None,
                    None,
                    "Matrix contains NaN or Inf.",
                )
            )
            return report
        report.checks.append(CheckResult("Finite", Status.OK, None, None, "OK."))

        # 3. Non-negative
        min_val = float(P.min())
        nn_ok = min_val >= 0.0
        report.checks.append(
            CheckResult(
                "NonNegative",
                Status.OK if nn_ok else Status.FAIL,
                min_val,
                0.0,
                "OK." if nn_ok else f"Min entry = {min_val:.3e} < 0.",
            )
        )
        if not nn_ok:
            return report

        # 4. Row sums to 1
        row_sums = P.sum(axis=1)
        max_err = float(np.max(np.abs(row_sums - 1.0)))
        rs_ok = max_err < _EPS_STOCH
        report.checks.append(
            CheckResult(
                "RowSumsToOne",
                Status.OK if rs_ok else Status.FAIL,
                max_err,
                _EPS_STOCH,
                "OK." if rs_ok else f"Max |row_sum - 1| = {max_err:.3e}.",
            )
        )

        return report

    def summary(self) -> None:
        """Print the diagnostic report to the console."""
        print(self.check())
