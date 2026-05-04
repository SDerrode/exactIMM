#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prg/utils/exceptions.py
-----------------------
Centralised exception hierarchy for the exactIMM project.

Tree
----
GSSError
├── ParamError
├── NumericalError
│   └── CovarianceError
└── SimulationError
"""

__all__ = [
    "GSSError",
    "ParamError",
    "NumericalError",
    "CovarianceError",
    "SimulationError",
]


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------


class GSSError(Exception):
    """Root of all exactIMM exceptions."""

    def __repr__(self) -> str:
        msg = self.args[0] if self.args else ""
        return f"{self.__class__.__name__}({msg!r})"


# ---------------------------------------------------------------------------
# Parameter errors
# ---------------------------------------------------------------------------


class ParamError(GSSError):
    """
    Invalid parameter supplied to a class or method.

    Raised for example if ``N`` is not a strictly positive integer,
    if ``K`` < 2, or if a list of matrices has the wrong length.
    """


# ---------------------------------------------------------------------------
# Numerical errors
# ---------------------------------------------------------------------------


class NumericalError(GSSError):
    """
    Generic numerical error related to matrix computations.

    Parameters
    ----------
    message : str
        Human-readable description of the error.
    matrix_name : str, optional
        Name of the matrix concerned (default ``""``).
    step : int, optional
        Time step index where the error occurred (default ``-1``).

    Attributes
    ----------
    matrix_name : str
    step : int  (-1 means unknown)
    """

    def __init__(self, message: str, matrix_name: str = "", step: int = -1) -> None:
        super().__init__(message)
        self.matrix_name = matrix_name
        self.step = step

    def __str__(self) -> str:
        parts = [self.args[0] if self.args else ""]
        if self.step != -1:
            parts.append(f"step={self.step}")
        if self.matrix_name:
            parts.append(f"matrix={self.matrix_name!r}")
        return " | ".join(parts)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"step={self.step}, "
            f"matrix={self.matrix_name!r}, "
            f"msg={self.args[0]!r})"
        )


class CovarianceError(NumericalError):
    """
    Invalid covariance matrix.

    Raised when a matrix is not symmetric positive definite,
    or when a Cholesky factorisation fails.
    """


# ---------------------------------------------------------------------------
# Simulation errors
# ---------------------------------------------------------------------------


class SimulationError(GSSError):
    """
    Error occurring during simulation execution.

    Raised for example if the RNG state is inconsistent or if
    a simulation step produces non-finite values.
    """
