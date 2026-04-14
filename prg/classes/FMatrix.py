#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prg/classes/FMatrix.py
======================
Block-structured transition matrix F(k) for the GSS model (equation 7).

For each switching state k in {0, ..., K-1}:

    F(k) = | A_k  B_k |   shape (q+s, q+s)
            | C_k  D_k |

where A_k in R^{q x q}, B_k in R^{q x s}, C_k in R^{s x q}, D_k in R^{s x s}.
"""

from __future__ import annotations

import numpy as np

from prg.utils.exceptions import ParamError

__all__ = ["FMatrix"]


class FMatrix:
    """
    Stores and provides access to the block transition matrices F(k).

    Parameters
    ----------
    K : int
        Number of switching states.  Must be >= 2.
    q : int
        Dimension of the hidden state X.  Must be >= 1.
    s : int
        Dimension of the observation Y.  Must be >= 1.
    A_list : list of K arrays, each of shape (q, q)
    B_list : list of K arrays, each of shape (q, s)
    C_list : list of K arrays, each of shape (s, q)
    D_list : list of K arrays, each of shape (s, s)

    Raises
    ------
    ParamError
        If any dimension is invalid, any list has the wrong length,
        or any array has the wrong shape.

    Examples
    --------
    >>> import numpy as np
    >>> fm = FMatrix(
    ...     K=2, q=1, s=1,
    ...     A_list=[np.array([[0.8]]), np.array([[0.5]])],
    ...     B_list=[np.array([[0.1]]), np.array([[0.3]])],
    ...     C_list=[np.array([[0.2]]), np.array([[0.1]])],
    ...     D_list=[np.array([[0.7]]), np.array([[0.6]])],
    ... )
    >>> fm.F(0).shape
    (2, 2)
    >>> fm.A(1)
    array([[0.5]])
    """

    def __init__(
        self,
        K: int,
        q: int,
        s: int,
        A_list: list[np.ndarray],
        B_list: list[np.ndarray],
        C_list: list[np.ndarray],
        D_list: list[np.ndarray],
    ) -> None:
        if __debug__:
            self._validate_dims(K, q, s)
            self._validate_blocks(K, q, s, A_list, B_list, C_list, D_list)

        self._K = int(K)
        self._q = int(q)
        self._s = int(s)
        self._dim_z = q + s

        # Store as float64, indexed [k]
        self._A = [np.array(A, dtype=float) for A in A_list]
        self._B = [np.array(B, dtype=float) for B in B_list]
        self._C = [np.array(C, dtype=float) for C in C_list]
        self._D = [np.array(D, dtype=float) for D in D_list]

        # Pre-build full F(k) matrices and cache them
        self._F: list[np.ndarray] = [
            np.block([[self._A[k], self._B[k]],
                      [self._C[k], self._D[k]]])
            for k in range(self._K)
        ]

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_dims(K: int, q: int, s: int) -> None:
        if not (isinstance(K, int) and K >= 2):
            raise ParamError(f"K must be an integer >= 2, got {K!r}.")
        if not (isinstance(q, int) and q >= 1):
            raise ParamError(f"q must be an integer >= 1, got {q!r}.")
        if not (isinstance(s, int) and s >= 1):
            raise ParamError(f"s must be an integer >= 1, got {s!r}.")

    @staticmethod
    def _validate_blocks(
        K: int, q: int, s: int,
        A_list: list, B_list: list, C_list: list, D_list: list,
    ) -> None:
        expected = {"A_list": (q, q), "B_list": (q, s),
                    "C_list": (s, q), "D_list": (s, s)}
        for name, lst in zip(expected, [A_list, B_list, C_list, D_list]):
            if not isinstance(lst, (list, tuple)) or len(lst) != K:
                raise ParamError(
                    f"{name} must be a list of {K} arrays, got length {len(lst)}."
                )
            shape = expected[name]
            for k, arr in enumerate(lst):
                arr = np.asarray(arr)
                if arr.shape != shape:
                    raise ParamError(
                        f"{name}[{k}] must have shape {shape}, "
                        f"got {arr.shape}."
                    )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def K(self) -> int:
        """Number of switching states."""
        return self._K

    @property
    def q(self) -> int:
        """Dimension of X."""
        return self._q

    @property
    def s(self) -> int:
        """Dimension of Y."""
        return self._s

    @property
    def dim_z(self) -> int:
        """Dimension of Z = [X; Y], equals q + s."""
        return self._dim_z

    def F(self, k: int) -> np.ndarray:
        """
        Full transition matrix for state k.

        Returns
        -------
        ndarray of shape (q+s, q+s)
        """
        return self._F[k]

    def A(self, k: int) -> np.ndarray:
        """Upper-left block A_k, shape (q, q)."""
        return self._A[k]

    def B(self, k: int) -> np.ndarray:
        """Upper-right block B_k, shape (q, s)."""
        return self._B[k]

    def C(self, k: int) -> np.ndarray:
        """Lower-left block C_k, shape (s, q)."""
        return self._C[k]

    def D(self, k: int) -> np.ndarray:
        """Lower-right block D_k, shape (s, s)."""
        return self._D[k]

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def summary(self) -> None:
        """Print all block matrices for each state k."""

        def fmt(M: np.ndarray) -> str:
            return np.array2string(M, formatter={"float_kind": lambda x: f"{x:8.4f}"})

        print(f"=== FMatrix (K={self._K}, q={self._q}, s={self._s}) ===")
        for k in range(self._K):
            print(f"\n--- State k={k} ---")
            print(f"F({k}):\n{fmt(self._F[k])}")
            print(f"  A({k}): {fmt(self._A[k])}")
            print(f"  B({k}): {fmt(self._B[k])}")
            print(f"  C({k}): {fmt(self._C[k])}")
            print(f"  D({k}): {fmt(self._D[k])}")
        print("=" * 40)

    def __repr__(self) -> str:
        return f"<FMatrix(K={self._K}, q={self._q}, s={self._s})>"
