#!/usr/bin/env python3
"""
prg/classes/NoiseCovariance.py
==============================
Correlated Gaussian noise covariance Sigma_W(k) for the GSS model (eq. 7ter).

For each switching state k in {0, ..., K-1}:

    Sigma_W(k) = | Sigma_U_k   Delta_k   |   shape (q+s, q+s)
                 | Delta_k^T   Sigma_V_k  |

where:
    Sigma_U_k in R^{q x q}  — covariance of U_{n+1}  (noise on X)
    Delta_k   in R^{q x s}  — cross-covariance
    Sigma_V_k in R^{s x s}  — covariance of V_{n+1}  (noise on Y)

Sigma_W(k) must be symmetric positive definite for all k.
"""

from __future__ import annotations

import numpy as np

from prg.utils.exceptions import CovarianceError, ParamError
from prg.utils.matrix_checks import CovarianceMatrix

__all__ = ["GSSNoiseCovariance"]


class GSSNoiseCovariance:
    """
    Stores and provides access to the block noise covariance matrices Sigma_W(k).

    Parameters
    ----------
    K : int
        Number of switching states.  Must be >= 2.
    q : int
        Dimension of X (and of Sigma_U_k).  Must be >= 1.
    s : int
        Dimension of Y (and of Sigma_V_k).  Must be >= 1.
    Sigma_U_list : list of K arrays, each of shape (q, q)
        Covariance of the noise on X for each state.
    Delta_list : list of K arrays, each of shape (q, s)
        Cross-covariance between noise on X and noise on Y.
    Sigma_V_list : list of K arrays, each of shape (s, s)
        Covariance of the noise on Y for each state.

    Raises
    ------
    ParamError
        If any dimension is invalid or any list has the wrong length/shape.
    CovarianceError
        If Sigma_W(k) is not symmetric positive definite for some k.

    Notes
    -----
    Cholesky factors are computed once at construction and cached.
    Use ``chol_W(k)`` to retrieve L_k such that Sigma_W(k) = L_k @ L_k.T.

    Examples
    --------
    >>> import numpy as np
    >>> nc = GSSNoiseCovariance(
    ...     K=2, q=1, s=1,
    ...     Sigma_U_list=[np.array([[0.1]]), np.array([[0.2]])],
    ...     Delta_list  =[np.array([[0.05]]), np.array([[0.02]])],
    ...     Sigma_V_list=[np.array([[0.1]]), np.array([[0.15]])],
    ... )
    >>> nc.Sigma_W(0).shape
    (2, 2)
    >>> nc.chol_W(0).shape
    (2, 2)
    """

    def __init__(
        self,
        K: int,
        q: int,
        s: int,
        Sigma_U_list: list[np.ndarray],
        Delta_list: list[np.ndarray],
        Sigma_V_list: list[np.ndarray],
    ) -> None:
        if __debug__:
            self._validate_dims(K, q, s)
            self._validate_blocks(K, q, s, Sigma_U_list, Delta_list, Sigma_V_list)

        self._K = int(K)
        self._q = int(q)
        self._s = int(s)
        self._dim_z = q + s

        self._Sigma_U = [np.array(M, dtype=float) for M in Sigma_U_list]
        self._Delta = [np.array(M, dtype=float) for M in Delta_list]
        self._Sigma_V = [np.array(M, dtype=float) for M in Sigma_V_list]

        # Build full Sigma_W(k) for each state
        self._Sigma_W: list[np.ndarray] = [
            np.block(
                [
                    [self._Sigma_U[k], self._Delta[k]],
                    [self._Delta[k].T, self._Sigma_V[k]],
                ]
            )
            for k in range(self._K)
        ]

        # Validate SPD and cache Cholesky factors
        self._chol: list[np.ndarray] = []
        for k in range(self._K):
            report = CovarianceMatrix(self._Sigma_W[k]).check()
            if not report.is_valid:
                raise CovarianceError(
                    f"Sigma_W({k}) is not symmetric positive definite.\n{report}",
                    matrix_name=f"Sigma_W({k})",
                )
            # np.linalg.cholesky returns L (lower triangular), Sigma_W = L @ L.T
            self._chol.append(np.linalg.cholesky(self._Sigma_W[k]))

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
        K: int,
        q: int,
        s: int,
        Sigma_U_list: list,
        Delta_list: list,
        Sigma_V_list: list,
    ) -> None:
        expected = {
            "Sigma_U_list": (q, q),
            "Delta_list": (q, s),
            "Sigma_V_list": (s, s),
        }
        for name, lst in zip(expected, [Sigma_U_list, Delta_list, Sigma_V_list]):
            if not isinstance(lst, (list, tuple)) or len(lst) != K:
                raise ParamError(f"{name} must be a list of {K} arrays, got length {len(lst)}.")
            shape = expected[name]
            for k, arr in enumerate(lst):
                arr = np.asarray(arr)
                if arr.shape != shape:
                    raise ParamError(f"{name}[{k}] must have shape {shape}, got {arr.shape}.")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def K(self) -> int:
        return self._K

    @property
    def q(self) -> int:
        return self._q

    @property
    def s(self) -> int:
        return self._s

    @property
    def dim_z(self) -> int:
        return self._dim_z

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    def Sigma_W(self, k: int) -> np.ndarray:
        """
        Full noise covariance matrix for state k.

        Returns
        -------
        ndarray of shape (q+s, q+s), symmetric positive definite.
        """
        return self._Sigma_W[k]

    def Sigma_U(self, k: int) -> np.ndarray:
        """Upper-left block, shape (q, q): covariance of noise on X."""
        return self._Sigma_U[k]

    def Delta(self, k: int) -> np.ndarray:
        """Upper-right block, shape (q, s): cross-covariance."""
        return self._Delta[k]

    def Sigma_V(self, k: int) -> np.ndarray:
        """Lower-right block, shape (s, s): covariance of noise on Y."""
        return self._Sigma_V[k]

    # ------------------------------------------------------------------
    # Closed-form regime-conditional moments (AB / NGH-MSM family)
    # ------------------------------------------------------------------

    def M(self, k: int) -> np.ndarray:
        """
        Closed-form regime-conditional gain ``M_k = Δ_k Σ_V_k⁻¹``, shape (q, s).

        Under the AB constraint that defines the NGH-MSM family
        (``A_k = Δ_k Σ_V_k⁻¹ C_k``, ``B_k = Δ_k Σ_V_k⁻¹ D_k``), the hidden
        state is, conditionally on the regime, an affine function of the
        *current* observation:

            E[X_n | r_n = k, y_n] = M_k y_n     (constant in n).

        Σ_V(k) is symmetric positive definite (guaranteed at construction
        via the SPD check on Σ_W(k)), so the solve is well posed.
        """
        # solve(Σ_V, Δ^T) = Σ_V⁻¹ Δ^T  (s × q); transpose → Δ Σ_V⁻¹  (Σ_V symmetric)
        return np.linalg.solve(self._Sigma_V[k], self._Delta[k].T).T

    def Gamma(self, k: int) -> np.ndarray:
        """
        Closed-form regime-conditional covariance ``Γ_k`` (Schur complement),
        shape (q, q):

            Γ_k = Σ_U_k − Δ_k Σ_V_k⁻¹ Δ_k^T.

        Under the AB constraint, ``Cov[X_n | r_n = k, y_n] = Γ_k`` (constant
        in n).  ``Γ_k ⪰ 0`` since the joint noise covariance Σ_W(k) is SPD.
        """
        return self._Sigma_U[k] - self._Delta[k] @ np.linalg.solve(
            self._Sigma_V[k], self._Delta[k].T
        )

    def chol_W(self, k: int) -> np.ndarray:
        """
        Cached lower-triangular Cholesky factor L_k such that
        Sigma_W(k) = L_k @ L_k.T.

        Returns
        -------
        ndarray of shape (q+s, q+s), lower triangular.
        """
        return self._chol[k]

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def summary(self) -> None:
        """Print all covariance matrices for each state k."""

        def fmt(M: np.ndarray) -> str:
            return np.array2string(M, formatter={"float_kind": lambda x: f"{x:8.4f}"})

        print(f"=== GSSNoiseCovariance (K={self._K}, q={self._q}, s={self._s}) ===")
        for k in range(self._K):
            print(f"\n--- State k={k} ---")
            print(f"Sigma_W({k}):\n{fmt(self._Sigma_W[k])}")
            print(f"  Sigma_U({k}): {fmt(self._Sigma_U[k])}")
            print(f"  Delta({k}):   {fmt(self._Delta[k])}")
            print(f"  Sigma_V({k}): {fmt(self._Sigma_V[k])}")
            print(f"  chol_W({k}):  {fmt(self._chol[k])}")
        print("=" * 40)

    def __repr__(self) -> str:
        return f"<GSSNoiseCovariance(K={self._K}, q={self._q}, s={self._s})>"
