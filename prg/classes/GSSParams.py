#!/usr/bin/env python3
"""
prg/classes/GSSParams.py
========================
Aggregates all parameters of the GSS model (equations 7, 7bis, 7ter).

The model equation is:
  Z_{n+1} = F(r_{n+1}) Z_n + b(r_{n+1}) + W_{n+1}
where b(k) is the regime-dependent drift bias (zero by default).

GSSParams is the single source of truth passed to GSSSimulator (and
later to filters/smoothers).  It validates every parameter at
construction time so that downstream code can assume correctness.
"""

from __future__ import annotations

import warnings

import numpy as np

from prg.classes.FMatrix import FMatrix
from prg.classes.NoiseCovariance import GSSNoiseCovariance
from prg.utils.exceptions import CovarianceError, ParamError
from prg.utils.matrix_checks import CovarianceMatrix, StochasticMatrix

__all__ = ["GSSParams", "NGHMSMParams"]


class GSSParams:
    """
    Aggregates and validates all parameters of the GSS model.

    Parameters
    ----------
    K : int
        Number of switching states.  R_n in {0, ..., K-1}.
    q : int
        Dimension of the hidden state X.
    s : int
        Dimension of the observation Y.
    P : ndarray (K, K)
        Row-stochastic Markov transition matrix.
        P[i, j] = P(R_{n+1} = j | R_n = i).
    f_matrix : FMatrix
        Block transition matrices F(k) for each state k.
    noise_cov : GSSNoiseCovariance
        Block noise covariance matrices Sigma_W(k) for each state k.
    pi0 : ndarray (K,) or None
        Initial distribution of R_0.
        If None, the stationary distribution of P is computed and used.
    mu_z0_list : list of K ndarrays, each shape (q+s, 1)
        Initial mean of Z_0 given R_0 = k.
    Sigma_z0_list : list of K ndarrays, each shape (q+s, q+s)
        Initial covariance of Z_0 given R_0 = k.
    b_list : list of K ndarrays, each shape (q+s, 1), optional
        Regime-dependent drift bias b(k) in the transition equation
        Z_{n+1} = F(k) Z_n + b(k) + W_{n+1}.
        If None or omitted, b(k) = 0 for all k.

    Raises
    ------
    ParamError
        If dimensions are inconsistent, P is not row-stochastic, pi0
        does not sum to 1, or list lengths are wrong.
    CovarianceError
        If any Sigma_z0(k) is not symmetric positive definite.

    Class method
    ------------
    GSSParams.from_model(model)
        Convenience constructor from a BaseGSSModel instance.
    """

    def __init__(
        self,
        K: int,
        q: int,
        s: int,
        P: np.ndarray,
        f_matrix: FMatrix,
        noise_cov: GSSNoiseCovariance,
        pi0: np.ndarray | None,
        mu_z0_list: list[np.ndarray],
        Sigma_z0_list: list[np.ndarray],
        b_list: list[np.ndarray] | None = None,
    ) -> None:
        # Structural validation always runs: gating it behind ``if __debug__``
        # silently disables it under ``python -O``, letting malformed parameters
        # through. The cost is negligible — validation happens once at
        # construction, never inside the filter loop.
        self._validate_scalars(K, q, s)
        self._validate_P(P, K)
        self._validate_fmatrix(f_matrix, K, q, s)
        self._validate_noise_cov(noise_cov, K, q, s)
        self._validate_pi0(pi0, K)
        self._validate_initial_conditions(mu_z0_list, Sigma_z0_list, K, q + s)

        self._K = int(K)
        self._q = int(q)
        self._s = int(s)
        self._dim_z = q + s
        self._P = np.array(P, dtype=float)
        self._f_matrix = f_matrix
        self._noise_cov = noise_cov

        # Resolve pi0: None → stationary distribution
        if pi0 is None:
            self._pi0 = self._compute_stationary(self._P)
        else:
            self._pi0 = np.array(pi0, dtype=float)

        self._mu_z0 = [np.array(m, dtype=float) for m in mu_z0_list]
        self._Sigma_z0 = [np.array(S, dtype=float) for S in Sigma_z0_list]

        # Drift bias b(k): None → zero vector for all k
        zero = np.zeros((q + s, 1), dtype=float)
        if b_list is None:
            self._b = [zero.copy() for _ in range(K)]
        else:
            self._b = [np.array(b, dtype=float).reshape(q + s, 1) for b in b_list]

        # Cache Cholesky factors of Sigma_z0 for efficient sampling
        self._chol_z0: list[np.ndarray] = []
        for k in range(self._K):
            report = CovarianceMatrix(self._Sigma_z0[k]).check()
            if not report.is_valid:
                raise CovarianceError(
                    f"Sigma_z0({k}) is not symmetric positive definite.\n{report}",
                    matrix_name=f"Sigma_z0({k})",
                )
            self._chol_z0.append(np.linalg.cholesky(self._Sigma_z0[k]))

    # ------------------------------------------------------------------
    # Class method factory
    # ------------------------------------------------------------------

    @classmethod
    def from_model(cls, model) -> GSSParams:
        """
        Build a GSSParams from a BaseGSSModel instance.

        Parameters
        ----------
        model : BaseGSSModel
            The model whose ``get_params()`` dict is consumed.

        Returns
        -------
        GSSParams
        """
        p = model.get_params()
        f_matrix = FMatrix(
            K=p["K"],
            q=p["q"],
            s=p["s"],
            A_list=p["A_list"],
            B_list=p["B_list"],
            C_list=p["C_list"],
            D_list=p["D_list"],
        )
        noise_cov = GSSNoiseCovariance(
            K=p["K"],
            q=p["q"],
            s=p["s"],
            Sigma_U_list=p["Sigma_U_list"],
            Delta_list=p["Delta_list"],
            Sigma_V_list=p["Sigma_V_list"],
        )
        return cls(
            K=p["K"],
            q=p["q"],
            s=p["s"],
            P=p["P"],
            f_matrix=f_matrix,
            noise_cov=noise_cov,
            pi0=p["pi0"],
            mu_z0_list=p["mu_z0_list"],
            Sigma_z0_list=p["Sigma_z0_list"],
            b_list=p.get("b_list", None),
        )

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_scalars(K: int, q: int, s: int) -> None:
        if not (isinstance(K, int) and K >= 2):
            raise ParamError(f"K must be an integer >= 2, got {K!r}.")
        if not (isinstance(q, int) and q >= 1):
            raise ParamError(f"q must be an integer >= 1, got {q!r}.")
        if not (isinstance(s, int) and s >= 1):
            raise ParamError(f"s must be an integer >= 1, got {s!r}.")

    @staticmethod
    def _validate_P(P: np.ndarray, K: int) -> None:
        P = np.asarray(P)
        if P.shape != (K, K):
            raise ParamError(f"P must have shape ({K}, {K}), got {P.shape}.")
        report = StochasticMatrix(P).check()
        if not report.is_valid:
            raise ParamError(f"Transition matrix P is not row-stochastic.\n{report}")

    @staticmethod
    def _validate_fmatrix(f_matrix: FMatrix, K: int, q: int, s: int) -> None:
        if not isinstance(f_matrix, FMatrix):
            raise ParamError("f_matrix must be an FMatrix instance.")
        if f_matrix.K != K or f_matrix.q != q or f_matrix.s != s:
            raise ParamError(
                f"f_matrix dimensions (K={f_matrix.K}, q={f_matrix.q}, "
                f"s={f_matrix.s}) do not match (K={K}, q={q}, s={s})."
            )

    @staticmethod
    def _validate_noise_cov(noise_cov: GSSNoiseCovariance, K: int, q: int, s: int) -> None:
        if not isinstance(noise_cov, GSSNoiseCovariance):
            raise ParamError("noise_cov must be a GSSNoiseCovariance instance.")
        if noise_cov.K != K or noise_cov.q != q or noise_cov.s != s:
            raise ParamError(
                f"noise_cov dimensions (K={noise_cov.K}, q={noise_cov.q}, "
                f"s={noise_cov.s}) do not match (K={K}, q={q}, s={s})."
            )

    @staticmethod
    def _validate_pi0(pi0: np.ndarray | None, K: int) -> None:
        if pi0 is None:
            return
        pi0 = np.asarray(pi0, dtype=float)
        if pi0.shape != (K,):
            raise ParamError(f"pi0 must have shape ({K},), got {pi0.shape}.")
        if np.any(pi0 < 0.0):
            raise ParamError("pi0 must have non-negative entries.")
        total = float(pi0.sum())
        if abs(total - 1.0) > 1e-10:
            raise ParamError(f"pi0 must sum to 1, got sum = {total:.6f}.")

    @staticmethod
    def _validate_initial_conditions(
        mu_z0_list: list, Sigma_z0_list: list, K: int, dim_z: int
    ) -> None:
        for name, lst, shape in [
            ("mu_z0_list", mu_z0_list, (dim_z, 1)),
            ("Sigma_z0_list", Sigma_z0_list, (dim_z, dim_z)),
        ]:
            if not isinstance(lst, (list, tuple)) or len(lst) != K:
                raise ParamError(f"{name} must be a list of {K} arrays, got length {len(lst)}.")
            for k, arr in enumerate(lst):
                arr = np.asarray(arr)
                if arr.shape != shape:
                    raise ParamError(f"{name}[{k}] must have shape {shape}, got {arr.shape}.")

    # ------------------------------------------------------------------
    # Stationary distribution
    # ------------------------------------------------------------------

    # Tolerance for "eigenvalue ≈ 1" and sign-consistency checks below.
    _STATIONARY_TOL = 1e-8

    @staticmethod
    def _compute_stationary(P: np.ndarray) -> np.ndarray:
        """
        Compute the stationary distribution pi of P.

        pi is the left eigenvector of P for eigenvalue 1, i.e. the
        right eigenvector of P.T for eigenvalue 1.

        A *unique* stationary distribution exists only when the eigenvalue 1
        is simple (irreducible chain).  This method emits a ``RuntimeWarning``
        when that assumption looks violated, so that a silently-arbitrary
        choice cannot propagate unnoticed into filtering results:

        * **non-unique** — several eigenvalues cluster at 1 (reducible chain);
        * **mixed signs** — the selected eigenvector has both signs (taking
          ``|·|`` would mask a non-distribution vector);
        * **degenerate** — the eigenvector sums to ~0 (fallback: uniform).

        Returns
        -------
        ndarray of shape (K,), non-negative, summing to 1.
        """
        K = P.shape[0]
        eigenvalues, eigenvectors = np.linalg.eig(P.T)

        dist_to_one = np.abs(eigenvalues - 1.0)
        idx = int(np.argmin(dist_to_one))

        # Guard 1 — uniqueness: a simple eigenvalue 1 ⇔ unique stationary law.
        n_near_one = int(np.sum(dist_to_one < GSSParams._STATIONARY_TOL))
        if n_near_one > 1:
            warnings.warn(
                f"Transition matrix P has {n_near_one} eigenvalues ≈ 1: the "
                f"Markov chain is reducible and its stationary distribution is "
                f"not unique. Returning one admissible distribution; "
                f"downstream results may depend on this arbitrary choice.",
                RuntimeWarning,
                stacklevel=2,
            )

        pi = eigenvectors[:, idx].real

        # Guard 2 — sign consistency: a genuine stationary eigenvector has a
        # single sign; mixed signs mean |·| below would hide an invalid vector.
        scale = float(np.max(np.abs(pi)))
        if scale > 0.0:
            tol = GSSParams._STATIONARY_TOL * scale
            if np.any(pi > tol) and np.any(pi < -tol):
                warnings.warn(
                    "Stationary-distribution eigenvector has mixed signs; "
                    "the |·|-and-renormalise result may not be a valid "
                    "stationary distribution of P. Check that P is a proper, "
                    "irreducible row-stochastic transition matrix.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        pi = np.abs(pi)  # eigenvector sign is arbitrary
        total = float(pi.sum())

        # Guard 3 — degenerate normalisation.
        if total <= 0.0 or not np.isfinite(total):
            warnings.warn(
                "Stationary-distribution eigenvector summed to a non-positive "
                "value; falling back to the uniform distribution.",
                RuntimeWarning,
                stacklevel=2,
            )
            return np.full(K, 1.0 / K)

        return pi / total

    def stationary_distribution(self) -> np.ndarray:
        """
        Return the stationary distribution of the Markov chain.

        Returns
        -------
        ndarray of shape (K,)
        """
        return self._compute_stationary(self._P)

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

    @property
    def P(self) -> np.ndarray:
        """Row-stochastic transition matrix, shape (K, K)."""
        return self._P

    @property
    def pi0(self) -> np.ndarray:
        """Initial distribution of R_0, shape (K,)."""
        return self._pi0

    @property
    def f_matrix(self) -> FMatrix:
        return self._f_matrix

    @property
    def noise_cov(self) -> GSSNoiseCovariance:
        return self._noise_cov

    def b(self, k: int) -> np.ndarray:
        """Drift bias for regime k, shape (q+s, 1).  Zero if not set."""
        return self._b[k]

    def mu_z0(self, k: int) -> np.ndarray:
        """Initial mean of Z_0 given R_0 = k, shape (q+s, 1)."""
        return self._mu_z0[k]

    def Sigma_z0(self, k: int) -> np.ndarray:
        """Initial covariance of Z_0 given R_0 = k, shape (q+s, q+s)."""
        return self._Sigma_z0[k]

    def chol_z0(self, k: int) -> np.ndarray:
        """
        Cached lower-triangular Cholesky factor L_k such that
        Sigma_z0(k) = L_k @ L_k.T, shape (q+s, q+s).
        """
        return self._chol_z0[k]

    # ------------------------------------------------------------------
    # NGH-MSM validity (corrected CNS of Proposition 2)
    # ------------------------------------------------------------------

    def check_ngh_msm(
        self,
        *,
        tol: float = 1e-6,
        raise_on_fail: bool = True,
    ) -> list[str]:
        """
        Validate that this model is a valid NGH-MSM (corrected CNS, Prop. 2).

        Checks the AB / (H5) constraint together with the structural
        hypotheses (C_k ≠ 0, D_k invertible, Σ_V_k ≻ 0,
        Γ_k ⪰ 0). See :func:`prg.utils.h5_constraint.validate_ngh_msm`.

        Parameters
        ----------
        tol : float
            Tolerance on the relative pairwise (H5) residual.
        raise_on_fail : bool
            If True (default) and the model violates at least one condition,
            raise :class:`~prg.utils.exceptions.ParamError`. If False, the
            list of issues is returned for the caller to handle (e.g. GUI).

        Returns
        -------
        list[str]
            Empty iff the model is a valid NGH-MSM; otherwise one string per
            violated condition.

        Raises
        ------
        ParamError
            If ``raise_on_fail`` and the model is not a valid NGH-MSM.

        Notes
        -----
        This validity is intentionally *not* enforced at construction: the same
        class also represents non-(H5) models used by ``mode="imm_general"``.
        """
        from prg.utils.h5_constraint import validate_ngh_msm

        issues = validate_ngh_msm(self, tol=tol)
        if issues and raise_on_fail:
            raise ParamError(
                "Model is not a valid NGH-MSM (corrected CNS of Prop. 2):\n  - "
                + "\n  - ".join(issues)
            )
        return issues

    def is_ngh_msm(self, *, tol: float = 1e-6) -> bool:
        """``True`` iff this model is a valid NGH-MSM (cf. :meth:`check_ngh_msm`)."""
        return not self.check_ngh_msm(tol=tol, raise_on_fail=False)

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def summary(self) -> None:
        """Print a human-readable summary of all parameters."""

        def fmt(M: np.ndarray) -> str:
            return np.array2string(M, formatter={"float_kind": lambda x: f"{x:8.4f}"})

        print("=" * 50)
        print(f"{type(self).__name__}  (K={self._K}, q={self._q}, s={self._s})")
        print("=" * 50)
        print(f"\nTransition matrix P:\n{fmt(self._P)}")
        print(f"\nInitial distribution pi0: {self._pi0}")
        self._f_matrix.summary()
        self._noise_cov.summary()
        print("\nInitial conditions:")
        for k in range(self._K):
            print(f"  k={k}  mu_z0: {self._mu_z0[k].ravel()}   Sigma_z0:\n{fmt(self._Sigma_z0[k])}")
        print("\nDrift bias b(k):")
        for k in range(self._K):
            print(f"  k={k}  b: {self._b[k].ravel()}")
        print("=" * 50)

    def __repr__(self) -> str:
        return (
            f"<GSSParams(K={self._K}, q={self._q}, s={self._s}, "
            f"pi0={'stationary' if self._pi0 is None else 'given'})>"
        )


class NGHMSMParams(GSSParams):
    """
    A :class:`GSSParams` restricted to the NGH-MSM (AB) family.

    Holding an ``NGHMSMParams`` is a *type-level guarantee* that the model
    satisfies the corrected CNS of Proposition 2: the AB constraint
    ``A_k = Δ_k Σ_V_k⁻¹ C_k``, ``B_k = Δ_k Σ_V_k⁻¹ D_k`` together with the
    structural hypotheses (C_k ≠ 0, D_k invertible, Σ_V_k ≻ 0,
    Γ_k ⪰ 0).  Such a model admits the exact fast filter (``mode="h5_exact"``).

    Two ways to construct it:

    * :meth:`from_free_blocks` — pass only the *free* blocks
      (C, D, Σ_U, Δ, Σ_V); A, B are **derived** via the AB constraint, so a
      non-AB instance is unrepresentable through this entry point.  This is the
      recommended path for presets, the paper models, and the GUI.
    * ``NGHMSMParams(...)`` — same signature as the base class (an already
      assembled ``FMatrix`` carrying A, B); the CNS is **validated** and a
      :class:`~prg.utils.exceptions.ParamError` is raised on any violation,
      so hand-built blocks that are off the AB manifold are rejected.

    Notes
    -----
    * The CNS / (H5) validity is independent of the drift bias ``b(k)``, which
      is passed through unconstrained (beyond the base shape checks).
    * Validity is intentionally *not* imposed on the base :class:`GSSParams`,
      which also represents general (non-(H5)) models handled by
      ``mode="imm_general"``.  ``NGHMSMParams`` is a *restriction* of the base
      type (it adds guarantees, never changes behaviour), so it substitutes
      for a ``GSSParams`` everywhere.
    """

    def __init__(
        self,
        K: int,
        q: int,
        s: int,
        P: np.ndarray,
        f_matrix: FMatrix,
        noise_cov: GSSNoiseCovariance,
        pi0: np.ndarray | None,
        mu_z0_list: list[np.ndarray],
        Sigma_z0_list: list[np.ndarray],
        b_list: list[np.ndarray] | None = None,
        *,
        tol: float = 1e-6,
    ) -> None:
        super().__init__(K, q, s, P, f_matrix, noise_cov, pi0, mu_z0_list, Sigma_z0_list, b_list)
        # Validate the corrected CNS; reuses all of validate_ngh_msm's logic.
        from prg.utils.h5_constraint import validate_ngh_msm

        issues = validate_ngh_msm(self, tol=tol)
        if issues:
            raise ParamError(
                "NGHMSMParams requires a valid NGH-MSM (corrected CNS of Prop. 2):\n  - "
                + "\n  - ".join(issues)
            )

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def from_free_blocks(
        cls,
        K: int,
        q: int,
        s: int,
        P: np.ndarray,
        C_list: list[np.ndarray],
        D_list: list[np.ndarray],
        Sigma_U_list: list[np.ndarray],
        Delta_list: list[np.ndarray],
        Sigma_V_list: list[np.ndarray],
        pi0: np.ndarray | None,
        mu_z0_list: list[np.ndarray],
        Sigma_z0_list: list[np.ndarray],
        b_list: list[np.ndarray] | None = None,
        *,
        tol: float = 1e-6,
    ) -> NGHMSMParams:
        """
        Build an NGH-MSM from its *free* blocks, **deriving** A, B.

        ``A_k, B_k`` are computed as ``Δ_k Σ_V_k⁻¹ C_k`` and ``Δ_k Σ_V_k⁻¹ D_k``
        (``compute_AB``), so the AB constraint holds by construction — a non-AB
        model is unrepresentable through this path.  The remaining structural
        conditions (C_k ≠ 0, D_k invertible, Σ_V_k ≻ 0, Γ_k ⪰ 0)
        are checked by ``__init__`` and raise :class:`ParamError` if violated.
        """
        from prg.classes.FMatrix import FMatrix as _FMatrix
        from prg.classes.NoiseCovariance import GSSNoiseCovariance as _NoiseCov
        from prg.utils.h5_constraint import compute_AB

        A_list, B_list = [], []
        for C, D, Dt, SV in zip(C_list, D_list, Delta_list, Sigma_V_list, strict=True):
            A, B = compute_AB(
                np.asarray(C, dtype=float),
                np.asarray(D, dtype=float),
                np.asarray(Dt, dtype=float),
                np.asarray(SV, dtype=float),
            )
            A_list.append(A)
            B_list.append(B)
        f_matrix = _FMatrix(
            K=K, q=q, s=s, A_list=A_list, B_list=B_list, C_list=C_list, D_list=D_list
        )
        noise_cov = _NoiseCov(
            K=K,
            q=q,
            s=s,
            Sigma_U_list=Sigma_U_list,
            Delta_list=Delta_list,
            Sigma_V_list=Sigma_V_list,
        )
        return cls(
            K=K,
            q=q,
            s=s,
            P=P,
            f_matrix=f_matrix,
            noise_cov=noise_cov,
            pi0=pi0,
            mu_z0_list=mu_z0_list,
            Sigma_z0_list=Sigma_z0_list,
            b_list=b_list,
            tol=tol,
        )

    @classmethod
    def from_model(cls, model) -> NGHMSMParams:
        """
        Build a validated NGH-MSM from a model, **deriving** A, B from its free
        blocks (any model-supplied A, B are ignored — for an NGH-MSM model they
        are redundant with the derivation).
        """
        p = model.get_params()
        return cls.from_free_blocks(
            K=p["K"],
            q=p["q"],
            s=p["s"],
            P=p["P"],
            C_list=p["C_list"],
            D_list=p["D_list"],
            Sigma_U_list=p["Sigma_U_list"],
            Delta_list=p["Delta_list"],
            Sigma_V_list=p["Sigma_V_list"],
            pi0=p["pi0"],
            mu_z0_list=p["mu_z0_list"],
            Sigma_z0_list=p["Sigma_z0_list"],
            b_list=p.get("b_list", None),
        )

    # ------------------------------------------------------------------
    # Closed-form regime-conditional moments (discoverable on the type)
    # ------------------------------------------------------------------

    def M(self, k: int) -> np.ndarray:
        """Closed-form gain ``M_k = Δ_k Σ_V_k⁻¹`` (delegates to NoiseCovariance)."""
        return self._noise_cov.M(k)

    def Gamma(self, k: int) -> np.ndarray:
        """Closed-form covariance ``Γ_k = Σ_U_k − Δ_k Σ_V_k⁻¹ Δ_k^T``."""
        return self._noise_cov.Gamma(k)

    def __repr__(self) -> str:
        return f"<NGHMSMParams(K={self._K}, q={self._q}, s={self._s})>"
