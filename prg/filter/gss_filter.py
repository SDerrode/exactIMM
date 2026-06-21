#!/usr/bin/env python3
"""
prg/filter/gss_filter.py
========================
IMM-style optimal filter for the GSS model with two operating modes.

Modes
-----

``mode="h5_exact"`` (default) — exact IMM recursion under hypothesis (H5)
    Under (H5) the mode-conditional state posterior collapses on the
    current regime and observation alone:

        p(X_{n+1} | r_{n+1}, y_{1:n+1}) = p(X_{n+1} | r_{n+1}, y_{n+1})

    Consequently the per-regime moments

        µ(k) := E[Z_n | r_n = k]          (constant in n under stationarity)
        P(k) := Var[Z_n | r_n = k]        (constant in n under stationarity)

    are **observation-free** and depend only on the model parameters
    and on the **stationary distribution π_∞** of the regime chain.
    They are pre-computed once at filter construction. The recursion
    then reduces to:

      Step (I)   regime moments = pre-computed constants;
      Step (II)  mode-probability update via pair-conditional likelihoods;
      Step (III) mode-conditional Kalman update against the constant
                 moments of regime k;
      Step (IV)  combination using π_{n+1}(k).

    This mode is mathematically exact under (H5), the algebraic constraint

        Δ(k)ᵀ A(k)ᵀ + Σ_V(k) B(k)ᵀ = P(k) M(k)⁻¹ W(k),

    tying together all 7 parameter matrices (A, B, C, D, Σ_U, Σ_V, Δ) of
    each regime. (H5) is **not** equivalent to B(k) = 0 — A(k) and B(k)
    are determined together from (C, D, Δ, Σ_V) by the closed-form
    "AB constraint" A = Δ Σ_V⁻¹ C, B = Δ Σ_V⁻¹ D;
    see :mod:`prg.utils.h5_constraint`.
    When (H5) is violated a ``RuntimeWarning`` is emitted at construction
    (relative residual above ``H5_TOL``).

``mode="imm_general"`` — IMM-style recursion, no (H5) required
    The full F(k) = [[A_k, B_k], [C_k, D_k]] is used. Per-regime moments
    µ_n(r), P_n(r) are propagated **at each step** following the recursion
    of the companion paper CS_FinaleBis eqs (17ter), (17quater), (13')–(15),
    (18), (21')–(22). This matches the behaviour of ``exactIMM ≤ v0.9.0``
    and is appropriate for any GSS model, in particular non-(H5) models.

    **IMM approximation note.** The paper's recursion (17ter)/(17quater)
    uses the *prior* marginal p(r_n) to form the time-reversed transition
    p(r_n | r_{n+1}).  This implementation instead uses the *filtered*
    posterior π_n = p(r_n | y_{1:n}) — i.e.

        p(r_n=j | r_{n+1}=k, y_{1:n})  ∝  π_n(j) · P(j, k)

    which is the standard IMM heuristic and more responsive to the data.
    Concretely this means µ_n(r), P_n(r) are *prior* (observation-free)
    quantities propagated via the IMM mixing weights rather than the
    unconditional marginals of the paper.  The difference is negligible
    when the chain is near-stationary but can be significant in transient
    regimes.  For the exact (H5)-optimal filter use ``mode="h5_exact"``.

Joseph form (optional, h5_exact only)
-------------------------------------
The mode-conditional posterior covariance admits the standard short form

    P_{n+1|n+1}^{(k)} = Σ_XX(k) - K_k S_k K_k^T,

or the numerically-preferable Joseph form (paper App. E)

    P_{n+1|n+1}^{(k)} = (I - K_k H_k) Σ_XX(k) (I - K_k H_k)^T
                       + K_k R_k K_k^T,

with H_k := Σ_YX(k) Σ_XX(k)^{-1} and R_k := S_k - H_k Σ_XX(k) H_k^T.
Both forms give the same constant covariance under stationarity; Joseph
preserves symmetry / PSD under finite-precision arithmetic. The Joseph
form is selected by passing ``joseph=True`` to the constructor (default
``joseph=False``).
"""

from __future__ import annotations

import csv
import dataclasses
import logging
import pathlib
import time
import warnings

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

from prg.classes.GSSParams import GSSParams
from prg.classes.GSSSimulator import GSSSimulator

__all__ = ["GSSFilter", "FilterResult"]

FILTER_MODES = ("h5_exact", "imm_general")
# Relative residual ‖F(k)‖_F / max(‖Z(k)‖_F, 1) of the (H5) constraint
# F(k) = Z(k) − P(k) M(k)⁻¹ W(k) below this → (H5) holds. 1e-6 is the
# practical floor for float64 matrix arithmetic.
H5_TOL = 1e-6

# log(2π), used by the fast Gaussian log-pdf (mirrors scipy's _LOG_2PI).
_LOG_2PI = float(np.log(2.0 * np.pi))

logger = logging.getLogger("exactIMM.filter")


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class FilterResult:
    """
    Filtered estimates returned by :meth:`GSSFilter.step` at time step n.

    Attributes
    ----------
    n : int
        Time index (0-based).
    E_x : ndarray (q, 1)
        E[X_n | y_{1:n}]
    E_xx : ndarray (q, q)
        E[X_n X_n^T | y_{1:n}]   (uncentred second moment)
    pi : ndarray (K,)
        p(r_n | y_{1:n})
    innovation : ndarray (s, 1)
        ν_n = y_n − ŷ_{n|n−1}  (prediction residual).
        For n=0 the prior predictive mean Σ_r π_0(r) μ_Y(r) is used.
    log_lik : float
        Incremental log-likelihood log p(y_n | y_{1:n-1}) (or log p(y_1) for n=0).
    """

    n: int
    E_x: np.ndarray  # (q, 1)
    E_xx: np.ndarray  # (q, q)
    pi: np.ndarray  # (K,)
    innovation: np.ndarray  # (s, 1)
    log_lik: float = 0.0

    @property
    def Var_x(self) -> np.ndarray:
        """Posterior variance Var[X_n | y_{1:n}] = E[XX^T] − E[X]E[X]^T."""
        return self.E_xx - self.E_x @ self.E_x.T


# ---------------------------------------------------------------------------
# Filter
# ---------------------------------------------------------------------------


class GSSFilter:
    """
    IMM-style optimal filter for GSS models.

    Parameters
    ----------
    params : GSSParams
        Validated model parameters.
    joseph : bool, default False
        (Only used when ``mode="h5_exact"``.) If True, the mode-conditional
        posterior covariance uses the Joseph form (paper App. E) instead of
        the short form. Mathematically equivalent under stationarity;
        numerically more stable.
    mode : {"imm_general", "h5_exact"}, default "imm_general"
        Filter recursion variant.

        * ``"imm_general"`` (default) — IMM recursion with per-step
          moment propagation; no (H5) requirement. Matches
          ``exactIMM ≤ v0.9.0`` and is the correct choice for any GSS
          model, in particular those that do not satisfy (H5).
        * ``"h5_exact"`` — exact IMM under hypothesis (H5), with
          stationary pre-computed regime moments. Requires (H5), i.e.
          the (H5) algebraic constraint holds for every regime k;
          emits a ``RuntimeWarning`` at construction when the relative
          residual exceeds ``H5_TOL``. To enforce (H5) automatically,
          use :func:`prg.utils.h5_constraint.apply_AB_constraint`,
          which recomputes A(k), B(k) from (C, D, Δ, Σ_V) via the
          closed form A = Δ Σ_V⁻¹ C, B = Δ Σ_V⁻¹ D.

    Examples
    --------
    Step-by-step, (H5)-exact::

        filt = GSSFilter(params)                    # default mode, short form
        for y in observations:
            res = filt.step(y)

    General IMM for a non-(H5) model::

        filt = GSSFilter(params, mode="imm_general")
        sim_path, df = filt.run(N=1000, seed=42)
    """

    def __init__(
        self,
        params: GSSParams,
        joseph: bool = False,
        mode: str = "imm_general",
    ) -> None:
        if mode not in FILTER_MODES:
            raise ValueError(f"Unknown mode {mode!r}. Expected one of: {FILTER_MODES}.")
        self._p = params
        self._joseph = bool(joseph)
        self._mode = mode

        if mode == "h5_exact":
            # An NGHMSMParams already guarantees the (H5) / CNS precondition by
            # its type, so the residual recheck is redundant; only re-check
            # (and possibly warn) when handed a base GSSParams.
            from prg.classes.GSSParams import NGHMSMParams

            if not isinstance(params, NGHMSMParams):
                self._check_h5()
            self._precompute()  # calls _precompute_stationary() internally
        else:
            # imm_general: no (H5) pre-computation; joseph flag is ignored
            if self._joseph:
                logger.warning(
                    "joseph=True has no effect in mode='imm_general' "
                    "(the per-step posterior covariance is already PSD-floored)."
                )
            # Compute stationary moments so the filter can start at stationarity
            self._precompute_stationary()
        self._reset_state()

    # ------------------------------------------------------------------
    # (H5) check
    # ------------------------------------------------------------------

    def _check_h5(self) -> None:
        """Emit a RuntimeWarning if (H5) is violated.

        (H5) — the Markovianity of (R, Y) — requires, for **every** ordered
        regime pair (j, k), that the regression of X_{n+1} on (Y_n, Y_{n+1})
        given (r_n=j, r_{n+1}=k) not load on Y_n. We evaluate the relative
        pairwise residual

            r(j, k) = ‖β₁(j, k)‖_F / ‖[β₁ β₂](j, k)‖_F

        over all K² pairs and warn if  max_{j,k} r(j, k) > H5_TOL. (The
        same-regime algebraic residual alone — compute_h5_residual — is
        necessary but **not** sufficient.)
        """
        from prg.utils.h5_constraint import h5_residual_max

        max_rel, (worst_j, worst_k) = h5_residual_max(self._p, relative=True)

        if not np.isfinite(max_rel):
            warnings.warn(
                f"mode='h5_exact': could not evaluate the (H5) residual for "
                f"regime pair (j={worst_j}, k={worst_k}) — the conditional "
                f"(Y_n, Y_{{n+1}}) covariance is singular. The filter may be "
                f"biased. Use mode='imm_general' if in doubt.",
                RuntimeWarning,
                stacklevel=3,
            )
            return

        if max_rel > H5_TOL:
            warnings.warn(
                f"mode='h5_exact' assumes (H5) — the Markovianity of (R, Y), "
                f"requiring that for every regime pair the predictive law of "
                f"Y_{{n+1}} not depend on Y_n through X_n — but the model has "
                f"max relative pairwise residual = {max_rel:.3g} (worst pair "
                f"(j={worst_j}, k={worst_k})). The filter will be biased. Use "
                f"mode='imm_general' for non-(H5) models, or apply "
                f"prg.utils.h5_constraint.apply_AB_constraint to enforce (H5) "
                f"by recomputing A(k), B(k) via the closed form "
                f"A = Δ Σ_V⁻¹ C, B = Δ Σ_V⁻¹ D.",
                RuntimeWarning,
                stacklevel=3,
            )

    # ------------------------------------------------------------------
    # Pre-computation of stationary moments and IMM constants
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Pre-computation of stationary moments (shared by both modes)
    # ------------------------------------------------------------------

    def _precompute_stationary(self) -> None:
        """
        Pre-compute and cache the stationary regime moments µ(k), Σ(k) and
        related quantities.  Called once at construction for **both** filter
        modes so that the initial step can start at the stationary distribution.

        Stores
        ------
        self._pi_inf   : (K,)                  stationary regime distribution
        self._p_rev    : (K, K)                time-reversed transition
                                               p_rev[j,k] = p(r_n=j | r_{n+1}=k)
        self._mu_z     : list[K] of (dim_z,1)  E[Z_n | r_n=k] at stationarity
        self._P_z_stat : list[K] of (dim_z,dim_z)  E[Z_n Z_n^T | r_n=k] (uncentred)
        self._Sigma    : list[K] of (dim_z,dim_z)  Var[Z_n | r_n=k] (centred)
        self._mu_X     : list[K] of (q,1)      E[X_n | r_n=k]
        self._mu_Y     : list[K] of (s,1)      E[Y_n | r_n=k]
        self._S_XX     : list[K] of (q,q)      Σ_XX(k)
        self._S_XY     : list[K] of (q,s)      Σ_XY(k)
        self._S_YY     : list[K] of (s,s)      Σ_YY(k)
        """
        p = self._p
        K, q = p.K, p.q

        # --- Stationary distribution and time-reversed transition ----------
        self._pi_inf = p.stationary_distribution()  # (K,)
        joint_inf = self._pi_inf[:, None] * p.P  # p(r_n=j, r_{n+1}=k)
        marg_rnp1 = joint_inf.sum(axis=0)  # = π_∞ again
        safe_marg = np.where(marg_rnp1 > 0, marg_rnp1, 1.0)
        # p_rev[j, k] = p(r_n=j | r_{n+1}=k)  under stationarity
        self._p_rev = joint_inf / safe_marg[None, :]  # (K, K)

        # --- Fixed-point iteration for µ(k), P(k) --------------------------
        # µ(k) = F_k Σ_j p_rev[j,k] µ(j) + b_k
        # P(k) = F_k Σ_j p_rev[j,k] P(j) F_k^T
        #      + F_k (Σ_j p_rev[j,k] µ(j)) b_k^T
        #      + b_k (Σ_j p_rev[j,k] µ(j))^T F_k^T
        #      + b_k b_k^T + Σ_W(k)
        mu_list = [p.mu_z0(k).copy() for k in range(K)]
        P_list = [p.Sigma_z0(k) + p.mu_z0(k) @ p.mu_z0(k).T for k in range(K)]

        MAX_ITER = 1000
        TOL = 1e-12
        diff = np.inf
        for it in range(MAX_ITER):
            mu_new: list[np.ndarray] = []
            P_new: list[np.ndarray] = []
            for k in range(K):
                F = p.f_matrix.F(k)
                b = p.b(k)
                w_mu = sum(self._p_rev[j, k] * mu_list[j] for j in range(K))
                w_P = sum(self._p_rev[j, k] * P_list[j] for j in range(K))
                Fw_mu = F @ w_mu
                mu_new.append(Fw_mu + b)
                P_new.append(
                    _sym(
                        F @ w_P @ F.T + Fw_mu @ b.T + b @ Fw_mu.T + b @ b.T + p.noise_cov.Sigma_W(k)
                    )
                )
            diff = max(np.abs(mu_new[k] - mu_list[k]).max() for k in range(K))
            diff = max(diff, max(np.abs(P_new[k] - P_list[k]).max() for k in range(K)))
            mu_list, P_list = mu_new, P_new
            if diff < TOL:
                logger.debug("Stationary moments converged in %d iterations (Δ=%.2e)", it + 1, diff)
                break
        else:
            logger.warning(
                "Stationary moments did not fully converge after %d iterations "
                "(Δ=%.2e). Filter outputs may be slightly biased.",
                MAX_ITER,
                diff,
            )

        # Cache stationary moments and their X/Y blocks
        self._mu_z = mu_list  # µ(k),                  (dim_z, 1)
        self._P_z_stat = P_list  # uncentred 2nd moment,  (dim_z, dim_z)
        self._Sigma = [
            _psd_floor(_sym(P_list[k] - mu_list[k] @ mu_list[k].T)) for k in range(K)
        ]  # centred Σ(k),          (dim_z, dim_z)
        self._mu_X = [self._mu_z[k][:q] for k in range(K)]
        self._mu_Y = [self._mu_z[k][q:] for k in range(K)]
        self._S_XX = [self._Sigma[k][:q, :q] for k in range(K)]
        self._S_XY = [self._Sigma[k][:q, q:] for k in range(K)]
        self._S_YY = [self._Sigma[k][q:, q:] for k in range(K)]

    # ------------------------------------------------------------------
    # Pre-computation of IMM constants (h5_exact only)
    # ------------------------------------------------------------------

    def _precompute(self) -> None:
        """
        Pre-compute all IMM constants for mode='h5_exact': stationary moments
        (via :meth:`_precompute_stationary`), per-regime Kalman gain, posterior
        covariance, and pair-conditional regime-update constants.

        All quantities computed here are time-invariant; the per-step
        recursion only manipulates the K-vector π_n and the previous
        observation y_{n-1}.
        """
        p = self._p
        K, q = p.K, p.q

        # --- Stationary distribution, p_rev, µ(k), Σ(k) and X/Y blocks ---
        self._precompute_stationary()

        # --- Per-regime Kalman gain and posterior covariance (constant) ----
        # K_k    = Σ_XY(k) Σ_YY(k)^{-1}
        self._K_gain = [
            _safe_solve(self._S_YY[k].T, self._S_XY[k].T).T  # (q, s)
            for k in range(K)
        ]
        # P_post(k): short form OR Joseph form (selected at construction).
        # Note: under stationarity this is *constant in n*.
        self._P_post: list[np.ndarray] = []
        for k in range(K):
            Kg = self._K_gain[k]
            if self._joseph:
                # H_k = Σ_YX(k) Σ_XX(k)^{-1}              (s, q)
                # R_k = Σ_YY(k) - H_k Σ_XX(k) H_k^T       (s, s)
                # P_post(k) = (I - K H) Σ_XX (I - K H)^T + K R K^T
                H = _safe_solve(self._S_XX[k].T, self._S_XY[k]).T
                R = _psd_floor(_sym(self._S_YY[k] - H @ self._S_XX[k] @ H.T))
                IKH = np.eye(q) - Kg @ H
                P_post_k = _psd_floor(_sym(IKH @ self._S_XX[k] @ IKH.T + Kg @ R @ Kg.T))
            else:
                # Short form: P_post(k) = Σ_XX(k) - K_k Σ_YY(k) K_k^T
                P_post_k = _psd_floor(_sym(self._S_XX[k] - Kg @ self._S_YY[k] @ Kg.T))
            self._P_post.append(P_post_k)

        # --- Pair-conditional regime-update constants ----------------------
        # For each (j, k):
        #   µ_Y(j, k) = [C_k, D_k] µ(j) + b_Y(k)               (s, 1)
        #   Cov(Y_{n+1}, Y_n | j, k) = (F_k Σ(j))[q:, q:]      (s, s)
        #   M̃_{j,k} = Cov × Σ_YY(j)^{-1}                       (s, s)  [used for the mean]
        #   Γ(j, k) = C_k P_post(j) C_k^T + Σ_V(k)             (s, s)
        #
        # The pair-conditional predictive covariance Γ(j,k) is derived by
        # integrating out the filtered state X_n | (r_n=j, y_{1:n}) which,
        # under (H5), is N(µ_X(j) + K_j(y_n − µ_Y(j)), P_post(j)):
        #
        #   Var(Y_{n+1} | r_n=j, r_{n+1}=k, y_n) = C_k P_post(j) C_k^T + Σ_V(k)
        #
        # where C_k = F_k[q:, :q] and Σ_V(k) = noise_cov[q:, q:].
        # Note: using Σ_YY(k) in place of this (the stationary *marginal*
        # variance of Y_{n+1} under regime k) introduces a systematic error
        # of F_k^Y [Σ_bar(k) − Σ(j)] F_k^{Y,T} that biases one regime to
        # always win when the regime covariances differ.
        self._mu_Y_jk: list[list[np.ndarray]] = [[None] * K for _ in range(K)]
        self._M_t: list[list[np.ndarray]] = [[None] * K for _ in range(K)]
        self._Gamma: list[list[np.ndarray]] = [[None] * K for _ in range(K)]
        # Pre-factorised Γ(j,k) for the fast per-step Gaussian log-pdf.
        # These are time-invariant under (H5), so factorising them once here
        # avoids an eigen-decomposition on every (j,k) at every step.
        self._Gamma_prec: list[list[np.ndarray]] = [[None] * K for _ in range(K)]
        self._Gamma_lognorm: list[list[float]] = [[0.0] * K for _ in range(K)]
        for k in range(K):
            F = p.f_matrix.F(k)
            b_Y = p.b(k)[q:]
            C_k = F[q:, :q]  # C block of F_k  (s, q)
            SV_k = p.noise_cov.Sigma_V(k)  # Σ_V(k)          (s, s)
            for j in range(K):
                mu_Y_jk = F[q:, :] @ self._mu_z[j] + b_Y
                Cov = (F @ self._Sigma[j])[q:, q:]  # (s, s) — used for mean
                M_t = _safe_solve(self._S_YY[j].T, Cov.T).T  # (s, s) — used for mean
                Gamma = _psd_floor(_sym(C_k @ self._P_post[j] @ C_k.T + SV_k))  # (s, s)
                self._mu_Y_jk[j][k] = mu_Y_jk
                self._M_t[j][k] = M_t
                self._Gamma[j][k] = Gamma
                prec, log_norm = _precompute_gaussian_logpdf(Gamma)
                self._Gamma_prec[j][k] = prec
                self._Gamma_lognorm[j][k] = log_norm

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def _reset_state(self) -> None:
        """Reset the per-step dynamic state (pre-computed constants stay cached)."""
        p = self._p
        # Posterior regime weights — start at the stationary distribution π_∞
        # (computed by _precompute_stationary(), called during __init__).
        self._pi: np.ndarray = self._pi_inf.copy()
        # Previous observation — needed from step 1 onward
        self._y_prev: np.ndarray | None = None
        self._n: int = 0
        self._initialized: bool = False

        # Mode-specific per-step moments (only used in mode="imm_general").
        # Start at the stationary moments (from _precompute_stationary()).
        if self._mode == "imm_general":
            self._mu: list[np.ndarray] = [self._mu_z[k].copy() for k in range(p.K)]
            self._P_z: list[np.ndarray] = [self._P_z_stat[k].copy() for k in range(p.K)]

    def reset(self) -> None:
        """Reset the filter to n = 0 (call before re-processing a sequence)."""
        self._reset_state()
        logger.debug("Filter reset.")

    # ------------------------------------------------------------------
    # Public step interface
    # ------------------------------------------------------------------

    def __iter__(self) -> GSSFilter:
        return self

    def step(self, y: np.ndarray) -> FilterResult:
        """
        Process one observation and return filtered estimates.

        Parameters
        ----------
        y : ndarray, shape (s,) or (s, 1)
            Observation Y_n.

        Returns
        -------
        FilterResult
        """
        y = np.asarray(y, dtype=float).reshape(-1, 1)  # (s, 1)

        if self._mode == "h5_exact":
            init_fn, step_fn = self._init_step_h5, self._update_step_h5
        else:  # "imm_general"
            init_fn, step_fn = self._init_step_general, self._update_step_general

        if not self._initialized:
            result = init_fn(y)
            self._initialized = True
        else:
            result = step_fn(y)

        self._y_prev = y
        self._n += 1
        return result

    # ==================================================================
    # MODE = "h5_exact" — exact IMM under hypothesis (H5)
    # ==================================================================
    #
    # The initial step uses the *stationary* prior (µ_z[r], Σ[r], π_∞[r]),
    # i.e. the same pre-computed moments that drive every subsequent step.
    # This ensures the filter starts in steady state and avoids the
    # transient that would arise from an arbitrary user-supplied π_0 / Σ_z0.
    # ------------------------------------------------------------------

    def _init_step_h5(self, y1: np.ndarray) -> FilterResult:
        p = self._p
        K, q = p.K, p.q

        log_w: np.ndarray = np.empty(K)
        E_x_r: list[np.ndarray] = []
        Var_x_r: list[np.ndarray] = []

        for r in range(K):
            # All moments are pre-computed at stationarity
            mu_Y_r = self._mu_Y[r]  # (s, 1)

            # log p(y_1 | r_1=r) = log π_∞(r) + log N(y_1; µ_Y(r), Σ_YY(r))
            log_w[r] = np.log(self._pi_inf[r] + 1e-300) + multivariate_normal.logpdf(
                y1.ravel(),
                mean=mu_Y_r.ravel(),
                cov=self._S_YY[r],
                allow_singular=True,
            )

            # Kalman update using pre-computed gain and posterior covariance
            e_x = self._mu_X[r] + self._K_gain[r] @ (y1 - mu_Y_r)  # (q, 1)
            var_x = self._P_post[r]  # constant

            E_x_r.append(e_x)
            Var_x_r.append(var_x)

        # Incremental log-likelihood
        log_lik = float(logsumexp(log_w)) if np.isfinite(log_w).any() else -np.inf

        # Posterior weights π_1
        log_max = log_w.max()
        if not np.isfinite(log_max):
            pi1 = self._pi_inf.copy()
            logger.warning("Filter init: log-weights all -inf; falling back to π_∞.")
        else:
            log_w -= log_max
            pi1 = np.exp(log_w)
            s_pi = pi1.sum()
            if not np.isfinite(s_pi) or s_pi <= 0.0:
                pi1 = self._pi_inf.copy()
                logger.warning("Filter init: invalid posterior sum; falling back to π_∞.")
            else:
                pi1 /= s_pi

        # Marginal filtered estimates
        E_x, E_xx = _mix(pi1, E_x_r, Var_x_r, K)

        # Innovation: ν_1 = y_1 − Σ_r π_∞(r) µ_Y(r)
        y_pred = sum(self._pi_inf[r] * self._mu_Y[r] for r in range(K))
        innov = y1 - y_pred

        self._pi = pi1
        return FilterResult(
            n=self._n, E_x=E_x, E_xx=E_xx, pi=pi1, innovation=innov, log_lik=log_lik
        )

    # ------------------------------------------------------------------
    # Recursion  n → n+1   (paper §3 — IMM steps II, III, IV)
    # ------------------------------------------------------------------

    def _update_step_h5(self, y_new: np.ndarray) -> FilterResult:
        p = self._p
        K, q = p.K, p.q
        y_prev = self._y_prev  # (s, 1) — previous observation

        # ---- Step (II): mode-probability update ---------------------------
        # Joint p(r_n=j, r_{n+1}=k | y_{1:n}) = π_n(j) · P(j, k)
        joint = self._pi[:, None] * p.P  # (K, K)
        marg_rnp1 = joint.sum(axis=0)  # (K,)

        # Pair-conditional likelihoods Λ(j, k) using *constant* moments
        log_alpha = np.full(K, -np.inf)
        y_pred_acc = np.zeros_like(y_new)  # for marginal innovation

        for k in range(K):
            log_terms: list[float] = []
            for j in range(K):
                w = joint[j, k]
                if w < 1e-300:
                    continue
                # Pair-conditional predictive mean (eq. y_pred_jk)
                mean_jk = self._mu_Y_jk[j][k] + self._M_t[j][k] @ (y_prev - self._mu_Y[j])

                # Accumulate marginal predicted observation
                y_pred_acc += w * mean_jk

                # Fast log N(y_new; mean_jk, Γ(j,k)) using the pre-factorised
                # Γ(j,k) (identical to multivariate_normal.logpdf, no per-step
                # factorisation).
                log_lik_c = _fast_logpdf(
                    (y_new - mean_jk).ravel(),
                    self._Gamma_prec[j][k],
                    self._Gamma_lognorm[j][k],
                )
                log_terms.append(np.log(w) + log_lik_c)

            if log_terms:
                log_alpha[k] = float(logsumexp(log_terms))

        # Incremental log-likelihood log p(y_{n+1} | y_{1:n})
        log_lik = float(logsumexp(log_alpha)) if np.isfinite(log_alpha).any() else -np.inf

        # Normalise to π_{n+1}
        log_max = log_alpha.max()
        if not np.isfinite(log_max):
            pi_np1 = marg_rnp1.copy()
            logger.warning(
                "Filter step %d: log_alpha all -inf; falling back to marginal p(r_{n+1}|y_{1:n}).",
                self._n,
            )
        else:
            log_alpha -= log_max
            pi_np1 = np.exp(log_alpha)
            s_pi = pi_np1.sum()
            if not np.isfinite(s_pi) or s_pi <= 0.0:
                pi_np1 = marg_rnp1.copy()
                logger.warning(
                    "Filter step %d: invalid posterior sum; falling back to marg_rnp1.",
                    self._n,
                )
            else:
                pi_np1 /= s_pi

        # ---- Step (III): mode-conditional Kalman update --------------------
        # Uses ONLY single-regime stationary moments. Per-regime posterior
        # covariance is constant (pre-computed in self._P_post).
        E_x_r: list[np.ndarray] = []
        Var_x_r: list[np.ndarray] = []
        for k in range(K):
            nu_k = y_new - self._mu_Y[k]  # (s, 1)
            e_x = self._mu_X[k] + self._K_gain[k] @ nu_k  # (q, 1)
            E_x_r.append(e_x)
            Var_x_r.append(self._P_post[k])  # constant!

        # ---- Step (IV): combination ---------------------------------------
        E_x, E_xx = _mix(pi_np1, E_x_r, Var_x_r, K)
        innov = y_new - y_pred_acc

        # ---- Update dynamic state -----------------------------------------
        self._pi = pi_np1

        return FilterResult(
            n=self._n, E_x=E_x, E_xx=E_xx, pi=pi_np1, innovation=innov, log_lik=log_lik
        )

    # ==================================================================
    # MODE = "imm_general" — IMM-style recursion, no (H5) required
    # ==================================================================
    #
    # This is the filter of CS_FinaleBis (eqs 17ter, 16–17, 13'–15, 18,
    # 21'–22, 8–9) — i.e. the exactIMM ≤ v0.9.0 implementation. The full
    # F(k) = [[A_k, B_k], [C_k, D_k]] is used, per-regime moments
    # µ_n(r), P_n(r) are propagated at each step from the filtered π_n
    # (*not* the stationary π_∞), and the Kalman update uses the
    # up-to-date Σ_n(r) rather than a constant Σ(k). No (H5) requirement.
    # ------------------------------------------------------------------

    def _init_step_general(self, y1: np.ndarray) -> FilterResult:
        """Initial step using the stationary prior µ_z[r], Σ[r], π_∞[r]."""
        p = self._p
        K, q = p.K, p.q

        log_w: np.ndarray = np.empty(K)
        E_x_r: list[np.ndarray] = []
        Var_x_r: list[np.ndarray] = []

        for r in range(K):
            # Use stationary moments (from _precompute_stationary())
            Sig = self._Sigma[r]  # (dim_z, dim_z) centred cov
            mu = self._mu_z[r]  # (dim_z, 1)
            mu_X = mu[:q]
            mu_Y = mu[q:]
            S_XX = Sig[:q, :q]
            S_XY = Sig[:q, q:]
            S_YY = Sig[q:, q:]

            log_w[r] = np.log(self._pi_inf[r] + 1e-300) + multivariate_normal.logpdf(
                y1.ravel(),
                mean=mu_Y.ravel(),
                cov=S_YY,
                allow_singular=True,
            )

            M_r = _safe_solve(S_YY.T, S_XY.T).T  # (q, s)
            e_x = mu_X + M_r @ (y1 - mu_Y)  # (q, 1)
            var_x = _psd_floor(_sym(S_XX - M_r @ S_XY.T))  # (q, q)

            E_x_r.append(e_x)
            Var_x_r.append(var_x)

        log_lik = float(logsumexp(log_w)) if np.isfinite(log_w).any() else -np.inf

        log_max = log_w.max()
        if not np.isfinite(log_max):
            pi1 = self._pi_inf.copy()
            logger.warning("Filter init: log-weights all -inf; falling back to π_∞.")
        else:
            log_w -= log_max
            pi1 = np.exp(log_w)
            s_pi = pi1.sum()
            if not np.isfinite(s_pi) or s_pi <= 0.0:
                pi1 = self._pi_inf.copy()
                logger.warning("Filter init: invalid posterior sum; falling back to π_∞.")
            else:
                pi1 /= s_pi

        E_x, E_xx = _mix(pi1, E_x_r, Var_x_r, K)

        # Innovation: ν_1 = y_1 − Σ_r π_∞(r) µ_Y(r)
        y_pred = sum(self._pi_inf[r] * self._mu_Y[r] for r in range(K))
        innov = y1 - y_pred

        self._pi = pi1
        return FilterResult(
            n=self._n, E_x=E_x, E_xx=E_xx, pi=pi1, innovation=innov, log_lik=log_lik
        )

    def _update_step_general(self, y_new: np.ndarray) -> FilterResult:
        p = self._p
        K, q = p.K, p.q
        y_prev = self._y_prev  # (s, 1)

        # ---- (17bis)  joint[rn, rnp1] = π_n[rn] · P[rn, rnp1] -------------
        joint = self._pi[:, None] * p.P  # (K, K)
        marg_rnp1 = joint.sum(axis=0)  # (K,)
        safe_marg = np.where(marg_rnp1 > 0, marg_rnp1, 1.0)
        p_rn_rnp1 = joint / safe_marg[None, :]  # (K, K): [rn, rnp1]

        # ---- (17ter) + (17)  Mean and second-moment propagation -----------
        mu_np1: list[np.ndarray] = []
        P_np1: list[np.ndarray] = []
        for rnp1 in range(K):
            F = p.f_matrix.F(rnp1)
            b = p.b(rnp1)  # (q+s, 1)
            w_mu = sum(p_rn_rnp1[rn, rnp1] * self._mu[rn] for rn in range(K))
            w_P = sum(p_rn_rnp1[rn, rnp1] * self._P_z[rn] for rn in range(K))

            Fw_mu = F @ w_mu
            mu_np1.append(Fw_mu + b)  # (17ter)
            P_np1.append(
                _sym(
                    F @ w_P @ F.T + Fw_mu @ b.T + b @ Fw_mu.T + b @ b.T + p.noise_cov.Sigma_W(rnp1)
                )
            )

        # ---- (13'),(14),(15),(18)  Transition density → π_{n+1} -----------
        log_alpha = np.full(K, -np.inf)
        y_pred_acc = np.zeros_like(y_new)  # (s, 1)

        for rnp1 in range(K):
            F = p.f_matrix.F(rnp1)

            log_terms: list[float] = []
            for rn in range(K):
                w = joint[rn, rnp1]
                if w < 1e-300:
                    continue

                Sig_n = _sym(self._P_z[rn] - self._mu[rn] @ self._mu[rn].T)
                mu_Y_n = self._mu[rn][q:]
                S_YY_n = Sig_n[q:, q:]

                Cov_Ynp1_Yn = (F @ Sig_n)[q:, q:]  # (16)
                M_t = _safe_solve(S_YY_n.T, Cov_Ynp1_Yn.T).T  # (14)
                # Pair-conditional predictive variance Var(Y_{n+1} | r_n=rn, r_{n+1}=rnp1):
                # the block of F Sigma_n(rn) F^T + Sigma_W(rnp1) — NOT the r_n-marginal
                # Sigma_{YY,n+1}(rnp1). With it, Gamma is the Schur complement of S_YY_n and
                # is PSD by construction (no spurious mixing of conditionings). (eq. 22)
                S_YY_pair = (F @ Sig_n @ F.T + p.noise_cov.Sigma_W(rnp1))[q:, q:]
                Gamma = _psd_floor(_sym(S_YY_pair - M_t @ Cov_Ynp1_Yn.T))

                mu_Ynp1 = F[q:, :] @ self._mu[rn] + p.b(rnp1)[q:]
                mean_c = mu_Ynp1 + M_t @ (y_prev - mu_Y_n)

                y_pred_acc += w * mean_c

                log_lik_c = multivariate_normal.logpdf(
                    y_new.ravel(),
                    mean=mean_c.ravel(),
                    cov=Gamma,
                    allow_singular=True,
                )
                log_terms.append(np.log(w) + log_lik_c)

            if log_terms:
                log_alpha[rnp1] = float(logsumexp(log_terms))

        log_lik = float(logsumexp(log_alpha)) if np.isfinite(log_alpha).any() else -np.inf

        log_max = log_alpha.max()
        if not np.isfinite(log_max):
            pi_np1 = marg_rnp1.copy()
            logger.warning(
                "Filter step %d: log_alpha all -inf; falling back to marg_rnp1.",
                self._n,
            )
        else:
            log_alpha -= log_max
            pi_np1 = np.exp(log_alpha)
            s_pi = pi_np1.sum()
            if not np.isfinite(s_pi) or s_pi <= 0.0:
                pi_np1 = marg_rnp1.copy()
                logger.warning(
                    "Filter step %d: invalid posterior sum; falling back to marg_rnp1.",
                    self._n,
                )
            else:
                pi_np1 /= s_pi

        # ---- (21'),(22)  Kalman update ------------------------------------
        E_x_r: list[np.ndarray] = []
        Var_x_r: list[np.ndarray] = []
        for rnp1 in range(K):
            Sig = _sym(P_np1[rnp1] - mu_np1[rnp1] @ mu_np1[rnp1].T)
            mu_X = mu_np1[rnp1][:q]
            mu_Y = mu_np1[rnp1][q:]
            S_XX = Sig[:q, :q]
            S_XY = Sig[:q, q:]
            S_YY = Sig[q:, q:]

            M_r = _safe_solve(S_YY.T, S_XY.T).T  # (21')
            e_x = mu_X + M_r @ (y_new - mu_Y)
            var_x = _psd_floor(_sym(S_XX - M_r @ S_XY.T))  # (22)

            E_x_r.append(e_x)
            Var_x_r.append(var_x)

        # ---- (8),(9)  Marginal estimates ----------------------------------
        E_x, E_xx = _mix(pi_np1, E_x_r, Var_x_r, K)
        innov = y_new - y_pred_acc

        # ---- State update --------------------------------------------------
        self._pi = pi_np1
        self._P_z = P_np1
        self._mu = mu_np1

        return FilterResult(
            n=self._n, E_x=E_x, E_xx=E_xx, pi=pi_np1, innovation=innov, log_lik=log_lik
        )

    # ------------------------------------------------------------------
    # Batch run methods
    # ------------------------------------------------------------------

    def run_csv(self, csv_path: str | pathlib.Path) -> pd.DataFrame:
        """
        Run the filter on observations stored in a CSV file.

        The file must have columns ``y_0, …, y_{s-1}`` (and optionally
        ``x_0, …, x_{q-1}`` to compute per-step squared error).

        Returns
        -------
        pd.DataFrame
            One row per time step; columns defined in :meth:`run`.
        """
        df_in = pd.read_csv(pathlib.Path(csv_path))
        return self._filter_df(df_in)

    def run(
        self,
        N: int,
        seed: int | None = None,
        output_dir: str | pathlib.Path | None = None,
        model_name: str = "gss",
    ) -> tuple[pathlib.Path | None, pd.DataFrame]:
        """
        Simulate N steps and run the filter jointly.

        Returns
        -------
        (sim_path, filter_df)
            sim_path  : Path to the simulation CSV, or None.
            filter_df : DataFrame with columns
                n, E_x_*, V_x_*, p_r_*, sq_err.
        """
        p = self._p
        q, s = p.q, p.s
        self.reset()

        sim = GSSSimulator(p, N=N, seed=seed)

        sim_rows: list[list] = []
        filter_rows: list[dict] = []
        t0 = time.perf_counter()

        for n, r, x, y in sim:
            result = self.step(y)
            e_x = result.E_x.ravel()
            var_x = result.Var_x.diagonal()

            sim_rows.append([n, r] + x.ravel().tolist() + y.ravel().tolist())

            row: dict = {"n": n}
            row.update({f"E_x_{i}": float(e_x[i]) for i in range(q)})
            row.update({f"V_x_{i}": float(var_x[i]) for i in range(q)})
            row.update({f"p_r_{k}": float(result.pi[k]) for k in range(p.K)})
            row["sq_err"] = float(np.sum((x.ravel() - e_x) ** 2))
            filter_rows.append(row)

        logger.info("Filter complete: %d steps in %.3f s.", N, time.perf_counter() - t0)

        # Optionally save simulation CSV
        sim_path: pathlib.Path | None = None
        if output_dir is not None:
            output_dir = pathlib.Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            seed_str = str(seed) if seed is not None else "random"
            fname = f"simulated_{model_name}_N{N}_seed{seed_str}.csv"
            sim_path = output_dir / fname
            header = ["n", "r"] + [f"x_{i}" for i in range(q)] + [f"y_{i}" for i in range(s)]
            with sim_path.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow(header)
                writer.writerows(sim_rows)
            logger.info("Simulation CSV: %s", sim_path)

        return sim_path, pd.DataFrame(filter_rows)

    def _filter_df(self, df_in: pd.DataFrame) -> pd.DataFrame:
        """Internal helper: filter from an already-loaded DataFrame."""
        p = self._p
        q, s = p.q, p.s
        self.reset()

        y_cols = [f"y_{i}" for i in range(s)]
        x_cols = [f"x_{i}" for i in range(q)]
        has_x = all(c in df_in.columns for c in x_cols)

        rows: list[dict] = []
        for _, row_in in df_in.iterrows():
            y = row_in[y_cols].to_numpy(dtype=float).reshape(-1, 1)
            result = self.step(y)
            e_x = result.E_x.ravel()
            var_x = result.Var_x.diagonal()

            row: dict = {"n": int(row_in["n"]) if "n" in row_in.index else result.n}
            row.update({f"E_x_{i}": float(e_x[i]) for i in range(q)})
            row.update({f"V_x_{i}": float(var_x[i]) for i in range(q)})
            row.update({f"p_r_{k}": float(result.pi[k]) for k in range(p.K)})
            if has_x:
                x_true = row_in[x_cols].to_numpy(dtype=float)
                row["sq_err"] = float(np.sum((x_true - e_x) ** 2))
            rows.append(row)

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Properties / repr
    # ------------------------------------------------------------------

    @property
    def params(self) -> GSSParams:
        """Model parameters."""
        return self._p

    @property
    def joseph(self) -> bool:
        """Whether the Joseph-form covariance update is enabled (h5_exact only)."""
        return self._joseph

    @property
    def mode(self) -> str:
        """Filter mode: ``'h5_exact'`` or ``'imm_general'``."""
        return self._mode

    @property
    def n(self) -> int:
        """Number of observations processed so far."""
        return self._n

    @property
    def stationary_distribution(self) -> np.ndarray:
        """Pre-computed stationary distribution π_∞ of the regime chain."""
        return self._pi_inf

    def __repr__(self) -> str:
        joseph_str = ", joseph=True" if self._joseph else ""
        return (
            f"<GSSFilter(K={self._p.K}, q={self._p.q}, "
            f"s={self._p.s}, mode={self._mode!r}, n={self._n}{joseph_str})>"
        )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _safe_solve(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Solve A X = B robustly.

    Tries :func:`numpy.linalg.solve` first (fast LU). If A is singular or
    ill-conditioned, falls back to :func:`numpy.linalg.lstsq` which uses
    an SVD-based pseudo-inverse and handles rank-deficient A gracefully.
    """
    try:
        return np.linalg.solve(A, B)
    except np.linalg.LinAlgError:
        logger.warning("_safe_solve: singular matrix, falling back to lstsq.")
        X, *_ = np.linalg.lstsq(A, B, rcond=None)
        return X


def _sym(M: np.ndarray) -> np.ndarray:
    """Symmetrise a square matrix to counteract floating-point drift."""
    return 0.5 * (M + M.T)


def _precompute_gaussian_logpdf(cov: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Pre-factorise a covariance for fast, repeated Gaussian log-density.

    Returns ``(prec, log_norm)`` such that, for a deviation
    ``dev = (x − mean).ravel()``,

        log N(x; mean, cov) = log_norm − 0.5 · ‖prec @ dev‖².

    This is mathematically identical to
    :func:`scipy.stats.multivariate_normal.logpdf` with
    ``allow_singular=True``, but the expensive factorisation of *cov* is
    done **once** here instead of on every call. The hot per-step loop
    then only needs one small matrix–vector product (see
    :func:`_fast_logpdf`), which removes the O(N·K²) eigen-decompositions
    that ``multivariate_normal.logpdf`` would otherwise perform.

    The common path uses the Cholesky factor (``cov`` is positive definite
    here: every Γ(j,k) is PSD-floored at construction). If Cholesky fails
    on a (near-)singular covariance, it falls back to an eigen-decomposition
    pseudo-inverse with the same tolerance scipy uses for
    ``allow_singular=True`` — so degenerate models keep the previous
    behaviour.
    """
    s = cov.shape[0]
    try:
        L = np.linalg.cholesky(cov)  # cov = L Lᵀ, lower-triangular
        prec = np.linalg.inv(L)  # ‖prec @ dev‖² = devᵀ cov⁻¹ dev
        log_det = 2.0 * float(np.sum(np.log(np.diag(L))))
        log_norm = -0.5 * (s * _LOG_2PI + log_det)
        return prec, log_norm
    except np.linalg.LinAlgError:
        # Pseudo-inverse fallback mirroring scipy's allow_singular=True.
        vals, vecs = np.linalg.eigh(_sym(cov))
        eps = 1e6 * np.finfo(np.float64).eps * max(float(vals.max()), 0.0)
        keep = vals > eps
        d = vals[keep]
        # prec = (vecs[:, keep] / sqrt(d))ᵀ  ⇒  ‖prec @ dev‖² = Mahalanobis.
        prec = (vecs[:, keep] / np.sqrt(d)).T
        log_norm = -0.5 * (int(keep.sum()) * _LOG_2PI + float(np.sum(np.log(d))))
        return prec, log_norm


def _fast_logpdf(dev: np.ndarray, prec: np.ndarray, log_norm: float) -> float:
    """
    Gaussian log-density from a pre-factorised covariance.

    Parameters
    ----------
    dev : ndarray (s,)
        Centred observation ``(x − mean).ravel()``.
    prec, log_norm
        Output of :func:`_precompute_gaussian_logpdf` for the covariance.
    """
    z = prec @ dev
    return log_norm - 0.5 * float(z @ z)


def _psd_floor(M: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """
    Project M onto the PSD cone by flooring negative eigenvalues at *eps*.

    For 1×1 matrices the operation reduces to a simple max().
    """
    if M.shape == (1, 1):
        return np.maximum(M, eps)
    vals, vecs = np.linalg.eigh(_sym(M))
    if vals.min() >= eps:
        return M  # already PSD — nothing to do
    return vecs @ np.diag(np.maximum(vals, eps)) @ vecs.T


def _mix(
    pi: np.ndarray,
    E_x_r: list[np.ndarray],
    Var_x_r: list[np.ndarray],
    K: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Marginalise over regimes.

    E[X|y]     = Σ_r π(r) E[X|r,y]
    E[XX^T|y]  = Σ_r π(r) (Var[X|r,y] + E[X|r,y] E[X|r,y]^T)
    """
    E_x = sum(pi[r] * E_x_r[r] for r in range(K))
    E_xx = sum(pi[r] * (Var_x_r[r] + E_x_r[r] @ E_x_r[r].T) for r in range(K))
    return E_x, _sym(E_xx)
