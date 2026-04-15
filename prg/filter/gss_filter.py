#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prg/filter/gss_filter.py
========================
Fast optimal filter for the GSS model (Option B — general non-zero mean).

Implements the recursion from CS_FinaleBis:

  Initialisation — eqs (I.1)–(I.3)
    p(r1|y1), E[Z1|r1,y1], P_z0(r)  from π0, μ_z0, Σ_z0.

  Step (i) — eqs (17ter), (16)–(17), (13')–(15), (18)
    Propagate means μ_{n+1} and second moments P_{n+1};
    evaluate transition density; update p(r_{n+1}|y_{1:n+1}).

  Step (ii) — eqs (21')–(22)
    Kalman update: E[X_{n+1}|r_{n+1}, y_{n+1}] and Γ_{r_{n+1}}.

  Step (iii) — eqs (8)–(9)
    Marginalise over regimes to get E[X_{n+1}|y_{1:n+1}]
    and E[X_{n+1}X_{n+1}^T|y_{1:n+1}].

Option B vs the paper's zero-mean formulation
----------------------------------------------
The paper (CS_FinaleBis eqs 13–22) uses uncentred second moments
as if they were covariances, which is exact only when E[Z_n|r_n]=0.
Option B tracks the marginal means μ_n(r) and uses centred covariances
Σ_n(r) = P_n(r) − μ_n(r)μ_n(r)^T throughout.  For models with
μ_z0(r)=0 (e.g. model_gss_K2_q1_s1) both formulations coincide.
"""

from __future__ import annotations

import csv
import dataclasses
import logging
import pathlib
import time

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

from prg.classes.GSSParams import GSSParams
from prg.classes.GSSSimulator import GSSSimulator

__all__ = ["GSSFilter", "FilterResult"]

logger = logging.getLogger("fofgss.filter")


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
    """

    n:    int
    E_x:  np.ndarray   # (q, 1)
    E_xx: np.ndarray   # (q, q)
    pi:   np.ndarray   # (K,)

    @property
    def Var_x(self) -> np.ndarray:
        """
        Posterior variance Var[X_n | y_{1:n}] = E[XX^T] − E[X]E[X]^T.

        Shape (q, q).  Diagonal entries give per-component MSE.
        """
        return self.E_xx - self.E_x @ self.E_x.T


# ---------------------------------------------------------------------------
# Filter
# ---------------------------------------------------------------------------


class GSSFilter:
    """
    Fast optimal filter for GSS models (Option B, general non-zero mean).

    Parameters
    ----------
    params : GSSParams
        Validated model parameters.

    Examples
    --------
    Step-by-step (iterator style)::

        filt = GSSFilter(params)
        for y in observations:          # y shape (s,) or (s, 1)
            res = filt.step(y)
            # res.E_x  shape (q, 1)
            # res.pi   shape (K,)

    From a simulation CSV::

        _, df = filt.run_csv("data/simulated/sim.csv")

    Simulate and filter jointly::

        sim_path, df = filt.run(N=1000, seed=42,
                                output_dir="data/simulated")
    """

    def __init__(self, params: GSSParams) -> None:
        self._p = params
        self._reset_state()

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def _reset_state(self) -> None:
        """Initialise (or re-initialise) all internal state variables."""
        p = self._p
        # Prior second moments  P_z_0(r) = Σ_z0(r) + μ_z0(r) μ_z0(r)^T   (eq I.1)
        self._P_z: list[np.ndarray] = [
            p.Sigma_z0(k) + p.mu_z0(k) @ p.mu_z0(k).T
            for k in range(p.K)
        ]
        # Prior marginal means  μ_0(r) = μ_z0(r)
        self._mu: list[np.ndarray] = [p.mu_z0(k).copy() for k in range(p.K)]
        # Posterior regime weights (start at π_0)
        self._pi: np.ndarray = p.pi0.copy()
        # Previous observation — needed for transition density from step 1 onward
        self._y_prev: np.ndarray | None = None
        self._n: int = 0
        self._initialized: bool = False

    def reset(self) -> None:
        """Reset the filter to n = 0 (call before re-processing a sequence)."""
        self._reset_state()
        logger.debug("Filter reset.")

    # ------------------------------------------------------------------
    # Public step interface
    # ------------------------------------------------------------------

    def __iter__(self) -> "GSSFilter":
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
        y = np.asarray(y, dtype=float).reshape(-1, 1)   # (s, 1)

        if not self._initialized:
            result = self._init_step(y)
            self._initialized = True
        else:
            result = self._update_step(y)

        self._y_prev = y
        self._n += 1
        return result

    # ------------------------------------------------------------------
    # Initialisation  n = 0  (eqs I.2, I.3)
    # ------------------------------------------------------------------

    def _init_step(self, y1: np.ndarray) -> FilterResult:
        p = self._p
        K, q = p.K, p.q

        log_w:   np.ndarray          = np.empty(K)
        E_x_r:   list[np.ndarray]    = []
        Var_x_r: list[np.ndarray]    = []

        for r in range(K):
            Sig  = p.Sigma_z0(r)              # (dim_z, dim_z) centred cov
            mu   = p.mu_z0(r)                 # (dim_z, 1)
            mu_X = mu[:q];  mu_Y = mu[q:]
            S_XX = Sig[:q, :q]
            S_XY = Sig[:q, q:]
            S_YY = Sig[q:, q:]

            # log p(y1 | r1=r)  =  log π0(r)  +  log N(y1; μ_Y, S_YY)   (eq I.2)
            log_w[r] = (
                np.log(p.pi0[r] + 1e-300)
                + multivariate_normal.logpdf(y1.ravel(), mean=mu_Y.ravel(), cov=S_YY)
            )

            # Kalman gain  M_r = S_XY S_YY^{-1}   (eq I.3)
            M_r   = np.linalg.solve(S_YY.T, S_XY.T).T        # (q, s)
            e_x   = mu_X + M_r @ (y1 - mu_Y)                 # (q, 1)
            var_x = _psd_floor(_sym(S_XX - M_r @ S_XY.T))                # (q, q)

            E_x_r.append(e_x)
            Var_x_r.append(var_x)

        # Posterior weights  (eq I.2 — normalisation)
        log_w -= log_w.max()
        pi1 = np.exp(log_w);  pi1 /= pi1.sum()

        # Marginal filtered estimates  (eqs 8–9)
        E_x, E_xx = _mix(pi1, E_x_r, Var_x_r, K)

        self._pi = pi1
        return FilterResult(n=self._n, E_x=E_x, E_xx=E_xx, pi=pi1)

    # ------------------------------------------------------------------
    # Recursion  n → n+1  (eqs 17ter, 16–17, 13'–15, 18, 21'–22, 8–9)
    # ------------------------------------------------------------------

    def _update_step(self, y_new: np.ndarray) -> FilterResult:
        p = self._p
        K, q = p.K, p.q
        y_prev = self._y_prev          # (s, 1) — previous observation

        # ---- (17bis)  joint[rn, rnp1] = π_n[rn] · P[rn, rnp1] ----------
        joint      = self._pi[:, None] * p.P           # (K, K)
        marg_rnp1  = joint.sum(axis=0)                 # (K,)
        safe_marg  = np.where(marg_rnp1 > 0, marg_rnp1, 1.0)
        p_rn_rnp1  = joint / safe_marg[None, :]        # (K, K): [rn, rnp1]

        # ---- (17ter) + (17)  Mean and second-moment propagation -----------
        #
        # With bias b(r_{n+1}) the second-moment recursion becomes:
        #   P_{n+1}(r) = F w_P F^T + F w_µ b^T + b w_µ^T F^T + b b^T + Σ_W
        # which ensures that the centred covariance
        #   Σ_{n+1}(r) = P_{n+1}(r) − µ_{n+1}(r) µ_{n+1}(r)^T
        #              = F (w_P − w_µ w_µ^T) F^T + Σ_W  ≥ 0
        # is always positive semi-definite regardless of b.
        # When b = 0 the formula reduces to the original (17).
        mu_np1: list[np.ndarray] = []
        P_np1:  list[np.ndarray] = []
        for rnp1 in range(K):
            F    = p.f_matrix.F(rnp1)
            b    = p.b(rnp1)                                      # (q+s, 1)
            w_mu = sum(p_rn_rnp1[rn, rnp1] * self._mu[rn] for rn in range(K))
            w_P  = sum(p_rn_rnp1[rn, rnp1] * self._P_z[rn] for rn in range(K))

            # (17ter)  µ_{n+1}(r) = F w_µ + b
            Fw_mu = F @ w_mu
            mu_np1.append(Fw_mu + b)

            # (17) corrected for non-zero bias
            P_np1.append(_sym(
                F @ w_P @ F.T
                + Fw_mu @ b.T + b @ Fw_mu.T
                + b @ b.T
                + p.noise_cov.Sigma_W(rnp1)
            ))

        # ---- (13'),(14),(15),(18)  Transition density → π_{n+1} ----------
        log_alpha = np.full(K, -np.inf)

        for rnp1 in range(K):
            # Centred covariance of Z_{n+1}  (used for Γ̃ and Kalman later)
            Sig_np1  = _sym(P_np1[rnp1] - mu_np1[rnp1] @ mu_np1[rnp1].T)
            S_YY_np1 = Sig_np1[q:, q:]                        # (s, s)

            log_terms: list[float] = []
            for rn in range(K):
                w = joint[rn, rnp1]
                if w < 1e-300:
                    continue

                # Centred covariance of Z_n
                Sig_n  = _sym(self._P_z[rn] - self._mu[rn] @ self._mu[rn].T)
                mu_Y_n = self._mu[rn][q:]                      # (s, 1)
                S_YY_n = Sig_n[q:, q:]                         # (s, s)

                F = p.f_matrix.F(rnp1)

                # Cov[Z_{n+1}, Z_n | r_{n:n+1}]  =  F Σ_n  (eq 16, centred)
                Cov_Ynp1_Yn = (F @ Sig_n)[q:, q:]             # (s, s)

                # M̃_{r_{n:n+1}}  =  Cov · S_YY_n^{-1}   (eq 14 — centred)
                M_t = np.linalg.solve(S_YY_n.T, Cov_Ynp1_Yn.T).T  # (s, s)

                # Γ̃_{r_{n:n+1}}  =  S_YY_{n+1}(rnp1) − M̃ Cov^T  (eq 15 — centred)
                # _psd_floor guards against tiny negative values from
                # catastrophic cancellation (near-unit-root systems).
                Gamma = _psd_floor(_sym(S_YY_np1 - M_t @ Cov_Ynp1_Yn.T))

                # Conditional mean of Y_{n+1}  (eq 13')  µ_{Y,n+1} = [C,D] µ_n + b_Y
                mu_Ynp1 = F[q:, :] @ self._mu[rn] + p.b(rnp1)[q:]   # (s, 1)
                mean_c  = mu_Ynp1 + M_t @ (y_prev - mu_Y_n)

                log_lik = multivariate_normal.logpdf(
                    y_new.ravel(), mean=mean_c.ravel(), cov=Gamma
                )
                log_terms.append(np.log(w) + log_lik)

            if log_terms:
                log_alpha[rnp1] = float(logsumexp(log_terms))

        # Normalise  (eq 18)
        log_alpha -= log_alpha.max()
        pi_np1 = np.exp(log_alpha);  pi_np1 /= pi_np1.sum()

        # ---- (21'),(22)  Kalman update ------------------------------------
        E_x_r:   list[np.ndarray] = []
        Var_x_r: list[np.ndarray] = []

        for rnp1 in range(K):
            Sig  = _sym(P_np1[rnp1] - mu_np1[rnp1] @ mu_np1[rnp1].T)
            mu_X = mu_np1[rnp1][:q];  mu_Y = mu_np1[rnp1][q:]
            S_XX = Sig[:q, :q];  S_XY = Sig[:q, q:];  S_YY = Sig[q:, q:]

            M_r   = np.linalg.solve(S_YY.T, S_XY.T).T        # (q, s)  eq (21')
            e_x   = mu_X + M_r @ (y_new - mu_Y)               # (q, 1)
            var_x = _psd_floor(_sym(S_XX - M_r @ S_XY.T))                 # (q, q)  eq (22)

            E_x_r.append(e_x)
            Var_x_r.append(var_x)

        # ---- (8),(9)  Marginal estimates ----------------------------------
        E_x, E_xx = _mix(pi_np1, E_x_r, Var_x_r, K)

        # ---- State update --------------------------------------------------
        self._pi  = pi_np1
        self._P_z = P_np1
        self._mu  = mu_np1

        return FilterResult(n=self._n, E_x=E_x, E_xx=E_xx, pi=pi_np1)

    # ------------------------------------------------------------------
    # Batch run methods
    # ------------------------------------------------------------------

    def run_csv(self, csv_path: str | pathlib.Path) -> pd.DataFrame:
        """
        Run the filter on observations stored in a CSV file.

        The file must have columns ``y_0, …, y_{s-1}`` (and optionally
        ``x_0, …, x_{q-1}`` to compute per-step squared error).

        Parameters
        ----------
        csv_path : str or Path

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
        seed:       int | None              = None,
        output_dir: str | pathlib.Path | None = None,
        model_name: str                     = "gss",
    ) -> tuple[pathlib.Path | None, pd.DataFrame]:
        """
        Simulate N steps and run the filter jointly.

        Parameters
        ----------
        N          : number of time steps
        seed       : RNG seed (None = non-deterministic)
        output_dir : if given, write the simulation CSV there
        model_name : label used in the CSV filename

        Returns
        -------
        (sim_path, filter_df)
            sim_path   : Path to the simulation CSV, or None.
            filter_df  : DataFrame — columns described below.

        Output columns
        --------------
        n              time index
        E_x_0…E_x_{q-1}   filtered mean E[X_n | y_{1:n}]
        V_x_0…V_x_{q-1}   posterior variance diagonal (EQMM)
        p_r_0…p_r_{K-1}   regime probabilities p(r_n | y_{1:n})
        sq_err             ||X_n − E[X_n|y_{1:n}]||²
        """
        p = self._p
        q, s = p.q, p.s
        self.reset()

        sim = GSSSimulator(p, N=N, seed=seed)

        sim_rows:    list[list] = []
        filter_rows: list[dict] = []
        t0 = time.perf_counter()

        for n, r, x, y in sim:
            result = self.step(y)
            e_x    = result.E_x.ravel()
            var_x  = result.Var_x.diagonal()

            sim_rows.append([n, r] + x.ravel().tolist() + y.ravel().tolist())

            row: dict = {"n": n}
            row.update({f"E_x_{i}": float(e_x[i])       for i in range(q)})
            row.update({f"V_x_{i}": float(var_x[i])      for i in range(q)})
            row.update({f"p_r_{k}": float(result.pi[k])  for k in range(p.K)})
            row["sq_err"] = float(np.sum((x.ravel() - e_x) ** 2))
            filter_rows.append(row)

        logger.info("Filter complete: %d steps in %.3f s.", N, time.perf_counter() - t0)

        # Optionally save simulation CSV
        sim_path: pathlib.Path | None = None
        if output_dir is not None:
            output_dir = pathlib.Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            seed_str = str(seed) if seed is not None else "random"
            fname    = f"simulated_{model_name}_N{N}_seed{seed_str}.csv"
            sim_path = output_dir / fname
            header   = (
                ["n", "r"]
                + [f"x_{i}" for i in range(q)]
                + [f"y_{i}" for i in range(s)]
            )
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
        has_x  = all(c in df_in.columns for c in x_cols)

        rows: list[dict] = []
        for _, row_in in df_in.iterrows():
            y      = row_in[y_cols].to_numpy(dtype=float).reshape(-1, 1)
            result = self.step(y)
            e_x    = result.E_x.ravel()
            var_x  = result.Var_x.diagonal()

            row: dict = {
                "n": int(row_in["n"]) if "n" in row_in.index else result.n
            }
            row.update({f"E_x_{i}": float(e_x[i])       for i in range(q)})
            row.update({f"V_x_{i}": float(var_x[i])      for i in range(q)})
            row.update({f"p_r_{k}": float(result.pi[k])  for k in range(p.K)})
            if has_x:
                x_true     = row_in[x_cols].to_numpy(dtype=float)
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
    def n(self) -> int:
        """Number of observations processed so far."""
        return self._n

    def __repr__(self) -> str:
        return (
            f"<GSSFilter(K={self._p.K}, q={self._p.q}, "
            f"s={self._p.s}, n={self._n})>"
        )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _sym(M: np.ndarray) -> np.ndarray:
    """Symmetrise a square matrix to counteract floating-point drift."""
    return 0.5 * (M + M.T)


def _psd_floor(M: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """
    Project M onto the PSD cone by flooring negative eigenvalues at *eps*.

    This corrects the tiny negative values that arise from catastrophic
    cancellation in Schur-complement computations (e.g. near-unit-root
    systems where a large variance is subtracted from another large variance).
    For 1×1 matrices the operation reduces to a simple max().
    """
    if M.shape == (1, 1):
        return np.maximum(M, eps)
    vals, vecs = np.linalg.eigh(_sym(M))
    if vals.min() >= eps:
        return M                        # already PSD — nothing to do
    return vecs @ np.diag(np.maximum(vals, eps)) @ vecs.T


def _mix(
    pi:      np.ndarray,
    E_x_r:   list[np.ndarray],
    Var_x_r: list[np.ndarray],
    K:       int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Marginalise over regimes (eqs 8–9).

    E[X|y]     = Σ_r π(r) E[X|r,y]
    E[XX^T|y]  = Σ_r π(r) (Var[X|r,y] + E[X|r,y] E[X|r,y]^T)
    """
    E_x  = sum(pi[r] * E_x_r[r] for r in range(K))
    E_xx = sum(
        pi[r] * (Var_x_r[r] + E_x_r[r] @ E_x_r[r].T) for r in range(K)
    )
    return E_x, _sym(E_xx)
