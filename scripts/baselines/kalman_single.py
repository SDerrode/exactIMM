#!/usr/bin/env python3
"""
scripts/baselines/kalman_single.py
==================================
Baseline B of the paper — single-regime Kalman filter (K=1, no regime
switching). Fitted on training data by aggregating all regimes into one,
then run on the test data.

The model reduces to the pairwise-Markov linear-Gaussian GSS with K=1:
    Z_{n+1} = F Z_n + b + W_{n+1}
where Z = [X; Y], F = [[A, B], [C, D]], W ~ N(0, Σ_W).

Interface mirrors the minimal subset of :class:`GSSFilter` needed by the
E2 driver: an iterable of one-step filter updates exposing
``E_x`` (q,1), ``log_lik`` (float) per step.
"""

from __future__ import annotations

import dataclasses

import numpy as np

__all__ = ["SingleKalmanFilter"]


@dataclasses.dataclass
class _KalmanStepResult:
    E_x:     np.ndarray   # (q, 1)
    log_lik: float


class SingleKalmanFilter:
    """
    Single-regime Kalman filter for the GSS model reduced to K=1.

    The state to filter is X_n ∈ R^q. At step n+1 we observe Y_{n+1} but
    the previous observation Y_n is also needed (it enters the dynamics
    via B and the emission via D), so the filter keeps the previous Y
    internally.

    Parameters
    ----------
    A, B : ndarray (q,q) , (q,s)
        Dynamics of X_{n+1} = A X_n + B Y_n + b_X + U.
    C, D : ndarray (s,q) , (s,s)
        Emission     Y_{n+1} = C X_n + D Y_n + b_Y + V.
    Sigma_U : ndarray (q,q)
    Delta   : ndarray (q,s)
    Sigma_V : ndarray (s,s)
    b_X     : ndarray (q,1)
    b_Y     : ndarray (s,1)
    mu_x0   : ndarray (q,1)
    Sigma_x0: ndarray (q,q)
    """

    def __init__(
        self,
        A: np.ndarray, B: np.ndarray,
        C: np.ndarray, D: np.ndarray,
        Sigma_U: np.ndarray, Delta: np.ndarray, Sigma_V: np.ndarray,
        b_X: np.ndarray, b_Y: np.ndarray,
        mu_x0: np.ndarray, Sigma_x0: np.ndarray,
        mu_y0: np.ndarray, Sigma_y0: np.ndarray,
        Sigma_xy0: np.ndarray,
    ) -> None:
        self.A, self.B, self.C, self.D = map(np.asarray, (A, B, C, D))
        self.Sigma_U = np.asarray(Sigma_U)
        self.Delta   = np.asarray(Delta)
        self.Sigma_V = np.asarray(Sigma_V)
        self.b_X = np.asarray(b_X).reshape(-1, 1)
        self.b_Y = np.asarray(b_Y).reshape(-1, 1)
        self.q = A.shape[0]
        self.s = C.shape[0]

        # Priors for n=1 treated consistently with GSSFilter: condition
        # (X_1, Y_1) ~ N([μ_x0; μ_y0], [[Σ_x0, Σ_xy0], [Σ_xy0^T, Σ_y0]])
        # on Y_1 = y_1 to get μ_{1|1}, P_{1|1}.
        self._mu_x0   = np.asarray(mu_x0).reshape(-1, 1)
        self._Sx0     = np.asarray(Sigma_x0)
        self._mu_y0   = np.asarray(mu_y0).reshape(-1, 1)
        self._Sy0     = np.asarray(Sigma_y0)
        self._Sxy0    = np.asarray(Sigma_xy0)

        self._n  = 0
        self._mu = None      # (q,1) X_{n|n}
        self._P  = None      # (q,q) Var(X_n | y_{1:n})
        self._y_prev = None  # (s,1)

    # ------------------------------------------------------------------
    @classmethod
    def from_regressed(cls, xs: np.ndarray, ys: np.ndarray) -> SingleKalmanFilter:
        """
        Least-squares fit of F = [[A,B],[C,D]], b = [b_X; b_Y], Σ_W on
        the full training series, then build a single-Kalman baseline.
        """
        xs = np.asarray(xs)
        ys = np.asarray(ys)
        if xs.ndim == 1:
            xs = xs[:, None]
        if ys.ndim == 1:
            ys = ys[:, None]
        Z = np.hstack([xs, ys])
        Zc = Z[:-1]       # (N-1, q+s) = Z_n
        Zn = Z[1:]        # Z_{n+1}
        N  = Zc.shape[0]
        q  = xs.shape[1]
        s  = ys.shape[1]

        # Full-model design: [Z_n; 1]  (adds column of ones for intercept)
        X_design = np.hstack([Zc, np.ones((N, 1))])
        # OLS: Zn = X_design @ Θ   → Θ has shape (dim_z+1, dim_z)
        theta, *_ = np.linalg.lstsq(X_design, Zn, rcond=None)
        F  = theta[:q + s, :].T          # (dim_z, dim_z)
        b  = theta[q + s:, :].T          # (dim_z, 1)
        resid = Zn - X_design @ theta
        Sigma_W = (resid.T @ resid) / max(N - (q + s + 1), 1)

        A = F[:q, :q]
        B = F[:q, q:]
        C = F[q:, :q]
        D = F[q:, q:]
        Sigma_U = Sigma_W[:q, :q]
        Delta   = Sigma_W[:q, q:]
        Sigma_V = Sigma_W[q:, q:]
        b_X     = b[:q].reshape(q, 1)
        b_Y     = b[q:].reshape(s, 1)

        # Priors from sample moments of Z
        mu_z    = Z.mean(axis=0).reshape(-1, 1)
        Sigma_z = np.cov(Z, rowvar=False)
        Sigma_z = np.atleast_2d(Sigma_z)
        mu_x0    = mu_z[:q]
        mu_y0    = mu_z[q:]
        Sigma_x0 = Sigma_z[:q, :q]
        Sigma_y0 = Sigma_z[q:, q:]
        Sigma_xy0 = Sigma_z[:q, q:]

        return cls(
            A=A, B=B, C=C, D=D,
            Sigma_U=Sigma_U, Delta=Delta, Sigma_V=Sigma_V,
            b_X=b_X, b_Y=b_Y,
            mu_x0=mu_x0, Sigma_x0=Sigma_x0,
            mu_y0=mu_y0, Sigma_y0=Sigma_y0,
            Sigma_xy0=Sigma_xy0,
        )

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self._n  = 0
        self._mu = None
        self._P  = None
        self._y_prev = None

    # ------------------------------------------------------------------
    def step(self, y: np.ndarray) -> _KalmanStepResult:
        """Process one observation Y_{n+1}, return (E[X_{n+1}|y_{1:n+1}], log_lik_incr)."""
        y = np.asarray(y, dtype=float).reshape(-1, 1)

        if self._n == 0:
            # --- Initial step: condition prior on Y_1 = y ---
            Sy_inv = np.linalg.inv(self._Sy0)
            mu_x1 = self._mu_x0 + self._Sxy0 @ Sy_inv @ (y - self._mu_y0)
            P_x1  = self._Sx0 - self._Sxy0 @ Sy_inv @ self._Sxy0.T
            # log p(y_1) = N(μ_y0, Σ_y0)
            diff = y - self._mu_y0
            sign, logdet = np.linalg.slogdet(self._Sy0)
            quad = float((diff.T @ Sy_inv @ diff).ravel()[0])
            ll = -0.5 * (self.s * np.log(2 * np.pi) + logdet + quad)
            self._mu, self._P = mu_x1, P_x1
            self._y_prev = y
            self._n = 1
            return _KalmanStepResult(E_x=mu_x1.copy(), log_lik=ll)

        # --- Prediction of Z_{n+1} given y_{1:n} ---
        mu_x_pred = self.A @ self._mu + self.B @ self._y_prev + self.b_X
        mu_y_pred = self.C @ self._mu + self.D @ self._y_prev + self.b_Y
        A, C = self.A, self.C
        P = self._P
        P_xx = A @ P @ A.T + self.Sigma_U
        P_xy = A @ P @ C.T + self.Delta
        P_yy = C @ P @ C.T + self.Sigma_V
        P_yy = 0.5 * (P_yy + P_yy.T)

        # --- Kalman update on Y_{n+1} ---
        Sy_inv = np.linalg.inv(P_yy)
        innov = y - mu_y_pred
        mu_x_new = mu_x_pred + P_xy @ Sy_inv @ innov
        P_new    = P_xx - P_xy @ Sy_inv @ P_xy.T
        P_new    = 0.5 * (P_new + P_new.T)

        # --- Incremental log-likelihood log p(y_{n+1} | y_{1:n}) ---
        sign, logdet = np.linalg.slogdet(P_yy)
        quad = float((innov.T @ Sy_inv @ innov).ravel()[0])
        ll = -0.5 * (self.s * np.log(2 * np.pi) + logdet + quad)

        self._mu, self._P = mu_x_new, P_new
        self._y_prev = y
        self._n += 1
        return _KalmanStepResult(E_x=mu_x_new.copy(), log_lik=ll)
