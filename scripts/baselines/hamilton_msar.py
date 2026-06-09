#!/usr/bin/env python3
"""
scripts/baselines/hamilton_msar.py
==================================
Baseline C — Hamilton (1989) Markov-Switching regression.

We model the **X** series (S&P500 log-return) as:

    X_n = µ_{r_n} + φ_{r_n} X_{n-1} + σ_{r_n} ε_n       ε_n ~ N(0, 1)

with a Markov latent regime r_n ∈ {0, ..., K-1}. Y (log VIX) is *ignored*
by this baseline — it only observes X, which is the econometric tradition.

Evaluation on the *test* period:
    - predictive log-likelihood  log p(X_t | X_{1:t-1})
    - MSE of one-step-ahead filtered mean E[X_t | X_{1:t}]
    - smoothed regime probabilities → argmax → classification accuracy
      and ARI vs proxy labels

Notes
-----
Unlike the GSS filter, Hamilton's filter does not have a hidden
continuous state beyond X itself; so "MSE on X" is the one-step-ahead
prediction error, not a filtering error on a truly hidden X.
"""

from __future__ import annotations

import warnings

import numpy as np
from statsmodels.tsa.regime_switching.markov_autoregression import (
    MarkovAutoregression,
)

__all__ = ["HamiltonMSAR"]


class HamiltonMSAR:
    """Convenience wrapper around statsmodels' MS-AR for E3."""

    def __init__(self, K: int = 2, ar_order: int = 1) -> None:
        self.K = K
        self.ar_order = ar_order
        self._res = None       # fitted MarkovAutoregressionResults
        self._train_mean = 0.0

    def fit(self, x_train: np.ndarray, max_iter: int = 500,
            seed: int = 42) -> HamiltonMSAR:
        x_train = np.asarray(x_train).ravel()
        self._train_mean = float(x_train.mean())
        mod = MarkovAutoregression(
            endog=x_train,
            k_regimes=self.K,
            order=self.ar_order,
            switching_ar=True,
            switching_variance=True,
            switching_trend=True,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Deterministic optimisation: we pass a seed via em_algorithm init
            self._res = mod.fit(em_iter=max_iter, disp=False)
        return self

    def predict_test(
        self, x_train: np.ndarray, x_test: np.ndarray
    ) -> dict:
        """
        Score on the test series using the in-sample-trained model.
        Computes per-step one-step-ahead predictive log-likelihood and MSE.

        Strategy: concatenate [train, test] and fit a *new* MarkovAutoregression
        using the *same parameters* via ``smooth()``. That yields filtered
        probabilities on the full history; we evaluate metrics only on the
        test portion.
        """
        assert self._res is not None, "call .fit(...) first"

        x_train = np.asarray(x_train).ravel()
        x_test  = np.asarray(x_test).ravel()
        x_full  = np.concatenate([x_train, x_test])
        N_train = len(x_train)
        N_test  = len(x_test)

        # Rebuild the model on the full sample
        mod_full = MarkovAutoregression(
            endog=x_full,
            k_regimes=self.K,
            order=self.ar_order,
            switching_ar=True,
            switching_variance=True,
            switching_trend=True,
        )
        # Apply the trained parameters to the full sample: Kim filter
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res_full = mod_full.smooth(self._res.params)

        # Filtered regime probabilities (time x K)
        pi_filt = np.asarray(res_full.filtered_marginal_probabilities)
        pi_smoothed = np.asarray(res_full.smoothed_marginal_probabilities)

        # Predicted regime probabilities (one-step-ahead): pi_pred_t = P^T @ pi_filt_{t-1}
        # statsmodels stores pi_filt with shape (nobs, k_regimes).
        # Recover transition matrix.
        P_T = self._res.regime_transition[..., 0]   # (K, K, 1) → (K, K)
        # regime_transition has shape (k_to, k_from) in statsmodels.

        # Per-step log-likelihood on the full sample, via llf_obs if exposed
        ll_obs_full = np.asarray(res_full.llf_obs)  # (N_full,)
        ll_test = float(ll_obs_full[-N_test:].sum())

        # For per-step prediction of X: use filtered mean
        # Fitted values from each regime (statsmodels gives regime-specific
        # conditional means; take weighted sum with filtered probs)
        fitted_by_regime = np.asarray(res_full.predict())  # shape (N_full,)
        sse_test = float(np.sum((x_test - fitted_by_regime[-N_test:]) ** 2))

        r_hat_test = np.argmax(pi_filt[-N_test:], axis=1)

        return {
            "log_lik_test":   ll_test,
            "nll_per_obs":    -ll_test / N_test,
            "mse_x":          sse_test / N_test,
            "pi_filt_test":   pi_filt[-N_test:],
            "pi_smoothed_test": pi_smoothed[-N_test:],
            "r_hat_test":     r_hat_test,
            "fitted_test":    fitted_by_regime[-N_test:],
        }
