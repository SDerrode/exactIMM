#!/usr/bin/env python3
"""
prg/gui/workers.py
==================
Background ``QThread`` workers for the GSS GUI: ``_SimWorker`` (runs the
simulator) and ``_FilterWorker`` (runs the filter step-by-step and exposes
the pre-computed conditional moments for diagnostics). Extracted verbatim
from ``prg/gui/main_window.py``.
"""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from prg.classes.GSSParams import GSSParams
from prg.classes.GSSSimulator import GSSSimulator
from prg.filter.gss_filter import GSSFilter


class _SimWorker(QThread):
    """Run GSSSimulator in a background thread."""

    finished = pyqtSignal(list, list, object, object)  # ns, rs, xs, ys
    error = pyqtSignal(str)

    def __init__(self, params: GSSParams, N: int, seed: int | None, parent=None):
        super().__init__(parent)
        self._params = params
        self._N = N
        self._seed = seed

    def run(self) -> None:
        try:
            sim = GSSSimulator(self._params, N=self._N, seed=self._seed)
            ns, rs = [], []
            xs_rows, ys_rows = [], []
            # Check interruption every CHECK_EVERY iterations to keep overhead low
            CHECK_EVERY = 256
            for i, (n, r, x, y) in enumerate(sim):
                if (i & (CHECK_EVERY - 1)) == 0 and self.isInterruptionRequested():
                    return  # silent abort: no signal emitted
                ns.append(n)
                rs.append(r)
                xs_rows.append(x.ravel())
                ys_rows.append(y.ravel())
            xs = np.array(xs_rows)
            ys = np.array(ys_rows)
            self.finished.emit(ns, rs, xs, ys)
        except Exception as exc:  # noqa: BLE001
            self.error.emit(str(exc))


class _FilterWorker(QThread):
    """Run GSSFilter step-by-step in a background thread."""

    # E_xs (N,q), Var_xs (N,q), pis (N,K), innovations (N,s), log_lik_total (float)
    finished = pyqtSignal(object, object, object, object, float)
    progress = pyqtSignal(int, int)  # (n_done, N_total) — D8
    error = pyqtSignal(str)

    def __init__(
        self,
        params: GSSParams,
        ys: np.ndarray,  # (N, s)
        joseph: bool = False,
        mode: str = "imm_general",
        parent=None,
    ):
        super().__init__(parent)
        self._params = params
        self._ys = ys
        self._joseph = joseph
        self._mode = mode

    def run(self) -> None:
        try:
            filt = GSSFilter(self._params, joseph=self._joseph, mode=self._mode)
            E_xs_list: list[np.ndarray] = []
            Var_xs_list: list[np.ndarray] = []
            pis_list: list[np.ndarray] = []
            innov_list: list[np.ndarray] = []
            log_lik_total: float = 0.0
            N = len(self._ys)
            PROGRESS_EVERY = max(1, N // 50)  # D8: ~50 progress updates
            CHECK_EVERY = 256
            for i, y_row in enumerate(self._ys):
                if (i & (CHECK_EVERY - 1)) == 0 and self.isInterruptionRequested():
                    return
                if i % PROGRESS_EVERY == 0:
                    self.progress.emit(i, N)
                res = filt.step(y_row.reshape(-1, 1))
                E_xs_list.append(res.E_x.ravel())
                Var_xs_list.append(res.Var_x.diagonal())
                pis_list.append(res.pi)
                innov_list.append(res.innovation.ravel())
                if np.isfinite(res.log_lik):
                    log_lik_total += float(res.log_lik)

            # Expose pre-computed moments for diagnostics.
            # Stationary moments µ_Y[k] and Σ_YY(k) are always available (both modes
            # call _precompute_stationary()).  h5_exact-specific keys (mu_Y_jk, …) are
            # added only when the filter was run in h5_exact mode.
            self.cond_moments: dict = {}
            if hasattr(filt, "_mu_Y") and hasattr(filt, "_S_YY"):
                self.cond_moments["mu_Y"] = filt._mu_Y  # [K] ndarray (s,1)
                self.cond_moments["S_YY"] = filt._S_YY  # [K] ndarray (s,s)

            if hasattr(filt, "_mu_Y_jk"):
                # ── h5_exact: Signal 2 — exact Markov transition density ──────
                # Per the Markovianité Proposition (formulas (f) and (h)):
                #   μ₂(j,k) = b_Y(k) + (D_k + C_k Δ_j Σ_{V,j}^{-1}) y_n      (f)
                #   Γ₂(j,k) = Σ_{V,k} + C_k (Σ_{U,j} − Δ_j Σ_{V,j}^{-1} Δ_j^T) C_k^T  (h)
                p = self._params
                K, q = p.K, p.q
                nc = p.noise_cov
                M_simple = [[None] * K for _ in range(K)]
                Gamma2 = [[None] * K for _ in range(K)]
                b_Y = [p.b(k)[q:] for k in range(K)]  # [K] ndarray (s,1)
                for j in range(K):
                    SV_j = nc.Sigma_V(j)  # (s,s)
                    SV_j_inv = np.linalg.inv(SV_j)
                    D_j = nc.Delta(j)  # (q,s)
                    SU_j = nc.Sigma_U(j)  # (q,q)
                    Schur_j = SU_j - D_j @ SV_j_inv @ D_j.T  # (q,q)
                    for k in range(K):
                        F_k = p.f_matrix.F(k)
                        C_k = F_k[q:, :q]  # (s,q)
                        D_k = F_k[q:, q:]  # (s,s)
                        SV_k = nc.Sigma_V(k)  # (s,s)
                        M_simple[j][k] = D_k + C_k @ D_j @ SV_j_inv  # (s,s)
                        Gamma2[j][k] = SV_k + C_k @ Schur_j @ C_k.T  # (s,s)

                # Stationary mixture weights w_{jk} = π_∞(j) P(j,k)  (K,K)
                mix_w = filt._pi_inf[:, None] * p.P

                self.cond_moments.update(
                    {
                        "mu_Y_jk": filt._mu_Y_jk,
                        "M_t": filt._M_t,
                        "Gamma": filt._Gamma,
                        "M_simple": M_simple,  # (s,s) signal 2 coefficient matrix
                        "Gamma2": Gamma2,  # (s,s) signal 2 constant covariance
                        "b_Y": b_Y,  # [K]   ndarray (s,1) — signal 2 bias
                        "mix_w": mix_w,  # (K,K) stationary mixture weights
                    }
                )

            self.finished.emit(
                np.array(E_xs_list),
                np.array(Var_xs_list),
                np.array(pis_list),
                np.array(innov_list),
                log_lik_total,
            )
        except Exception as exc:  # noqa: BLE001
            self.error.emit(str(exc))
