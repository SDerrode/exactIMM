#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prg/gui/main_window.py
======================
GSSMainWindow — top-level application window.

Layout
------
  ┌─────────────────────┬───────────────────────────────────────┐
  │  ParamPanel         │  PlotPanel                            │
  │  (tabs F(k)/Σ_W(k)) │                                       │
  │  ─────────────────  │                                       │
  │  N  [spinbox]       │                                       │
  │  Seed  [lineedit]   │                                       │
  │  [Simuler]          │                                       │
  │  [Enregistrer CSV]  │                                       │
  └─────────────────────┴───────────────────────────────────────┘

Fixed (non-editable) parameters
--------------------------------
  P       = uniform (1/K) or model's P
  pi0     = None → stationary distribution
  mu_z0   = zeros
  Sigma_z0= I_{q+s}
"""

import csv
import datetime
import pathlib
import time
from dataclasses import dataclass, field
import numpy as np
from scipy.stats import chi2 as _chi2_dist
from scipy.stats import jarque_bera as _jb

from prg.filter.gss_filter import GSSFilter

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSettings
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QGridLayout,
    QLabel, QSpinBox, QLineEdit, QPushButton,
    QDialog, QMessageBox, QSizePolicy, QSplitter,
    QFileDialog, QFrame, QComboBox, QCheckBox, QProgressBar,
    QTabWidget,
)

# ---------------------------------------------------------------------------
# White-noise test (Ljung-Box)
# ---------------------------------------------------------------------------

_PILL_OK    = ("font-size: 10px; padding: 2px 8px; border-radius: 3px;"
               "background: #d4edda; color: #155724; border: 1px solid #c3e6cb;")
_PILL_WARN  = ("font-size: 10px; padding: 2px 8px; border-radius: 3px;"
               "background: #fff3cd; color: #856404; border: 1px solid #ffc107;")
_PILL_ERR   = ("font-size: 10px; padding: 2px 8px; border-radius: 3px;"
               "background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb;")


def _ljung_box(x: np.ndarray, lags: int | None = None) -> tuple[float, float, int]:
    """
    Ljung-Box portmanteau test for autocorrelation.

    H₀: no autocorrelation up to lag *h*.

    Returns (Q_stat, p_value, h_used).
    High p-value → fail to reject H₀ → innovation looks like white noise.
    """
    n = len(x)
    h = lags if lags is not None else min(20, max(5, n // 10))
    h = min(h, n - 1)
    x = x - x.mean()
    var = float(np.var(x))
    if var < 1e-30:                         # constant series
        return 0.0, 1.0, h
    # Sample autocorrelations ρ_k, k = 1…h
    rho_sq = np.array([
        (np.dot(x[k:], x[:-k]) / (n * var)) ** 2
        for k in range(1, h + 1)
    ])
    Q = float(n * (n + 2) * np.sum(rho_sq / (n - np.arange(1, h + 1))))
    p = float(1.0 - _chi2_dist.cdf(Q, df=h))
    return Q, p, h


def _shape_diagnostics(x: np.ndarray) -> tuple[float, float, float, float]:
    """
    Compute sample shape diagnostics for an innovation series.

    Returns ``(skewness, excess_kurtosis, JB_stat, p_value)``.

    - Skewness  S        : 3rd standardised moment.   |S| ≈ 0 ⇔ symmetric.
    - Excess kurtosis K  : 4th std. moment − 3.       |K| ≈ 0 ⇔ Gaussian tails.
    - Jarque–Bera (JB)   : combined test, JB ~ χ²(2) under H₀ "Gaussian".
                           Computed via SciPy's reference implementation.

    Applied to *standardised* innovations these diagnostics are meaningful for
    GSS models: if S ≈ 0 and K ≈ 0 the filter is well-calibrated.
    """
    n = len(x)
    if n < 4:
        return 0.0, 0.0, 0.0, 1.0
    xc = np.asarray(x, dtype=float) - float(np.mean(x))
    sigma2 = float(np.mean(xc ** 2))
    if sigma2 < 1e-30:                      # constant series
        return 0.0, 0.0, 0.0, 1.0
    sigma = float(np.sqrt(sigma2))
    z = xc / sigma
    S    = float(np.mean(z ** 3))
    Kurt = float(np.mean(z ** 4) - 3.0)
    try:
        jb_res = _jb(x)                    # SciPy returns (statistic, pvalue)
        JB     = float(jb_res[0])
        p      = float(jb_res[1])
    except Exception:                       # noqa: BLE001
        # Manual fallback (matches SciPy formula exactly)
        JB = n * (S ** 2 / 6.0 + Kurt ** 2 / 24.0)
        p  = float(1.0 - _chi2_dist.cdf(JB, df=2))
    return S, Kurt, JB, p


# ---------------------------------------------------------------------------
# Innovation standardisation (A12 / D1)
# ---------------------------------------------------------------------------


def _standardise_innovations(
    innovations: np.ndarray,
    mix_w: np.ndarray | None,
    Gamma: list | None,
    mu_Y_jk: list | None,
) -> np.ndarray:
    """
    Whiten innovations by the (approximate) marginal innovation covariance S.

    Two modes:

    * h5_exact (all three extra arguments provided):
      S = Σ_{j,k} w_{jk} [Γ(j,k) + δ_{jk} δ_{jk}ᵀ]
      where δ_{jk} = μ_{Y,jk} − Σ w μ_{Y} is the deviation of the
      component mean from the mixture mean.  S is the *stationary*
      marginal innovation covariance.

    * imm_general (extra arguments None):
      S is estimated from the sample covariance of the innovations.

    Returns  ν̃ = L⁻¹ ν   (shape same as *innovations*), where S = L Lᵀ.
    Under a well-calibrated filter each ν̃ᵢ is approximately N(0, 1).
    """
    s = innovations.shape[1]

    if mix_w is not None and Gamma is not None and mu_Y_jk is not None:
        # Stationary marginal covariance
        K_mix = mix_w.shape[0]
        mu_marg = np.zeros((s, 1))
        for j in range(K_mix):
            for k in range(K_mix):
                mu_marg += float(mix_w[j, k]) * mu_Y_jk[j][k]

        S = np.zeros((s, s))
        for j in range(K_mix):
            for k in range(K_mix):
                w = float(mix_w[j, k])
                if w < 1e-12:
                    continue
                delta = mu_Y_jk[j][k] - mu_marg       # (s, 1)
                S += w * (Gamma[j][k] + delta @ delta.T)
    else:
        # Sample covariance fallback (imm_general mode)
        raw = innovations.T    # (s, N)
        S = np.cov(raw) if s > 1 else np.array([[float(np.var(innovations[:, 0]))]])

    # Cholesky + solve: ν̃ = L⁻¹ ν  →  each column has unit variance
    try:
        L = np.linalg.cholesky(_sym_reg(S, 1e-10))
        return np.linalg.solve(L, innovations.T).T   # (N, s)
    except np.linalg.LinAlgError:
        return innovations   # give up and return raw


def _sym_reg(M: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Return (M + Mᵀ)/2 + eps·I  (symmetrised + regularised)."""
    n = M.shape[0]
    return 0.5 * (M + M.T) + eps * np.eye(n)


# ---------------------------------------------------------------------------
# Stationary distribution helper
# ---------------------------------------------------------------------------

def _stationary_dist(P: np.ndarray) -> np.ndarray | None:
    """Stationary distribution of a row-stochastic matrix (left eigenvector)."""
    try:
        vals, vecs = np.linalg.eig(P.T)
        idx = int(np.argmin(np.abs(vals - 1.0)))
        pi = np.real(vecs[:, idx])
        pi = np.maximum(pi, 0.0)
        s  = pi.sum()
        return pi / s if s > 1e-12 else None
    except np.linalg.LinAlgError:
        return None


# ---------------------------------------------------------------------------
# Histogram dialogs
# ---------------------------------------------------------------------------

class _InnovHistDialog(QDialog):
    """Non-modal dialog: innovation histograms, ACF, and scatter (D4/D5/D12).

    Tabs
    ----
    • Histograms — raw innovation distributions + theoretical Gaussian mixture
      (h5_exact mode) or best-fit N (imm_general mode).
    • ACF        — sample autocorrelation function per component with
                   Ljung-Box 95 % confidence bands (±1.96 / √N).
    • Scatter    — pairwise scatter plot (only shown when s ≥ 2).

    When *mix_params* is supplied (h5_exact mode), the histogram shows:
        p(ν) = Σ_{j,k} w_{jk} N(ν ; δ̃_{jk}, Γ_{jk})
    where δ̃_{jk} = μ_{Y,jk}[i] − Σ_{k'} P(j,k') μ_{Y,jk'}[i] is the
    within-previous-regime centred component mean.
    """

    def __init__(
        self,
        innovations: np.ndarray,
        mix_params: dict | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Innovation diagnostics")
        self.setModal(False)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self._innovations = innovations

        from matplotlib.backends.backend_qtagg import (
            FigureCanvasQTAgg, NavigationToolbar2QT,
        )
        from matplotlib.figure import Figure
        from scipy.stats import norm as _norm

        s      = innovations.shape[1]
        N      = innovations.shape[0]
        layout = QVBoxLayout(self)

        inner_tabs = QTabWidget()
        layout.addWidget(inner_tabs)

        colours = ["#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

        # ─────────────────────────────────────────────────────────────
        # Tab 0: Histograms
        # ─────────────────────────────────────────────────────────────
        hist_w = QWidget()
        hist_l = QVBoxLayout(hist_w)
        hist_l.setContentsMargins(0, 0, 0, 0)
        fig_h   = Figure(figsize=(max(4, 3.5 * s), 3.8), tight_layout=True)
        can_h   = FigureCanvasQTAgg(fig_h)
        hist_l.addWidget(NavigationToolbar2QT(can_h, hist_w))
        hist_l.addWidget(can_h)

        for i in range(s):
            ax = fig_h.add_subplot(1, s, i + 1)
            x  = innovations[:, i]
            ax.hist(x, bins=40, density=True,
                    color=colours[i % len(colours)], alpha=0.70,
                    label="empirical")

            # x-grid slightly wider than the data range
            pad = 0.5 * float(x.std()) if x.std() > 1e-10 else 0.5
            xg  = np.linspace(float(x.min()) - pad, float(x.max()) + pad, 400)

            if mix_params is not None:
                # ── Theoretical Gaussian mixture ──────────────────────────
                # p(ν) = Σ_{j,k} w_{jk} N(ν ; δ̃_{jk}, Γ_{jk})
                #
                # δ̃_{jk} = μ_{Y,jk}[i] − Σ_{k'} P(j,k') μ_{Y,jk'}[i]
                #         = within-previous-regime centred mean
                w     = mix_params["w"]        # (K, K) — already normalised
                K_mix = w.shape[0]
                Gam   = mix_params["Gamma"]    # [K][K] (s, s)
                muYjk = mix_params["mu_Y_jk"]  # [K][K] (s, 1)

                pdf_mix = np.zeros_like(xg)
                for j in range(K_mix):
                    pi_j = max(float(w[j, :].sum()), 1e-12)
                    prev_mean_j_i = sum(
                        float(w[j, kk]) * float(muYjk[j][kk][i, 0])
                        for kk in range(K_mix)
                    ) / pi_j
                    for k in range(K_mix):
                        wjk = float(w[j, k])
                        if wjk < 1e-10:
                            continue
                        delta = float(muYjk[j][k][i, 0]) - prev_mean_j_i
                        var_i = float(Gam[j][k][i, i])
                        sig   = float(np.sqrt(max(var_i, 1e-12)))
                        pdf_mix += wjk * _norm.pdf(xg, delta, sig)

                ax.plot(xg, pdf_mix, "k-", linewidth=1.8,
                        label=rf"$\sum_{{jk}}w_{{jk}}\,\mathcal{{N}}(\tilde{{\delta}}_{{jk}},\Gamma_{{jk}})$"
                              f"  ({K_mix}² terms)")
            else:
                # ── Fallback: best-fit single Gaussian ───────────────────
                mu_e, sig_e = float(x.mean()), float(x.std())
                if sig_e > 1e-10:
                    ax.plot(xg, _norm.pdf(xg, mu_e, sig_e),
                            "k--", linewidth=1.5,
                            label=f"N({mu_e:.3f}, {sig_e:.3f}²)")

            ax.set_title(rf"$\nu^{i}$   (N = {N})", fontsize=10)
            ax.set_xlabel("value", fontsize=9)
            ax.set_ylabel("density", fontsize=9)
            ax.legend(fontsize=8)
            ax.grid(True, linestyle=":", alpha=0.4)

        inner_tabs.addTab(hist_w, "Histograms")

        # ─────────────────────────────────────────────────────────────
        # Tab 1: ACF (D5)
        # ─────────────────────────────────────────────────────────────
        acf_w = QWidget()
        acf_l = QVBoxLayout(acf_w)
        acf_l.setContentsMargins(0, 0, 0, 0)
        max_lag = min(40, max(5, N // 10))
        fig_a   = Figure(figsize=(max(4, 3.5 * s), 3.2), tight_layout=True)
        can_a   = FigureCanvasQTAgg(fig_a)
        acf_l.addWidget(NavigationToolbar2QT(can_a, acf_w))
        acf_l.addWidget(can_a)
        conf95 = 1.96 / float(np.sqrt(N))

        for i in range(s):
            ax = fig_a.add_subplot(1, s, i + 1)
            x  = innovations[:, i]
            xc = x - x.mean()
            c0 = float(np.dot(xc, xc)) / N
            if c0 > 1e-30:
                acf_vals = np.array([
                    float(np.dot(xc[:N - lag], xc[lag:])) / (N * c0)
                    for lag in range(1, max_lag + 1)
                ])
            else:
                acf_vals = np.zeros(max_lag)
            lags = np.arange(1, max_lag + 1)
            ax.bar(lags, acf_vals, color=colours[i % len(colours)],
                   alpha=0.75, width=0.8)
            ax.axhline(0,       color="k",       linewidth=0.6)
            ax.axhline( conf95, color="#999999",  linewidth=1.0,
                        linestyle="--", label=f"±1.96/√N")
            ax.axhline(-conf95, color="#999999",  linewidth=1.0,
                        linestyle="--")
            ax.set_xlim(0, max_lag + 1)
            ax.set_ylim(-1.05, 1.05)
            ax.set_xlabel("lag", fontsize=9)
            ax.set_ylabel("ACF", fontsize=9)
            ax.set_title(rf"ACF  $\nu^{i}$", fontsize=10)
            ax.legend(fontsize=7)
            ax.grid(True, linestyle=":", alpha=0.4)

        inner_tabs.addTab(acf_w, "ACF")

        # ─────────────────────────────────────────────────────────────
        # Tab 2: Pairwise scatter (D4, only when s ≥ 2)
        # ─────────────────────────────────────────────────────────────
        if s >= 2:
            sc_w = QWidget()
            sc_l = QVBoxLayout(sc_w)
            sc_l.setContentsMargins(0, 0, 0, 0)
            n_pairs = s * (s - 1) // 2
            ncols   = min(n_pairs, 3)
            nrows   = (n_pairs + ncols - 1) // ncols
            fig_s   = Figure(figsize=(max(4, 3.5 * ncols), 3.2 * nrows),
                             tight_layout=True)
            can_s   = FigureCanvasQTAgg(fig_s)
            sc_l.addWidget(NavigationToolbar2QT(can_s, sc_w))
            sc_l.addWidget(can_s)

            idx = 1
            for a_i in range(s):
                for b_i in range(a_i + 1, s):
                    ax = fig_s.add_subplot(nrows, ncols, idx)
                    ax.scatter(innovations[:, a_i], innovations[:, b_i],
                               s=4, alpha=0.40, color="#555555")
                    rho = float(np.corrcoef(
                        innovations[:, a_i], innovations[:, b_i]
                    )[0, 1])
                    ax.set_xlabel(rf"$\nu^{a_i}$", fontsize=9)
                    ax.set_ylabel(rf"$\nu^{b_i}$", fontsize=9)
                    ax.set_title(
                        rf"$\nu^{a_i}$ vs $\nu^{b_i}$   ρ = {rho:+.3f}",
                        fontsize=9,
                    )
                    ax.grid(True, linestyle=":", alpha=0.4)
                    idx += 1

            inner_tabs.addTab(sc_w, "Scatter")

        # ─────────────────────────────────────────────────────────────
        # Export CSV button (D12)
        # ─────────────────────────────────────────────────────────────
        export_row = QHBoxLayout()
        btn_csv = QPushButton("Export innovations CSV…")
        btn_csv.setFixedHeight(28)
        btn_csv.clicked.connect(self._on_export_csv)
        export_row.addStretch()
        export_row.addWidget(btn_csv)
        layout.addLayout(export_row)

        self.resize(max(480, 360 * min(s, 3)), 480)

    def _on_export_csv(self) -> None:
        """Save innovation columns to a CSV file."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Export innovations",
            "innovations.csv",
            "CSV files (*.csv);;All files (*)",
        )
        if not path:
            return
        try:
            innov = self._innovations
            s = innov.shape[1]
            header = ["n"] + [f"nu_{i}" for i in range(s)]
            with open(path, "w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow(header)
                for n, row in enumerate(innov):
                    w.writerow([n] + list(row))
            QMessageBox.information(
                self, "Export OK",
                f"Innovations saved to:\n{pathlib.Path(path).resolve()}",
            )
        except Exception as exc:   # noqa: BLE001
            QMessageBox.critical(self, "Export error", str(exc))


# ---------------------------------------------------------------------------

from prg.classes.FMatrix import FMatrix
from prg.classes.GSSParams import GSSParams
from prg.classes.GSSSimulator import GSSSimulator
from prg.classes.NoiseCovariance import GSSNoiseCovariance
from prg.models.presets import PRESETS
from prg.gui.matrix_widget import StochasticMatrixWidget
from prg.gui.param_panel import ParamPanel
from prg.gui.plot_panel import PlotPanel, PredYPanel


# ---------------------------------------------------------------------------
# Session state — single source of truth for everything produced by Simulate /
# Filter / Monte-Carlo / Load. Replacing the previous handful of scattered
# `_last_*` attributes with one dataclass makes invariants explicit (e.g. the
# innovation array is meaningless without the data that produced it) and
# narrows the API: every state mutation goes through one of the methods below.
# ---------------------------------------------------------------------------


@dataclass
class _SessionState:
    """Holds everything the UI may need to display about the current session.

    Two independent slots:

    * data        — produced by Simulate (single) or Load CSV
    * innovations — produced by Filter, only meaningful with `data` AND `params`

    `params` and `params_signature` are captured at the moment Simulate /
    Load runs, NOT live from the widgets. The Filter step uses them as-is
    even if the widgets have since been edited (drift is signalled in the
    UI but not silently corrected).
    """

    data: tuple | None = None              # (ns, rs, xs, ys, seed_used)
    params: object | None = None           # GSSParams (avoids circular import)
    params_signature: tuple | None = None  # bytes signature of GUI at capture
    innovations: np.ndarray | None = None  # (N, s)

    # ----- Predicates --------------------------------------------------

    def has_data(self) -> bool:
        return self.data is not None

    def has_filter(self) -> bool:
        return self.innovations is not None

    def can_filter(self) -> bool:
        return self.has_data() and self.params is not None

    # ----- Atomic mutations -------------------------------------------

    def reset(self) -> None:
        """Forget everything — called from Reset button."""
        self.data = None
        self.params = None
        self.params_signature = None
        self.innovations = None

    def begin_simulation(self, params: object, signature: tuple | None) -> None:
        """About to launch a new Simulate: capture params, drop stale results."""
        self.params = params
        self.params_signature = signature
        self.innovations = None      # filter result no longer matches

    def store_data(self, ns, rs, xs, ys, seed) -> None:
        self.data = (ns, rs, xs, ys, seed)

    def store_innovations(self, innov: np.ndarray) -> None:
        self.innovations = innov

    def load_external(self, ns, rs, xs, ys, params: object,
                      signature: tuple | None) -> None:
        """User loaded an external CSV: store it with the live GUI params."""
        self.data = (ns, rs, xs, ys, None)
        self.params = params
        self.params_signature = signature
        self.innovations = None


# ---------------------------------------------------------------------------
# Background workers
# ---------------------------------------------------------------------------

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
                    return                       # silent abort: no signal emitted
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
    progress = pyqtSignal(int, int)   # (n_done, N_total) — D8
    error = pyqtSignal(str)

    def __init__(
        self,
        params: GSSParams,
        ys: np.ndarray,   # (N, s)
        joseph: bool = False,
        mode:   str  = "imm_general",
        parent=None,
    ):
        super().__init__(parent)
        self._params = params
        self._ys = ys
        self._joseph = joseph
        self._mode   = mode

    def run(self) -> None:
        try:
            filt = GSSFilter(self._params, joseph=self._joseph, mode=self._mode)
            E_xs_list:    list[np.ndarray] = []
            Var_xs_list:  list[np.ndarray] = []
            pis_list:     list[np.ndarray] = []
            innov_list:   list[np.ndarray] = []
            log_lik_total: float = 0.0
            N = len(self._ys)
            PROGRESS_EVERY = max(1, N // 50)   # D8: ~50 progress updates
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

            # Expose pre-computed moments for the p(y_{n+1}|r_n,r_{n+1},y_n) tab.
            # These attributes only exist in "h5_exact" mode (produced by _precompute()).
            # In "imm_general" mode they are absent: _on_filter_finished checks hasattr()
            # and leaves the tab grayed out.
            if hasattr(filt, "_mu_Y_jk"):
                # ── Signal 2: direct moments from the model matrices ──────────
                # μ₂(j,k) = b_Y(k) + (D_k + C_k Δ_j Σ_{V,j}^{-1}) y_n
                # Γ₂(j,k) = Σ_{V,k} + C_k (Σ_{U,j} − Δ_j Σ_{V,j}^{-1} Δ_j^T) C_k^T
                p   = self._params
                K, q = p.K, p.q
                nc  = p.noise_cov
                M_simple  = [[None] * K for _ in range(K)]
                Gamma2    = [[None] * K for _ in range(K)]
                b_Y       = [p.b(k)[q:]  for k in range(K)]  # [K] ndarray (s,1)
                for j in range(K):
                    SV_j     = nc.Sigma_V(j)                        # (s,s)
                    SV_j_inv = np.linalg.inv(SV_j)
                    D_j      = nc.Delta(j)                          # (q,s)
                    SU_j     = nc.Sigma_U(j)                        # (q,q)
                    Schur_j  = SU_j - D_j @ SV_j_inv @ D_j.T       # (q,q)
                    for k in range(K):
                        F_k = p.f_matrix.F(k)
                        C_k = F_k[q:, :q]                           # (s,q)
                        D_k = F_k[q:, q:]                           # (s,s)
                        SV_k = nc.Sigma_V(k)                        # (s,s)
                        M_simple[j][k] = D_k + C_k @ D_j @ SV_j_inv   # (s,s)
                        Gamma2[j][k]   = SV_k + C_k @ Schur_j @ C_k.T # (s,s)

                # Stationary mixture weights w_{jk} = π_∞(j) P(j,k)  (K,K)
                mix_w = filt._pi_inf[:, None] * p.P

                self.cond_moments: dict = {
                    "mu_Y_jk": filt._mu_Y_jk,
                    "M_t":     filt._M_t,
                    "Gamma":   filt._Gamma,
                    "mu_Y":    filt._mu_Y,
                    "M_simple": M_simple,   # (s,s) signal 2 coefficient matrix
                    "Gamma2":   Gamma2,     # (s,s) signal 2 constant covariance
                    "b_Y":      b_Y,        # [K]   ndarray (s,1) — signal 2 bias
                    "mix_w":    mix_w,      # (K,K) stationary mixture weights
                }

            self.finished.emit(
                np.array(E_xs_list),
                np.array(Var_xs_list),
                np.array(pis_list),
                np.array(innov_list),
                log_lik_total,
            )
        except Exception as exc:  # noqa: BLE001
            self.error.emit(str(exc))



# ---------------------------------------------------------------------------
# Wait dialog (modal, no close button)
# ---------------------------------------------------------------------------

class _WaitDialog(QDialog):
    """Modal wait dialog with progress bar and optional Cancel button.

    Defaults to indeterminate (busy) mode; `set_progress(m, M)` switches to
    a determinate percentage bar and updates it.

    Parameters
    ----------
    message : str
        Title and label text shown in the dialog.
    on_cancel : callable | None
        If provided, a Cancel button is shown. Clicking it calls *on_cancel()*
        then closes the dialog.  Typically ``GSSMainWindow._on_reset``.
    """

    def __init__(self, message: str = "Please wait…", on_cancel=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(message)
        self.setModal(True)
        self.setWindowFlag(Qt.WindowType.WindowCloseButtonHint, False)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(8)

        self._msg_lbl = QLabel(message)
        self._msg_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._msg_lbl)

        self._bar = QProgressBar()
        self._bar.setRange(0, 0)            # indeterminate by default
        self._bar.setTextVisible(True)
        self._bar.setFormat("%p%")
        self._bar.setMinimumWidth(260)
        layout.addWidget(self._bar)

        self._prog_lbl = QLabel("")
        self._prog_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._prog_lbl.setStyleSheet("font-size: 10px; color: #555555;")
        layout.addWidget(self._prog_lbl)

        # D7: optional Cancel button
        if on_cancel is not None:
            btn_cancel = QPushButton("Cancel")
            btn_cancel.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            btn_cancel.setStyleSheet(
                "padding: 4px 16px; font-size: 11px;"
            )
            btn_cancel.clicked.connect(on_cancel)
            btn_cancel.clicked.connect(self.reject)
            layout.addWidget(btn_cancel, alignment=Qt.AlignmentFlag.AlignCenter)
            self.setFixedSize(340, 165)
        else:
            self.setFixedSize(340, 130)

        self._t0 = time.perf_counter()

    def set_progress(self, m: int, M: int) -> None:
        """Update the progress bar (switches to determinate mode)."""
        if self._bar.maximum() == 0 and M > 0:
            self._bar.setRange(0, M)
        self._bar.setValue(m)
        elapsed = time.perf_counter() - self._t0
        # ETA estimate
        eta_txt = ""
        if m > 0 and m < M:
            eta = elapsed * (M - m) / m
            eta_txt = f"  |  ETA ~ {eta:4.1f}s"
        self._prog_lbl.setText(f"Run {m} / {M}   ({elapsed:4.1f}s elapsed{eta_txt})")


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class GSSMainWindow(QMainWindow):
    """Interactive GSS simulator window."""

    # Emitted when the user picks a preset with different K/q/s.
    # The slot in main.py closes this window and reopens with the new model.
    restart_with_model = pyqtSignal(object)

    def __init__(
        self,
        K: int,
        q: int,
        s: int,
        P: np.ndarray | None = None,
        model=None,
        parent=None,
    ):
        super().__init__(parent)
        self._K = K
        self._q = q
        self._s = s
        # Transition matrix (fixed, not editable)
        self._P = P if P is not None else np.full((K, K), 1.0 / K)

        # Single source of truth for everything Simulate / Filter / MC / Load
        # produces. See _SessionState above for the invariants.
        self._state = _SessionState()

        self._worker: _SimWorker | None = None
        self._filter_worker: _FilterWorker | None = None
        self._wait_dlg: _WaitDialog | None = None
        self._op_t0: float = time.perf_counter()   # C4: initialised in constructor

        self.setWindowTitle(
            f"FofGss — GSS Simulator  (K={K}, q={q}, s={s})"
        )

        # ── central widget ────────────────────────────────────────────
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self._splitter = splitter
        main_layout.addWidget(splitter)

        # ── left panel ───────────────────────────────────────────────
        left = QWidget()
        left.setMinimumWidth(540)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)

        # ── Preset selector ───────────────────────────────────────────
        self._presets = PRESETS
        preset_row = QHBoxLayout()
        preset_row.setSpacing(6)
        preset_row.addWidget(QLabel("Preset :"))
        self._preset_combo = QComboBox()
        self._preset_combo.addItem("— select a model —")
        for entry in PRESETS:
            self._preset_combo.addItem(entry.label)
            self._preset_combo.setItemData(
                self._preset_combo.count() - 1,
                entry.tooltip,
                Qt.ItemDataRole.ToolTipRole,
            )
        self._preset_combo.setToolTip("Load a predefined model")
        self._preset_combo.activated.connect(self._on_preset_selected)
        preset_row.addWidget(self._preset_combo, stretch=1)
        left_layout.addLayout(preset_row)

        self._param_panel = ParamPanel(K, q, s)
        self._param_panel.validity_changed.connect(self._on_validity_changed)
        self._param_panel.value_changed.connect(self._refresh_filter_button_drift_indicator)
        self._param_panel.constraint_toggled.connect(self._on_reset)
        left_layout.addWidget(self._param_panel)  # no stretch: natural height

        # Transition matrix P
        p_section = QHBoxLayout()
        p_label = QLabel("Transition matrix:")
        p_label.setStyleSheet("font-size: 11px;")
        p_section.addWidget(p_label)
        self._p_widget = StochasticMatrixWidget(K, title=f"P  ({K}×{K})")
        self._p_widget.set_matrix(self._P)
        self._p_widget.validity_changed.connect(self._on_validity_changed)
        self._p_widget.value_changed.connect(self._update_stationary_display)
        self._p_widget.value_changed.connect(self._refresh_filter_button_drift_indicator)
        p_section.addWidget(self._p_widget)

        self._stationary_label = QLabel("")
        self._stationary_label.setStyleSheet("font-size: 10px; color: #444444;")
        p_section.addWidget(self._stationary_label)

        p_section.addStretch()
        left_layout.addLayout(p_section)
        self._update_stationary_display()   # initialise avec la valeur par défaut

        # N field
        n_row = QHBoxLayout()
        n_row.addWidget(QLabel("N (steps) :"))
        self._n_spin = QSpinBox()
        self._n_spin.setRange(1, 1_000_000)
        self._n_spin.setValue(1000)
        self._n_spin.setSingleStep(100)
        self._n_spin.valueChanged.connect(self._on_sim_params_changed)
        n_row.addWidget(self._n_spin)
        left_layout.addLayout(n_row)

        # Seed field
        seed_row = QHBoxLayout()
        seed_row.addWidget(QLabel("Seed:"))
        self._seed_edit = QLineEdit()
        self._seed_edit.setPlaceholderText("empty = random")
        self._seed_edit.setMaximumWidth(120)
        self._seed_edit.textChanged.connect(self._on_seed_text_changed)   # A2: validate
        self._seed_edit.editingFinished.connect(self._on_sim_params_changed)  # A1: not per-keystroke
        seed_row.addWidget(self._seed_edit)
        left_layout.addLayout(seed_row)

        # Auto-filter row
        auto_row = QHBoxLayout()
        self._auto_filter_check = QCheckBox("Auto-filter after simulate")
        self._auto_filter_check.setToolTip(
            "When enabled, the filter runs automatically after every single simulation.\n"
            "The filter uses the parameters captured at simulate time — editing the\n"
            "widgets afterwards does not affect the auto-filter run.\n"
            "The Filter mode (IMM / H5) and Joseph form settings below apply as usual.\n"
            "No effect in Monte-Carlo mode."
        )
        auto_row.addWidget(self._auto_filter_check)
        auto_row.addStretch()
        left_layout.addLayout(auto_row)

        # Filter-mode row
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Filter mode:"))
        self._mode_combo = QComboBox()
        self._mode_combo.addItem("Approximate IMM - H5 not required", "imm_general")
        self._mode_combo.addItem("Exact IMM - H5 required",          "h5_exact")
        self._mode_combo.setToolTip(
            "Approximate IMM — per-step moment propagation from the filtered π_n.\n"
            "               Works for any GSS model, with or without (H5)\n"
            "               (matches fofgss ≤ v0.9.0).\n"
            "Exact IMM    — stationary pre-computed moments. Exact when (H5)\n"
            "               holds: the algebraic constraint of paper eq. (4.4)\n"
            "               linking A, B, C, D, Σ_U, Σ_V, Δ. Emits a warning\n"
            "               when the residual exceeds the tolerance. Use\n"
            "               'Apply (H5)' to enforce it (recomputes B(k))."
        )
        mode_row.addWidget(self._mode_combo)
        mode_row.addStretch()
        left_layout.addLayout(mode_row)

        # Joseph form row (only meaningful for h5_exact mode)
        joseph_row = QHBoxLayout()
        self._joseph_check = QCheckBox("Joseph form (covariance update)")
        self._joseph_check.setToolTip(
            "Only used in 'Exact IMM under (H5)' mode. When checked, the\n"
            "mode-conditional posterior covariance is computed via the Joseph\n"
            "form (numerically stable, symmetric and PSD by construction).\n"
            "Mathematically equivalent to the short form under stationarity.\n"
            "See paper Appendix E."
        )
        joseph_row.addWidget(self._joseph_check)
        joseph_row.addStretch()
        left_layout.addLayout(joseph_row)

        # Gray out the Joseph checkbox unless mode is h5_exact
        def _sync_joseph_enabled():
            self._joseph_check.setEnabled(
                self._mode_combo.currentData() == "h5_exact"
            )
        self._mode_combo.currentIndexChanged.connect(lambda _: _sync_joseph_enabled())
        _sync_joseph_enabled()

        left_layout.addStretch()   # pushes buttons to the bottom of the panel

        # Buttons — grille 2×2
        #   [Simulate]        [Filter]
        #   [Save CSV]        [Export model…]
        #   [Reset ——————————————————————————]
        btn_grid = QGridLayout()
        btn_grid.setSpacing(6)

        self._btn_simulate = QPushButton("Simulate")
        self._btn_simulate.setFixedHeight(36)
        self._btn_simulate.setToolTip("Run a new simulation  (Ctrl+R)")           # B2
        self._btn_simulate.clicked.connect(self._on_simulate)

        self._btn_filter = QPushButton("Filter")
        self._btn_filter.setFixedHeight(36)
        self._btn_filter.setEnabled(False)
        self._btn_filter.setToolTip("Run the optimal filter on the simulation  (Ctrl+F)")  # B3
        self._btn_filter.clicked.connect(self._on_filter)

        self._btn_save = QPushButton("Save CSV")
        self._btn_save.setFixedHeight(36)
        self._btn_save.setEnabled(False)
        self._btn_save.setToolTip("Save simulation data to a CSV file  (Ctrl+S)")  # B4
        self._btn_save.clicked.connect(self._on_save)

        self._btn_load = QPushButton("Load CSV…")
        self._btn_load.setFixedHeight(36)
        self._btn_load.setToolTip("Load a previously saved simulation CSV  (Ctrl+O)")  # B5
        self._btn_load.clicked.connect(self._on_load_data)

        self._btn_export = QPushButton("Export model…")
        self._btn_export.setFixedHeight(36)
        self._btn_export.setEnabled(False)
        self._btn_export.setToolTip("Export current parameters as a Python model file  (Ctrl+E)")  # B6
        self._btn_export.clicked.connect(self._on_export_model)

        self._btn_export_plots = QPushButton("Export plots…")
        self._btn_export_plots.setFixedHeight(36)
        self._btn_export_plots.setEnabled(False)
        self._btn_export_plots.setToolTip("Save current plots to PDF / PNG / SVG  (Ctrl+Shift+E)")  # B6
        self._btn_export_plots.clicked.connect(self._on_export_plots)

        self._btn_innov_hist = QPushButton("📊 Innovation histograms…")
        self._btn_innov_hist.setFixedHeight(36)
        self._btn_innov_hist.setEnabled(False)
        self._btn_innov_hist.setToolTip(
            "Show innovation histograms, ACF, and scatter plots.  (Ctrl+I)"  # A8
        )
        self._btn_innov_hist.clicked.connect(self._on_innov_hist)

        self._btn_reset = QPushButton("⟳  Reset")
        self._btn_reset.setFixedHeight(36)
        self._btn_reset.setToolTip("Clear all results and reset the plots  (Ctrl+Shift+R)")  # B7
        self._btn_reset.clicked.connect(self._on_reset)

        btn_grid.addWidget(self._btn_simulate,    0, 0)
        btn_grid.addWidget(self._btn_filter,      0, 1)
        btn_grid.addWidget(self._btn_save,        1, 0)
        btn_grid.addWidget(self._btn_load,        1, 1)
        btn_grid.addWidget(self._btn_export,      2, 0)
        btn_grid.addWidget(self._btn_export_plots,2, 1)
        btn_grid.addWidget(self._btn_innov_hist,  3, 0, 1, 2)
        btn_grid.addWidget(self._btn_reset,       4, 0, 1, 2)

        left_layout.addLayout(btn_grid)

        # B1: keyboard shortcut hint — small one-liner for discoverability
        _sc_hint = QLabel(
            "Ctrl+R Simulate  ·  Ctrl+F Filter  ·  Ctrl+I Innovations  ·  Ctrl+Shift+R Reset"
        )
        _sc_hint.setStyleSheet("font-size: 9px; color: #888888; padding: 1px 0;")
        _sc_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_layout.addWidget(_sc_hint)

        # Thin separator between buttons and result panels (B11)
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        sep.setStyleSheet("color: #cccccc;")
        left_layout.addWidget(sep)

        # ── MSE display box ──────────────────────────────────────────
        self._mse_frame = QFrame()
        self._mse_frame.setFrameShape(QFrame.Shape.StyledPanel)
        self._mse_frame.setVisible(False)
        mse_layout = QVBoxLayout(self._mse_frame)
        mse_layout.setContentsMargins(8, 6, 8, 6)
        mse_layout.setSpacing(3)

        self._mse_title = QLabel("Filter quality")
        self._mse_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        mse_layout.addWidget(self._mse_title)

        self._loglik_label = QLabel("")
        self._loglik_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._loglik_label.setToolTip(
            "Total log-likelihood  log L = Σₙ log p(yₙ | y₁:ₙ₋₁)  (nats).\n"
            "Mean per step:  ℓ̄ = log L / N.\n\n"
            "Model-selection criteria (smaller = better fit adjusted for complexity):\n"
            "  BIC = −2 log L + d · log N\n"
            "  AIC = −2 log L + 2d\n"
            "where d = number of free parameters.\n\n"
            "Under the true model, ℓ̄ → −h  (entropy rate) as N → ∞."
        )
        mse_layout.addWidget(self._loglik_label)

        self._mse_global_label = QLabel("")
        self._mse_global_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        mse_layout.addWidget(self._mse_global_label)

        self._rmse_label = QLabel("")
        self._rmse_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        mse_layout.addWidget(self._rmse_label)

        # Per-component labels (created dynamically in _on_filter_finished)
        self._mse_comp_labels: list[QLabel] = []

        left_layout.addWidget(self._mse_frame)

        # ── Innovation diagnostics (Ljung-Box + Jarque-Bera) ─────────
        self._innov_frame = QFrame()
        self._innov_frame.setFrameShape(QFrame.Shape.StyledPanel)
        self._innov_frame.setVisible(False)
        innov_layout = QVBoxLayout(self._innov_frame)
        innov_layout.setContentsMargins(8, 6, 8, 6)
        innov_layout.setSpacing(4)

        innov_title = QLabel("Innovation diagnostics")
        innov_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        innov_title.setStyleSheet("font-size: 10px; font-weight: bold;")
        innov_title.setToolTip(
            "Ljung-Box: whiteness test on raw innovations\n"
            "           (no autocorrelation up to lag h).\n"
            "Shape:    skewness S and excess kurtosis K of STANDARDISED\n"
            "           innovations ν̃ = S^{-1/2} ν.\n"
            "           (h5_exact: whitened by stationary Γ mixture;\n"
            "            imm_general: whitened by sample covariance)\n"
            "           For a well-tuned filter: |S|≈0, |K|≈0."
        )
        innov_layout.addWidget(innov_title)

        innov_grid = QGridLayout()
        innov_grid.setSpacing(4)
        header_style = "font-size: 9px; color: #555555; font-style: italic;"
        h_lb = QLabel("Ljung-Box")
        h_jb = QLabel("Skew · Kurt")
        h_lb.setStyleSheet(header_style);  h_lb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        h_jb.setStyleSheet(header_style);  h_jb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        innov_grid.addWidget(h_lb, 0, 1)
        innov_grid.addWidget(h_jb, 0, 2)

        self._innov_lb_badges: list[QLabel] = []
        self._innov_jb_badges: list[QLabel] = []
        for i in range(s):
            name = QLabel(f"ν^{i}")
            name.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            name.setStyleSheet("font-size: 10px;")
            lb = QLabel();  lb.setAlignment(Qt.AlignmentFlag.AlignCenter);  lb.setFixedHeight(20)
            jb = QLabel();  jb.setAlignment(Qt.AlignmentFlag.AlignCenter);  jb.setFixedHeight(20)
            innov_grid.addWidget(name, i + 1, 0)
            innov_grid.addWidget(lb,   i + 1, 1)
            innov_grid.addWidget(jb,   i + 1, 2)
            self._innov_lb_badges.append(lb)
            self._innov_jb_badges.append(jb)
        innov_grid.setColumnStretch(1, 1)
        innov_grid.setColumnStretch(2, 1)
        innov_layout.addLayout(innov_grid)

        left_layout.addWidget(self._innov_frame)

        self._status_bar = QLabel("")
        self._status_bar.setWordWrap(True)
        self._status_bar.setStyleSheet("font-size: 10px; color: #444444;")  # A10: neutral default
        left_layout.addWidget(self._status_bar)

        splitter.addWidget(left)

        # ── right panel — QTabWidget avec deux onglets ───────────────
        self._right_tabs = QTabWidget()
        self._right_tabs.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

        # Onglet 0 : trajectoires simulation / filtre (inchangé)
        self._plot_panel = PlotPanel(q, s)
        self._right_tabs.addTab(self._plot_panel, "Simulation / Filtre")

        # Onglet 1 : densité conditionnelle p(y_{n+1} | r_n, r_{n+1}, y_n)
        self._pred_y_panel = PredYPanel(K, q, s)
        self._right_tabs.addTab(
            self._pred_y_panel,
            "p(yₙ₊₁ | rₙ, rₙ₊₁, yₙ)",
        )
        self._right_tabs.setTabEnabled(1, False)
        self._right_tabs.setTabToolTip(
            1,
            "Exact conditional Gaussian density p(y_{n+1} | r_n=j, r_{n+1}=k, y_n).\n"
            "Available after filtering in 'Exact IMM – H5 required' mode.",
        )

        splitter.addWidget(self._right_tabs)

        # Left panel takes ~420px initially, right panel takes the rest
        splitter.setSizes([540, 800])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        # ── Menu bar + status bar + persisted settings ────────────────
        self._build_menu_bar()
        self._build_status_bar()

        # If a model was passed, load its F and Σ_W
        if model is not None:
            self._load_model(model)

        self._refresh_simulate_button()
        self._load_settings()
        self._refresh_session_summary()

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_validity_changed(self, _: bool) -> None:
        self._refresh_simulate_button()
        self._refresh_filter_button_drift_indicator()

    def _refresh_filter_button_drift_indicator(self) -> None:
        """Mark the Filter button when GUI params differ from those captured.

        A small ⚠ prefix on the button label, plus a tooltip, makes it
        obvious that re-clicking Filter will use the *captured* parameters,
        not the ones currently shown in the widgets. Cleared as soon as
        the user re-runs Simulate.
        """
        base_label = "Filter"
        if self._state.params_signature is None:
            self._btn_filter.setText(base_label)
            self._btn_filter.setToolTip("")
            return
        live_sig = self._params_signature()
        if live_sig is not None and live_sig != self._state.params_signature:
            self._btn_filter.setText("⚠ " + base_label)
            self._btn_filter.setToolTip(
                "Parameters in the panel have changed since the last Simulate.\n"
                "Filter will use the parameters captured at Simulate, not the\n"
                "current GUI values. Re-run Simulate to use the new ones."
            )
        else:
            self._btn_filter.setText(base_label)
            self._btn_filter.setToolTip("")

    def _on_sim_params_changed(self) -> None:
        """Called when N or Seed changes: invalidate current results."""
        if not self._state.has_data():
            return
        self._on_reset()

    def _on_seed_text_changed(self, text: str) -> None:
        """Show red border when seed text cannot be parsed as an integer (A2)."""
        stripped = text.strip()
        if not stripped:
            self._seed_edit.setStyleSheet("")       # empty → random, neutral
        else:
            try:
                int(stripped)
                self._seed_edit.setStyleSheet("")   # valid integer
            except ValueError:
                self._seed_edit.setStyleSheet(
                    "border: 1px solid #cc0000; background-color: #fff0f0;"
                )

    def _on_reset(self) -> None:
        """Clear all simulation / filter results and reset the interface."""
        # Stop in-flight workers FIRST so their late `finished` signal
        # cannot resurrect the state we are about to clear.
        self._cancel_active_workers()
        # Also dismiss any wait dialog from a cancelled operation
        if getattr(self, "_wait_dlg", None) is not None:
            self._wait_dlg.close()
            self._wait_dlg = None
        # Re-enable the simulate button (a cancelled MC may have left it disabled)
        self._btn_simulate.setEnabled(True)

        self._state.reset()
        self._refresh_filter_button_drift_indicator()
        self._btn_filter.setEnabled(False)
        self._btn_save.setEnabled(False)
        self._btn_export.setEnabled(False)
        self._btn_export_plots.setEnabled(False)
        self._btn_innov_hist.setEnabled(False)
        self._sync_menu_actions()
        self._mse_frame.setVisible(False)
        self._innov_frame.setVisible(False)
        self._set_status("")
        self._plot_panel.clear()
        # Réinitialiser et désactiver l'onglet p(y_{n+1} | …)
        self._pred_y_panel.clear()
        self._right_tabs.setTabEnabled(1, False)
        self._right_tabs.setCurrentIndex(0)
        self.statusBar().showMessage("Reset — ready.", 4000)

    def _cancel_active_workers(self) -> None:
        """Disconnect signals and request interruption on every running worker.

        Disconnecting first ensures that any `finished` / `error` / `progress`
        signal already queued on the event loop will be discarded by Qt
        instead of running our handlers and corrupting freshly-reset state.
        """
        for attr in ("_worker", "_filter_worker"):
            w = getattr(self, attr, None)
            if w is None:
                continue
            try:
                w.finished.disconnect()
            except (TypeError, RuntimeError):
                pass
            try:
                w.error.disconnect()
            except (TypeError, RuntimeError):
                pass
            if hasattr(w, "progress"):
                try:
                    w.progress.disconnect()
                except (TypeError, RuntimeError):
                    pass
            if w.isRunning():
                w.requestInterruption()
                w.quit()
                # Don't block the UI: the worker checks isInterruptionRequested()
                # at fixed checkpoints and aborts silently. We let it die alone.
            setattr(self, attr, None)

    def _on_simulate(self) -> None:
        self._on_simulate_single()

    def _on_simulate_single(self) -> None:
        params = self._build_gss_params()
        if params is None:
            return

        N = self._n_spin.value()
        seed = self._parse_seed()

        self._state.begin_simulation(params, self._params_signature())
        self._refresh_filter_button_drift_indicator()
        self._btn_simulate.setEnabled(False)
        self._btn_save.setEnabled(False)
        self._btn_filter.setEnabled(False)
        self._btn_innov_hist.setEnabled(False)
        self._sync_menu_actions()
        self._mse_frame.setVisible(False)
        self._innov_frame.setVisible(False)
        self._plot_panel.clear_filter_overlay()
        self._set_status("")
        self._op_t0 = time.perf_counter()
        self.statusBar().showMessage(f"Simulating  N = {N}…")

        self._wait_dlg = _WaitDialog(
            f"Simulating  N = {N}…", on_cancel=self._on_reset, parent=self)
        self._wait_dlg.show()

        self._worker = _SimWorker(params, N, seed, parent=self)
        self._worker.finished.connect(self._on_sim_finished)
        self._worker.error.connect(self._on_sim_error)
        self._worker.start()

    def _on_sim_finished(
        self,
        ns: list[int],
        rs: list[int],
        xs: np.ndarray,
        ys: np.ndarray,
    ) -> None:
        # Drop signals that arrived after a Reset / new operation: the worker
        # was disconnected & nulled out, so any leftover queued signal is stale.
        if self.sender() is not self._worker:
            return
        if self._wait_dlg:
            self._wait_dlg.accept()
            self._wait_dlg = None

        seed = self._parse_seed()
        self._state.store_data(ns, rs, xs, ys, seed)
        self._plot_panel.update_plots(ns, rs, xs, ys, self._K)

        self._btn_save.setEnabled(True)
        self._btn_filter.setEnabled(True)
        self._btn_export_plots.setEnabled(True)
        self._sync_menu_actions()
        self._refresh_simulate_button()

        elapsed = time.perf_counter() - getattr(self, "_op_t0", time.perf_counter())
        self.statusBar().showMessage(
            f"Simulation complete — N = {len(ns)} steps in {elapsed:.2f}s.", 6000
        )

        # Auto-filter chaining
        if self._auto_filter_check.isChecked() and self._state.can_filter():
            self._on_filter()

    def _on_sim_error(self, msg: str) -> None:
        if self.sender() is not self._worker:
            return
        if self._wait_dlg:
            self._wait_dlg.reject()
            self._wait_dlg = None

        self._set_status(f"Error: {msg}", error=True)
        self.statusBar().showMessage(f"Simulation error: {msg}", 8000)
        self._refresh_simulate_button()

    def _on_filter(self) -> None:
        if not self._state.can_filter():
            return

        _, _, _, ys, _ = self._state.data

        # Warn the user if the parameters displayed on screen have drifted
        # from those captured at Simulate (or Load CSV) — the filter uses
        # the captured ones, not whatever is in the widgets right now.
        live_sig = self._params_signature()
        params_drifted = (
            live_sig is not None
            and self._state.params_signature is not None
            and live_sig != self._state.params_signature
        )

        self._btn_filter.setEnabled(False)
        self._sync_menu_actions()
        self._mse_frame.setVisible(False)
        self._set_status("")
        self._op_t0 = time.perf_counter()
        msg = f"Filtering  N = {len(ys)}…"
        if params_drifted:
            msg += "  ⚠ using parameters captured at last Simulate (GUI values differ)"
        self.statusBar().showMessage(msg)

        self._wait_dlg = _WaitDialog(
            "Filtering…", on_cancel=self._on_reset, parent=self)
        self._wait_dlg.show()

        self._filter_worker = _FilterWorker(
            self._state.params, ys,
            joseph=self._joseph_check.isChecked(),
            mode=self._mode_combo.currentData(),
            parent=self,
        )
        self._filter_worker.finished.connect(self._on_filter_finished)  # type: ignore[arg-type]
        self._filter_worker.error.connect(self._on_filter_error)
        # D8: wire progress → wait dialog progress bar
        if self._wait_dlg is not None:
            self._filter_worker.progress.connect(
                lambda n, tot, dlg=self._wait_dlg: dlg.set_progress(n, tot)
            )
        self._filter_worker.start()

    def _on_filter_finished(
        self,
        E_xs:          np.ndarray,   # (N, q)
        Var_xs:        np.ndarray,   # (N, q)
        pis:           np.ndarray,   # (N, K)
        innovations:   np.ndarray,   # (N, s)
        log_lik_total: float,
    ) -> None:
        if self.sender() is not self._filter_worker:
            return
        if not self._state.has_data():
            return                       # state cleared while filter was running
        if self._wait_dlg:
            self._wait_dlg.accept()
            self._wait_dlg = None

        ns, rs, xs, ys, _ = self._state.data
        N = len(ns)

        # Overlay filtered estimates + regime posterior + innovation sequence
        self._state.store_innovations(innovations)
        self._plot_panel.add_filter_overlay(ns, E_xs, Var_xs)
        self._plot_panel.add_pi_overlay(ns, pis, self._K)
        self._plot_panel.update_innovations(ns, innovations)

        # ── Filter quality frame — always visible after filter ───────
        mean_ll = log_lik_total / N if N > 0 else float("nan")
        self._loglik_label.setText(
            f"log L = {log_lik_total:.4g}   (mean = {mean_ll:.4g})"
        )

        # MSE / RMSE only when ground-truth X is available
        if xs is not None:
            err = xs - E_xs                              # (N, q)
            mse_per_comp = np.mean(err ** 2, axis=0)     # (q,)
            mse_global   = float(mse_per_comp.mean())
            rmse_global  = float(np.sqrt(mse_global))

            self._mse_global_label.setText(f"MSE  = {mse_global:.5g}")
            self._rmse_label.setText(f"RMSE = {rmse_global:.5g}")

            # Color the frame: green if RMSE is small relative to signal std
            sig_std = float(xs.std()) if xs.std() > 0 else 1.0
            ratio   = rmse_global / sig_std
            if ratio < 0.20:
                bg, fg, border = "#d4edda", "#155724", "#c3e6cb"   # green
                quality_icon = "✓"
            elif ratio < 0.50:
                bg, fg, border = "#fff3cd", "#856404", "#ffc107"   # amber
                quality_icon = "~"
            else:
                bg, fg, border = "#f8d7da", "#721c24", "#f5c6cb"   # red
                quality_icon = "✗"
            title_text = (
                f"Filter quality  {quality_icon}"
                f"  (RMSE/σ = {ratio:.2f})"
            )
        else:
            # No ground truth: neutral styling, hide MSE/RMSE rows
            self._mse_global_label.setText("")
            self._rmse_label.setText("")
            bg, fg, border = "#eef2f7", "#333333", "#c8d0d8"
            title_text = "Filter quality  (log L only)"

        self._mse_title.setText(title_text)
        self._mse_frame.setStyleSheet(
            f"QFrame {{ background-color: {bg}; border: 1px solid {border};"
            f" border-radius: 4px; }}"
        )
        title_style = f"font-weight: bold; font-size: 11px; color: {fg};"
        value_style = f"font-size: 10px; color: {fg};"
        self._mse_title.setStyleSheet(title_style)
        self._loglik_label.setStyleSheet(value_style)
        self._mse_global_label.setStyleSheet(value_style)
        self._rmse_label.setStyleSheet(value_style)

        # Per-component rows (rebuild if q changed)
        mse_vbox = self._mse_frame.layout()
        for lbl in self._mse_comp_labels:
            mse_vbox.removeWidget(lbl)
            lbl.deleteLater()
        self._mse_comp_labels.clear()

        if xs is not None and self._q > 1:
            for i, v in enumerate(np.mean((xs - E_xs) ** 2, axis=0)):
                lbl = QLabel(f"  MSE(X^{i}) = {v:.5g}")
                lbl.setStyleSheet(f"font-size: 9px; color: {fg};")
                mse_vbox.addWidget(lbl)
                self._mse_comp_labels.append(lbl)

        self._mse_frame.setVisible(True)

        self._btn_filter.setEnabled(True)
        self._btn_innov_hist.setEnabled(True)
        self._sync_menu_actions()

        # ── Tab: p(y_{n+1} | r_n, r_{n+1}, y_n) ───────────────────────────
        if (
            self._filter_worker is not None
            and hasattr(self._filter_worker, "cond_moments")
        ):
            cm = self._filter_worker.cond_moments
            _, _, _, ys, _ = self._state.data
            self._pred_y_panel.set_data(
                cm["mu_Y_jk"], cm["M_t"], cm["Gamma"], cm["mu_Y"], ys,
                M_simple=cm.get("M_simple"),
                Gamma2=cm.get("Gamma2"),
                b_Y=cm.get("b_Y"),
            )
            self._right_tabs.setTabEnabled(1, True)

        elapsed = time.perf_counter() - getattr(self, "_op_t0", time.perf_counter())
        self.statusBar().showMessage(
            f"Filter complete — N = {N}, log L = {log_lik_total:.4g}  "
            f"({elapsed:.2f}s).",
            8000,
        )

        # ── Innovation diagnostics: Ljung-Box + shape of standardised ν̃ ────
        # Compute standardised innovations once (A12 / D1):
        #   h5_exact  → whiten by stationary marginal covariance Σ w_{jk} Γ_{jk}
        #   imm_general → whiten by sample covariance
        _mix_w    = None
        _Gamma_cm = None
        _muY_jk   = None
        if (
            self._filter_worker is not None
            and hasattr(self._filter_worker, "cond_moments")
        ):
            cm_diag = self._filter_worker.cond_moments
            if all(k in cm_diag for k in ("mix_w", "Gamma", "mu_Y_jk")):
                _mix_w    = cm_diag["mix_w"]
                _Gamma_cm = cm_diag["Gamma"]
                _muY_jk   = cm_diag["mu_Y_jk"]

        try:
            innov_std = _standardise_innovations(
                innovations, _mix_w, _Gamma_cm, _muY_jk
            )
        except Exception:   # noqa: BLE001
            innov_std = innovations   # safe fallback

        std_mode = "h5 S" if _mix_w is not None else "sample S"

        # D2: Bonferroni correction — running 2·s independent tests (s LB + s shape).
        # Per-test significance level = α / (2·s) so the family-wise error rate ≤ α=0.05.
        n_tests   = max(1, 2 * self._s)
        alpha_fam = 0.05                             # target FWER
        alpha_lb  = alpha_fam / n_tests              # per LB test
        alpha_sk  = alpha_fam / n_tests              # per shape test
        # "warn" zone = [alpha, 2·alpha]; "ok" zone = (2·alpha, 1]
        thresh_lb_ok   = 2.0 * alpha_lb
        thresh_lb_warn = alpha_lb
        bonf_note = (
            f"Bonferroni-corrected threshold: α_per = {alpha_lb:.4g} "
            f"(family-wise α={alpha_fam}, {n_tests} tests)."
        )

        for i in range(self._s):
            # Ljung-Box on raw innovation (autocorrelation test is scale-independent)
            _, p_lb, h_lb = _ljung_box(innovations[:, i])
            if p_lb > thresh_lb_ok:
                style, icon, verdict = _PILL_OK,   "✓", "white"
            elif p_lb > thresh_lb_warn:
                style, icon, verdict = _PILL_WARN, "~", "border"
            else:
                style, icon, verdict = _PILL_ERR,  "✗", "autocor."
            self._innov_lb_badges[i].setText(
                f"{icon} {verdict}  p={p_lb:.3f}"
            )
            self._innov_lb_badges[i].setStyleSheet(style)
            self._innov_lb_badges[i].setToolTip(
                f"Ljung-Box: Q stat with h = {h_lb} lags.\n"
                f"p = {p_lb:.4g}   (OK if p > {thresh_lb_ok:.4g}).\n"
                f"{bonf_note}"
            )
            # Shape diagnostics on STANDARDISED innovation
            S, K, JB, p_jb = _shape_diagnostics(innov_std[:, i])
            sk_ok = abs(S) < 0.25 and abs(K) < 0.50
            sk_wn = abs(S) < 0.50 and abs(K) < 1.00
            if sk_ok:
                style, icon = _PILL_OK,   "✓"
            elif sk_wn:
                style, icon = _PILL_WARN, "~"
            else:
                style, icon = _PILL_ERR,  "✗"
            self._innov_jb_badges[i].setText(
                f"{icon}  S={S:+.2f}  K={K:+.2f}"
            )
            self._innov_jb_badges[i].setStyleSheet(style)
            self._innov_jb_badges[i].setToolTip(
                f"Standardised innovation ν̃  ({std_mode})\n"
                f"Skewness  S = {S:+.4f}   (target ≈ 0)\n"
                f"Excess kurtosis  K = {K:+.4f}   (target ≈ 0 for Gaussian)\n"
                f"Jarque-Bera  JB = {JB:.3f}   p = {p_jb:.4g}\n"
                "Standardisation removes the scale-mixing effect of regime\n"
                "switching, so S ≈ 0 and K ≈ 0 are achievable for a\n"
                f"well-calibrated filter.\n{bonf_note}"
            )
        self._innov_frame.setVisible(True)

    def _on_filter_error(self, msg: str) -> None:
        if self.sender() is not self._filter_worker:
            return
        if self._wait_dlg:
            self._wait_dlg.reject()
            self._wait_dlg = None

        self._set_status(f"Filter error: {msg}", error=True)
        self.statusBar().showMessage(f"Filter error: {msg}", 8000)
        self._btn_filter.setEnabled(True)
        self._sync_menu_actions()

    def _on_export_model(self) -> None:
        """Open a save-file dialog and write a ready-to-use Python model file."""
        # A14: guard against exporting with invalid parameters
        if not (self._param_panel.is_valid() and self._p_widget.is_valid()):
            QMessageBox.warning(
                self, "Invalid parameters",
                "One or more parameters are currently invalid.\n"
                "Fix all highlighted cells before exporting the model.",
            )
            return
        default_name = (
            f"model_gss_K{self._K}_q{self._q}_s{self._s}_custom.py"
        )
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Python model",
            str(pathlib.Path("prg/models") / default_name),
            "Python files (*.py);;All files (*)",
        )
        if not path:
            return   # user cancelled

        code = self._generate_model_code(pathlib.Path(path).stem)
        try:
            pathlib.Path(path).write_text(code, encoding="utf-8")
            QMessageBox.information(
                self,
                "Model exported",
                f"File saved:\n{pathlib.Path(path).resolve()}",
            )
        except OSError as exc:
            QMessageBox.critical(self, "Write error", str(exc))

    def _on_save(self) -> None:
        if not self._state.has_data():
            return
        ns, rs, xs, ys, seed_used = self._state.data

        output_dir = pathlib.Path("data/simulated")
        output_dir.mkdir(parents=True, exist_ok=True)

        N = len(ns)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        seed_str = str(seed_used) if seed_used is not None else "rnd"
        filename = f"simulated_gui_N{N}_seed{seed_str}_{ts}.csv"
        filepath = output_dir / filename

        q, s = self._q, self._s
        header = ["n", "r"] + [f"x_{i}" for i in range(q)] + [f"y_{i}" for i in range(s)]

        with filepath.open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(header)
            for n, r, x_row, y_row in zip(ns, rs, xs, ys):
                writer.writerow([n, r] + list(x_row) + list(y_row))

        QMessageBox.information(
            self,
            "CSV saved",
            f"File saved:\n{filepath.resolve()}",
        )

    def _on_preset_selected(self, index: int) -> None:
        """Load a preset model; restart the window if K/q/s differ."""
        if index == 0:          # placeholder item
            return
        entry = self._presets[index - 1]
        try:
            model = entry.load()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Preset error", str(exc))
            self._preset_combo.setCurrentIndex(0)
            return

        if entry.K == self._K and entry.q == self._q and entry.s == self._s:
            # Same dimensions — load in place and reset plots
            self._load_model(model)
            self._on_reset()
        else:
            answer = QMessageBox.question(
                self,
                "Dimension change",
                f"This model requires K={entry.K}, q={entry.q}, s={entry.s}.\n"
                f"The window will be recreated. Continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if answer == QMessageBox.StandardButton.Yes:
                self.restart_with_model.emit(model)
            else:
                self._preset_combo.setCurrentIndex(0)

    def _on_export_plots(self) -> None:
        """Save the current plot panel figure to a PNG or PDF file."""
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export plots",
            str(pathlib.Path("data/simulated") / "plots.pdf"),
            "PDF (*.pdf);;PNG (*.png);;SVG (*.svg);;All files (*)",
        )
        if not path:
            return
        try:
            self._plot_panel.save_figure(path)
            QMessageBox.information(
                self, "Export plots",
                f"Figure saved:\n{pathlib.Path(path).resolve()}",
            )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Export error", str(exc))

    def _on_load_data(self) -> None:
        """Open a CSV file and display it without running a simulation."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load simulation data",
            str(pathlib.Path("data/simulated")),
            "CSV files (*.csv);;All files (*)",
        )
        if not path:
            return

        try:
            with open(path, newline="", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                rows = list(reader)

            if not rows:
                QMessageBox.warning(self, "Load error", "File is empty.")
                return

            headers = list(rows[0].keys())

            # --- Validate required columns ---
            required = ["n", "r"] + [f"y_{i}" for i in range(self._s)]
            missing = [c for c in required if c not in headers]
            if missing:
                QMessageBox.warning(
                    self, "Load error",
                    f"Missing column(s): {missing}\n"
                    f"Expected at least: n, r, y_0 … y_{self._s - 1}\n"
                    f"Found: {headers}",
                )
                return

            # --- Parse ---
            ns = [int(float(row["n"])) for row in rows]
            rs = [int(float(row["r"])) for row in rows]
            ys = np.array(
                [[float(row[f"y_{i}"]) for i in range(self._s)] for row in rows]
            )

            x_cols = [f"x_{i}" for i in range(self._q)]
            has_x  = all(c in headers for c in x_cols)
            xs     = (
                np.array([[float(row[f"x_{i}"]) for i in range(self._q)]
                          for row in rows])
                if has_x else None
            )

        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Load error", str(exc))
            return

        # --- Update state ---
        live_params = self._build_gss_params()  # uses current GUI params
        self._state.load_external(
            ns, rs, xs, ys,
            params=live_params,
            signature=self._params_signature(),
        )
        self._refresh_filter_button_drift_indicator()

        self._plot_panel.clear_filter_overlay()
        self._plot_panel.update_plots(ns, rs, xs, ys, self._K)

        self._btn_filter.setEnabled(self._state.can_filter())
        self._btn_save.setEnabled(False)        # nothing new generated
        self._btn_export_plots.setEnabled(True)
        self._sync_menu_actions()
        self._mse_frame.setVisible(False)

        info = f"Loaded {len(ns)} steps from '{pathlib.Path(path).name}'"
        if not has_x:
            info += "  (no ground-truth X)"
        self._set_status(info)
        self.statusBar().showMessage(info, 6000)

    # ------------------------------------------------------------------
    # Menu bar, status bar, settings
    # ------------------------------------------------------------------

    def _build_menu_bar(self) -> None:
        """Create the application menu bar with keyboard shortcuts."""
        mb = self.menuBar()

        # ── File menu ────────────────────────────────────────────────
        file_menu = mb.addMenu("&File")

        self._act_save = QAction("Save &CSV", self)
        self._act_save.setShortcut(QKeySequence("Ctrl+S"))
        self._act_save.setStatusTip("Save current simulation as CSV")
        self._act_save.setEnabled(False)
        self._act_save.triggered.connect(self._on_save)
        file_menu.addAction(self._act_save)

        self._act_load = QAction("&Load CSV…", self)
        self._act_load.setShortcut(QKeySequence("Ctrl+O"))
        self._act_load.setStatusTip("Load a previously saved simulation CSV")
        self._act_load.triggered.connect(self._on_load_data)
        file_menu.addAction(self._act_load)

        file_menu.addSeparator()

        self._act_export = QAction("Export &model…", self)
        self._act_export.setShortcut(QKeySequence("Ctrl+E"))
        self._act_export.setStatusTip("Export current parameters as a Python model file")
        self._act_export.setEnabled(False)
        self._act_export.triggered.connect(self._on_export_model)
        file_menu.addAction(self._act_export)

        self._act_export_plots = QAction("Export &plots…", self)
        self._act_export_plots.setShortcut(QKeySequence("Ctrl+Shift+E"))
        self._act_export_plots.setStatusTip("Save current plot panel as PDF/PNG/SVG")
        self._act_export_plots.setEnabled(False)
        self._act_export_plots.triggered.connect(self._on_export_plots)
        file_menu.addAction(self._act_export_plots)

        file_menu.addSeparator()

        act_quit = QAction("&Quit", self)
        act_quit.setShortcut(QKeySequence.StandardKey.Quit)
        act_quit.setStatusTip("Quit the application")
        act_quit.triggered.connect(self.close)
        file_menu.addAction(act_quit)

        # ── Simulation menu ──────────────────────────────────────────
        sim_menu = mb.addMenu("&Simulation")

        self._act_simulate = QAction("Si&mulate", self)
        self._act_simulate.setShortcut(QKeySequence("Ctrl+R"))
        self._act_simulate.setStatusTip("Run simulation (single or Monte-Carlo)")
        self._act_simulate.triggered.connect(self._on_simulate)
        sim_menu.addAction(self._act_simulate)

        self._act_filter = QAction("&Filter", self)
        self._act_filter.setShortcut(QKeySequence("Ctrl+F"))
        self._act_filter.setStatusTip("Run the optimal filter on the current simulation")
        self._act_filter.setEnabled(False)
        self._act_filter.triggered.connect(self._on_filter)
        sim_menu.addAction(self._act_filter)

        sim_menu.addSeparator()

        self._act_reset = QAction("&Reset", self)
        self._act_reset.setShortcut(QKeySequence("Ctrl+Shift+R"))
        self._act_reset.setStatusTip("Clear all results and reset the plots")
        self._act_reset.triggered.connect(self._on_reset)
        sim_menu.addAction(self._act_reset)

        # ── View menu ────────────────────────────────────────────────
        view_menu = mb.addMenu("&View")

        self._act_innov_hist = QAction("&Innovation histograms…", self)
        self._act_innov_hist.setShortcut(QKeySequence("Ctrl+I"))
        self._act_innov_hist.setStatusTip("Show histogram of each innovation component")
        self._act_innov_hist.setEnabled(False)
        self._act_innov_hist.triggered.connect(self._on_innov_hist)
        view_menu.addAction(self._act_innov_hist)


    def _build_status_bar(self) -> None:
        """Permanent status bar: ephemeral messages + session summary."""
        sb = self.statusBar()
        sb.setSizeGripEnabled(False)

        # Permanent right-side session summary label
        self._sb_session_lbl = QLabel("")
        self._sb_session_lbl.setStyleSheet("font-size: 10px; color: #333333;")
        sb.addPermanentWidget(self._sb_session_lbl)

        # Wire signals that change session parameters to refresh the summary
        self._n_spin.valueChanged.connect(self._refresh_session_summary)
        self._seed_edit.textChanged.connect(self._refresh_session_summary)
        self._auto_filter_check.toggled.connect(self._refresh_session_summary)
        self._joseph_check.toggled.connect(self._refresh_session_summary)
        self._mode_combo.currentIndexChanged.connect(
            lambda _: self._refresh_session_summary()
        )

    def _refresh_session_summary(self) -> None:
        """Update the right-hand permanent label in the status bar."""
        if not hasattr(self, "_sb_session_lbl"):
            return
        seed = self._seed_edit.text().strip() or "random"
        auto = "auto-filter" if self._auto_filter_check.isChecked() else ""
        mode = self._mode_combo.currentData()
        mode_short = "H5-exact" if mode == "h5_exact" else "IMM-general"
        parts = [
            f"K={self._K}·q={self._q}·s={self._s}",
            f"N={self._n_spin.value()}",
            f"seed={seed}",
            f"filter={mode_short}",
        ]
        # Joseph flag is only meaningful in h5_exact mode
        if mode == "h5_exact":
            joseph = "Joseph" if self._joseph_check.isChecked() else "short"
            parts.append(f"cov={joseph}")
        if auto:
            parts.append(auto)
        self._sb_session_lbl.setText("  |  ".join(parts))

    # -- Settings persistence via QSettings --------------------------------

    def _settings(self) -> QSettings:
        return QSettings("FofGss", "Simulator")

    def _load_settings(self) -> None:
        s = self._settings()
        # N is intentionally not restored: every launch starts with N=1000
        # so a previous large-N session can't make the next launch sluggish.
        seed = s.value("seed", self._seed_edit.text())
        if seed is not None:
            self._seed_edit.setText(str(seed))
        self._auto_filter_check.setChecked(s.value("auto_filter", False, type=bool))
        self._joseph_check.setChecked(s.value("joseph", False, type=bool))
        saved_mode = s.value("filter_mode", "imm_general")
        idx = self._mode_combo.findData(saved_mode)
        if idx >= 0:
            self._mode_combo.setCurrentIndex(idx)

        geom = s.value("geometry")
        if geom is not None:
            try:
                self.restoreGeometry(geom)
            except Exception:                                       # noqa: BLE001
                pass
        split = s.value("splitter")
        if split is not None:
            try:
                self._splitter.restoreState(split)
            except Exception:                                       # noqa: BLE001
                pass

    def _save_settings(self) -> None:
        s = self._settings()
        # N intentionally not saved — see _load_settings.
        s.setValue("seed", self._seed_edit.text())
        s.setValue("auto_filter", self._auto_filter_check.isChecked())
        s.setValue("joseph", self._joseph_check.isChecked())
        s.setValue("filter_mode", self._mode_combo.currentData())
        s.setValue("geometry", self.saveGeometry())
        s.setValue("splitter", self._splitter.saveState())
        # Clean up legacy keys from previous versions so they can't resurface
        s.remove("N")
        s.remove("mc_on")
        s.remove("M")

    def closeEvent(self, event) -> None:                            # noqa: N802
        self._cancel_active_workers()
        self._save_settings()
        super().closeEvent(event)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _update_stationary_display(self) -> None:
        """Recompute and display the stationary distribution π* of P."""
        P = self._p_widget.get_matrix()
        if P is None:
            self._stationary_label.setText("")
            return
        pi = _stationary_dist(P)
        if pi is None:
            self._stationary_label.setText("π* = ?")
            return
        parts = "   ".join(f"π<sub>{k}</sub> = {pi[k]:.3f}" for k in range(len(pi)))
        self._stationary_label.setText(f"π* :  {parts}")

    def _on_innov_hist(self) -> None:
        """Open innovation histogram dialog."""
        if not self._state.has_filter():
            return
        mix_params = None
        if (
            self._filter_worker is not None
            and hasattr(self._filter_worker, "cond_moments")
        ):
            cm = self._filter_worker.cond_moments
            if all(k in cm for k in ("mix_w", "Gamma", "mu_Y_jk")):
                mix_params = {
                    "w":       cm["mix_w"],      # (K,K)
                    "Gamma":   cm["Gamma"],      # [K][K] (s,s)
                    "mu_Y_jk": cm["mu_Y_jk"],   # [K][K] (s,1)
                }
        dlg = _InnovHistDialog(
            self._state.innovations, mix_params=mix_params, parent=self,
        )
        dlg.show()

    def _refresh_simulate_button(self) -> None:
        valid = self._param_panel.is_valid() and self._p_widget.is_valid()
        self._btn_simulate.setEnabled(valid)
        self._btn_export.setEnabled(valid)
        if valid:
            self._btn_simulate.setStyleSheet("")
            self._set_status("")
        else:
            self._btn_simulate.setStyleSheet(
                "border: 2px solid #cc0000; color: #cc0000;"
            )
            self._set_status("Invalid parameter(s) — fix before simulating.", error=True)
        self._sync_menu_actions()

    def _sync_menu_actions(self) -> None:
        """Mirror every QPushButton's enabled state into its matching QAction."""
        if not hasattr(self, "_act_simulate"):
            return   # menu bar not built yet
        self._act_simulate.setEnabled(self._btn_simulate.isEnabled())
        self._act_filter.setEnabled(self._btn_filter.isEnabled())
        self._act_save.setEnabled(self._btn_save.isEnabled())
        self._act_export.setEnabled(self._btn_export.isEnabled())
        self._act_export_plots.setEnabled(self._btn_export_plots.isEnabled())
        self._act_innov_hist.setEnabled(self._btn_innov_hist.isEnabled())

    def _set_status(self, msg: str, *, error: bool = False) -> None:
        """Update the left-panel status label with appropriate styling (A10).

        Parameters
        ----------
        msg   : message text (empty string clears the label).
        error : if True, use red error styling; if False, use neutral grey.
        """
        self._status_bar.setText(msg)
        color = "#cc0000" if error else "#444444"
        self._status_bar.setStyleSheet(f"font-size: 10px; color: {color};")

    def _parse_seed(self) -> int | None:
        text = self._seed_edit.text().strip()
        if not text:
            return None
        try:
            return int(text)
        except ValueError:
            return None

    def _params_signature(self) -> tuple | None:
        """Compact byte-level signature of the GUI parameters.

        Two simulations done with identical widget values produce the same
        signature; any edit invalidates it. Used to warn the user when the
        parameters displayed on screen no longer match those captured at
        the last Simulate (and thus used by Filter).

        Returns None if any widget is currently invalid.
        """
        F  = self._param_panel.get_F_list()
        S  = self._param_panel.get_Sigma_W_list()
        mu = self._param_panel.get_mu_z0_list()
        b  = self._param_panel.get_b_list()
        P  = self._p_widget.get_matrix()
        if any(x is None for x in (F, S, mu, b, P)):
            return None
        return (
            tuple(np.ascontiguousarray(m).tobytes() for m in F),
            tuple(np.ascontiguousarray(m).tobytes() for m in S),
            tuple(np.ascontiguousarray(m).tobytes() for m in mu),
            tuple(np.ascontiguousarray(m).tobytes() for m in b),
            np.ascontiguousarray(P).tobytes(),
        )

    def _build_gss_params(self) -> GSSParams | None:
        """Collect GUI values and build a GSSParams object."""
        K, q, s = self._K, self._q, self._s

        F_list = self._param_panel.get_F_list()
        Sigma_W_list = self._param_panel.get_Sigma_W_list()
        mu_z0_list = self._param_panel.get_mu_z0_list()
        b_list = self._param_panel.get_b_list()
        P = self._p_widget.get_matrix()
        if F_list is None or Sigma_W_list is None or mu_z0_list is None or b_list is None or P is None:
            self._set_status("Invalid parameter(s).", error=True)
            return None

        # Decompose Sigma_W(k) into blocks Sigma_U, Delta, Sigma_V
        Sigma_U_list = [sw[:q, :q] for sw in Sigma_W_list]
        Delta_list   = [sw[:q, q:] for sw in Sigma_W_list]
        Sigma_V_list = [sw[q:, q:] for sw in Sigma_W_list]

        # Decompose F(k) into blocks A, B, C, D
        A_list = [f[:q, :q] for f in F_list]
        B_list = [f[:q, q:] for f in F_list]
        C_list = [f[q:, :q] for f in F_list]
        D_list = [f[q:, q:] for f in F_list]

        try:
            f_matrix = FMatrix(K, q, s, A_list, B_list, C_list, D_list)
            noise_cov = GSSNoiseCovariance(K, q, s, Sigma_U_list, Delta_list, Sigma_V_list)

            Sigma_z0_list = [np.eye(q + s) for _ in range(K)]

            params = GSSParams(
                K=K, q=q, s=s,
                P=P,
                f_matrix=f_matrix,
                noise_cov=noise_cov,
                pi0=None,               # stationary
                mu_z0_list=mu_z0_list,  # from GUI
                Sigma_z0_list=Sigma_z0_list,
                b_list=b_list,          # from GUI
            )
        except Exception as exc:  # noqa: BLE001
            self._set_status(f"Parameter error: {exc}", error=True)
            return None

        return params

    def _generate_model_code(self, file_stem: str) -> str:
        """
        Generate a complete Python model file from the current GUI parameters.

        Parameters
        ----------
        file_stem : str
            Stem of the target file (e.g. ``'model_gss_K2_q1_s1_custom'``).
            Used to derive the class name.
        """
        K, q, s = self._K, self._q, self._s

        # Class name: CamelCase from file stem
        class_name = "".join(w.capitalize() for w in file_stem.split("_"))

        # Collect parameters from GUI
        F_list        = self._param_panel.get_F_list()
        Sigma_W_list  = self._param_panel.get_Sigma_W_list()
        mu_z0_list    = self._param_panel.get_mu_z0_list()
        b_list        = self._param_panel.get_b_list()
        P             = self._p_widget.get_matrix()

        # Decompose blocks
        A_list       = [f[:q,  :q]  for f in F_list]
        B_list       = [f[:q,  q:]  for f in F_list]
        C_list       = [f[q:,  :q]  for f in F_list]
        D_list       = [f[q:,  q:]  for f in F_list]
        Sigma_U_list = [sw[:q, :q]  for sw in Sigma_W_list]
        Delta_list   = [sw[:q, q:]  for sw in Sigma_W_list]
        Sigma_V_list = [sw[q:, q:]  for sw in Sigma_W_list]

        def _fmt_arr(arr: np.ndarray) -> str:
            """Format a 2-D numpy array as a compact np.array(...) literal.

            All rows after the first are aligned under the opening ``[``.
            e.g.  np.array([[0.8, 0.1],
                             [0.0, 0.7]])
            """
            prefix = "np.array(["
            align  = " " * len(prefix)         # align continuation rows
            rows   = []
            for r in range(arr.shape[0]):
                vals = ", ".join(f"{v:.8g}" for v in arr[r])
                rows.append(f"[{vals}]")
            inner = (",\n" + align).join(rows)
            return f"{prefix}{inner}])"

        def _fmt_list(arrays, field_indent: int = 4) -> str:
            """Format a list of arrays with each item on its own line.

            field_indent : spaces before each item (matches class body indent).
            """
            pad = " " * (field_indent + 1)   # one extra to align past '['
            items = [_fmt_arr(a) for a in arrays]
            if len(items) == 1:
                return f"[{items[0]}]"
            joined = (",\n" + pad).join(items)
            return f"[{joined}]"

        lines: list[str] = []
        lines += [
            "#!/usr/bin/env python3",
            "# -*- coding: utf-8 -*-",
            f'"""',
            f"prg/models/{file_stem}.py",
            f"{'=' * (len('prg/models/') + len(file_stem) + 3)}",
            f"GSS model: K={K} states, q={q} (hidden), s={s} (observed).",
            "",
            "Generated by FofGss GUI.",
            f'"""',
            "",
            "from __future__ import annotations",
            "",
            "import numpy as np",
            "",
            "from prg.models.base_gss_model import BaseGSSModel",
            "",
            f'__all__ = ["{class_name}"]',
            "",
            "",
            f"class {class_name}(BaseGSSModel):",
            f'    """GSS model exported from the GUI (K={K}, q={q}, s={s})."""',
            "",
            f"    K: int = {K}",
            f"    q: int = {q}",
            f"    s: int = {s}",
            "",
            "    # --- Markov chain ---",
            f"    P: np.ndarray = {_fmt_arr(P)}",
            "",
            "    # --- Dynamics: F(k) = [[A_k, B_k], [C_k, D_k]] ---",
            f"    A_list: list[np.ndarray] = {_fmt_list(A_list)}",
            f"    B_list: list[np.ndarray] = {_fmt_list(B_list)}",
            f"    C_list: list[np.ndarray] = {_fmt_list(C_list)}",
            f"    D_list: list[np.ndarray] = {_fmt_list(D_list)}",
            "",
            "    # --- Noise covariances: Sigma_W(k) = [[Sigma_U, Delta], [Delta^T, Sigma_V]] ---",
            f"    Sigma_U_list: list[np.ndarray] = {_fmt_list(Sigma_U_list)}",
            f"    Delta_list:   list[np.ndarray] = {_fmt_list(Delta_list)}",
            f"    Sigma_V_list: list[np.ndarray] = {_fmt_list(Sigma_V_list)}",
            "",
            "    # --- Initial conditions ---",
            "    pi0: np.ndarray | None = None   # None → stationary distribution",
            "",
            f"    mu_z0_list: list[np.ndarray] = {_fmt_list(mu_z0_list)}",
            f"    Sigma_z0_list: list[np.ndarray] = {_fmt_list([np.eye(q + s)] * K)}",
            f"    b_list: list[np.ndarray] = {_fmt_list(b_list)}",
            "",
            "    # ------------------------------------------------------------------",
            "",
            "    def get_params(self) -> dict:",
            "        return {",
            '            "K": self.K, "q": self.q, "s": self.s, "P": self.P,',
            '            "A_list": self.A_list, "B_list": self.B_list,',
            '            "C_list": self.C_list, "D_list": self.D_list,',
            '            "Sigma_U_list": self.Sigma_U_list,',
            '            "Delta_list": self.Delta_list,',
            '            "Sigma_V_list": self.Sigma_V_list,',
            '            "pi0": self.pi0,',
            '            "mu_z0_list": self.mu_z0_list,',
            '            "Sigma_z0_list": self.Sigma_z0_list,',
            '            "b_list": self.b_list,',
            "        }",
            "",
        ]
        return "\n".join(lines)

    def _load_model(self, model) -> None:
        """Pre-fill tables from a BaseGSSModel instance.

        Signals on ``_param_panel`` and ``_p_widget`` are blocked during the
        bulk update so that validity / value signals don't fire repeatedly for
        each individual state — preventing the double-refresh of the Simulate
        button and the flicker of intermediate validity states (A11 / A16).
        Callers must invoke ``_refresh_simulate_button()`` after this method.
        """
        # A11 / A16: block validity/value signals during bulk loading
        self._param_panel.blockSignals(True)
        self._p_widget.blockSignals(True)
        try:
            p = model.get_params()
            K, q, s = p["K"], p["q"], p["s"]

            # Build block noise cov to get full Σ_W
            nc = GSSNoiseCovariance(K, q, s, p["Sigma_U_list"], p["Delta_list"], p["Sigma_V_list"])
            fm = FMatrix(K, q, s, p["A_list"], p["B_list"], p["C_list"], p["D_list"])

            mu_list = p.get("mu_z0_list")
            b_list  = p.get("b_list")
            for k in range(K):
                mu = np.asarray(mu_list[k]) if mu_list is not None else None
                b  = np.asarray(b_list[k])  if b_list  is not None else None
                self._param_panel.set_state_params(k, fm.F(k), nc.Sigma_W(k), mu, b)

            if p.get("P") is not None:
                self._P = np.asarray(p["P"])
                self._p_widget.set_matrix(self._P)
                self._update_stationary_display()
        finally:
            self._param_panel.blockSignals(False)
            self._p_widget.blockSignals(False)
