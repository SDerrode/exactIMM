#!/usr/bin/env python3
"""
prg/gui/dialogs.py
==================
Auxiliary dialogs for the GSS GUI: ``_InnovHistDialog`` (innovation
histograms / ACF / scatter / per-regime residuals), ``_RegimeDiagDialog``
(confusion matrix + regime-duration histograms), and ``_WaitDialog`` (the
modal progress dialog). Extracted verbatim from ``prg/gui/main_window.py``.

matplotlib is imported lazily inside the plotting dialogs, exactly as
before, so importing this module stays cheap.
"""

from __future__ import annotations

import csv
import pathlib
import time

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from scipy.stats import norm as _norm_dist

from prg.gui.diagnostics import _shape_diagnostics


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
        pis: np.ndarray | None = None,  # D11: (N, K) filtered posteriors
        mu_Y: list | None = None,  # [K] ndarray (s,1) — stationary means
        S_YY: list | None = None,  # [K] ndarray (s,s) — stationary cov
        ys: np.ndarray | None = None,  # (N, s) observations
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Innovation diagnostics")
        self.setModal(False)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self._innovations = innovations

        from matplotlib.backends.backend_qtagg import (
            FigureCanvasQTAgg,
            NavigationToolbar2QT,
        )
        from matplotlib.figure import Figure

        _norm = _norm_dist  # C1/C3: use module-level import

        s = innovations.shape[1]
        N = innovations.shape[0]
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
        fig_h = Figure(figsize=(max(4, 3.5 * s), 3.8), tight_layout=True)
        can_h = FigureCanvasQTAgg(fig_h)
        can_h.setFocusPolicy(Qt.FocusPolicy.ClickFocus)  # D10
        hist_l.addWidget(NavigationToolbar2QT(can_h, hist_w))
        hist_l.addWidget(can_h)

        for i in range(s):
            ax = fig_h.add_subplot(1, s, i + 1)
            x = innovations[:, i]
            ax.hist(
                x,
                bins=40,
                density=True,
                color=colours[i % len(colours)],
                alpha=0.70,
                label="empirical",
            )

            # x-grid slightly wider than the data range
            pad = 0.5 * float(x.std()) if x.std() > 1e-10 else 0.5
            xg = np.linspace(float(x.min()) - pad, float(x.max()) + pad, 400)

            if mix_params is not None:
                # ── Theoretical Gaussian mixture ──────────────────────────
                # p(ν) = Σ_{j,k} w_{jk} N(ν ; δ̃_{jk}, Γ_{jk})
                #
                # δ̃_{jk} = μ_{Y,jk}[i] − Σ_{k'} P(j,k') μ_{Y,jk'}[i]
                #         = within-previous-regime centred mean
                w = mix_params["w"]  # (K, K) — already normalised
                K_mix = w.shape[0]
                Gam = mix_params["Gamma"]  # [K][K] (s, s)
                muYjk = mix_params["mu_Y_jk"]  # [K][K] (s, 1)

                pdf_mix = np.zeros_like(xg)
                for j in range(K_mix):
                    pi_j = max(float(w[j, :].sum()), 1e-12)
                    prev_mean_j_i = (
                        sum(float(w[j, kk]) * float(muYjk[j][kk][i, 0]) for kk in range(K_mix))
                        / pi_j
                    )
                    for k in range(K_mix):
                        wjk = float(w[j, k])
                        if wjk < 1e-10:
                            continue
                        delta = float(muYjk[j][k][i, 0]) - prev_mean_j_i
                        var_i = float(Gam[j][k][i, i])
                        sig = float(np.sqrt(max(var_i, 1e-12)))
                        pdf_mix += wjk * _norm.pdf(xg, delta, sig)

                ax.plot(
                    xg,
                    pdf_mix,
                    "k-",
                    linewidth=1.8,
                    label=rf"$\sum_{{jk}}w_{{jk}}\,\mathcal{{N}}(\tilde{{\delta}}_{{jk}},\Gamma_{{jk}})$"
                    f"  ({K_mix}² terms)",
                )
            else:
                # ── Fallback: best-fit single Gaussian ───────────────────
                mu_e, sig_e = float(x.mean()), float(x.std())
                if sig_e > 1e-10:
                    ax.plot(
                        xg,
                        _norm.pdf(xg, mu_e, sig_e),
                        "k--",
                        linewidth=1.5,
                        label=f"N({mu_e:.3f}, {sig_e:.3f}²)",
                    )

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
        fig_a = Figure(figsize=(max(4, 3.5 * s), 3.2), tight_layout=True)
        can_a = FigureCanvasQTAgg(fig_a)
        can_a.setFocusPolicy(Qt.FocusPolicy.ClickFocus)  # D10
        acf_l.addWidget(NavigationToolbar2QT(can_a, acf_w))
        acf_l.addWidget(can_a)
        conf95 = 1.96 / float(np.sqrt(N))

        for i in range(s):
            ax = fig_a.add_subplot(1, s, i + 1)
            x = innovations[:, i]
            xc = x - x.mean()
            c0 = float(np.dot(xc, xc)) / N
            if c0 > 1e-30:
                acf_vals = np.array(
                    [
                        float(np.dot(xc[: N - lag], xc[lag:])) / (N * c0)
                        for lag in range(1, max_lag + 1)
                    ]
                )
            else:
                acf_vals = np.zeros(max_lag)
            lags = np.arange(1, max_lag + 1)
            ax.bar(lags, acf_vals, color=colours[i % len(colours)], alpha=0.75, width=0.8)
            ax.axhline(0, color="k", linewidth=0.6)
            ax.axhline(conf95, color="#999999", linewidth=1.0, linestyle="--", label="±1.96/√N")
            ax.axhline(-conf95, color="#999999", linewidth=1.0, linestyle="--")
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
            ncols = min(n_pairs, 3)
            nrows = (n_pairs + ncols - 1) // ncols
            fig_s = Figure(figsize=(max(4, 3.5 * ncols), 3.2 * nrows), tight_layout=True)
            can_s = FigureCanvasQTAgg(fig_s)
            can_s.setFocusPolicy(Qt.FocusPolicy.ClickFocus)  # D10
            sc_l.addWidget(NavigationToolbar2QT(can_s, sc_w))
            sc_l.addWidget(can_s)

            idx = 1
            for a_i in range(s):
                for b_i in range(a_i + 1, s):
                    ax = fig_s.add_subplot(nrows, ncols, idx)
                    ax.scatter(
                        innovations[:, a_i], innovations[:, b_i], s=4, alpha=0.40, color="#555555"
                    )
                    rho = float(np.corrcoef(innovations[:, a_i], innovations[:, b_i])[0, 1])
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
        # Tab 3: Per-regime soft-weighted residuals (D11)
        #
        # Hard assignment (argmax π_n = k) contaminates each regime's
        # histogram with observations from regime j at transition times,
        # producing bimodal, heavily-skewed distributions.
        #
        # Soft assignment: each observation y_n contributes to regime k
        # with weight π_n(k) → the weighted distribution converges to
        # N(0, Σ_YY(k)) if the model is correct.
        # N_eff = Σ_n π_n(k) is shown in the title.
        #
        # Fallback (no ys / mu_Y): hard-assigned marginal innovations.
        # ─────────────────────────────────────────────────────────────
        if pis is not None and pis.ndim == 2 and pis.shape[0] == N:
            K = pis.shape[1]
            regime_colours = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

            # Decide whether we can build soft-weighted residuals
            use_resid = (
                mu_Y is not None and ys is not None and len(mu_Y) == K and ys.shape == (N, s)
            )
            has_theory = use_resid and S_YY is not None and len(S_YY) == K

            preg_w = QWidget()
            preg_l = QVBoxLayout(preg_w)
            preg_l.setContentsMargins(2, 2, 2, 2)

            nrows = K
            ncols = s
            fig_pr = Figure(
                figsize=(max(4, 3.5 * ncols), max(2.5, 2.5 * nrows)),
                tight_layout=True,
            )
            can_pr = FigureCanvasQTAgg(fig_pr)
            can_pr.setFocusPolicy(Qt.FocusPolicy.ClickFocus)

            # Pre-compute residuals for all regimes (shape (N, s) each)
            if use_resid:
                resid_all = [
                    ys - mu_Y[k].ravel()[np.newaxis, :]  # (N, s)
                    for k in range(K)
                ]

            for k in range(K):
                c = regime_colours[k % len(regime_colours)]

                if use_resid:
                    w_k = pis[:, k]  # (N,) soft weights π_n(k)
                    n_eff = float(w_k.sum())  # effective count
                    xi_all = resid_all[k]  # (N, s)  y_n − µ_Y(k)
                    w_norm = w_k / max(n_eff, 1e-12)  # normalised
                else:
                    # Fallback: hard-assigned marginal innovations
                    mask = np.argmax(pis, axis=1) == k
                    n_eff = float(mask.sum())
                    xi_all = innovations[mask]  # (N_k, s)
                    w_k = None
                    w_norm = None

                for i in range(s):
                    ax = fig_pr.add_subplot(nrows, ncols, k * ncols + i + 1)
                    if n_eff > 1:
                        xi = xi_all[:, i]  # (N,) or (N_k,)

                        # ── Axis limits: clip to ±5σ of theoretical (or ±3 empirical σ)
                        if has_theory:
                            sig_th = float(np.sqrt(max(float(S_YY[k][i, i]), 1e-12)))
                            x_lo, x_hi = -5.0 * sig_th, 5.0 * sig_th
                        else:
                            if w_norm is not None:
                                mu_e = float(np.dot(w_norm, xi))
                                std_e = float(np.sqrt(np.dot(w_norm, (xi - mu_e) ** 2)))
                            else:
                                mu_e, std_e = float(xi.mean()), float(xi.std())
                            std_e = max(std_e, 1e-3)
                            x_lo = mu_e - 3.5 * std_e
                            x_hi = mu_e + 3.5 * std_e

                        n_bins = max(10, min(int(n_eff) // 50, 300))
                        ax.hist(
                            xi,
                            bins=n_bins,
                            range=(x_lo, x_hi),
                            weights=w_k,
                            density=True,
                            color=c,
                            alpha=0.5,
                        )
                        ax.set_xlim(x_lo, x_hi)

                        xg = np.linspace(x_lo, x_hi, 400)
                        if has_theory:
                            ax.plot(
                                xg,
                                _norm.pdf(xg, 0.0, sig_th),
                                "k-",
                                linewidth=1.6,
                                label=rf"$\mathcal{{N}}(0,\,\Sigma_{{YY}}({k}))$",
                            )
                            ax.legend(fontsize=7)
                        else:
                            if std_e > 1e-10:
                                ax.plot(xg, _norm.pdf(xg, mu_e, std_e), color=c, linewidth=1.5)

                        # ── Weighted shape statistics ──────────────────────
                        if w_norm is not None:
                            mu_w = float(np.dot(w_norm, xi))
                            xc = xi - mu_w
                            std_w = float(np.sqrt(np.dot(w_norm, xc**2)))
                            if std_w > 1e-10:
                                S2 = float(np.dot(w_norm, (xc / std_w) ** 3))
                                K2 = float(np.dot(w_norm, (xc / std_w) ** 4)) - 3.0
                            else:
                                S2, K2 = 0.0, 0.0
                            stats_str = f"S={S2:+.2f}  K={K2:+.2f}  μ={mu_w:+.2f}"
                        else:
                            S2, K2, _, _ = _shape_diagnostics(xi)
                            stats_str = f"S={S2:+.2f}  K={K2:+.2f}"

                        kind = "soft-resid" if use_resid else "ν"
                        ax.set_title(
                            f"k={k}  {kind}^{i}   N_eff={n_eff:.0f}\n{stats_str}",
                            fontsize=8,
                        )
                    else:
                        ax.set_title(f"k={k}   N_eff={n_eff:.0f}", fontsize=8)
                        ax.text(
                            0.5,
                            0.5,
                            "n/a",
                            transform=ax.transAxes,
                            ha="center",
                            va="center",
                            fontsize=9,
                            color="#aaa",
                        )

                    if use_resid:
                        ax.set_xlabel(rf"$y^{i} - \mu_{{Y,{k}}}^{i}$", fontsize=8)
                    else:
                        ax.set_xlabel(rf"$\nu^{i}$", fontsize=8)
                    ax.grid(True, linestyle=":", alpha=0.4)
                    ax.tick_params(labelsize=7)

            preg_l.addWidget(NavigationToolbar2QT(can_pr, preg_w))
            preg_l.addWidget(can_pr)
            inner_tabs.addTab(preg_w, "Per-regime")

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
            self,
            "Export innovations",
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
                self,
                "Export OK",
                f"Innovations saved to:\n{pathlib.Path(path).resolve()}",
            )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Export error", str(exc))


# ---------------------------------------------------------------------------
# Regime diagnostics dialog (B7 + B8)
# ---------------------------------------------------------------------------


class _RegimeDiagDialog(QDialog):
    """Non-modal dialog: confusion matrix and regime-duration histograms.

    Tabs
    ----
    • Confusion  — K×K table of  argmax_k π_n(k)  vs  true r_n
                   (only when filter results are available).
    • Durations  — per-regime histogram of consecutive run lengths
                   overlaid with the theoretical Geom(1−P_{kk}) PMF.
    """

    def __init__(
        self,
        K: int,
        rs: list,  # true regime sequence, length N
        P: np.ndarray,  # (K, K) transition matrix
        pis: np.ndarray | None = None,  # (N, K) filtered posteriors (None → no filter)
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Regime diagnostics")
        self.setModal(False)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)

        from matplotlib.backends.backend_qtagg import (
            FigureCanvasQTAgg,
            NavigationToolbar2QT,
        )
        from matplotlib.figure import Figure

        rs_arr = np.asarray(rs, dtype=int)
        N = len(rs_arr)

        layout = QVBoxLayout(self)
        tabs = QTabWidget()
        layout.addWidget(tabs)

        colours = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

        # ── Tab 1: Confusion matrix ───────────────────────────────────
        if pis is not None:
            conf_w = QWidget()
            conf_l = QVBoxLayout(conf_w)
            conf_l.setContentsMargins(4, 4, 4, 4)

            r_pred = np.argmax(pis, axis=1)  # (N,) predicted regime

            # K×K confusion matrix (rows = true, cols = predicted)
            conf = np.zeros((K, K), dtype=int)
            for t, p in zip(rs_arr, r_pred):
                conf[t, p] += 1

            accuracy = np.trace(conf) / max(N, 1)

            # ── Title label ──
            # Pill-style label so text is readable on both light and dark themes
            if accuracy > 0.85:
                acc_bg, acc_fg, acc_bd = "#d4edda", "#155724", "#c3e6cb"
            elif accuracy > 0.70:
                acc_bg, acc_fg, acc_bd = "#fff3cd", "#856404", "#ffc107"
            else:
                acc_bg, acc_fg, acc_bd = "#f8d7da", "#721c24", "#f5c6cb"
            acc_lbl = QLabel(
                f"Overall regime accuracy:  {accuracy:.1%}  (N = {N},  argmax π_n  vs  r_n)"
            )
            acc_lbl.setStyleSheet(
                f"font-weight: bold; font-size: 11px; color: {acc_fg};"
                f"background: {acc_bg}; border: 1px solid {acc_bd};"
                "padding: 3px 12px; border-radius: 4px;"
            )
            acc_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            conf_l.addWidget(acc_lbl)

            # ── Matplotlib heatmap ──
            fig_c = Figure(figsize=(max(3.5, 1.8 * K), max(3.0, 1.6 * K)), tight_layout=True)
            can_c = FigureCanvasQTAgg(fig_c)
            can_c.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
            ax_c = fig_c.add_subplot(1, 1, 1)

            # Normalize by true-count per row
            row_sums = conf.sum(axis=1, keepdims=True).clip(1)
            conf_norm = conf / row_sums

            im = ax_c.imshow(conf_norm, vmin=0, vmax=1, cmap="Blues", aspect="equal")
            fig_c.colorbar(im, ax=ax_c, fraction=0.046, pad=0.04, label="Recall per row")

            for i in range(K):
                for j in range(K):
                    n_ij = conf[i, j]
                    pct = conf_norm[i, j]
                    txt = f"{n_ij}\n({pct:.1%})"
                    col = "white" if pct > 0.6 else "#333333"
                    ax_c.text(j, i, txt, ha="center", va="center", fontsize=9, color=col)

            ticks = list(range(K))
            ax_c.set_xticks(ticks)
            ax_c.set_yticks(ticks)
            ax_c.set_xticklabels([f"Pred k={k}" for k in ticks])
            ax_c.set_yticklabels([f"True k={k}" for k in ticks])
            ax_c.set_xlabel("Predicted (argmax π_n)", fontsize=10)
            ax_c.set_ylabel("True (r_n)", fontsize=10)
            ax_c.set_title(f"Confusion matrix  —  accuracy = {accuracy:.1%}", fontsize=10)

            conf_l.addWidget(NavigationToolbar2QT(can_c, conf_w))
            conf_l.addWidget(can_c)

            # ── Per-regime recall/precision text ──
            for k in range(K):
                recall = conf[k, k] / max(conf[k, :].sum(), 1)
                precision = conf[k, k] / max(conf[:, k].sum(), 1)
                lbl = QLabel(
                    f"  k={k}:  recall = {recall:.1%}"
                    f"   precision = {precision:.1%}"
                    f"   support = {conf[k, :].sum()}"
                )
                # Explicit foreground so the label reads on both light & dark themes
                lbl.setStyleSheet(
                    "font-size: 10px; color: palette(windowText);background: transparent;"
                )
                conf_l.addWidget(lbl)

            tabs.addTab(conf_w, "Confusion matrix")

        # ── Tab 2: Duration histograms ────────────────────────────────
        dur_w = QWidget()
        dur_l = QVBoxLayout(dur_w)
        dur_l.setContentsMargins(4, 4, 4, 4)

        ncols = min(K, 3)
        nrows = (K + ncols - 1) // ncols
        fig_d = Figure(figsize=(max(4, 4.5 * ncols), 3.5 * nrows), tight_layout=True)
        can_d = FigureCanvasQTAgg(fig_d)
        can_d.setFocusPolicy(Qt.FocusPolicy.ClickFocus)

        for k in range(K):
            # Extract run lengths in regime k
            runs: list[int] = []
            run_len = 0
            for r in rs_arr:
                if r == k:
                    run_len += 1
                else:
                    if run_len > 0:
                        runs.append(run_len)
                    run_len = 0
            if run_len > 0:
                runs.append(run_len)

            ax_d = fig_d.add_subplot(nrows, ncols, k + 1)
            c = colours[k % len(colours)]

            if runs:
                runs_arr = np.asarray(runs, dtype=float)
                max_len = int(runs_arr.max())
                bins = np.arange(0.5, max_len + 2.5, 1)
                ax_d.hist(runs_arr, bins=bins, density=True, color=c, alpha=0.55, label="observed")

                # Theoretical Geom(p) where p = 1 − P_kk
                p_kk = float(P[k, k])
                q_kk = max(1e-9, 1.0 - p_kk)
                xs_t = np.arange(1, max_len + 2)
                pmf = (1 - q_kk) ** (xs_t - 1) * q_kk  # Geom(q_kk) PMF
                ax_d.step(
                    xs_t - 0.5,
                    pmf,
                    where="post",
                    color=c,
                    linewidth=1.5,
                    linestyle="--",
                    label=f"Geom(1−P_{k}{k}={q_kk:.3f})",
                )
                ax_d.set_xlabel("Run length", fontsize=9)
                ax_d.set_ylabel("Density", fontsize=9)
                mean_obs = float(runs_arr.mean())
                mean_th = 1.0 / q_kk if q_kk > 0 else float("inf")
                ax_d.set_title(
                    f"Regime k={k}   ({len(runs)} runs,  mean {mean_obs:.1f} / th. {mean_th:.1f})",
                    fontsize=9,
                )
                ax_d.legend(fontsize=8)
            else:
                ax_d.text(
                    0.5,
                    0.5,
                    f"No runs in regime k={k}",
                    transform=ax_d.transAxes,
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="#aaa",
                )
                ax_d.set_title(f"Regime k={k}", fontsize=9)
            ax_d.grid(True, linestyle=":", alpha=0.5)

        dur_l.addWidget(NavigationToolbar2QT(can_d, dur_w))
        dur_l.addWidget(can_d)
        tabs.addTab(dur_w, "Duration histograms")

        tabs.setCurrentIndex(0)
        self.resize(max(520, 440 * min(K, 3)), 500)


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
        self._bar.setRange(0, 0)  # indeterminate by default
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
            btn_cancel.setStyleSheet("padding: 4px 16px; font-size: 11px;")
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
