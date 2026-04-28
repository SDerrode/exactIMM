#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prg/gui/plot_panel.py
=====================
PlotPanel — matplotlib canvas embedded in Qt.

Layout:  1 (R_n step plot) + q (X_i) + s (Y_i) subplots stacked vertically.
"""

import numpy as np
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg,
    NavigationToolbar2QT,
)
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QSpinBox, QDoubleSpinBox,
    QRadioButton, QButtonGroup, QGroupBox, QSizePolicy, QTabWidget,
)


class PlotPanel(QWidget):
    """Matplotlib figure stacked vertically.

    Layout (top → bottom):
      [0]                R_n   step plot (simulation only)
      [1 … q]            X^i   hidden components
      [1+q … q+s]        Y^i   observed components
      [1+q+s … q+2s]     ν^i   filter innovations (shorter axes, after Filter)

    NOTE: the π_n(k) (filtered regime posterior) subplot has been
    commented out — see ``_pi_offset = None`` below. The ``add_pi_overlay``
    / ``clear_pi_overlay`` API is kept as a no-op so the main window code
    that calls it does not need to change.
    """

    def __init__(self, q: int, s: int, parent=None):
        super().__init__(parent)
        self._q = q
        self._s = s
        # Subplot index offsets for readable indexing.
        # π_n(k) axis is hidden; _pi_offset is kept as None as a marker.
        self._r_offset     = 0
        self._pi_offset    = None      # was: 1  (π_n(k) plot — hidden)
        self._x_offset     = 1         # was: 2
        self._y_offset     = 1 + q     # was: 2 + q
        self._innov_offset = 1 + q + s # was: 2 + q + s
        self._n_axes       = self._innov_offset + s

        # ν^i axes are 55 % the height of the regular axes
        height_ratios = (
            [1.0]                  # R_n      (π_n(k) row removed)
            + [1.0] * (q + s)      # X^i, Y^i
            + [0.55] * s           # innovations
        )
        fig_h = 2.2 * (1 + q + s) + 1.3 * s
        self._fig = Figure(figsize=(7, fig_h), tight_layout=True)
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._toolbar = NavigationToolbar2QT(self._canvas, self)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._toolbar)
        layout.addWidget(self._canvas)

        self._axes = self._fig.subplots(
            self._n_axes, 1, sharex=True,
            gridspec_kw={"height_ratios": height_ratios},
        )
        if self._n_axes == 1:
            self._axes = [self._axes]
        self._draw_empty()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def update_plots(
        self,
        ns: list[int],
        rs: list[int],
        xs: np.ndarray | None,   # shape (N, q), or None if no ground truth
        ys: np.ndarray,          # shape (N, s)
        K: int,
    ) -> None:
        """Redraw all subplots with fresh simulation data."""
        for ax in self._axes:
            ax.cla()

        ns_arr = np.asarray(ns)

        # --- R_n step plot ---
        ax_r = self._axes[self._r_offset]
        ax_r.step(ns_arr, rs, where="post", color="#555555", linewidth=1.2)
        ax_r.set_ylabel(r"$R_n$", fontsize=10)
        ax_r.set_yticks(range(K))
        ax_r.set_ylim(-0.5, K - 0.5)
        ax_r.grid(True, linestyle=":", alpha=0.5)
        ax_r.set_title("GSS Simulation", fontsize=10)

        # --- π_n(k) placeholder (HIDDEN — see _pi_offset = None) ---
        # ax_pi = self._axes[self._pi_offset]
        # ax_pi.set_ylabel(r"$\pi_n(k)$", fontsize=9)
        # ax_pi.set_yticks([])
        # ax_pi.grid(True, linestyle=":", alpha=0.4)
        # ax_pi.tick_params(labelsize=7)

        # --- X^i components ---
        colours_x = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
        for i in range(self._q):
            ax = self._axes[self._x_offset + i]
            ax.set_ylabel(rf"$X^{i}$", fontsize=10)
            ax.grid(True, linestyle=":", alpha=0.5)
            if xs is not None:
                ax.plot(
                    ns_arr, xs[:, i],
                    color=colours_x[i % len(colours_x)],
                    linewidth=0.9,
                    label=rf"$X^{i}$",
                )
            else:
                ax.set_yticks([])
                ax.text(
                    0.5, 0.5, "no ground truth",
                    transform=ax.transAxes,
                    ha="center", va="center",
                    fontsize=8, color="#aaaaaa", style="italic",
                )

        # --- Y^i components ---
        colours_y = ["#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
        for i in range(self._s):
            ax = self._axes[self._y_offset + i]
            ax.plot(
                ns_arr, ys[:, i],
                color=colours_y[i % len(colours_y)],
                linewidth=0.9,
                label=rf"$Y^{i}$",
            )
            ax.set_ylabel(rf"$Y^{i}$", fontsize=10)
            ax.grid(True, linestyle=":", alpha=0.5)

        self._axes[-1].set_xlabel(r"$n$", fontsize=10)
        # Force x-axis limits to the new data range (sharex propagates).
        # Needed because cla() on multiple shared axes can leave autoscale_x
        # disabled so subsequent plot() calls don't drive the limits.
        if len(ns_arr) > 0:
            self._set_shared_xlim(ns_arr)
        self._canvas.draw_idle()

    def add_filter_overlay(
        self,
        ns: list[int],
        E_xs: np.ndarray,    # (N, q)
        Var_xs: np.ndarray,  # (N, q)
    ) -> None:
        """Overlay filtered estimates on X_i subplots (dashed + ±2σ band)."""
        self.clear_filter_overlay()
        ns_arr = np.asarray(ns)
        colour_filt = "#d62728"   # red

        self._filter_artists: list = []
        for i in range(self._q):
            ax = self._axes[self._x_offset + i]
            mu = E_xs[:, i]
            sigma = np.sqrt(np.maximum(Var_xs[:, i], 0.0))

            line, = ax.plot(
                ns_arr, mu,
                color=colour_filt, linewidth=1.2,
                linestyle="--", label=rf"$\mathbb{{E}}[X^{i}\mid y]$",
                zorder=3,
            )
            fill = ax.fill_between(
                ns_arr, mu - 2 * sigma, mu + 2 * sigma,
                color=colour_filt, alpha=0.15,
                label=r"$\pm 2\sigma$",
                zorder=2,
            )
            ax.legend(fontsize=7, loc="upper right")
            self._filter_artists.extend([line, fill])

        self._canvas.draw_idle()

    def add_pi_overlay(
        self,
        ns: list[int],
        pis: np.ndarray,   # shape (N, K)
        K: int,
    ) -> None:
        """No-op stub — π_n(k) subplot has been hidden.

        Kept so callers in main_window.py do not need to be changed.
        Original implementation preserved below for future re-enablement.
        """
        self._pi_artists = []
        # --- ORIGINAL IMPLEMENTATION (HIDDEN) ---
        # self.clear_pi_overlay()
        # ns_arr = np.asarray(ns)
        # ax_pi  = self._axes[self._pi_offset]
        # ax_pi.cla()
        # colours_pi = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        #               "#8c564b", "#e377c2"]
        # self._pi_artists: list = []
        # for k in range(K):
        #     line, = ax_pi.plot(
        #         ns_arr, pis[:, k],
        #         color=colours_pi[k % len(colours_pi)],
        #         linewidth=1.0, linestyle="-", alpha=0.9,
        #         label=rf"$\pi_n({k})$",
        #     )
        #     self._pi_artists.append(line)
        # ax_pi.set_ylim(-0.05, 1.05)
        # ax_pi.set_ylabel(r"$\pi_n(k)$", fontsize=9)
        # ax_pi.tick_params(labelsize=7)
        # ax_pi.grid(True, linestyle=":", alpha=0.4)
        # ax_pi.legend(fontsize=7, loc="center right", ncol=min(K, 4))
        # self._canvas.draw_idle()

    def clear_pi_overlay(self) -> None:
        """No-op stub — π_n(k) subplot has been hidden."""
        self._pi_artists = []
        # --- ORIGINAL IMPLEMENTATION (HIDDEN) ---
        # ax_pi = self._axes[self._pi_offset]
        # ax_pi.cla()
        # ax_pi.set_ylabel(r"$\pi_n(k)$", fontsize=9)
        # ax_pi.set_yticks([])
        # ax_pi.grid(True, linestyle=":", alpha=0.4)
        # ax_pi.tick_params(labelsize=7)
        # self._canvas.draw_idle()

    def update_innovations(
        self,
        ns: list[int],
        innovations: np.ndarray,   # shape (N, s)
    ) -> None:
        """Plot the filter innovation sequence on the bottom axes."""
        self.clear_innovations()
        ns_arr = np.asarray(ns)
        colours = ["#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

        self._innov_artists: list = []
        for i in range(self._s):
            ax = self._axes[self._innov_offset + i]
            ax.cla()
            ax.axhline(0, color="#999999", linewidth=0.8, linestyle="--")
            line, = ax.plot(
                ns_arr, innovations[:, i],
                color=colours[i % len(colours)],
                linewidth=0.7,
            )
            ax.set_ylabel(rf"$\nu^{i}$", fontsize=9)
            ax.tick_params(labelsize=7)
            ax.grid(True, linestyle=":", alpha=0.4)
            self._innov_artists.append(line)

        self._axes[-1].set_xlabel(r"$n$", fontsize=10)
        if len(ns_arr) > 0:
            self._set_shared_xlim(ns_arr)
        self._canvas.draw_idle()

    def clear_innovations(self) -> None:
        """Clear innovation plots and restore empty labels."""
        for a in getattr(self, "_innov_artists", []):
            try:
                a.remove()
            except (ValueError, NotImplementedError, AttributeError):
                # Artist may already be detached after a parent ax.cla().
                pass
        self._innov_artists = []
        for i in range(self._s):
            ax = self._axes[self._innov_offset + i]
            ax.cla()
            ax.set_ylabel(rf"$\nu^{i}$", fontsize=9)
            ax.set_yticks([])
            ax.grid(True, linestyle=":", alpha=0.4)
            ax.tick_params(labelsize=7)
        self._axes[-1].set_xlabel(r"$n$", fontsize=10)
        self._canvas.draw_idle()

    def update_mc_plots(
        self,
        ns: list[int],
        mean_xs: np.ndarray,      # (N, q)
        std_xs: np.ndarray,       # (N, q)
        median_xs: np.ndarray,    # (N, q)
        mean_ys: np.ndarray,      # (N, s)
        std_ys: np.ndarray,       # (N, s)
        median_ys: np.ndarray,    # (N, s)
        regime_freqs: np.ndarray, # (N, K)
        K: int,
        M: int,
    ) -> None:
        """Redraw all subplots with Monte-Carlo statistics (mean ± 2σ + median)."""
        self._clear_mc_plots()
        self.clear_filter_overlay()
        self.clear_innovations()
        for ax in self._axes:
            ax.cla()

        ns_arr = np.asarray(ns)

        # --- R_n: regime frequencies ---
        ax_r = self._axes[self._r_offset]
        colours_r = ["#555555", "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
        for k in range(K):
            ax_r.plot(
                ns_arr, regime_freqs[:, k],
                color=colours_r[k % len(colours_r)],
                linewidth=1.0, label=f"P(R={k})",
            )
        ax_r.set_ylabel("P(R=k)", fontsize=10)
        ax_r.set_ylim(-0.05, 1.05)
        ax_r.legend(fontsize=7, loc="upper right")
        ax_r.grid(True, linestyle=":", alpha=0.5)
        ax_r.set_title(f"GSS Monte Carlo  (M = {M})", fontsize=10)

        # --- π_n(k) axis HIDDEN — see _pi_offset = None ---
        # ax_pi = self._axes[self._pi_offset]
        # ax_pi.set_ylabel(r"$\pi_n(k)$", fontsize=9)
        # ax_pi.set_yticks([])
        # ax_pi.grid(True, linestyle=":", alpha=0.4)
        # ax_pi.tick_params(labelsize=7)

        # --- X^i components: mean ± 2σ + median ---
        colours_x = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
        self._mc_artists: list = []
        for i in range(self._q):
            ax = self._axes[self._x_offset + i]
            mu     = mean_xs[:, i]
            sigma  = std_xs[:, i]
            med    = median_xs[:, i]
            c = colours_x[i % len(colours_x)]
            line_mean, = ax.plot(ns_arr, mu, color=c, linewidth=1.0,
                                 label=rf"$\bar{{X}}^{i}$  (mean)")
            line_med,  = ax.plot(ns_arr, med, color=c, linewidth=1.0,
                                 linestyle="--",
                                 label=rf"$\tilde{{X}}^{i}$  (median)")
            fill = ax.fill_between(
                ns_arr, mu - 2 * sigma, mu + 2 * sigma,
                color=c, alpha=0.18, label=r"$\pm 2\sigma$",
            )
            ax.set_ylabel(rf"$X^{i}$", fontsize=10)
            ax.legend(fontsize=7, loc="upper right")
            ax.grid(True, linestyle=":", alpha=0.5)
            self._mc_artists.extend([line_mean, line_med, fill])

        # --- Y^i components: mean ± 2σ + median ---
        colours_y = ["#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
        for i in range(self._s):
            ax = self._axes[self._y_offset + i]
            mu    = mean_ys[:, i]
            sigma = std_ys[:, i]
            med   = median_ys[:, i]
            c = colours_y[i % len(colours_y)]
            line_mean, = ax.plot(ns_arr, mu, color=c, linewidth=1.0,
                                 label=rf"$\bar{{Y}}^{i}$  (mean)")
            line_med,  = ax.plot(ns_arr, med, color=c, linewidth=1.0,
                                 linestyle="--",
                                 label=rf"$\tilde{{Y}}^{i}$  (median)")
            fill = ax.fill_between(
                ns_arr, mu - 2 * sigma, mu + 2 * sigma,
                color=c, alpha=0.18, label=r"$\pm 2\sigma$",
            )
            ax.set_ylabel(rf"$Y^{i}$", fontsize=10)
            ax.legend(fontsize=7, loc="upper right")
            ax.grid(True, linestyle=":", alpha=0.5)
            self._mc_artists.extend([line_mean, line_med, fill])

        # Innovation axes stay empty in MC mode
        for i in range(self._s):
            ax = self._axes[self._innov_offset + i]
            ax.set_ylabel(rf"$\nu^{i}$", fontsize=9)
            ax.set_yticks([])
            ax.grid(True, linestyle=":", alpha=0.4)
            ax.tick_params(labelsize=7)

        self._axes[-1].set_xlabel(r"$n$", fontsize=10)
        if len(ns_arr) > 0:
            self._set_shared_xlim(ns_arr)
        self._canvas.draw_idle()

    def _clear_mc_plots(self) -> None:
        """Remove Monte-Carlo overlay artists."""
        for a in getattr(self, "_mc_artists", []):
            try:
                a.remove()
            except (ValueError, NotImplementedError, AttributeError):
                pass
        self._mc_artists = []

    def save_figure(self, path: str) -> None:
        """Save the current figure to *path* (PNG, PDF, SVG… via extension)."""
        self._fig.savefig(path, dpi=150, bbox_inches="tight")

    def clear(self) -> None:
        """Clear all plots and restore the empty-state message."""
        self._clear_mc_plots()
        self.clear_filter_overlay()
        self.clear_pi_overlay()
        self.clear_innovations()
        for ax in self._axes:
            ax.cla()
        self._draw_empty()

    def clear_filter_overlay(self) -> None:
        """Remove previously drawn filter overlay artists."""
        artists = getattr(self, "_filter_artists", [])
        for a in artists:
            try:
                a.remove()
            except (ValueError, NotImplementedError, AttributeError):
                pass
        self._filter_artists = []
        # Also clear legends that may reference removed artists
        for i in range(self._q):
            leg = self._axes[self._x_offset + i].get_legend()
            if leg is not None:
                leg.remove()
        # π_n(k) is always produced together with the filter overlay,
        # so tear it down here as well.
        self.clear_pi_overlay()
        self._canvas.draw_idle()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _set_shared_xlim(self, ns_arr: np.ndarray) -> None:
        """Force xlim on the shared x-axis to match the new data range.

        After cla() on multiple sharex=True axes, autoscale on x can be
        left disabled, so plotted data alone does not drive the limits
        (an old, larger N keeps the old span). Setting xlim explicitly
        on one axis propagates to all linked axes.
        """
        x0, x1 = float(ns_arr[0]), float(ns_arr[-1])
        if x0 == x1:
            x0, x1 = x0 - 0.5, x1 + 0.5
        # Re-enable autoscaling so future plots behave normally,
        # then pin the limits to the current data span.
        for ax in self._axes:
            ax.set_autoscalex_on(True)
        self._axes[0].set_xlim(x0, x1)

    def _draw_empty(self) -> None:
        """Show labelled empty axes (no data, no message)."""
        ax_r = self._axes[self._r_offset]
        ax_r.set_ylabel(r"$R_n$", fontsize=10)
        ax_r.set_yticks([])
        ax_r.grid(True, linestyle=":", alpha=0.4)
        ax_r.set_title("GSS Simulation", fontsize=10)

        # π_n(k) axis HIDDEN — see _pi_offset = None
        # ax_pi = self._axes[self._pi_offset]
        # ax_pi.set_ylabel(r"$\pi_n(k)$", fontsize=9)
        # ax_pi.set_yticks([])
        # ax_pi.grid(True, linestyle=":", alpha=0.4)
        # ax_pi.tick_params(labelsize=7)

        for i in range(self._q):
            ax = self._axes[self._x_offset + i]
            ax.set_ylabel(rf"$X^{i}$", fontsize=10)
            ax.set_yticks([])
            ax.grid(True, linestyle=":", alpha=0.4)

        for i in range(self._s):
            ax = self._axes[self._y_offset + i]
            ax.set_ylabel(rf"$Y^{i}$", fontsize=10)
            ax.set_yticks([])
            ax.grid(True, linestyle=":", alpha=0.4)

        for i in range(self._s):
            ax = self._axes[self._innov_offset + i]
            ax.set_ylabel(rf"$\nu^{i}$", fontsize=9)
            ax.set_yticks([])
            ax.grid(True, linestyle=":", alpha=0.4)
            ax.tick_params(labelsize=7)

        self._axes[-1].set_xlabel(r"$n$", fontsize=10)
        self._canvas.draw_idle()


# ---------------------------------------------------------------------------
# PredYPanel — p(y_{n+1} | r_n=j, r_{n+1}=k, y_n)
# ---------------------------------------------------------------------------

class PredYPanel(QWidget):
    """Onglet dédié à la distribution conditionnelle p(y_{n+1} | r_n=j, r_{n+1}=k, y_n).

    Disponible après filtrage en mode « h5_exact ». Contient deux sous-onglets :

    • Trajectoire — espérance E[y_{n+1} | j, k, y_n] tracée sur toute la
      durée de la simulation, avec enveloppe ±1σ / ±2σ et overlay de y_{n+1}
      observé. La covariance Γ(j,k) est constante (indépendante de n).

    • Densité — densité gaussienne exacte p(y_{n+1} | j, k, y_n) pour un
      instant n choisi (depuis la trajectoire ou valeur libre).

    Rappel mathématique :
        Moyenne  : μ(n) = mu_Y_jk[j][k] + M_t[j][k] @ (y_n − mu_Y[j])   (affine en y_n)
        Covariance: Γ(j,k)                                                 (constante en n)
    """

    _COLOURS = ["#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#1f77b4", "#ff7f0e"]

    def __init__(self, K: int, q: int, s: int, parent=None):
        super().__init__(parent)
        self._K = K
        self._q = q
        self._s = s

        # Moments pré-calculés (fournis par le filtre après son exécution)
        self._mu_Y_jk:  list | None = None   # [K][K] ndarray (s,1)
        self._M_t:      list | None = None   # [K][K] ndarray (s,s)
        self._Gamma:    list | None = None   # [K][K] ndarray (s,s)
        self._mu_Y:     list | None = None   # [K]    ndarray (s,1)
        self._M_simple: list | None = None   # [K][K] ndarray (s,s) — coeff signal 2
        self._Gamma2:   list | None = None   # [K][K] ndarray (s,s) — cov. signal 2
        self._ys:       np.ndarray | None = None  # (N, s)

        self._build_ui()
        self._draw_traj_empty()
        self._draw_density_empty()

    # ------------------------------------------------------------------
    # Construction de l'interface
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        K, s = self._K, self._s
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # ── Contrôles partagés : sélection du couple de régimes ─────────
        ctrl = QHBoxLayout()
        ctrl.setSpacing(12)

        regime_box = QGroupBox("Régimes")
        regime_box.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        rr = QHBoxLayout(regime_box)
        rr.setSpacing(8)
        rr.addWidget(QLabel("r_n = j :"))
        self._j_combo = QComboBox()
        for jj in range(K):
            self._j_combo.addItem(str(jj))
        self._j_combo.setFixedWidth(60)
        rr.addWidget(self._j_combo)
        rr.addSpacing(10)
        rr.addWidget(QLabel("r_{n+1} = k :"))
        self._k_combo = QComboBox()
        for kk in range(K):
            self._k_combo.addItem(str(kk))
        self._k_combo.setFixedWidth(60)
        rr.addWidget(self._k_combo)
        ctrl.addWidget(regime_box)
        ctrl.addStretch()
        root.addLayout(ctrl)

        # ── Sous-onglets : Trajectoire / Densité ────────────────────────
        self._inner_tabs = QTabWidget()
        root.addWidget(self._inner_tabs)

        # --- Onglet 0 : Trajectoire ---
        traj_widget = QWidget()
        traj_layout = QVBoxLayout(traj_widget)
        traj_layout.setContentsMargins(0, 0, 0, 0)
        self._fig_traj    = Figure(tight_layout=True)
        self._canvas_traj = FigureCanvasQTAgg(self._fig_traj)
        traj_layout.addWidget(NavigationToolbar2QT(self._canvas_traj, traj_widget))
        traj_layout.addWidget(self._canvas_traj)
        self._inner_tabs.addTab(traj_widget, "Trajectoire")

        # --- Onglet 1 : Densité ---
        dens_widget = QWidget()
        dens_layout = QVBoxLayout(dens_widget)
        dens_layout.setContentsMargins(4, 4, 4, 4)
        dens_layout.setSpacing(4)

        # Contrôles y_n (propres à l'onglet Densité)
        src_box = QGroupBox("Valeur de y_n")
        src_layout = QVBoxLayout(src_box)
        src_layout.setSpacing(4)

        traj_row = QHBoxLayout()
        self._src_traj = QRadioButton("Depuis la trajectoire, n =")
        self._src_traj.setChecked(True)
        traj_row.addWidget(self._src_traj)
        self._n_spin = QSpinBox()
        self._n_spin.setRange(0, 0)
        self._n_spin.setValue(0)
        self._n_spin.setFixedWidth(80)
        self._n_spin.setToolTip("Indice temporel n (y_n lue dans la trajectoire simulée)")
        traj_row.addWidget(self._n_spin)
        traj_row.addStretch()
        src_layout.addLayout(traj_row)

        free_row = QHBoxLayout()
        self._src_free = QRadioButton("Valeur libre :")
        free_row.addWidget(self._src_free)
        self._yn_spins: list[QDoubleSpinBox] = []
        for i in range(s):
            lbl = QLabel(f"y^{i} =")
            sp  = QDoubleSpinBox()
            sp.setRange(-1e6, 1e6)
            sp.setValue(0.0)
            sp.setDecimals(4)
            sp.setFixedWidth(100)
            sp.setEnabled(False)
            sp.setToolTip(f"Composante y^{i} de y_n (saisie libre)")
            free_row.addWidget(lbl)
            free_row.addWidget(sp)
            self._yn_spins.append(sp)
        free_row.addStretch()
        src_layout.addLayout(free_row)

        self._src_group = QButtonGroup(self)
        self._src_group.addButton(self._src_traj, 0)
        self._src_group.addButton(self._src_free,  1)
        dens_layout.addWidget(src_box)

        self._fig_dens    = Figure(tight_layout=True)
        self._canvas_dens = FigureCanvasQTAgg(self._fig_dens)
        dens_layout.addWidget(NavigationToolbar2QT(self._canvas_dens, dens_widget))
        dens_layout.addWidget(self._canvas_dens)
        self._inner_tabs.addTab(dens_widget, "Densité")

        # ── Connexions ────────────────────────────────────────────────────
        self._j_combo.currentIndexChanged.connect(self._refresh_both)
        self._k_combo.currentIndexChanged.connect(self._refresh_both)
        self._n_spin.valueChanged.connect(self._on_n_changed)
        self._src_traj.toggled.connect(self._on_src_toggled)
        for sp in self._yn_spins:
            sp.valueChanged.connect(self._refresh_density)

    # ------------------------------------------------------------------
    # API publique
    # ------------------------------------------------------------------

    def set_data(
        self,
        mu_Y_jk:  list,
        M_t:      list,
        Gamma:    list,
        mu_Y:     list,
        ys:       np.ndarray,   # (N, s)
        M_simple: list | None = None,   # [K][K] ndarray (s,s) — coeff signal 2
        Gamma2:   list | None = None,   # [K][K] ndarray (s,s) — cov. signal 2
    ) -> None:
        """Charge les moments du filtre et la trajectoire observée."""
        self._mu_Y_jk  = mu_Y_jk
        self._M_t      = M_t
        self._Gamma    = Gamma
        self._mu_Y     = mu_Y
        self._M_simple = M_simple
        self._Gamma2   = Gamma2
        self._ys       = ys
        N = len(ys)
        # n ∈ {0…N-2} : y_n→ y_{n+1}
        self._n_spin.setRange(0, max(0, N - 2))
        self._n_spin.setValue(min(self._n_spin.value(), N - 2))
        self._refresh_both()

    def clear(self) -> None:
        """Remet le panneau à son état initial (pas de données)."""
        self._mu_Y_jk  = None
        self._M_t      = None
        self._Gamma    = None
        self._mu_Y     = None
        self._M_simple = None
        self._Gamma2   = None
        self._ys       = None
        self._draw_traj_empty()
        self._draw_density_empty()

    # ------------------------------------------------------------------
    # Slots internes
    # ------------------------------------------------------------------

    def _refresh_both(self) -> None:
        self._refresh_traj()
        self._refresh_density()

    def _on_n_changed(self) -> None:
        """Mise à jour de la densité lorsque n change."""
        self._refresh_density()

    def _on_src_toggled(self, _: bool) -> None:
        traj = self._src_traj.isChecked()
        self._n_spin.setEnabled(traj)
        for sp in self._yn_spins:
            sp.setEnabled(not traj)
        self._refresh_density()

    def _get_yn(self) -> np.ndarray:
        """Retourne y_n sous forme (s, 1)."""
        if self._src_traj.isChecked() and self._ys is not None:
            n = self._n_spin.value()
            return self._ys[n].reshape(-1, 1)
        return np.array([[sp.value()] for sp in self._yn_spins], dtype=float)

    # ------------------------------------------------------------------
    # Sous-onglet 0 : Trajectoire
    # ------------------------------------------------------------------

    _COL_SIG1 = "#1f77b4"   # bleu — signal 1  (H5 exact)
    _COL_SIG2 = "#ff7f0e"   # orange — signal 2 (approx. sans H5)

    def _refresh_traj(self) -> None:
        """Trace les deux signaux conditionnels sur toute la trajectoire.

        Signal 1 — exact sous (H5) :
            μ₁(n) = μ_Y_jk[j][k] + M_t[j][k] @ (y_n − μ_Y[j])
            Cov₁  = Γ(j,k)  (constante)

        Signal 2 — approximation sans (H5) :
            μ₂(n) = M_simple[j][k] @ y_n
            Cov₂  = Γ₂(j,k)  (constante)

        Seule l'enveloppe ±2σ est tracée (pas de ±1σ). Pas de ligne verticale.
        """
        if self._mu_Y_jk is None or self._ys is None:
            return

        j = self._j_combo.currentIndex()
        k = self._k_combo.currentIndex()
        s = self._s
        ys = self._ys           # (N, s)
        N  = len(ys)
        ns = np.arange(N - 1)  # indices n = 0 … N-2

        # ── Signal 1 ────────────────────────────────────────────────────
        diffs  = ys[:-1] - self._mu_Y[j].ravel()              # (N-1, s)
        means1 = self._mu_Y_jk[j][k].ravel() + (self._M_t[j][k] @ diffs.T).T  # (N-1, s)
        sigs1  = np.sqrt(np.maximum(np.diag(self._Gamma[j][k]), 1e-12))         # (s,)

        # ── Signal 2 (si disponible) ────────────────────────────────────
        has_sig2 = self._M_simple is not None and self._Gamma2 is not None
        if has_sig2:
            means2 = (self._M_simple[j][k] @ ys[:-1].T).T   # (N-1, s)
            sigs2  = np.sqrt(np.maximum(np.diag(self._Gamma2[j][k]), 1e-12))   # (s,)

        # y_{n+1} observé
        y_obs = ys[1:]   # (N-1, s)

        self._fig_traj.clf()
        axes = (self._fig_traj.subplots(s, 1, sharex=True)
                if s > 1 else [self._fig_traj.add_subplot(1, 1, 1)])

        c1 = self._COL_SIG1
        c2 = self._COL_SIG2

        for i in range(s):
            ax = axes[i]

            # ── Signal 1 ────────────────────────────────────────────────
            mu1_i  = means1[:, i]
            sig1_i = sigs1[i]
            env1 = ax.fill_between(ns, mu1_i - 2 * sig1_i, mu1_i + 2 * sig1_i,
                                   color=c1, alpha=0.15)
            line1, = ax.plot(ns, mu1_i, color=c1, linewidth=1.5)

            # ── Signal 2 ────────────────────────────────────────────────
            if has_sig2:
                mu2_i  = means2[:, i]
                sig2_i = sigs2[i]
                env2 = ax.fill_between(ns, mu2_i - 2 * sig2_i, mu2_i + 2 * sig2_i,
                                       color=c2, alpha=0.15)
                line2, = ax.plot(ns, mu2_i, color=c2, linewidth=1.5)

            # ── y_{n+1} observé ─────────────────────────────────────────
            obs_line, = ax.plot(ns, y_obs[:, i], color="#333333", linewidth=0.8,
                                alpha=0.6, linestyle="-")

            # ── Légende 2 colonnes : gauche = espérances + obs,
            #                         droite = enveloppes ──────────────────
            # Ordre interleaved pour que ncol=2 aligne correctement les cols
            leg_h = [line1,                          env1]
            leg_l = [rf"$\mu_1$ (H5 exact)",         rf"$\pm 2\sigma_1={sig1_i:.3g}$"]
            if has_sig2:
                leg_h += [line2,                     env2]
                leg_l += [rf"$\mu_2$ (approx.)",     rf"$\pm 2\sigma_2={sigs2[i]:.3g}$"]
            leg_h += [obs_line]
            leg_l += [rf"$y^{i}_{{n+1}}$ observé"]
            ax.legend(leg_h, leg_l, fontsize=8, loc="upper right", ncol=2)

            ax.set_ylabel(rf"$y^{i}_{{n+1}}$", fontsize=10)
            ax.grid(True, linestyle=":", alpha=0.4)

        axes[-1].set_xlabel(r"$n$", fontsize=10)
        self._fig_traj.suptitle(
            rf"$p(y_{{n+1}} \mid r_n={j},\; r_{{n+1}}={k},\; y_n)$"
            r"  —  enveloppes $\pm 2\sigma$",
            fontsize=10,
        )
        self._canvas_traj.draw_idle()

    def _draw_traj_empty(self) -> None:
        self._fig_traj.clf()
        ax = self._fig_traj.add_subplot(1, 1, 1)
        ax.set_xticks([]); ax.set_yticks([])
        ax.text(
            0.5, 0.5,
            "Lancez le filtre en mode\n"
            "« Exact IMM - H5 required »\n"
            "pour afficher la trajectoire\n"
            r"$\mathbb{E}[y_{n+1} \mid r_n,\; r_{n+1},\; y_n]$",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=11, color="#999999", style="italic",
        )
        ax.grid(True, linestyle=":", alpha=0.3)
        self._canvas_traj.draw_idle()

    # ------------------------------------------------------------------
    # Sous-onglet 1 : Densité
    # ------------------------------------------------------------------

    def _refresh_density(self) -> None:
        """Recalcule et redessine la (les) densité(s) gaussienne(s) pour le n courant."""
        if self._mu_Y_jk is None:
            return
        j = self._j_combo.currentIndex()
        k = self._k_combo.currentIndex()
        y_n = self._get_yn()

        # Signal 1 — exact sous (H5)
        mu1    = self._mu_Y_jk[j][k] + self._M_t[j][k] @ (y_n - self._mu_Y[j])
        Gamma1 = self._Gamma[j][k]

        # Signal 2 — approximation (si disponible)
        has_sig2 = self._M_simple is not None and self._Gamma2 is not None
        mu2    = self._M_simple[j][k] @ y_n if has_sig2 else None
        Gamma2 = self._Gamma2[j][k]         if has_sig2 else None

        self._fig_dens.clf()
        if self._s == 1:
            self._plot_1d(mu1, Gamma1, mu2, Gamma2, j, k, y_n)
        elif self._s == 2:
            self._plot_2d(mu1, Gamma1, mu2, Gamma2, j, k, y_n)
        else:
            self._plot_nd(mu1, Gamma1, mu2, Gamma2, j, k, y_n)
        self._canvas_dens.draw_idle()

    def _plot_1d(
        self,
        mu1: np.ndarray, Gamma1: np.ndarray,
        mu2: np.ndarray | None, Gamma2: np.ndarray | None,
        j: int, k: int, y_n: np.ndarray,
    ) -> None:
        from scipy.stats import norm as _norm
        c1 = self._COL_SIG1
        c2 = self._COL_SIG2

        m1   = float(mu1[0, 0])
        sig1 = float(np.sqrt(max(float(Gamma1[0, 0]), 1e-12)))

        # x range : union des ±4.5σ des deux signaux
        x_lo, x_hi = m1 - 4.5 * sig1, m1 + 4.5 * sig1
        if mu2 is not None:
            m2   = float(mu2[0, 0])
            sig2 = float(np.sqrt(max(float(Gamma2[0, 0]), 1e-12)))
            x_lo = min(x_lo, m2 - 4.5 * sig2)
            x_hi = max(x_hi, m2 + 4.5 * sig2)

        x = np.linspace(x_lo, x_hi, 600)
        ax = self._fig_dens.add_subplot(1, 1, 1)

        # ── Signal 1 ────────────────────────────────────────────────────
        pdf1 = _norm.pdf(x, m1, sig1)
        ax.plot(x, pdf1, color=c1, linewidth=2.0,
                label=rf"$p_1$  H5 exact  ($\mu_1={m1:.4g}$, $\sigma_1={sig1:.4g}$)")
        ax.fill_between(x, pdf1, where=(np.abs(x - m1) <= sig1),
                        color=c1, alpha=0.30, label=r"$\pm 1\sigma_1$")
        ax.fill_between(x, pdf1, where=(np.abs(x - m1) <= 2 * sig1),
                        color=c1, alpha=0.15, label=r"$\pm 2\sigma_1$")
        ax.axvline(m1, color=c1, linewidth=1.2, linestyle="--", alpha=0.8)

        # ── Signal 2 ────────────────────────────────────────────────────
        if mu2 is not None:
            pdf2 = _norm.pdf(x, m2, sig2)
            ax.plot(x, pdf2, color=c2, linewidth=2.0, linestyle="--",
                    label=rf"$p_2$  approx.  ($\mu_2={m2:.4g}$, $\sigma_2={sig2:.4g}$)")
            ax.fill_between(x, pdf2, where=(np.abs(x - m2) <= sig2),
                            color=c2, alpha=0.20, label=r"$\pm 1\sigma_2$")
            ax.fill_between(x, pdf2, where=(np.abs(x - m2) <= 2 * sig2),
                            color=c2, alpha=0.10, label=r"$\pm 2\sigma_2$")
            ax.axvline(m2, color=c2, linewidth=1.2, linestyle="--", alpha=0.8)

        # ── y_n ─────────────────────────────────────────────────────────
        if self._src_traj.isChecked() and self._ys is not None:
            yn_val = float(y_n[0, 0])
            ax.axvline(yn_val, color="#333333", linewidth=1.0,
                       linestyle=":", alpha=0.8, label=rf"$y_n = {yn_val:.4g}$")

        ax.set_xlabel(r"$y^0_{n+1}$", fontsize=11)
        ax.set_ylabel("densité", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, linestyle=":", alpha=0.4)
        n_lbl = f"  n = {self._n_spin.value()}" if self._src_traj.isChecked() else ""
        ax.set_title(
            rf"$p(y_{{n+1}} \mid r_n={j},\; r_{{n+1}}={k},\; y_n)${n_lbl}",
            fontsize=10,
        )

    def _plot_2d(
        self,
        mu1: np.ndarray, Gamma1: np.ndarray,
        mu2: np.ndarray | None, Gamma2: np.ndarray | None,
        j: int, k: int, y_n: np.ndarray,
    ) -> None:
        from scipy.stats import norm as _norm, multivariate_normal as _mvn
        c1 = self._COL_SIG1
        c2 = self._COL_SIG2

        m1_0 = float(mu1[0, 0]); m1_1 = float(mu1[1, 0])
        s1_0 = float(np.sqrt(max(float(Gamma1[0, 0]), 1e-12)))
        s1_1 = float(np.sqrt(max(float(Gamma1[1, 1]), 1e-12)))

        has2 = mu2 is not None
        if has2:
            m2_0 = float(mu2[0, 0]); m2_1 = float(mu2[1, 0])
            s2_0 = float(np.sqrt(max(float(Gamma2[0, 0]), 1e-12)))
            s2_1 = float(np.sqrt(max(float(Gamma2[1, 1]), 1e-12)))

        def _marginal(ax, dim, m1v, s1v, m2v=None, s2v=None, xlabel=""):
            lo = m1v - 4.5 * s1v; hi = m1v + 4.5 * s1v
            if m2v is not None:
                lo = min(lo, m2v - 4.5 * s2v); hi = max(hi, m2v + 4.5 * s2v)
            x   = np.linspace(lo, hi, 400)
            pdf = _norm.pdf(x, m1v, s1v)
            ax.plot(x, pdf, color=c1, linewidth=2.0,
                    label=rf"$p_1$  ($\mu={m1v:.3g}$, $\sigma={s1v:.3g}$)")
            ax.fill_between(x, pdf, where=(np.abs(x - m1v) <= s1v),
                            color=c1, alpha=0.30, label=r"$\pm 1\sigma_1$")
            ax.fill_between(x, pdf, where=(np.abs(x - m1v) <= 2 * s1v),
                            color=c1, alpha=0.15, label=r"$\pm 2\sigma_1$")
            ax.axvline(m1v, color=c1, linewidth=1.0, linestyle="--", alpha=0.8)
            if m2v is not None:
                pdf2 = _norm.pdf(x, m2v, s2v)
                ax.plot(x, pdf2, color=c2, linewidth=2.0, linestyle="--",
                        label=rf"$p_2$  ($\mu={m2v:.3g}$, $\sigma={s2v:.3g}$)")
                ax.fill_between(x, pdf2, where=(np.abs(x - m2v) <= s2v),
                                color=c2, alpha=0.20, label=r"$\pm 1\sigma_2$")
                ax.fill_between(x, pdf2, where=(np.abs(x - m2v) <= 2 * s2v),
                                color=c2, alpha=0.10, label=r"$\pm 2\sigma_2$")
                ax.axvline(m2v, color=c2, linewidth=1.0, linestyle="--", alpha=0.8)
            ax.set_xlabel(xlabel, fontsize=10); ax.set_ylabel("densité", fontsize=9)
            ax.legend(fontsize=7); ax.grid(True, linestyle=":", alpha=0.4)

        ax0 = self._fig_dens.add_subplot(1, 3, 1)
        _marginal(ax0, 0, m1_0, s1_0,
                  m2_0 if has2 else None, s2_0 if has2 else None,
                  xlabel=r"$y^0_{n+1}$")
        ax0.set_title(rf"Marginale $y^0$", fontsize=9)

        ax1 = self._fig_dens.add_subplot(1, 3, 2)
        _marginal(ax1, 1, m1_1, s1_1,
                  m2_1 if has2 else None, s2_1 if has2 else None,
                  xlabel=r"$y^1_{n+1}$")
        ax1.set_title(rf"Marginale $y^1$", fontsize=9)

        ax2 = self._fig_dens.add_subplot(1, 3, 3)
        # Signal 1 — contours bleus
        gx = np.linspace(m1_0 - 4 * s1_0, m1_0 + 4 * s1_0, 100)
        gy = np.linspace(m1_1 - 4 * s1_1, m1_1 + 4 * s1_1, 100)
        if has2:
            gx = np.linspace(min(m1_0 - 4*s1_0, m2_0 - 4*s2_0),
                             max(m1_0 + 4*s1_0, m2_0 + 4*s2_0), 100)
            gy = np.linspace(min(m1_1 - 4*s1_1, m2_1 - 4*s2_1),
                             max(m1_1 + 4*s1_1, m2_1 + 4*s2_1), 100)
        XX, YY = np.meshgrid(gx, gy)
        pos = np.stack([XX, YY], axis=-1)
        G1sym = (Gamma1 + Gamma1.T) / 2
        try:
            ZZ1 = _mvn.pdf(pos, mean=[m1_0, m1_1], cov=G1sym)
            ax2.contourf(XX, YY, ZZ1, levels=8, cmap="Blues", alpha=0.50)
            ax2.contour(XX, YY, ZZ1, levels=8, colors=[c1], alpha=0.60, linewidths=0.8)
        except Exception:
            pass
        ax2.scatter([m1_0], [m1_1], color=c1, s=50, zorder=5,
                    label=rf"$\mu_1 = ({m1_0:.3g},\,{m1_1:.3g})$")
        # Signal 2 — contours oranges (superposés)
        if has2:
            G2sym = (Gamma2 + Gamma2.T) / 2
            try:
                ZZ2 = _mvn.pdf(pos, mean=[m2_0, m2_1], cov=G2sym)
                ax2.contourf(XX, YY, ZZ2, levels=8, cmap="Oranges", alpha=0.35)
                ax2.contour(XX, YY, ZZ2, levels=8, colors=[c2], alpha=0.70, linewidths=0.8)
            except Exception:
                pass
            ax2.scatter([m2_0], [m2_1], color=c2, s=50, zorder=5, marker="D",
                        label=rf"$\mu_2 = ({m2_0:.3g},\,{m2_1:.3g})$")
        ax2.set_xlabel(r"$y^0_{n+1}$", fontsize=10); ax2.set_ylabel(r"$y^1_{n+1}$", fontsize=10)
        ax2.set_title("Densité jointe", fontsize=9)
        ax2.legend(fontsize=8); ax2.grid(True, linestyle=":", alpha=0.3)
        self._fig_dens.suptitle(
            rf"$p(y_{{n+1}} \mid r_n={j},\; r_{{n+1}}={k},\; y_n)$", fontsize=10)

    def _plot_nd(
        self,
        mu1: np.ndarray, Gamma1: np.ndarray,
        mu2: np.ndarray | None, Gamma2: np.ndarray | None,
        j: int, k: int, y_n: np.ndarray,
    ) -> None:
        from scipy.stats import norm as _norm
        c1 = self._COL_SIG1
        c2 = self._COL_SIG2
        has2 = mu2 is not None
        s = self._s; ncols = min(s, 3); nrows = (s + ncols - 1) // ncols
        for i in range(s):
            ax   = self._fig_dens.add_subplot(nrows, ncols, i + 1)
            m1v  = float(mu1[i, 0])
            s1v  = float(np.sqrt(max(float(Gamma1[i, i]), 1e-12)))
            lo, hi = m1v - 4.5 * s1v, m1v + 4.5 * s1v
            if has2:
                m2v = float(mu2[i, 0])
                s2v = float(np.sqrt(max(float(Gamma2[i, i]), 1e-12)))
                lo  = min(lo, m2v - 4.5 * s2v)
                hi  = max(hi, m2v + 4.5 * s2v)
            x    = np.linspace(lo, hi, 400)
            pdf1 = _norm.pdf(x, m1v, s1v)
            ax.plot(x, pdf1, color=c1, linewidth=2.0,
                    label=rf"$p_1$  ($\mu={m1v:.3g}$, $\sigma={s1v:.3g}$)")
            ax.fill_between(x, pdf1, where=(np.abs(x - m1v) <= s1v),
                            color=c1, alpha=0.30, label=r"$\pm 1\sigma_1$")
            ax.fill_between(x, pdf1, where=(np.abs(x - m1v) <= 2 * s1v),
                            color=c1, alpha=0.15, label=r"$\pm 2\sigma_1$")
            ax.axvline(m1v, color=c1, linewidth=1.0, linestyle="--", alpha=0.8)
            if has2:
                pdf2 = _norm.pdf(x, m2v, s2v)
                ax.plot(x, pdf2, color=c2, linewidth=2.0, linestyle="--",
                        label=rf"$p_2$  ($\mu={m2v:.3g}$, $\sigma={s2v:.3g}$)")
                ax.fill_between(x, pdf2, where=(np.abs(x - m2v) <= s2v),
                                color=c2, alpha=0.20, label=r"$\pm 1\sigma_2$")
                ax.fill_between(x, pdf2, where=(np.abs(x - m2v) <= 2 * s2v),
                                color=c2, alpha=0.10, label=r"$\pm 2\sigma_2$")
                ax.axvline(m2v, color=c2, linewidth=1.0, linestyle="--", alpha=0.8)
            ax.set_xlabel(rf"$y^{i}_{{n+1}}$", fontsize=10)
            ax.set_ylabel("densité", fontsize=9)
            ax.set_title(rf"Marginale $y^{i}$", fontsize=9)
            ax.legend(fontsize=7); ax.grid(True, linestyle=":", alpha=0.4)
        self._fig_dens.suptitle(
            rf"$p(y_{{n+1}} \mid r_n={j},\; r_{{n+1}}={k},\; y_n)$  [marginales]",
            fontsize=10)

    def _draw_density_empty(self) -> None:
        self._fig_dens.clf()
        ax = self._fig_dens.add_subplot(1, 1, 1)
        ax.set_xticks([]); ax.set_yticks([])
        ax.text(0.5, 0.5,
                "Lancez le filtre en mode\n"
                "« Exact IMM - H5 required »\n"
                "pour afficher\n"
                r"$p(y_{n+1} \mid r_n,\; r_{n+1},\; y_n)$",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=11, color="#999999", style="italic")
        ax.grid(True, linestyle=":", alpha=0.3)
        self._canvas_dens.draw_idle()
