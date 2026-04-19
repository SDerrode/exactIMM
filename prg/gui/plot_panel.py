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
from PyQt6.QtWidgets import QWidget, QVBoxLayout


class PlotPanel(QWidget):
    """Matplotlib figure stacked vertically.

    Layout (top → bottom):
      [0]                R_n   step plot (simulation only)
      [1]                π_n(k) filtered regime posterior (shown after Filter)
      [2 … 1+q]          X^i   hidden components
      [2+q … 1+q+s]      Y^i   observed components
      [2+q+s … 1+q+2s]   ν^i   filter innovations (shorter axes, after Filter)
    """

    def __init__(self, q: int, s: int, parent=None):
        super().__init__(parent)
        self._q = q
        self._s = s
        # Subplot index offsets for readable indexing
        self._r_offset     = 0
        self._pi_offset    = 1
        self._x_offset     = 2
        self._y_offset     = 2 + q
        self._innov_offset = 2 + q + s
        self._n_axes       = self._innov_offset + s

        # π_n(k) and ν^i axes are 55 % the height of the regular axes
        height_ratios = (
            [1.0, 0.55]            # R_n, π_n(k)
            + [1.0] * (q + s)      # X^i, Y^i
            + [0.55] * s           # innovations
        )
        fig_h = 2.2 * (1 + q + s) + 1.3 * (1 + s)
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

        # --- π_n(k) placeholder (empty until Filter runs) ---
        ax_pi = self._axes[self._pi_offset]
        ax_pi.set_ylabel(r"$\pi_n(k)$", fontsize=9)
        ax_pi.set_yticks([])
        ax_pi.grid(True, linestyle=":", alpha=0.4)
        ax_pi.tick_params(labelsize=7)

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
        """Populate the dedicated π_n(k) subplot with one line per regime.

        R_n stays a clean step plot — π_n(k) has its own axis (range 0–1)
        stacked directly below R_n, sharing the x-axis.
        """
        self.clear_pi_overlay()
        ns_arr = np.asarray(ns)
        ax_pi  = self._axes[self._pi_offset]
        ax_pi.cla()

        colours_pi = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                      "#8c564b", "#e377c2"]
        self._pi_artists: list = []
        for k in range(K):
            line, = ax_pi.plot(
                ns_arr, pis[:, k],
                color=colours_pi[k % len(colours_pi)],
                linewidth=1.0, linestyle="-", alpha=0.9,
                label=rf"$\pi_n({k})$",
            )
            self._pi_artists.append(line)
        ax_pi.set_ylim(-0.05, 1.05)
        ax_pi.set_ylabel(r"$\pi_n(k)$", fontsize=9)
        ax_pi.tick_params(labelsize=7)
        ax_pi.grid(True, linestyle=":", alpha=0.4)
        ax_pi.legend(fontsize=7, loc="center right", ncol=min(K, 4))
        self._canvas.draw_idle()

    def clear_pi_overlay(self) -> None:
        """Clear the π_n(k) subplot back to its empty placeholder."""
        ax_pi = self._axes[self._pi_offset]
        ax_pi.cla()
        ax_pi.set_ylabel(r"$\pi_n(k)$", fontsize=9)
        ax_pi.set_yticks([])
        ax_pi.grid(True, linestyle=":", alpha=0.4)
        ax_pi.tick_params(labelsize=7)
        self._pi_artists = []
        self._canvas.draw_idle()

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

        # --- π_n(k) axis stays empty in MC mode ---
        ax_pi = self._axes[self._pi_offset]
        ax_pi.set_ylabel(r"$\pi_n(k)$", fontsize=9)
        ax_pi.set_yticks([])
        ax_pi.grid(True, linestyle=":", alpha=0.4)
        ax_pi.tick_params(labelsize=7)

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

        ax_pi = self._axes[self._pi_offset]
        ax_pi.set_ylabel(r"$\pi_n(k)$", fontsize=9)
        ax_pi.set_yticks([])
        ax_pi.grid(True, linestyle=":", alpha=0.4)
        ax_pi.tick_params(labelsize=7)

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
