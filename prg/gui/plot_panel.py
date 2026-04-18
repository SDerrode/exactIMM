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
    """Matplotlib figure with 1 + q + s subplots."""

    def __init__(self, q: int, s: int, parent=None):
        super().__init__(parent)
        self._q = q
        self._s = s
        self._n_axes = 1 + q + s

        self._fig = Figure(figsize=(7, 2.2 * self._n_axes), tight_layout=True)
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._toolbar = NavigationToolbar2QT(self._canvas, self)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._toolbar)
        layout.addWidget(self._canvas)

        self._axes = self._fig.subplots(self._n_axes, 1, sharex=True)
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
        xs: np.ndarray,   # shape (N, q)
        ys: np.ndarray,   # shape (N, s)
        K: int,
    ) -> None:
        """Redraw all subplots with fresh simulation data."""
        for ax in self._axes:
            ax.cla()

        ns_arr = np.asarray(ns)

        # --- R_n step plot ---
        ax_r = self._axes[0]
        ax_r.step(ns_arr, rs, where="post", color="#555555", linewidth=1.2)
        ax_r.set_ylabel(r"$R_n$", fontsize=10)
        ax_r.set_yticks(range(K))
        ax_r.set_ylim(-0.5, K - 0.5)
        ax_r.grid(True, linestyle=":", alpha=0.5)
        ax_r.set_title("GSS Simulation", fontsize=10)

        # --- X^i components ---
        colours_x = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
        for i in range(self._q):
            ax = self._axes[1 + i]
            ax.plot(
                ns_arr, xs[:, i],
                color=colours_x[i % len(colours_x)],
                linewidth=0.9,
                label=rf"$X^{i}$",
            )
            ax.set_ylabel(rf"$X^{i}$", fontsize=10)
            ax.grid(True, linestyle=":", alpha=0.5)

        # --- Y^i components ---
        colours_y = ["#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
        for i in range(self._s):
            ax = self._axes[1 + self._q + i]
            ax.plot(
                ns_arr, ys[:, i],
                color=colours_y[i % len(colours_y)],
                linewidth=0.9,
                label=rf"$Y^{i}$",
            )
            ax.set_ylabel(rf"$Y^{i}$", fontsize=10)
            ax.grid(True, linestyle=":", alpha=0.5)

        self._axes[-1].set_xlabel(r"$n$", fontsize=10)
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
            ax = self._axes[1 + i]
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

    def clear(self) -> None:
        """Clear all plots and restore the empty-state message."""
        self.clear_filter_overlay()
        for ax in self._axes:
            ax.cla()
        self._draw_empty()

    def clear_filter_overlay(self) -> None:
        """Remove previously drawn filter overlay artists."""
        artists = getattr(self, "_filter_artists", [])
        for a in artists:
            try:
                a.remove()
            except ValueError:
                pass
        self._filter_artists = []
        # Also clear legends that may reference removed artists
        for i in range(self._q):
            leg = self._axes[1 + i].get_legend()
            if leg is not None:
                leg.remove()
        self._canvas.draw_idle()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _draw_empty(self) -> None:
        for ax in self._axes:
            ax.set_visible(True)
            ax.text(
                0.5, 0.5,
                "Press [Simulate] to run a simulation",
                ha="center", va="center",
                transform=ax.transAxes,
                fontsize=9,
            )
            ax.set_xticks([])
            ax.set_yticks([])
        self._canvas.draw_idle()
