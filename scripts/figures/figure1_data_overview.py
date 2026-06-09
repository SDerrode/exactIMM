#!/usr/bin/env python3
"""
scripts/figures/figure1_data_overview.py
========================================
Produce *Figure 1* of the paper's experimental section:

Two-panel overview of the S&P500 / VIX dataset
    top    : X_n = 100 * log-return of S&P500 (daily)
    bottom : Y_n = log(VIX)

with
    - training period (2004-01 .. 2018-12)  : plain background
    - test     period (2019-01 .. 2024-12)  : light grey band
    - NBER recession bands                  : light red
    - horizontal dashed line on Y = training-median of log(VIX)
      (the L1 regime-label threshold used later in the paper)

Usage
-----
    python scripts/figures/figure1_data_overview.py

    python scripts/figures/figure1_data_overview.py \
        --csv data/real/sp500_vix.csv \
        --out paper/figures/fig01_data.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

DEFAULT_CSV = Path("data/real/sp500_vix.csv")
DEFAULT_OUT = Path("paper/figures/fig01_data.pdf")

# NBER recession periods covered by the 2004-2024 range.
NBER_RECESSIONS = [
    ("2007-12-01", "2009-06-30", "Great Recession"),
    ("2020-02-01", "2020-04-30", "COVID-19"),
]

# Train / test split (must match the rest of the experimental pipeline).
TRAIN_END = pd.Timestamp("2018-12-31")
TEST_START = pd.Timestamp("2019-01-01")


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    required = {"log_return", "log_vix"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")
    return df


def _add_recession_bands(ax, ymin: float, ymax: float) -> None:
    for start, end, _label in NBER_RECESSIONS:
        ax.axvspan(
            pd.Timestamp(start),
            pd.Timestamp(end),
            color="tab:red",
            alpha=0.12,
            zorder=0,
        )


def make_figure(df: pd.DataFrame, out: Path) -> None:
    train_log_vix_median = df.loc[df.index <= TRAIN_END, "log_vix"].median()

    xlim = (df.index.min(), df.index.max())

    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(9.0, 4.6),
        sharex=True,
        gridspec_kw={"hspace": 0.08},
    )

    # --- top panel : log-returns ---
    ax0 = axes[0]
    ax0.plot(df.index, df["log_return"], lw=0.5, color="tab:blue")
    ax0.set_ylabel(r"$X_n$ : 100$\,\cdot\,\log(P_n/P_{n-1})$")
    ax0.axhline(0.0, color="0.3", lw=0.6, ls=":")
    ax0.set_ylim(df["log_return"].min() * 1.05, df["log_return"].max() * 1.05)

    # --- bottom panel : log-VIX ---
    ax1 = axes[1]
    ax1.plot(df.index, df["log_vix"], lw=0.6, color="tab:orange")
    ax1.set_ylabel(r"$Y_n = \log(\mathrm{VIX}_n)$")
    ax1.axhline(
        train_log_vix_median,
        color="tab:green",
        lw=0.8,
        ls="--",
        label=rf"L1 threshold = median on train = {train_log_vix_median:.3f}",
    )
    ax1.legend(loc="upper right", fontsize=8, framealpha=0.85)
    ax1.set_xlabel("date")

    # --- shared overlays (order matters: bands first, lines remain visible) ---
    for ax in axes:
        ax.set_xlim(xlim)
        # Test-period band: from TEST_START to right edge.
        ax.axvspan(TEST_START, xlim[1], color="0.85", alpha=0.5, zorder=-1)
        # NBER recession bands.
        ylim = ax.get_ylim()
        _add_recession_bands(ax, *ylim)
        ax.set_ylim(ylim)
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # Train / test labels inside the top panel.
    y_top = axes[0].get_ylim()[1]
    axes[0].annotate(
        "train",
        xy=(pd.Timestamp("2011-06-01"), y_top),
        xytext=(0, -10),
        textcoords="offset points",
        ha="center",
        va="top",
        fontsize=9,
        color="0.3",
    )
    axes[0].annotate(
        "test",
        xy=(pd.Timestamp("2022-01-01"), y_top),
        xytext=(0, -10),
        textcoords="offset points",
        ha="center",
        va="top",
        fontsize=9,
        color="0.3",
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    # Also save a PNG preview for quick viewing.
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure1] wrote {out} and {out.with_suffix('.png')}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[3])
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    df = load_data(args.csv)
    print(f"[figure1] loaded {len(df)} rows, {df.index.min().date()} .. {df.index.max().date()}")
    make_figure(df, args.out)


if __name__ == "__main__":
    main()
