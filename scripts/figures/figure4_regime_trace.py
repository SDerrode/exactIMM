#!/usr/bin/env python3
"""
scripts/figures/figure4_regime_trace.py
=======================================
*Figure 4* — Filtered regime probability  π_n(r=1)  on the test period
(2019-2024), overlaid with the proxy labels L1 (VIX-median) and L2
(NBER recession).

Reads ``results/e3/regime_trace.csv`` (written by ``e3_bw_em.py``).
Plots one curve per BW-EM variant and marks L2 recession bands in red.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CSV = ROOT / "results/e3/regime_trace.csv"
DEFAULT_OUT = ROOT / "paper/figures/fig04_regime_trace.pdf"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()

    df = pd.read_csv(args.csv, parse_dates=["date"]).set_index("date")

    # Identify π_1 columns produced by E3 (one per variant).
    pi_cols = [c for c in df.columns if c.endswith("_pi1")]

    fig, axes = plt.subplots(
        nrows=len(pi_cols),
        ncols=1,
        figsize=(9.0, 1.5 * len(pi_cols) + 1.2),
        sharex=True,
        gridspec_kw={"hspace": 0.12},
    )
    if len(pi_cols) == 1:
        axes = [axes]

    # NBER recessions in the test window (COVID 2020).
    nber = [("2020-02-01", "2020-04-30")]

    for ax, col in zip(axes, pi_cols):
        vname = col.replace("_pi1", "")
        pi1 = df[col].to_numpy()
        ax.plot(df.index, pi1, lw=0.7, color="tab:blue", label=r"$\pi_n(r=1)$")
        # Light fill of high-regime indicator
        ax.fill_between(df.index, 0, pi1, alpha=0.15, color="tab:blue")
        # Ground-truth L2 regimes as red bands.
        for start, end in nber:
            ax.axvspan(
                pd.Timestamp(start), pd.Timestamp(end), color="tab:red", alpha=0.18, zorder=0
            )
        # Ground-truth L1 as a step trace (on a twin axis if crowded).
        ax.step(
            df.index,
            df["r_L1"],
            where="post",
            color="tab:green",
            lw=0.5,
            alpha=0.65,
            label="L1 (VIX-med.)",
        )
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel(vname, fontsize=9)
        ax.grid(ls=":", lw=0.4)
        ax.xaxis.set_major_locator(mdates.YearLocator(1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    axes[0].legend(loc="upper right", fontsize=8, framealpha=0.9)
    axes[-1].set_xlabel("date")
    axes[0].set_title("Filtered regime probability  on the 2019-2024 test period")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    fig.savefig(args.out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure4] wrote {args.out} and {args.out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
