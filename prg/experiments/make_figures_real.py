#!/usr/bin/env python3
"""
prg/experiments/make_figures_real.py
=====================================
Generate figures for §7 (real-data ENSO experiment).

Reads
-----
    data/real/enso_sst.csv             (raw indices + regime label)
    results/enso/regime_trace.csv      (filter posteriors on test)

Writes
------
    paper/figures/generated/fig_enso_overview.pdf
        Two-panel: top  = standardized Niño 3.4 over full period with regime
                          shading (La Niña / Neutral / El Niño from ONI).
                   bot  = train/test split marker + standardized Niño 1+2.

    paper/figures/generated/fig_enso_regime_trace.pdf
        Three-panel for the test period:
          (a) Niño 3.4 (Y) and Niño 1+2 (X) standardized
          (b) Filter posterior π_n(El Niño) and π_n(La Niña) for variant V0
          (c) Regime argmax (filter) vs ground truth from ONI
"""

from __future__ import annotations

import pathlib
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA_CSV = REPO_ROOT / "data" / "real" / "enso_sst.csv"
TRACE_CSV = REPO_ROOT / "results" / "enso" / "regime_trace.csv"
FIG_DIR = REPO_ROOT / "paper" / "figures" / "generated"

REGIME_COL = {0: "#3b8ec2", 1: "#cccccc", 2: "#d62728"}  # blue / grey / red
REGIME_NM = {0: "La Niña", 1: "Neutral", 2: "El Niño"}
TRAIN_END = pd.Timestamp("2010-12-01")


def _shade_regimes(ax, dates, regimes):
    """Color background by regime."""
    cur = regimes[0]
    start = dates[0]
    for i in range(1, len(regimes)):
        if regimes[i] != cur:
            ax.axvspan(start, dates[i - 1], color=REGIME_COL[cur], alpha=0.18, linewidth=0)
            cur = regimes[i]
            start = dates[i]
    ax.axvspan(start, dates[-1], color=REGIME_COL[cur], alpha=0.18, linewidth=0)


def make_overview(out_path: pathlib.Path):
    df = pd.read_csv(DATA_CSV, parse_dates=["date"]).set_index("date").sort_index()
    train = df.loc[df.index <= TRAIN_END]
    mu_x, sd_x = train["nino12"].mean(), train["nino12"].std(ddof=0)
    mu_y, sd_y = train["nino34"].mean(), train["nino34"].std(ddof=0)
    df["x_std"] = (df["nino12"] - mu_x) / sd_x
    df["y_std"] = (df["nino34"] - mu_y) / sd_y

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 4), sharex=True)

    # Top: standardized Niño 3.4 with regime shading
    _shade_regimes(ax1, df.index, df["regime"].to_numpy())
    ax1.plot(df.index, df["y_std"], color="black", lw=0.8)
    ax1.axvline(TRAIN_END, color="k", lw=0.8, ls="--", alpha=0.6)
    ax1.text(TRAIN_END, ax1.get_ylim()[1] * 0.85, "  test →", fontsize=8, ha="left", va="top")
    ax1.set_ylabel(r"$Y_n$ std.\ Niño\,3.4", fontsize=9)
    ax1.set_title("ENSO monthly indices, 1950–2026", fontsize=10)
    ax1.tick_params(axis="both", labelsize=8)

    # Bottom: standardized Niño 1+2
    _shade_regimes(ax2, df.index, df["regime"].to_numpy())
    ax2.plot(df.index, df["x_std"], color="#1f4068", lw=0.7)
    ax2.axvline(TRAIN_END, color="k", lw=0.8, ls="--", alpha=0.6)
    ax2.set_ylabel(r"$X_n$ std.\ Niño\,1+2", fontsize=9)
    ax2.set_xlabel("Year", fontsize=9)
    ax2.tick_params(axis="both", labelsize=8)
    ax2.xaxis.set_major_locator(mdates.YearLocator(10))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    handles = [
        mpatches.Patch(color=REGIME_COL[k], alpha=0.45, label=REGIME_NM[k]) for k in (0, 1, 2)
    ]
    ax1.legend(handles=handles, fontsize=8, loc="upper right", framealpha=0.9, ncol=3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def make_regime_trace(out_path: pathlib.Path, variant: str = "V0_unconstrained"):
    if not TRACE_CSV.exists():
        print(f"  SKIP: {TRACE_CSV} not found")
        return
    tr = pd.read_csv(TRACE_CSV, parse_dates=["date"]).set_index("date").sort_index()

    # Filter posteriors on test
    pi0 = tr[f"{variant}_pi0"].to_numpy()  # La Niña
    pi2 = tr[f"{variant}_pi2"].to_numpy()  # El Niño
    r_hat = np.argmax(
        np.column_stack([tr[f"{variant}_pi{k}"].to_numpy() for k in range(3)]), axis=1
    )

    # Best regime alignment vs ground truth
    from itertools import permutations

    r_true = tr["r_true"].to_numpy().astype(int)
    best_acc = -1
    best_perm = None
    for p in permutations(range(3)):
        pa = np.array(p)
        acc = float(np.mean(pa[r_hat] == r_true))
        if acc > best_acc:
            best_acc, best_perm = acc, pa
    r_hat_aligned = best_perm[r_hat]
    print(f"  Variant {variant}: best regime accuracy = {best_acc:.3f}")

    # Re-map filter posteriors to ground-truth regime indexing
    inv_perm = np.argsort(best_perm)
    pi_aligned = np.column_stack([tr[f"{variant}_pi{inv_perm[k]}"].to_numpy() for k in range(3)])

    fig, axes = plt.subplots(
        3, 1, figsize=(7, 4.5), sharex=True, gridspec_kw={"height_ratios": [1.2, 1.0, 0.5]}
    )

    # (a) Standardized Y and X
    axes[0].plot(tr.index, tr["y_obs"], color="black", lw=0.9, label=r"$Y_n$ Niño\,3.4")
    axes[0].plot(
        tr.index, tr["x_true"], color="#1f4068", lw=0.7, alpha=0.7, label=r"$X_n$ Niño\,1+2"
    )
    axes[0].axhline(0, color="grey", lw=0.5)
    axes[0].set_ylabel("Std anom.", fontsize=9)
    axes[0].legend(fontsize=8, ncol=2, loc="upper right", framealpha=0.85)
    axes[0].tick_params(axis="both", labelsize=8)

    # (b) Filter posteriors π(La Niña) and π(El Niño)
    axes[1].fill_between(
        tr.index, 0, pi_aligned[:, 0], color=REGIME_COL[0], alpha=0.55, label=r"$\pi_n($La Niña$)$"
    )
    axes[1].fill_between(
        tr.index, 0, -pi_aligned[:, 2], color=REGIME_COL[2], alpha=0.55, label=r"$\pi_n($El Niño$)$"
    )
    axes[1].axhline(0, color="grey", lw=0.5)
    axes[1].set_ylabel(r"$\pi_n$ (V0)", fontsize=9)
    axes[1].set_ylim(-1.0, 1.0)
    axes[1].legend(fontsize=8, ncol=2, loc="upper right", framealpha=0.85)
    axes[1].tick_params(axis="both", labelsize=8)

    # (c) Argmax regime vs ground truth
    _shade_regimes(axes[2], tr.index, r_true)
    axes[2].plot(tr.index, r_hat_aligned, color="black", lw=0.6, drawstyle="steps-mid")
    axes[2].set_yticks([0, 1, 2])
    axes[2].set_yticklabels(["LN", "Neu", "EN"], fontsize=7)
    axes[2].set_xlabel("Year", fontsize=9)
    axes[2].tick_params(axis="both", labelsize=8)
    axes[2].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.suptitle(f"ENSO regime detection on test (V0, acc={best_acc:.2f})", fontsize=10, y=1.00)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    make_overview(FIG_DIR / "fig_enso_overview.pdf")
    make_regime_trace(FIG_DIR / "fig_enso_regime_trace.pdf")
    print("All real-data figures generated.")


if __name__ == "__main__":
    sys.exit(main() or 0)
