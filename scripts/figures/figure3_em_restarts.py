#!/usr/bin/env python3
"""
scripts/figures/figure3_em_restarts.py
======================================
*Figure 3* — Boxplot of final train log-likelihoods over ``n_inits``
independent k-means-initialised BW-EM restarts, one box per variant.

Reads ``results/e3/table3.json`` (written by ``e3_bw_em.py``).

Note: V3 (GEM with each-iter AB projection) is plotted on a separate
panel because the constrained inner M-step makes the EM objective
non-monotone, so its absolute log-likelihood values fall on a very
different scale (~ -66k) than the post-hoc variants V0/V1/V2 (~ -2k).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_JSON = ROOT / "results/e3/table3.json"
DEFAULT_OUT = ROOT / "paper/figures/fig03_em_restarts.pdf"

# Nicer human names for the plot
LABELS = {
    "V0_unconstr": r"V0 $\;$unconstr.",
    "V1_posthoc_B": r"V1 $\;$post-hoc $B$",
    "V2_posthoc_A": r"V2 $\;$post-hoc $A$",
    "V3_GEM_B": r"V3 $\;$GEM $B$",
}

COLORS = {
    "V0_unconstr": "#d0e1f9",
    "V1_posthoc_B": "#f9d0d0",
    "V2_posthoc_A": "#d0f9db",
    "V3_GEM_B": "#f9ead0",
}


def _draw_panel(ax, names, all_logLs, best_logLs, *, title, with_legend):
    pos = np.arange(1, len(names) + 1)
    bp = ax.boxplot(
        all_logLs,
        positions=pos,
        widths=0.55,
        patch_artist=True,
        showfliers=True,
    )
    for patch, n in zip(bp["boxes"], names):
        patch.set_facecolor(COLORS.get(n, "0.85"))
        patch.set_edgecolor("0.3")

    rng = np.random.default_rng(0)
    for i, vals in enumerate(all_logLs):
        xj = pos[i] + rng.uniform(-0.08, 0.08, size=len(vals))
        ax.scatter(xj, vals, color="0.2", s=14, alpha=0.7, zorder=3)

    ax.scatter(
        pos,
        best_logLs,
        marker="*",
        color="tab:red",
        s=110,
        zorder=5,
        label="retained restart" if with_legend else None,
    )

    pretty = [LABELS.get(n, n) for n in names]
    ax.set_xticks(pos)
    ax.set_xticklabels(pretty, fontsize=9)
    ax.grid(axis="y", ls=":", lw=0.5)
    ax.set_title(title, fontsize=10)
    if with_legend:
        ax.legend(loc="best", fontsize=8)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--json", type=Path, default=DEFAULT_JSON)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()

    data = json.loads(args.json.read_text(encoding="utf-8"))
    # BW-EM variants only (skip Hamilton baseline, which has empty restarts)
    variants = [v for v in data["variants"] if v["name"] in LABELS and v.get("all_train_log_liks")]

    posthoc = [v for v in variants if v["name"] != "V3_GEM_B"]
    gem = [v for v in variants if v["name"] == "V3_GEM_B"]

    fig, (ax_l, ax_r) = plt.subplots(
        ncols=2,
        figsize=(7.6, 3.6),
        gridspec_kw={"width_ratios": [3, 1], "wspace": 0.35},
    )

    _draw_panel(
        ax_l,
        names=[v["name"] for v in posthoc],
        all_logLs=[v["all_train_log_liks"] for v in posthoc],
        best_logLs=[v["train_log_lik"] for v in posthoc],
        title="post-hoc / unconstrained variants (V0–V2)",
        with_legend=True,
    )
    ax_l.set_ylabel(r"final train $\log\hat L$")

    _draw_panel(
        ax_r,
        names=[v["name"] for v in gem],
        all_logLs=[v["all_train_log_liks"] for v in gem],
        best_logLs=[v["train_log_lik"] for v in gem],
        title="GEM / each-iter $B=0$",
        with_legend=False,
    )
    ax_r.set_ylabel(r"final train $\log\hat L$ (different scale)", fontsize=8)

    n_inits = len(posthoc[0]["all_train_log_liks"])
    fig.suptitle(
        rf"BW-EM: final log-likelihood over {n_inits} k-means restarts", fontsize=11, y=1.02
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    fig.savefig(args.out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure3] wrote {args.out} and {args.out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
