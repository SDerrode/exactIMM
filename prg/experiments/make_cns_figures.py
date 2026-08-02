"""Figures for the exactness-domains paper (docs/CNS-exactness).

F1  "Three conditions, three filters" (3 panels, double column). Each panel
    violates one elementary condition and shows which filter leaves the exact
    list, the others staying at machine precision:
      (a) sweep the coupling C away from 0, with AB held  -> the IMM leaves;
      (b) sweep the violation of (B), with (A) held and C != 0
          -> the constant-gain filter leaves;
      (c) sweep the violation eta of (A), at C != 0
          -> GPB2 leaves too, with an empirical slope of about 3.
    Read left to right, the exact filters drop out one at a time -- which is
    Table I of the paper, in action.

F2  "The uniformity requirement, stated then measured" (2 panels, double
    column). (a) The grid of per-regime choices: all four cells satisfy the
    per-regime disjunction, only the diagonal is exact. (b) The same statement
    measured: the GPB2 gap vanishes exactly on the line A_1 = 0, the boundary
    of the uniform family {A == MC}. One panel says what is true, the other
    shows it.

Every point is a median over N_SEEDS_FIG independent runs against the exact
K^N mixture filter, using the model builders and metric of cns_exactness.py.

Run:  python -m prg.experiments.make_cns_figures
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from prg.experiments.cns_exactness import (
    N_STEPS,
    SEED0,
    ab_model,
    gaps,
    mixed_branch_model,
    off_union_model,
)

OUTDIR = Path(__file__).resolve().parents[2] / "docs" / "CNS-exactness" / "figures"
N_SEEDS_FIG = 25          # medians over this many runs per point
FLOOR = 1e-16             # plotting floor, below which "machine precision"

STYLE = {
    "imm": dict(color="#C44E52", marker="o", ls="-", label="IMM (order 1)"),
    "gpb2": dict(color="#4C72B0", marker="s", ls="-", label="GPB2 (order 2)"),
    "cg": dict(color="#55A868", marker="^", ls="-", label="constant gain"),
}


def _med(params, with_cg=False):
    """Median state-mean gap per filter, over N_SEEDS_FIG runs."""
    g = gaps(params, with_cg=with_cg, n_seeds=N_SEEDS_FIG)
    return {k: max(v[0].med, FLOOR) for k, v in g.items()}


def _panel(ax, xs, series, xlabel, title):
    """One log-log panel: the violation parameter on x, the median gap on y."""
    for name in ("imm", "gpb2", "cg"):
        if name not in series:
            continue
        ax.plot(xs, series[name], ms=3.0, lw=1.1, **STYLE[name])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(3e-16, 3e-1)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_title(title, fontsize=8.5, pad=4)
    ax.tick_params(labelsize=7)
    ax.grid(True, which="major", ls=":", lw=0.5, alpha=0.6)
    ax.axhspan(3e-16, 1e-13, color="0.86", alpha=0.7, zorder=0)


def _fit_slope(xs, ys, lo, hi):
    """Least-squares log-log slope over the window [lo, hi]."""
    x = np.asarray(xs, float)
    y = np.asarray(ys, float)
    m = (x >= lo) & (x <= hi) & (y > 1e-13)
    if m.sum() < 3:
        return None, None, None
    p, logc = np.polyfit(np.log10(x[m]), np.log10(y[m]), 1)
    return p, 10.0 ** logc, (x[m].min(), x[m].max())


def _slope_guide(ax, xs, ys, lo, hi, below=8.0):
    """Fit the log-log slope on [lo,hi], draw the reference line, label it
    with the measured exponent (rounded to one decimal)."""
    p, c, rng = _fit_slope(xs, ys, lo, hi)
    if p is None:
        return None
    xg = np.array(rng)
    yg = c * xg ** p
    ax.plot(xg, yg / below, color="0.3", ls="--", lw=0.9, zorder=1)
    xm = float(np.sqrt(xg[0] * xg[1]))
    ax.text(xm, c * xm ** p / below / 9.0, rf"slope $\approx {p:.1f}$",
            fontsize=6.5, color="0.3", ha="center", va="top")
    return p


def _logsweep(lo, hi, n):
    return list(np.geomspace(lo, hi, n))


def figure1(outdir: Path):
    """F1 -- three conditions, three filters."""
    # (a) sweep C, AB held (dB = 0): only the IMM leaves
    Cs = _logsweep(2e-3, 0.6, 9)
    a = {"imm": [], "gpb2": [], "cg": []}
    for C in Cs:
        m = _med(ab_model(C, dB=0.0), with_cg=True)
        for k in a:
            a[k].append(m[k])

    # (b) sweep the (B) violation, (A) held and C != 0: the constant gain leaves
    dBs = _logsweep(2e-3, 0.4, 9)
    b = {"imm": [], "gpb2": [], "cg": []}
    for dB in dBs:
        m = _med(ab_model(0.4, dB=dB), with_cg=True)
        for k in b:
            b[k].append(m[k])

    # (c) sweep the (A) violation eta at C != 0, with (B) held: GPB2
    #     leaves too, and the constant gain -- already exact at eta=0 since
    #     AB holds there -- leaves at first order, GPB2 only at third
    etas = _logsweep(2e-3, 0.35, 9)
    c = {"imm": [], "gpb2": [], "cg": []}
    for eta in etas:
        m = _med(off_union_model(0.4, eta), with_cg=True)
        for k in c:
            c[k].append(m[k])

    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.35), sharey=True)

    _panel(axes[0], Cs, a, r"coupling $C$",
           r"(a) violating autonomy $C\equiv 0$")
    _panel(axes[1], dBs, b, r"violation of (B):  $B-MD$",
           r"(b) violating slaving (B)")
    _panel(axes[2], etas, c, r"violation of (A):  $\eta$",
           r"(c) violating slaving (A)")

    # measured scaling laws, fitted over the decades where they are clean
    slopes = {
        "IMM vs C": _slope_guide(axes[0], Cs, a["imm"], 2e-3, 2e-1),
        "CG vs B-MD": _slope_guide(axes[1], dBs, b["cg"], 2e-3, 1e-1),
        "GPB2 vs eta": _slope_guide(axes[2], etas, c["gpb2"], 2e-3, 1e-1),
        "CG vs eta": _slope_guide(axes[2], etas, c["cg"], 2e-3, 1e-1,
                                  below=22.0),
    }
    print("  measured log-log slopes: " + ", ".join(
        f"{k} = {v:.2f}" for k, v in slopes.items() if v is not None))

    axes[0].set_ylabel("median gap to the exact filter", fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=7.5,
               frameon=False, bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = outdir / "conditions_filters.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")
    return {"C": (Cs, a), "dB": (dBs, b), "eta": (etas, c)}


def figure_uniformity(outdir: Path):
    """F2 -- the uniformity requirement, stated then measured.

    (a) The conditions of Theorem 2 are per regime, so for K=2 a model is a
        *pair* of choices, and the natural picture is a grid rather than a Venn
        diagram: the domains are equality constraints with empty interior,
        which blobs would misrepresent. All four cells satisfy the per-regime
        disjunction; only the diagonal lies in the union of the two uniform
        families.
    (b) The same statement measured: the median GPB2 gap over a grid of
        (A_1, C_2) vanishes exactly on the line A_1 = 0 -- the boundary of the
        uniform family {A == MC}. The zero set of the bias IS the exactness
        domain.

    Regimes are numbered as in the paper, Omega = {1, ..., K}. (In the code of
    cns_exactness.py the same two regimes are the 0-indexed entries of the
    block lists, so `A0` there is A_1 here and `C1` there is C_2 here.)
    """
    # ---- panel (b) data -----------------------------------------------
    A1s = np.array([0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9])
    C2s = np.array([0.1, 0.25, 0.4, 0.55, 0.7, 0.85])
    Z = np.zeros((len(C2s), len(A1s)))
    for i, C2 in enumerate(C2s):
        for j, A1 in enumerate(A1s):
            Z[i, j] = max(_med(mixed_branch_model(A1, C2))["gpb2"], FLOOR)

    fig, (ax, bx) = plt.subplots(
        1, 2, figsize=(7.16, 2.72),
        gridspec_kw=dict(width_ratios=[1.0, 1.06], wspace=0.42))

    # ---- panel (a): the grid of per-regime choices ---------------------
    row = [r"$C_1=0$", r"$A_1=M_1C_1$"]
    col = [r"$C_2=0$", r"$A_2=M_2C_2$"]
    diag_txt = "GPB2 exact" "\n" r"$\sim\!10^{-15}$"
    off_txt = "GPB2 biased" "\n" r"up to $5.6\times10^{-3}$"
    ok, bad = "#DFF0E4", "#FBE0DE"
    okl, badl = "#2E7D4F", "#B3403A"

    for i in (0, 1):
        for j in (0, 1):
            same = (i == j)
            ax.add_patch(plt.Rectangle((j, 1 - i), 1, 1,
                                       facecolor=ok if same else bad,
                                       edgecolor="0.35", lw=1.0))
            ax.text(j + 0.5, 1 - i + 0.62, diag_txt if same else off_txt,
                    ha="center", va="center", fontsize=7.0,
                    color=okl if same else badl,
                    linespacing=1.35, fontweight="bold" if same else "normal")
            ax.text(j + 0.5, 1 - i + 0.22,
                    r"uniform: $\{C\equiv0\}$" if (same and i == 0) else
                    (r"uniform: $\{A\equiv MC\}$" if same else "mixed"),
                    ha="center", va="center", fontsize=6.3, color="0.35",
                    style="italic")
    for j in (0, 1):
        ax.text(j + 0.5, 2.04, col[j], ha="center", va="bottom", fontsize=7.4)
    for i in (0, 1):
        ax.text(-0.10, 1 - i + 0.5, row[i], ha="center", va="center",
                fontsize=7.4, rotation=90)
    ax.text(1.0, 2.26, "condition satisfied at regime 2", ha="center",
            va="bottom", fontsize=7.0, color="0.3")
    ax.text(-0.44, 1.0, "condition satisfied at regime 1", ha="center",
            va="center", fontsize=7.0, color="0.3", rotation=90)
    ax.text(1.0, -0.20, "all four cells satisfy the per-regime disjunction",
            ha="center", va="bottom", fontsize=6.6, color="0.3")
    ax.text(1.0, -0.44, "off-diagonal: strictly, i.e. the regime that "
                        r"satisfies $C=0$ has $A\neq0$",
            ha="center", va="bottom", fontsize=6.1, color="0.45")
    ax.set_xlim(-0.56, 2.02)
    ax.set_ylim(-0.56, 2.46)
    ax.axis("off")
    ax.set_title("(a) the statement", fontsize=8.5, pad=2)

    # ---- panel (b): the same thing measured ---------------------------
    pc = bx.pcolormesh(A1s, C2s, Z, norm=LogNorm(vmin=1e-15, vmax=Z.max()),
                       cmap="viridis", shading="nearest")
    cb = fig.colorbar(pc, ax=bx, pad=0.02)
    cb.set_label("median GPB2 gap", fontsize=7.5)
    cb.ax.tick_params(labelsize=6.5)
    bx.axvline(0.0, color="w", lw=1.6)
    bx.text(0.012, 0.88, r"$\{A\equiv MC\}$", color="w", fontsize=7,
            rotation=90, va="top", transform=bx.get_xaxis_transform())
    bx.set_xlabel(r"memory of the $C=0$ regime,  $A_1$", fontsize=8)
    bx.set_ylabel(r"channel of the slaved regime,  $C_2$", fontsize=8)
    bx.tick_params(labelsize=7)
    bx.set_title("(b) the same thing, measured", fontsize=8.5, pad=2)

    fig.tight_layout()
    out = outdir / "uniformity.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")
    return A1s, C2s, Z


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    print(f"Figures for the exactness-domains paper "
          f"(N={N_STEPS}, {N_SEEDS_FIG} seeds from {SEED0}, medians):")
    figure1(OUTDIR)
    figure_uniformity(OUTDIR)


if __name__ == "__main__":
    main()
