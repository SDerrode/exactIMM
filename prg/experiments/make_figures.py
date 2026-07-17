#!/usr/bin/env python3
"""
prg/experiments/make_figures.py
================================
Generate paper figures and LaTeX table fragments from the Monte-Carlo
results produced by :mod:`prg.experiments.run_simulations`.

Reads  ``data/experiments/mc_results.csv``
Writes ``paper/figures/generated/*.pdf``  (one file per figure)
       ``paper/figures/generated/*.tex``  (table rows for \\input{})

Figures produced
----------------
fig_rmse_vs_N.pdf   RMSE vs N (log-log) for M1, comparing ngh_kf and
                    gpb2 (Fig. 1 of §6.2).

Tables produced (LaTeX row fragments)
--------------------------------------
tab_filter_M1.tex   Rows for Table 2 (M1 filter benchmark):
                    N | RMSE ngh_kf | NEES ngh_kf | LB ngh_kf | CPU ngh_kf ||
                        RMSE imm | NEES imm | LB imm | CPU imm
tab_filter_M2M3.tex Rows for Table 3 (M2/M3 filter benchmark, N=2000).
tab_bic.tex         Rows for Table 7 (BIC model selection on M1).

Usage
-----
    python -m prg.experiments.make_figures
    # or
    python -m prg.experiments.make_figures --results data/experiments/mc_results.csv
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

# Matplotlib import (optional dependency)
try:
    import matplotlib

    matplotlib.use("Agg")  # headless rendering
    import matplotlib.pyplot as plt
    import matplotlib.ticker

    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False
    print("WARNING: matplotlib not available — figures will not be generated.", file=sys.stderr)

__all__ = [
    "make_all",
    "make_fig_rmse_vs_N",
    "make_tab_filter",
    "make_tab_bic",
    "make_fig_supervised_rmse",
    "make_tab_supervised",
    "make_fig_em_convergence",
    "make_tab_em_basin",
    "make_tab_em_restarts",
]

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]  # → exactIMM/
DEFAULT_IN = REPO_ROOT / "data" / "experiments" / "mc_results.csv"
DEFAULT_FIG_DIR = REPO_ROOT / "paper" / "figures" / "generated"

# ---------------------------------------------------------------------------
# Colour / style constants
# ---------------------------------------------------------------------------

COLORS = {
    "ngh_kf": "#1f77b4",  # blue
    "gpb2": "#ff7f0e",  # orange
}
MARKERS = {
    "ngh_kf": "o",
    "gpb2": "s",
}
MODE_LABELS = {
    "ngh_kf": r"NGH-MSM-KF",
    "gpb2": r"IMM-general",
}

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------


def _load(results_path: pathlib.Path) -> pd.DataFrame:
    df = pd.read_csv(results_path)
    # Coerce numeric columns
    for col in ("rmse", "nees", "lb_pval", "jb_pval", "log_lik", "bic", "cpu_s"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# Figure 1: RMSE vs N for M1
# ---------------------------------------------------------------------------


def make_fig_rmse_vs_N(
    df: pd.DataFrame,
    fig_dir: pathlib.Path,
) -> pathlib.Path | None:
    """
    RMSE vs sequence length N (log–log) for model M1.

    One curve per filter mode (ngh_kf, gpb2) with
    ± 1 standard error shading.
    """
    if not _HAS_MPL:
        return None

    fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / "fig_rmse_vs_N.pdf"

    sub = df[df["model"] == "M1"].copy()
    N_vals = sorted(sub["N"].unique())

    fig, ax = plt.subplots(figsize=(4.5, 3.2))

    for mode in ("ngh_kf", "gpb2"):
        means, sems, xs = [], [], []
        for N in N_vals:
            vals = sub[(sub["N"] == N) & (sub["mode"] == mode)]["rmse"].dropna()
            if len(vals) == 0:
                continue
            xs.append(N)
            means.append(vals.mean())
            sems.append(vals.sem())
        if not xs:
            continue
        xs = np.array(xs, dtype=float)
        means = np.array(means)
        sems = np.array(sems)
        c = COLORS[mode]
        ax.plot(xs, means, marker=MARKERS[mode], color=c, lw=1.5, label=MODE_LABELS[mode])
        ax.fill_between(xs, means - sems, means + sems, color=c, alpha=0.15)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Sequence length $N$", fontsize=10)
    ax.set_ylabel(r"RMSE", fontsize=10)
    ax.set_title(r"M1 ($K=2,\,q=s=1$) — filter RMSE vs $N$", fontsize=10)
    ax.legend(fontsize=9, framealpha=0.8)
    ax.set_xticks(N_vals)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.tick_params(axis="both", labelsize=9)
    ax.grid(True, which="both", ls="--", lw=0.5, alpha=0.5)

    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")
    return out


# ---------------------------------------------------------------------------
# Table 2: M1 filter benchmark (all N, both modes)
# ---------------------------------------------------------------------------


def _fmt(val: float, ndigits: int = 4) -> str:
    """Format a float for LaTeX; show '---' for NaN."""
    if not np.isfinite(val):
        return r"\text{---}"
    return f"{val:.{ndigits}f}"


_TAB_FILTER_HEADER = r"""  \caption{%s}
  \label{%s}
  \centering\small
  \renewcommand{\arraystretch}{1.2}
  \setlength{\tabcolsep}{4pt}
  \begin{tabular}{@{}%s@{}}
    \toprule
    %s \\
    \cmidrule(lr){2-5}\cmidrule(lr){6-9}
    %s \\
    \midrule
"""

_TAB_FILTER_FOOTER = r"""    \bottomrule
  \end{tabular}"""


def make_tab_filter(
    df: pd.DataFrame,
    fig_dir: pathlib.Path,
) -> pathlib.Path:
    """
    Complete LaTeX table environments for:
      - Table 2 (M1 filter benchmark, all N): tab_filter_M1.tex
      - Table 3 (M2/M3 filter benchmark, N=2000): tab_filter_M2M3.tex

    Each file is a self-contained ``\\begin{table}...\\end{table}`` block
    (use ``\\input`` from outside any tabular environment).

    Columns per method: RMSE | NEES | LB-pass(%) | CPU(µs/step)
    LB-pass = fraction of runs where LB p-value > 0.05 (not rejected).
    CPU in µs per step (cpu_s / N * 1e6).
    """
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_M1 = fig_dir / "tab_filter_M1.tex"
    out_M2M3 = fig_dir / "tab_filter_M2M3.tex"

    def _data_row(sub: pd.DataFrame, label: str = "") -> str:
        parts = [label] if label else []
        for mode in ("ngh_kf", "gpb2"):
            g = sub[sub["mode"] == mode]
            if g.empty:
                parts += ["---"] * 4
                continue
            N_val = g["N"].iloc[0] if "N" in g.columns and not g.empty else None
            lb_pass = float((g["lb_pval"] > 0.05).mean() * 100)
            cpu_s = g["cpu_s"].mean()
            cpu_us = (cpu_s / N_val * 1e6) if N_val else cpu_s * 1e6
            parts += [
                _fmt(g["rmse"].mean()),
                _fmt(g["nees"].mean(), 3),
                f"{lb_pass:.0f}\\%",
                _fmt(cpu_us, 1),
            ]
        return "    " + " & ".join(parts) + r" \\"

    two_method_header = r"& \multicolumn{4}{c}{NGH-MSM-KF} & \multicolumn{4}{c}{GPB2}"
    col_header = (
        r"$N$ / Model & RMSE & NEES & LB\% & CPU($\mu$s) "
        r"& RMSE & NEES & LB\% & CPU($\mu$s)"
    )
    col_spec = "rccrcccrc"

    # ---- Table 2: M1 ---------------------------------------------------
    sub_M1 = df[df["model"] == "M1"]
    data_rows = []
    for N in sorted(sub_M1["N"].unique()):
        data_rows.append(_data_row(sub_M1[sub_M1["N"] == N], label=str(N)))

    cap = (
        r"Filter benchmark on M1 ($K=2$, $q=s=1$, 100 MC runs). "
        r"LB pass: fraction of runs where the Ljung--Box whiteness test "
        r"(lag~20) does not reject at level~0.05. "
        r"CPU in \textmu s per step (single core, Apple M2 Pro)."
    )
    lbl = "tab:filter_M1"
    content = (
        "\\begin{table}[ht]\n"
        + _TAB_FILTER_HEADER % (cap, lbl, col_spec, two_method_header, col_header)
        + "\n".join(data_rows)
        + "\n"
        + _TAB_FILTER_FOOTER
        + "\n"
        + "\\end{table}\n"
    )
    out_M1.write_text(content, encoding="utf-8")
    print(f"  Saved: {out_M1}")

    # ---- Table 3: M2/M3 at N=2000 -------------------------------------
    data_rows = []
    for model in ("M2", "M3"):
        sub = df[(df["model"] == model) & (df["N"] == 2000)]
        data_rows.append(_data_row(sub, label=f"\\textbf{{{model}}}"))

    cap = (
        r"Filter benchmark on M2 ($K=2$, $q=s=2$) and M3 ($K=3$, $q=s=1$), "
        r"$N=2\,000$, 100 MC runs."
    )
    lbl = "tab:filter_M2M3"
    content = (
        "\\begin{table}[ht]\n"
        + _TAB_FILTER_HEADER % (cap, lbl, col_spec, two_method_header, col_header)
        + "\n".join(data_rows)
        + "\n"
        + _TAB_FILTER_FOOTER
        + "\n"
        + "\\end{table}\n"
    )
    out_M2M3.write_text(content, encoding="utf-8")
    print(f"  Saved: {out_M2M3}")

    return out_M1


# ---------------------------------------------------------------------------
# Table 7: BIC model-order selection on M1 (N=2000)
# ---------------------------------------------------------------------------


def make_tab_bic(
    df: pd.DataFrame,
    fig_dir: pathlib.Path,
) -> pathlib.Path:
    """
    BIC rows for Table 7: for model M1 at N=2000, compare BIC when
    fitting AB-GSS(K,q=1,s=1) for K in {1,2,3,4} (using ngh_kf mode
    results; the true K=2 should have the lowest mean BIC).

    Note: the BIC here comes from the *true* K=2 filter log-likelihood
    penalised by the DOF of a fitted K-regime model.  This is only a
    rough BIC approximation; for a proper BIC the EM should be run for
    each K.  We include it as a table-format placeholder; the values will
    be replaced by EM-based BICs once §6.4 is implemented.
    """
    from prg.experiments.metrics import dof_ab

    fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / "tab_bic.tex"

    # Use ngh_kf log-likelihoods at N=2000
    sub = df[(df["model"] == "M1") & (df["N"] == 2000) & (df["mode"] == "ngh_kf")]
    N_val = 2000
    q, s = 1, 1

    lines = []
    for K_test in (1, 2, 3, 4):
        # Re-penalise the log-lik with the DOF of K_test-regime model
        # (K_test=1 is handled specially: dof_ab requires K>=2 so use K=1 manually)
        if K_test >= 2:
            d = dof_ab(K_test, q, s)
        else:
            # K=1: A(q²)+C(qs)+Sigma_W((q+s)(q+s+1)/2)+b(q+s), K²-1=0
            d = 1 * (q**2 + q * s + (q + s) * (q + s + 1) // 2 + (q + s)) + 0

        bic_vals = sub["log_lik"].dropna().apply(lambda ll: d * np.log(N_val) - 2.0 * ll)
        mean_bic = bic_vals.mean()
        pct_sel = float("nan")  # placeholder: proper selection needs EM per K

        lines.append(
            f"{K_test} & {_fmt(mean_bic, 1)} & "
            r"\ph{XX\%}" + r" \\"
        )

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  Saved: {out}")
    return out


# ---------------------------------------------------------------------------
# Figure 2: Filter RMSE vs N_train (supervised, §6.3)
# ---------------------------------------------------------------------------


def make_fig_supervised_rmse(
    df_sup: pd.DataFrame,
    fig_dir: pathlib.Path,
) -> pathlib.Path | None:
    """
    Filter RMSE as a function of N_train for each projection choice,
    plus the oracle (true-params) curve.  Paper Fig. 2 (§6.3).
    """
    if not _HAS_MPL:
        return None

    fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / "fig_supervised_rmse.pdf"

    # Projection display order and colours
    PROJ_STYLES = {
        "none": dict(color="#999999", ls="--", marker="x", label="Free OLS"),
        "ab": dict(color="#1f77b4", ls="-", marker="o", label="AB constraint"),
    }

    N_vals = sorted(df_sup["N"].unique())

    fig, ax = plt.subplots(figsize=(4.5, 3.2))

    # Oracle curve (same for all projections — use mean of 'ab' oracle as representative)
    sub_b = df_sup[df_sup["projection"] == "ab"]
    ora_means, ora_sems, xs = [], [], []
    for N in N_vals:
        g = sub_b[sub_b["N"] == N]["rmse_oracle"].dropna()
        if len(g) == 0:
            continue
        xs.append(N)
        ora_means.append(g.mean())
        ora_sems.append(g.sem())
    if xs:
        ax.plot(
            xs,
            ora_means,
            color="k",
            lw=2,
            ls="-",
            marker="D",
            ms=5,
            label="Oracle (true params)",
            zorder=5,
        )

    # Per-projection curves
    for proj, style in PROJ_STYLES.items():
        sub = df_sup[df_sup["projection"] == proj]
        means, sems, xs_p = [], [], []
        for N in N_vals:
            g = sub[sub["N"] == N]["rmse_estimated"].dropna()
            if len(g) == 0:
                continue
            xs_p.append(N)
            means.append(g.mean())
            sems.append(g.sem())
        if not xs_p:
            continue
        xs_p = np.array(xs_p, dtype=float)
        means = np.array(means)
        sems = np.array(sems)
        ax.plot(xs_p, means, lw=1.4, ms=5, **style)
        ax.fill_between(xs_p, means - sems, means + sems, color=style["color"], alpha=0.12)

    ax.set_xscale("log")
    ax.set_xlabel(r"Training length $N_{\rm train}$", fontsize=10)
    ax.set_ylabel(r"Filter RMSE", fontsize=10)
    ax.set_title(r"M1 — supervised estimation: filter RMSE vs $N_{\rm train}$", fontsize=9)
    ax.legend(fontsize=8, framealpha=0.85, ncol=2)
    ax.set_xticks(N_vals)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.tick_params(axis="both", labelsize=9)
    ax.grid(True, which="both", ls="--", lw=0.5, alpha=0.5)

    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")
    return out


# ---------------------------------------------------------------------------
# Table 4: Supervised estimation errors (§6.3)
# ---------------------------------------------------------------------------


def make_tab_supervised(
    df_sup: pd.DataFrame,
    fig_dir: pathlib.Path,
    N_cols: tuple = (200, 500, 2000),
) -> pathlib.Path:
    """
    Complete LaTeX table environment for Table 4 (tab:supervised_M1):
    relative F error and AB residual for each projection × N.

    Generated file is a self-contained ``\\begin{table}...\\end{table}``
    block (use ``\\input`` from outside any tabular environment).
    """
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / "tab_supervised_M1.tex"

    PROJ_LABELS = {
        "none": "Free OLS",
        "ab": "AB constraint",
    }

    data_lines = []
    for proj, label in PROJ_LABELS.items():
        sub = df_sup[df_sup["projection"] == proj]
        parts = [label]
        for N in N_cols:
            g = sub[sub["N"] == N]["rel_err_F"].dropna()
            parts.append(_fmt(g.mean()) if len(g) else r"\text{---}")
        # AB residual at largest N
        g_ab = sub[sub["N"] == max(N_cols)]["ab_residual"].dropna()
        med_ab = float(g_ab.median()) if len(g_ab) else float("nan")
        if np.isfinite(med_ab) and med_ab > 0:
            exp = int(np.floor(np.log10(med_ab)))
            parts.append(f"$10^{{{exp}}}$")
        else:
            parts.append(r"\text{---}")
        data_lines.append("    " + " & ".join(parts) + r" \\")

    n_N = len(N_cols)
    cmidrule_end = 1 + n_N
    col_N_hdrs = " & ".join(f"$N={N}$" for N in N_cols)
    data_block = "\n".join(data_lines)

    cap = (
        r"Supervised estimation on M1."
        r" Columns: relative Frobenius error"
        r" $\|\hat F_k - F_k\|_F / \|F_k\|_F$ (mean, averaged over"
        r" the two regimes $k=1,2$), for $N \in \{200, 500, 2000\}$,"
        r" and median AB residual at $N=2\,000$."
        r" The AB constraint reduces the AB residual to machine precision"
        r" by construction."
    )
    lbl = "tab:supervised_M1"

    table_lines = [
        r"\begin{table}[ht]",
        r"  \caption{" + cap + "}",
        r"  \label{" + lbl + "}",
        r"  \centering\small",
        r"  \renewcommand{\arraystretch}{1.2}",
        r"  \begin{tabular}{@{}lcccc@{}}",
        r"    \toprule",
        (
            r"    & \multicolumn{" + str(n_N) + r"}{c}"
            r"{Rel.\ error $\|\hat F - F\|_F/\|F\|_F$} & AB resid. \\"
        ),
        r"    \cmidrule(lr){2-" + str(cmidrule_end) + "}",
        r"    Setting & " + col_N_hdrs + r" & ($N=2\,000$) \\",
        r"    \midrule",
        data_block,
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]
    out.write_text("\n".join(table_lines) + "\n", encoding="utf-8")
    print(f"  Saved: {out}")
    return out


# ---------------------------------------------------------------------------
# Figure 3: EM log-likelihood convergence (§6.4)
# ---------------------------------------------------------------------------


def make_fig_em_convergence(
    df_hist: pd.DataFrame,
    fig_dir: pathlib.Path,
    model: str = "M1",
    N_plot: int = 2000,
    n_curves: int = 5,
) -> pathlib.Path | None:
    """
    LL convergence over EM iterations for PH vs GEM, paper Fig. 3.

    Plots the first `n_curves` seeds (thin lines) plus the mean (thick).
    """
    if not _HAS_MPL:
        return None

    fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / "fig_em_convergence.pdf"

    sub = df_hist[(df_hist["model"] == model) & (df_hist["N"] == N_plot)]
    if sub.empty:
        print(f"  WARNING: no EM history for model={model} N={N_plot} — skipping Fig. 3")
        return None

    VARIANT_STYLE = {
        "PH": dict(color="#1f77b4", label="PH (post-hoc)"),
        "GEM": dict(color="#ff7f0e", label="GEM"),
    }

    fig, ax = plt.subplots(figsize=(4.5, 3.2))

    for var, style in VARIANT_STYLE.items():
        sv = sub[sub["variant"] == var]
        seeds = sorted(sv["seed"].unique())[:n_curves]

        all_iters = []
        for seed in seeds:
            sg = sv[sv["seed"] == seed].sort_values("iteration")
            iters = sg["iteration"].values
            lls = sg["log_lik"].values
            ax.plot(iters, lls, color=style["color"], lw=0.8, alpha=0.35)
            all_iters.append((iters, lls))

        # Mean curve (over all available seeds for this N)
        all_seeds = sorted(sv["seed"].unique())
        max_it = max(sv["iteration"].max(), 0)
        ll_mat = []
        for seed in all_seeds:
            sg = sv[sv["seed"] == seed].sort_values("iteration")
            ll_mat.append(sg["log_lik"].values[: max_it + 1])
        min_len = min(len(v) for v in ll_mat)
        ll_arr = np.array([v[:min_len] for v in ll_mat])
        ax.plot(
            np.arange(min_len),
            ll_arr.mean(axis=0),
            color=style["color"],
            lw=2.2,
            label=style["label"],
        )

    ax.set_xlabel("EM iteration", fontsize=10)
    ax.set_ylabel(r"$\log p(Z_{1:N})$", fontsize=10)
    ax.set_title(
        rf"M1, $N={N_plot}$ — EM convergence (PH vs GEM, AB constraint)",
        fontsize=9,
    )
    ax.legend(fontsize=9, framealpha=0.85)
    ax.tick_params(axis="both", labelsize=9)
    ax.grid(True, ls="--", lw=0.5, alpha=0.5)

    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")
    return out


# ---------------------------------------------------------------------------
# Table 5: EM basin selection rate (§6.4)
# ---------------------------------------------------------------------------


def make_tab_em_basin(
    df_em: pd.DataFrame,
    fig_dir: pathlib.Path,
) -> pathlib.Path:
    """
    Complete LaTeX table environment for Table 5 (tab:em_basin):
    basin selection rate for PH and GEM at each N (model M1).

    Rows: N=500, N=2000
    Columns: N | basin_rate PH | n_iter PH | basin_rate GEM | n_iter GEM

    Generated file is a self-contained ``\\begin{table}...\\end{table}``
    block (use ``\\input`` from outside any tabular environment).
    """
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / "tab_em_basin.tex"

    sub = df_em[df_em["model"] == "M1"]
    data_lines = []
    for N in sorted(sub["N"].unique()):
        parts = [str(N)]
        for var in ("PH", "GEM"):
            g = sub[(sub["N"] == N) & (sub["variant"] == var)]
            if g.empty:
                parts += [r"\text{---}", r"\text{---}"]
                continue
            br = g["basin_rate"].mean()
            nit = g["best_n_iter"].mean()
            parts.append(f"{br * 100:.1f}\\%")
            parts.append(f"{nit:.1f}")
        data_lines.append("    " + " & ".join(parts) + r" \\")

    data_block = "\n".join(data_lines)

    cap = (
        r"Semi-supervised EM on M1: basin selection rate (fraction"
        r" of MC runs reaching the best LL basin with $n_{\mathrm{init}}=5$"
        r" restarts) and mean number of EM iterations used (budget: 50)."
        r" PH = post-hoc AB projection; GEM = AB constraint at every M-step."
    )
    lbl = "tab:em_basin"

    table_lines = [
        r"\begin{table}[ht]",
        r"  \caption{" + cap + "}",
        r"  \label{" + lbl + "}",
        r"  \centering\small",
        r"  \renewcommand{\arraystretch}{1.2}",
        r"  \begin{tabular}{@{}r cc cc@{}}",
        r"    \toprule",
        r"    & \multicolumn{2}{c}{PH} & \multicolumn{2}{c}{GEM} \\",
        r"    \cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        (
            r"    $N$ & Basin (\%) & $n_{\mathrm{iter}}$"
            r" & Basin (\%) & $n_{\mathrm{iter}}$ \\"
        ),
        r"    \midrule",
        data_block,
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]
    out.write_text("\n".join(table_lines) + "\n", encoding="utf-8")
    print(f"  Saved: {out}")
    return out


# ---------------------------------------------------------------------------
# Table 6: EM basin rate vs number of restarts (§6.4)
# ---------------------------------------------------------------------------


def make_tab_em_restarts(
    df_em: pd.DataFrame,
    fig_dir: pathlib.Path,
    model: str = "M1",
    N_plot: int = 2000,
    variant: str = "PH",
    n_init_vals: tuple = (1, 2, 3, 5),
    basin_tol: float = 0.01,
) -> pathlib.Path:
    """
    Complete LaTeX table environment for Table 6 (tab:em_restarts):
    basin selection rate as a function of the number of EM restarts,
    computed post-hoc from ``all_log_liks``.

    Uses only the first k restarts (k ∈ n_init_vals) from the stored
    all_log_liks column; no extra simulations required.

    Generated file is a self-contained ``\\begin{table}...\\end{table}``
    block (use ``\\input`` from outside any tabular environment).
    """
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / "tab_em_restarts.tex"

    sub = df_em[(df_em["model"] == model) & (df_em["N"] == N_plot) & (df_em["variant"] == variant)]

    if sub.empty:
        # Fallback: minimal placeholder table
        content = (
            r"\begin{table}[ht]"
            "\n"
            r"  \caption{EM restarts --- no data available.}"
            "\n"
            r"  \label{tab:em_restarts}"
            "\n"
            r"  \centering\small"
            "\n"
            r"  \begin{tabular}{@{}cc@{}}"
            "\n"
            r"    \toprule"
            "\n"
            r"    $n_{\mathrm{init}}$ & Best-basin rate (\%) \\"
            "\n"
            r"    \midrule"
            "\n"
            r"    \multicolumn{2}{c}{\textit{(no data)}} \\"
            "\n"
            r"    \bottomrule"
            "\n"
            r"  \end{tabular}"
            "\n"
            r"\end{table}"
            "\n"
        )
        out.write_text(content, encoding="utf-8")
        print(f"  WARNING: no EM data for model={model} N={N_plot} variant={variant}")
        return out

    data_lines = []
    for n_init in n_init_vals:
        rates = []
        for _, row in sub.iterrows():
            raw = str(row.get("all_log_liks", ""))
            if not raw or raw == "nan":
                continue
            all_lls = [float(x) for x in raw.split(";") if x]
            if not all_lls or not any(np.isfinite(ll) for ll in all_lls):
                continue
            # Global best over ALL stored restarts
            global_best = max(ll for ll in all_lls if np.isfinite(ll))
            threshold = global_best - basin_tol * abs(global_best)
            # Check whether any of the FIRST n_init restarts reaches the basin
            lls = all_lls[:n_init]
            basin = any(ll >= threshold for ll in lls if np.isfinite(ll))
            rates.append(float(basin))

        br_pct = np.mean(rates) * 100 if rates else float("nan")
        br_str = f"{br_pct:.1f}\\%" if np.isfinite(br_pct) else r"\text{---}"
        data_lines.append(f"    {n_init} & {br_str}" + r" \\")

    data_block = "\n".join(data_lines)

    cap = (
        r"EM basin selection rate on M1 ($N = 2\,000$,"
        r" post-hoc AB constraint):"
        r" fraction of MC runs where at least one of the first"
        r" $n_{\mathrm{init}}$ restarts lands in the best"
        r" log-likelihood basin."
    )
    lbl = "tab:em_restarts"

    table_lines = [
        r"\begin{table}[ht]",
        r"  \caption{" + cap + "}",
        r"  \label{" + lbl + "}",
        r"  \centering\small",
        r"  \renewcommand{\arraystretch}{1.2}",
        r"  \begin{tabular}{@{}cc@{}}",
        r"    \toprule",
        r"    $n_{\mathrm{init}}$ & Best-basin rate (\%) \\",
        r"    \midrule",
        data_block,
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]
    out.write_text("\n".join(table_lines) + "\n", encoding="utf-8")
    print(f"  Saved: {out}")
    return out


def print_summary(df: pd.DataFrame) -> None:
    """Print a compact summary table of mean metrics."""
    print("\n" + "=" * 70)
    print("Monte-Carlo summary (mean over runs)")
    print("=" * 70)
    print(
        f"{'Model':>5}  {'N':>5}  {'Mode':>12}  "
        f"{'RMSE':>8}  {'ANEES':>7}  {'LB p':>7}  {'JB p':>7}  {'CPU(s)':>7}"
    )
    print("-" * 70)
    for (model, N, mode), g in df.groupby(["model", "N", "mode"]):
        print(
            f"{model:>5}  {N:>5}  {mode:>12}  "
            f"{g['rmse'].mean():8.4f}  "
            f"{g['nees'].mean():7.3f}  "
            f"{g['lb_pval'].mean():7.3f}  "
            f"{g['jb_pval'].mean():7.3f}  "
            f"{g['cpu_s'].mean():7.3f}"
        )
    print("=" * 70)


# ---------------------------------------------------------------------------
# Master entry point
# ---------------------------------------------------------------------------


def make_all(
    results_path: pathlib.Path = DEFAULT_IN,
    fig_dir: pathlib.Path = DEFAULT_FIG_DIR,
    supervised_path: pathlib.Path | None = None,
    em_path: pathlib.Path | None = None,
    em_history_path: pathlib.Path | None = None,
) -> None:
    """
    Load all available results files and generate figures + table fragments.

    Parameters
    ----------
    results_path    : mc_results.csv  (§6.2 filter benchmark)
    supervised_path : supervised_results.csv  (§6.3), optional
    em_path         : em_results.csv  (§6.4), optional
    em_history_path : em_ll_history.csv  (§6.4 LL curves), optional
    fig_dir         : output directory
    """
    # Auto-detect sibling files if not provided
    data_dir = results_path.parent
    if supervised_path is None:
        supervised_path = data_dir / "supervised_results.csv"
    if em_path is None:
        em_path = data_dir / "em_results.csv"
    if em_history_path is None:
        em_history_path = data_dir / "em_ll_history.csv"

    # --- §6.2: filter benchmark -----------------------------------------
    if not results_path.exists():
        print(f"WARNING: {results_path} not found — skipping §6.2 figures.", file=sys.stderr)
    else:
        print(f"Loading filter results: {results_path}")
        df = _load(results_path)
        print(
            f"  {len(df)} rows, {df['model'].nunique()} models, "
            f"{df['N'].nunique()} N values, {df['mode'].nunique()} modes."
        )
        print_summary(df)
        print(f"\nGenerating §6.2 outputs → {fig_dir}/")
        make_fig_rmse_vs_N(df, fig_dir)
        make_tab_filter(df, fig_dir)
        make_tab_bic(df, fig_dir)

    # --- §6.3: supervised estimation ------------------------------------
    if not supervised_path.exists():
        print(f"INFO: {supervised_path} not found — skipping §6.3 figures.")
    else:
        print(f"\nLoading supervised results: {supervised_path}")
        df_sup = pd.read_csv(supervised_path)
        for c in ("rel_err_F", "rel_err_b", "ab_residual", "rmse_estimated", "rmse_oracle"):
            df_sup[c] = pd.to_numeric(df_sup[c], errors="coerce")
        print(f"  {len(df_sup)} rows")
        print(f"Generating §6.3 outputs → {fig_dir}/")
        make_fig_supervised_rmse(df_sup, fig_dir)
        make_tab_supervised(df_sup, fig_dir)

    # --- §6.4: semi-supervised EM ----------------------------------------
    if not em_path.exists():
        print(f"INFO: {em_path} not found — skipping §6.4 figures.")
    else:
        print(f"\nLoading EM results: {em_path}")
        df_em = pd.read_csv(em_path)
        for c in ("best_log_lik", "basin_rate", "rel_err_F", "rmse_estimated", "rmse_oracle"):
            df_em[c] = pd.to_numeric(df_em[c], errors="coerce")
        print(f"  {len(df_em)} rows")
        print(f"Generating §6.4 outputs → {fig_dir}/")
        make_tab_em_basin(df_em, fig_dir)
        make_tab_em_restarts(df_em, fig_dir)

        if em_history_path.exists():
            df_hist = pd.read_csv(em_history_path)
            df_hist["log_lik"] = pd.to_numeric(df_hist["log_lik"], errors="coerce")
            make_fig_em_convergence(df_hist, fig_dir)

    print("\nAll done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate paper figures from MC results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results",
        default=str(DEFAULT_IN),
        help="Path to mc_results.csv (§6.2).",
    )
    parser.add_argument(
        "--supervised",
        default=None,
        help="Path to supervised_results.csv (§6.3). Auto-detected if omitted.",
    )
    parser.add_argument(
        "--em",
        default=None,
        help="Path to em_results.csv (§6.4). Auto-detected if omitted.",
    )
    parser.add_argument(
        "--em-history",
        default=None,
        help="Path to em_ll_history.csv. Auto-detected if omitted.",
    )
    parser.add_argument(
        "--fig-dir",
        default=str(DEFAULT_FIG_DIR),
        help="Output directory for figures and table fragments.",
    )
    args = parser.parse_args()
    make_all(
        results_path=pathlib.Path(args.results),
        fig_dir=pathlib.Path(args.fig_dir),
        supervised_path=pathlib.Path(args.supervised) if args.supervised else None,
        em_path=pathlib.Path(args.em) if args.em else None,
        em_history_path=pathlib.Path(args.em_history) if args.em_history else None,
    )
