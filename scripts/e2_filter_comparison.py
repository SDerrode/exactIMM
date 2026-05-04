#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/e2_filter_comparison.py
===============================
Experiment **E2** — Filter comparison at fixed parameters on S&P500 / VIX.

Parameters are fit by supervised OLS on the training period using label L1
(VIX-median); the filters are then evaluated on the test period (2019-2024).

Filters compared (extensible — baselines plug in via the FILTERS dict):
    h5_exact        : our filter under (H5) assumption (biased if B≠0)
    imm_general     : our filter, general IMM recursion
    kalman_k1       : single Kalman (K=1, no regime switching)
    imm_standard    : Blom-Bar-Shalom approximate IMM            [TODO]
    gpb2            : Generalized Pseudo-Bayesian order 2         [TODO]
    rbpf            : Rao-Blackwellised particle filter            [TODO]

Outputs
-------
    results/e2/table2.json   numerical results
    results/e2/table2.tex    LaTeX table
    results/e2/trace.csv     per-step X̂ / π_n from each filter
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from labels import (  # noqa: E402
    TRAIN_END,
    get_label,
    load_dataset,
    standardize_with_train_stats,
    train_test_split,
)
from params_utils import params_from_dict  # noqa: E402
from prg.filter.gss_filter import GSSFilter  # noqa: E402
from prg.learning.supervised import fit_supervised  # noqa: E402
from baselines.kalman_single import SingleKalmanFilter  # noqa: E402


@dataclass
class FilterScore:
    name:       str
    N:          int
    log_lik:    float
    mse_x:      float
    nll_per_obs: float
    time_s:     float


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run_filter(name: str, filt, xs_test: np.ndarray, ys_test: np.ndarray,
               trace_store: dict[str, np.ndarray] | None = None) -> FilterScore:
    t0 = time.perf_counter()
    ll_total = 0.0
    sse = 0.0
    N = len(ys_test)
    x_hat = np.empty(N)
    for i, y in enumerate(ys_test):
        y_col = np.asarray(y, dtype=float).reshape(-1, 1)
        r = filt.step(y_col)
        xi = float(r.E_x.ravel()[0])
        x_hat[i] = xi
        sse += (xs_test[i, 0] - xi) ** 2
        ll_total += r.log_lik
    dt = time.perf_counter() - t0
    if trace_store is not None:
        trace_store[f"{name}_x_hat"] = x_hat
    return FilterScore(
        name=name, N=N,
        log_lik=ll_total,
        mse_x=sse / N,
        nll_per_obs=-ll_total / N,
        time_s=dt,
    )


# ---------------------------------------------------------------------------
# LaTeX emission
# ---------------------------------------------------------------------------
def emit_tex_table(scores: list[FilterScore], out_path: Path) -> None:
    lines = [
        r"% Table 2 — filter comparison on S&P500/VIX test period (2019-2024)",
        r"% parameters fit by supervised OLS on train with label L1 (VIX-median)",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Filter & $\log\hat L$ & NLL/obs & MSE $X$ & time (s) \\",
        r"\midrule",
    ]
    for s in scores:
        lines.append(
            f"{s.name:15s} & {s.log_lik:9.1f} & {s.nll_per_obs:+.4f} & "
            f"{s.mse_x:.4f} & {s.time_s:.2f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[3])
    parser.add_argument("--csv", type=Path, default=ROOT / "data/real/sp500_vix.csv")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results/e2")
    parser.add_argument("--K", type=int, default=2)
    parser.add_argument("--label", choices=("L1", "L2"), default="L1")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ---- data ----
    df = load_dataset(args.csv)
    train, _test = train_test_split(df, TRAIN_END)
    df_std, stats = standardize_with_train_stats(
        df, train, cols=("log_return", "log_vix"),
    )
    train_std = df_std.loc[df_std.index <= TRAIN_END]
    test_std  = df_std.loc[df_std.index >  TRAIN_END]

    xs_tr = train_std[["log_return"]].to_numpy()
    ys_tr = train_std[["log_vix"]].to_numpy()
    xs_te = test_std[["log_return"]].to_numpy()
    ys_te = test_std[["log_vix"]].to_numpy()

    labels = get_label(df, train, args.label)
    rs_tr = labels.loc[labels.index <= TRAIN_END].to_numpy()

    # ---- fit GSS params (supervised) ----
    fit = fit_supervised(rs_tr, xs_tr, ys_tr, K=args.K, q=1, s=1, constraint=None)
    params = params_from_dict(fit)
    print(f"[e2] params fitted, K={params.K}, "
          f"pi_inf={params.stationary_distribution().round(3)}")

    # ---- build filter bank ----
    traces: dict[str, np.ndarray] = {}
    scores: list[FilterScore] = []

    # h5_exact (warns for non-H5 models, expected here)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        gss_h5 = GSSFilter(params, mode="h5_exact")
    scores.append(run_filter("h5_exact", gss_h5, xs_te, ys_te, traces))

    # imm_general
    gss_gen = GSSFilter(params, mode="imm_general")
    scores.append(run_filter("imm_general", gss_gen, xs_te, ys_te, traces))

    # Kalman K=1 — fit on full train regardless of regimes
    kf = SingleKalmanFilter.from_regressed(xs_tr, ys_tr)
    scores.append(run_filter("kalman_k1", kf, xs_te, ys_te, traces))

    # ---- print summary ----
    print("")
    hdr = f"{'filter':<15s} {'log_lik':>10s} {'NLL/obs':>10s} {'MSE':>8s} {'t(s)':>6s}"
    print(hdr)
    print("-" * len(hdr))
    for s in scores:
        print(f"{s.name:<15s} {s.log_lik:>10.1f} {s.nll_per_obs:>+10.4f} "
              f"{s.mse_x:>8.4f} {s.time_s:>6.2f}")

    # ---- persist ----
    (args.out_dir / "table2.json").write_text(
        json.dumps({
            "label": args.label,
            "K": args.K,
            "n_train": int(xs_tr.shape[0]),
            "n_test":  int(xs_te.shape[0]),
            "standardization": {
                "log_return": {"mean": stats["log_return"][0],
                               "std":  stats["log_return"][1]},
                "log_vix":    {"mean": stats["log_vix"][0],
                               "std":  stats["log_vix"][1]},
            },
            "scores": [s.__dict__ for s in scores],
        }, indent=2),
        encoding="utf-8",
    )
    emit_tex_table(scores, args.out_dir / "table2.tex")

    # ---- traces CSV ----
    trace_df = pd.DataFrame({
        "date":      test_std.index,
        "x_true":    xs_te[:, 0],
        "y_obs":     ys_te[:, 0],
        **traces,
    })
    trace_df.to_csv(args.out_dir / "trace.csv", index=False)
    print(f"\n[e2] wrote {args.out_dir / 'table2.json'}")
    print(f"[e2] wrote {args.out_dir / 'table2.tex'}")
    print(f"[e2] wrote {args.out_dir / 'trace.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
