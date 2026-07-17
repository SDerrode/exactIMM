#!/usr/bin/env python3
"""
scripts/e1_supervised_ab.py
===========================
Experiment **E1** — Empirical AB test on the S&P500 / VIX dataset.

For each proxy label (L1 = VIX median, L2 = NBER recession):
    1. Run supervised OLS per regime (unconstrained).
    2. Report   ‖B(k)‖_F   and Fisher p-value of   H0 : B(k) = 0.
    3. Re-run with constraint='b' (post-hoc AB projection on B).

The resulting Table 1 tells us whether B(k) ≠ 0 is statistically
detectable — i.e. whether AB is a non-trivial assumption on this data.

Outputs
-------
    results/e1/table1.json          numerical results (reproducibility)
    results/e1/table1.tex           LaTeX table ready for \\input{...}
    results/e1/console.log          plain-text summary
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Project root on sys.path so we can import prg.* regardless of CWD.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prg.learning.supervised import fit_supervised  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from labels import (  # noqa: E402
    TRAIN_END,
    get_label,
    load_dataset,
    standardize_with_train_stats,
    summarize_label,
    train_test_split,
)

DEFAULT_CSV = ROOT / "data/real/sp500_vix.csv"
DEFAULT_OUT = ROOT / "results/e1"


# ---------------------------------------------------------------------------
# Fisher test : H0 : B(k) = 0  on train segment, per regime
# ---------------------------------------------------------------------------
def fisher_test_B_zero(
    xs: np.ndarray, ys: np.ndarray, rs: np.ndarray, k: int
) -> tuple[float, float, int]:
    """
    Nested-model F-test of  H0 : B(k) = 0  on the rows where r_{n+1} = k.

    Full model   : X_{n+1} = a X_n + b Y_n + c           (q=s=1 case)
    Null  model  : X_{n+1} = a X_n          + c

    Returns
    -------
    F_stat       : float
    p_value      : float
    df_residual  : int
    """
    from scipy.stats import f as f_dist

    mask = rs[1:] == k
    x_curr = xs[:-1][mask].ravel()
    y_curr = ys[:-1][mask].ravel()
    x_next = xs[1:][mask].ravel()
    n = x_curr.size

    # Full-model design: [x, y, 1]
    X_full = np.column_stack([x_curr, y_curr, np.ones(n)])
    beta_full, *_ = np.linalg.lstsq(X_full, x_next, rcond=None)
    rss_full = float(np.sum((x_next - X_full @ beta_full) ** 2))

    # Null-model design: [x, 1]
    X_null = np.column_stack([x_curr, np.ones(n)])
    beta_null, *_ = np.linalg.lstsq(X_null, x_next, rcond=None)
    rss_null = float(np.sum((x_next - X_null @ beta_null) ** 2))

    df_resid = n - X_full.shape[1]
    df_diff = X_full.shape[1] - X_null.shape[1]  # == 1 here
    if df_resid <= 0 or rss_full <= 0.0:
        return float("nan"), float("nan"), df_resid

    F_stat = ((rss_null - rss_full) / df_diff) / (rss_full / df_resid)
    p_value = float(f_dist.sf(F_stat, df_diff, df_resid))
    return float(F_stat), p_value, df_resid


# ---------------------------------------------------------------------------
# Run supervised OLS (unconstrained + projected on B) on one label set
# ---------------------------------------------------------------------------
def run_one_label(train: pd.DataFrame, labels_train: np.ndarray, K: int) -> dict:
    xs = train[["log_return"]].to_numpy()
    ys = train[["log_vix"]].to_numpy()
    rs = labels_train.astype(int)

    result_raw = fit_supervised(rs, xs, ys, K=K, q=1, s=1, constraint=None)
    try:
        result_proj = fit_supervised(rs, xs, ys, K=K, q=1, s=1, constraint="b")
    except ValueError as exc:
        print(f"[warn] projection on B failed: {exc}", flush=True)
        result_proj = None

    per_regime = []
    for k in range(K):
        B_raw = result_raw["B_list"][k]
        B_norm = float(np.linalg.norm(B_raw, ord="fro"))
        F_stat, p_val, df = fisher_test_B_zero(xs, ys, rs, k)
        n_k = int(np.sum(rs[1:] == k))
        per_regime.append(
            {
                "k": k,
                "n_transitions": n_k,
                "B_entry": float(B_raw.ravel()[0]),
                "B_fro": B_norm,
                "F_stat": F_stat,
                "p_value": p_val,
                "df_residual": df,
            }
        )
    return {
        "K": K,
        "n_train": int(len(rs)),
        "per_regime": per_regime,
        "has_projection": result_proj is not None,
    }


# ---------------------------------------------------------------------------
# LaTeX table emitter
# ---------------------------------------------------------------------------
def _fmt_p(p: float) -> str:
    if np.isnan(p):
        return "n/a"
    if p < 1e-4:
        return r"$<\!10^{-4}$"
    return f"{p:.3g}"


def emit_tex_table(results: dict, out_path: Path) -> None:
    lines = [
        r"% Table 1 — empirical AB test on S&P500 / VIX (train period only)",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Label & Regime & $n_k$ & $B(k)$ & $\|B(k)\|_F$ & $p$-value \\",
        r"\midrule",
    ]
    for label_name, res in results.items():
        for i, row in enumerate(res["per_regime"]):
            lab_col = label_name if i == 0 else ""
            lines.append(
                f"{lab_col} & {row['k']} & {row['n_transitions']} & "
                f"{row['B_entry']:.4f} & {row['B_fro']:.4f} & {_fmt_p(row['p_value'])} \\\\"
            )
        lines.append(r"\midrule")
    # remove the last \midrule (replace with \bottomrule)
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}"]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[3])
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--K", type=int, default=2)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ---- data ----
    df = load_dataset(args.csv)
    train, test = train_test_split(df, TRAIN_END)

    # ---- standardize with train-only stats (same convention as paper) ----
    df_std, stats = standardize_with_train_stats(df, train, cols=("log_return", "log_vix"))
    train_std = df_std.loc[df_std.index <= TRAIN_END]

    # ---- labels ----
    all_results: dict = {}
    for kind in ("L1", "L2"):
        labels = get_label(df, train, kind)
        labels_train = labels.loc[labels.index <= TRAIN_END].to_numpy()
        print(summarize_label(labels, kind))
        res = run_one_label(train_std, labels_train, K=args.K)
        all_results[kind] = res
        for row in res["per_regime"]:
            print(
                f"  [{kind}] k={row['k']}  n_k={row['n_transitions']:5d}  "
                f"B={row['B_entry']:+.4f}  ||B||_F={row['B_fro']:.4f}  "
                f"F={row['F_stat']:7.2f}  p={row['p_value']:.3g}"
            )

    # ---- persist ----
    (args.out_dir / "table1.json").write_text(
        json.dumps(
            {
                "standardization": {
                    "log_return": {"mean": stats["log_return"][0], "std": stats["log_return"][1]},
                    "log_vix": {"mean": stats["log_vix"][0], "std": stats["log_vix"][1]},
                },
                "train_end": str(TRAIN_END.date()),
                "K": args.K,
                "results": all_results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    emit_tex_table(all_results, args.out_dir / "table1.tex")
    print(f"\n[e1] wrote {args.out_dir / 'table1.json'}")
    print(f"[e1] wrote {args.out_dir / 'table1.tex'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
