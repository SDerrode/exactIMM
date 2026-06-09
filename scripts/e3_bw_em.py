#!/usr/bin/env python3
"""
scripts/e3_bw_em.py
===================
Experiment **E3** — Semi-supervised BW-EM estimation of the GSS model
on the S&P500/VIX dataset, with four H5-projection variants.

Variants
--------
    V0  unconstrained           (constraint=None,  each_iter=False)
    V1  post-hoc projection B   (constraint='b',   each_iter=False)
    V2  post-hoc projection A   (constraint='a',   each_iter=False)
    V3  GEM (projection B at every M-step)
                                (constraint='b',   each_iter=True)

Each variant is run ``n_inits`` times with different k-means seeds; the
best-likelihood restart is retained.

Evaluation
----------
    - train log-lik (from EM, best restart)
    - test  log-lik via :class:`GSSFilter` with ``mode='imm_general'``
    - test  MSE on X
    - regime accuracy / ARI vs proxy labels L1 (VIX-median) and L2 (NBER)
      after Hungarian-like label alignment

Outputs
-------
    results/e3/table3.json     numeric results (all variants)
    results/e3/table3.tex      LaTeX table
    results/e3/em_history.json log-lik over iterations, best restart per variant
    results/e3/regime_trace.csv  filtered π_n(1), regime argmax, ground truth
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

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
from prg.learning.semi_supervised import fit_semi_supervised  # noqa: E402

VARIANTS = {
    "V0_unconstr": dict(constraint=None, constraint_each_iter=False),
    "V1_posthoc_B": dict(constraint="b", constraint_each_iter=False),
    "V2_posthoc_A": dict(constraint="a", constraint_each_iter=False),
    "V3_GEM_B": dict(constraint="b", constraint_each_iter=True),
}


@dataclass
class VariantResult:
    name: str
    train_log_lik: float
    test_log_lik: float
    test_nll_per_obs: float
    test_mse_x: float
    accuracy_L1: float
    accuracy_L2: float
    ari_L1: float
    ari_L2: float
    all_train_log_liks: list[float] = field(default_factory=list)
    n_iter: int = 0
    converged: bool = False
    time_s: float = 0.0


def _best_regime_alignment(r_hat: np.ndarray, r_true: np.ndarray, K: int) -> np.ndarray:
    """
    Return a permutation of {0..K-1} that maximises accuracy of
    ``permutation[r_hat]`` against ``r_true``. Brute force over K! (OK for K≤4).
    """
    from itertools import permutations

    best_perm = None
    best_acc = -1.0
    for perm in permutations(range(K)):
        perm_arr = np.array(perm, dtype=int)
        acc = float(np.mean(perm_arr[r_hat] == r_true))
        if acc > best_acc:
            best_acc = acc
            best_perm = perm_arr
    return best_perm


def run_filter_on_test(
    params_dict: dict, xs_te: np.ndarray, ys_te: np.ndarray, trace: dict | None = None
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Evaluate a fitted params dict on the test set via imm_general."""
    params = params_from_dict(params_dict)
    filt = GSSFilter(params, mode="imm_general")
    N = len(ys_te)
    K = params.K
    pi_n = np.empty((N, K))
    x_hat = np.empty(N)
    ll = 0.0
    sse = 0.0
    for i, y in enumerate(ys_te):
        r = filt.step(np.asarray(y, dtype=float).reshape(-1, 1))
        pi_n[i] = r.pi
        x_hat[i] = float(r.E_x.ravel()[0])
        sse += (xs_te[i, 0] - x_hat[i]) ** 2
        ll += r.log_lik
    if trace is not None:
        trace["pi_n"] = pi_n
        trace["x_hat"] = x_hat
    return ll, sse / N, pi_n, x_hat


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=ROOT / "data/real/sp500_vix.csv")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results/e3")
    parser.add_argument("--K", type=int, default=2)
    parser.add_argument("--n-inits", type=int, default=10)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS.keys()))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ---- data ----
    df = load_dataset(args.csv)
    train, _ = train_test_split(df, TRAIN_END)
    df_std, stats = standardize_with_train_stats(
        df,
        train,
        cols=("log_return", "log_vix"),
    )
    train_std = df_std.loc[df_std.index <= TRAIN_END]
    test_std = df_std.loc[df_std.index > TRAIN_END]

    xs_tr = train_std[["log_return"]].to_numpy()
    ys_tr = train_std[["log_vix"]].to_numpy()
    xs_te = test_std[["log_return"]].to_numpy()
    ys_te = test_std[["log_vix"]].to_numpy()

    # Ground-truth-style regime labels on the *test* period
    L1_all = get_label(df, train, "L1")
    L2_all = get_label(df, train, "L2")
    r_true_L1_test = L1_all.loc[L1_all.index > TRAIN_END].to_numpy()
    r_true_L2_test = L2_all.loc[L2_all.index > TRAIN_END].to_numpy()

    # ---- run each variant ----
    results: list[VariantResult] = []
    em_histories: dict[str, list[float]] = {}
    pi_traces: dict[str, np.ndarray] = {}

    for vname in args.variants:
        cfg = VARIANTS[vname]
        print(
            f"\n=== {vname}  (n_inits={args.n_inits})  "
            f"constraint={cfg['constraint']!r}  "
            f"each_iter={cfg['constraint_each_iter']}"
        )
        t0 = time.perf_counter()
        params, info = fit_semi_supervised(
            xs_tr,
            ys_tr,
            K=args.K,
            constraint=cfg["constraint"],
            constraint_each_iter=cfg["constraint_each_iter"],
            n_inits=args.n_inits,
            max_iter=args.max_iter,
            seed=args.seed,
            verbose=False,
        )
        dt = time.perf_counter() - t0

        # test evaluation
        trace: dict = {}
        test_ll, test_mse, pi_n, x_hat = run_filter_on_test(
            params,
            xs_te,
            ys_te,
            trace=trace,
        )
        r_hat_test = np.argmax(pi_n, axis=1)
        perm_L1 = _best_regime_alignment(r_hat_test, r_true_L1_test, args.K)
        perm_L2 = _best_regime_alignment(r_hat_test, r_true_L2_test, args.K)
        r_hat_L1 = perm_L1[r_hat_test]
        r_hat_L2 = perm_L2[r_hat_test]

        res = VariantResult(
            name=vname,
            train_log_lik=info["best_log_lik"],
            test_log_lik=test_ll,
            test_nll_per_obs=-test_ll / len(ys_te),
            test_mse_x=test_mse,
            accuracy_L1=float(np.mean(r_hat_L1 == r_true_L1_test)),
            accuracy_L2=float(np.mean(r_hat_L2 == r_true_L2_test)),
            ari_L1=float(adjusted_rand_score(r_true_L1_test, r_hat_test)),
            ari_L2=float(adjusted_rand_score(r_true_L2_test, r_hat_test)),
            all_train_log_liks=[float(x) for x in info["all_log_liks"]],
            n_iter=int(info["best_n_iter"]),
            converged=bool(info["best_converged"]),
            time_s=dt,
        )
        results.append(res)
        em_histories[vname] = [float(x) for x in info["log_lik_history"]]
        pi_traces[vname] = pi_n

        print(
            f"  train logL = {res.train_log_lik:+.1f}   "
            f"test logL = {res.test_log_lik:+.1f}   "
            f"MSE = {res.test_mse_x:.4f}   "
            f"acc(L1)= {res.accuracy_L1:.3f}   "
            f"ari(L2)= {res.ari_L2:+.3f}   "
            f"time {dt:.1f}s"
        )

    # ---- persist ----
    (args.out_dir / "table3.json").write_text(
        json.dumps(
            {
                "K": args.K,
                "n_inits": args.n_inits,
                "n_train": int(xs_tr.shape[0]),
                "n_test": int(xs_te.shape[0]),
                "standardization": {
                    "log_return": {"mean": stats["log_return"][0], "std": stats["log_return"][1]},
                    "log_vix": {"mean": stats["log_vix"][0], "std": stats["log_vix"][1]},
                },
                "variants": [r.__dict__ for r in results],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (args.out_dir / "em_history.json").write_text(
        json.dumps(em_histories, indent=2),
        encoding="utf-8",
    )

    # LaTeX table
    lines = [
        r"% Table 3 — BW-EM variants on S&P500/VIX (test evaluation)",
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"Variant & train $\log\hat L$ & test NLL/obs & test MSE $X$ & "
        r"acc(L1) & ARI(L1) & acc(L2) & ARI(L2) \\",
        r"\midrule",
    ]
    for r in results:
        lines.append(
            f"{r.name:14s} & {r.train_log_lik:+.1f} & {r.test_nll_per_obs:+.4f} & "
            f"{r.test_mse_x:.4f} & {r.accuracy_L1:.3f} & {r.ari_L1:+.3f} & "
            f"{r.accuracy_L2:.3f} & {r.ari_L2:+.3f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (args.out_dir / "table3.tex").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    # regime trace CSV for the best variant (highest train log-lik)
    best = max(results, key=lambda r: r.train_log_lik)
    trace_df = pd.DataFrame(
        {
            "date": test_std.index,
            "x_true": xs_te[:, 0],
            "y_obs": ys_te[:, 0],
            "r_L1": r_true_L1_test,
            "r_L2": r_true_L2_test,
            **{f"{vname}_pi1": pi_traces[vname][:, 1] for vname in pi_traces},
        }
    )
    trace_df.to_csv(args.out_dir / "regime_trace.csv", index=False)

    print(f"\n[e3] best variant by train-logL: {best.name} ({best.train_log_lik:+.1f})")
    print(f"[e3] wrote {args.out_dir / 'table3.json'}")
    print(f"[e3] wrote {args.out_dir / 'table3.tex'}")
    print(f"[e3] wrote {args.out_dir / 'em_history.json'}")
    print(f"[e3] wrote {args.out_dir / 'regime_trace.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
