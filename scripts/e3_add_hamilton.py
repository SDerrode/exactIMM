#!/usr/bin/env python3
"""
scripts/e3_add_hamilton.py
==========================
Append the Hamilton MS-AR baseline (K regimes, AR(1), univariate on X
only) to the E3 results and regenerate ``table3.tex``.

Rationale
---------
Hamilton is a *univariate* regime-switching baseline on X — it does not
observe Y (log-VIX). Its log-likelihood is therefore on a different
random variable than our (X, Y)-joint filter, so we only report the
comparable metrics: MSE on X, accuracy / ARI vs proxy regime labels.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from baselines.hamilton_msar import HamiltonMSAR  # noqa: E402
from labels import (  # noqa: E402
    TRAIN_END,
    get_label,
    load_dataset,
    standardize_with_train_stats,
    train_test_split,
)


def _best_perm_accuracy(r_hat: np.ndarray, r_true: np.ndarray, K: int) -> tuple[float, np.ndarray]:
    from itertools import permutations
    best_acc, best_perm = -1.0, None
    for perm in permutations(range(K)):
        perm_arr = np.array(perm, dtype=int)
        acc = float(np.mean(perm_arr[r_hat] == r_true))
        if acc > best_acc:
            best_acc = acc
            best_perm = perm_arr
    return best_acc, best_perm


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=ROOT / "data/real/sp500_vix.csv")
    ap.add_argument("--table", type=Path, default=ROOT / "results/e3/table3.json")
    ap.add_argument("--K", type=int, default=2)
    args = ap.parse_args()

    # ---- data ----
    df = load_dataset(args.csv)
    train, _ = train_test_split(df, TRAIN_END)
    df_std, _ = standardize_with_train_stats(df, train, cols=("log_return", "log_vix"))
    train_std = df_std.loc[df_std.index <= TRAIN_END]
    test_std  = df_std.loc[df_std.index >  TRAIN_END]

    x_tr = train_std["log_return"].to_numpy()
    x_te = test_std["log_return"].to_numpy()

    L1_all = get_label(df, train, "L1")
    L2_all = get_label(df, train, "L2")
    r_L1_te = L1_all.loc[L1_all.index > TRAIN_END].to_numpy()
    r_L2_te = L2_all.loc[L2_all.index > TRAIN_END].to_numpy()

    # ---- Hamilton MS-AR fit + predict ----
    print(f"[hamilton] fitting MS-AR(K={args.K}, AR=1) on train...", flush=True)
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        h = HamiltonMSAR(K=args.K, ar_order=1).fit(x_tr)
    scores = h.predict_test(x_tr, x_te)
    dt = time.perf_counter() - t0
    r_hat = scores["r_hat_test"]
    acc_L1, _ = _best_perm_accuracy(r_hat, r_L1_te, args.K)
    acc_L2, _ = _best_perm_accuracy(r_hat, r_L2_te, args.K)
    ari_L1 = float(adjusted_rand_score(r_L1_te, r_hat))
    ari_L2 = float(adjusted_rand_score(r_L2_te, r_hat))

    ham_entry = {
        "name":              "Hamilton_MSAR",
        "train_log_lik":     float("nan"),   # NB: on X only, not comparable
        "test_log_lik":      float(scores["log_lik_test"]),
        "test_nll_per_obs":  float(scores["nll_per_obs"]),
        "test_mse_x":        float(scores["mse_x"]),
        "accuracy_L1":       acc_L1,
        "accuracy_L2":       acc_L2,
        "ari_L1":            ari_L1,
        "ari_L2":            ari_L2,
        "all_train_log_liks": [],
        "n_iter":            -1,
        "converged":         True,
        "time_s":            dt,
    }
    print(f"[hamilton] MSE={scores['mse_x']:.4f}  acc(L1)={acc_L1:.3f}  "
          f"acc(L2)={acc_L2:.3f}  ARI(L1)={ari_L1:+.3f}  ARI(L2)={ari_L2:+.3f}  "
          f"time {dt:.1f}s")

    # ---- update table3.json ----
    data = json.loads(args.table.read_text(encoding="utf-8"))
    # Avoid duplicating if rerun
    data["variants"] = [v for v in data["variants"] if v["name"] != "Hamilton_MSAR"]
    data["variants"].append(ham_entry)
    args.table.write_text(json.dumps(data, indent=2), encoding="utf-8")

    # ---- regenerate table3.tex ----
    lines = [
        r"% Table 3 — BW-EM variants + Hamilton baseline on S&P500/VIX",
        r"% NB: Hamilton is univariate on X; its test log-lik is on X only",
        r"%     and not directly comparable with the joint-(X,Y) log-lik of V0-V3.",
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"Variant & train $\log\hat L$ & test NLL/obs & test MSE $X$ & "
        r"acc(L1) & ARI(L1) & acc(L2) & ARI(L2) \\",
        r"\midrule",
    ]
    for r in data["variants"]:
        if r["name"] == "Hamilton_MSAR":
            lines.append(r"\midrule")
            train_ll_str = "n/a"
        else:
            train_ll_str = f"{r['train_log_lik']:+.1f}"
        lines.append(
            f"{r['name']:14s} & {train_ll_str} & {r['test_nll_per_obs']:+.4f} & "
            f"{r['test_mse_x']:.4f} & {r['accuracy_L1']:.3f} & {r['ari_L1']:+.3f} & "
            f"{r['accuracy_L2']:.3f} & {r['ari_L2']:+.3f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (args.table.parent / "table3.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8",
    )
    print(f"[hamilton] updated {args.table}")
    print(f"[hamilton] rewrote {args.table.parent / 'table3.tex'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
