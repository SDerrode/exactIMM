#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prg/experiments/run_real_data.py
=================================
Real-data experiments on the NOAA ENSO dataset (§7).

Inputs
------
data/real/enso_sst.csv  with columns:  date, nino34, nino12, oni, regime
    where regime ∈ {0=La Niña, 1=Neutral, 2=El Niño} from ONI thresholds.

Modeling choice
---------------
    X_n = standardized Niño 1+2  (state)
    Y_n = standardized Niño 3.4  (observation)
    R_n ∈ {0, 1, 2}              (regime, from ONI)
    train: 1950-01 .. 2010-12     (732 months)
    test : 2011-01 .. 2026-02     (182 months)

Experiments
-----------
    E1  Empirical (H5) test       (Fisher H0:B(k)=0, K=3 regimes)
    E2  Filter comparison         (h5_exact vs imm_general vs Kalman_K=1)
    E3  Semi-supervised EM        (V0 unconstrained, V1 post-hoc B,
                                   V2 post-hoc A, V3 GEM B)

Outputs
-------
    results/enso/
        e1_table.json / .tex     H5 empirical test
        e2_table.json / .tex     filter comparison
        e3_table.json / .tex     EM variant comparison
        regime_trace.csv          test-period regime probs and predictions

CLI
---
    python -m prg.experiments.run_real_data
    python -m prg.experiments.run_real_data --max-iter 100 --n-inits 10

Reproducibility
---------------
Default seed = 42, default n_inits = 5, default max_iter = 50.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from dataclasses import dataclass, field, asdict
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import f as f_dist
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from prg.filter.gss_filter import GSSFilter           # noqa: E402
from prg.learning.supervised import fit_supervised    # noqa: E402
from prg.learning.semi_supervised import fit_semi_supervised  # noqa: E402
from params_utils import params_from_dict             # noqa: E402
from baselines.kalman_single import SingleKalmanFilter  # noqa: E402

DEFAULT_CSV = ROOT / "data/real/enso_sst.csv"
DEFAULT_OUT = ROOT / "results/enso"
DEFAULT_FIG_DIR = ROOT / "paper" / "figures" / "generated"
TRAIN_END   = pd.Timestamp("2010-12-01")
REGIME_NAMES = ("La Niña", "Neutral", "El Niño")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_enso(csv_path: Path, train_end: pd.Timestamp = TRAIN_END):
    df = pd.read_csv(csv_path, parse_dates=["date"]).set_index("date").sort_index()
    train = df.loc[df.index <= train_end]
    mu_x, sd_x = train["nino12"].mean(), train["nino12"].std(ddof=0)
    mu_y, sd_y = train["nino34"].mean(), train["nino34"].std(ddof=0)
    xs = ((df["nino12"] - mu_x) / sd_x).to_numpy().reshape(-1, 1)
    ys = ((df["nino34"] - mu_y) / sd_y).to_numpy().reshape(-1, 1)
    rs = df["regime"].to_numpy().astype(int)
    n_tr = int((df.index <= train_end).sum())
    return {
        "df":      df,
        "xs":      xs, "ys": ys, "rs": rs,
        "n_train": n_tr,
        "stats":   {"x": (float(mu_x), float(sd_x)),
                    "y": (float(mu_y), float(sd_y))},
    }


# ---------------------------------------------------------------------------
# E1 — Fisher test H0: B(k) = 0
# ---------------------------------------------------------------------------
def fisher_B_zero(xs, ys, rs, k):
    mask = rs[1:] == k
    x_curr = xs[:-1][mask].ravel()
    y_curr = ys[:-1][mask].ravel()
    x_next = xs[1:][mask].ravel()
    n = x_curr.size
    if n < 5:
        return float("nan"), float("nan"), n
    X_full = np.column_stack([x_curr, y_curr, np.ones(n)])
    bf, *_ = np.linalg.lstsq(X_full, x_next, rcond=None)
    rss_full = float(np.sum((x_next - X_full @ bf) ** 2))
    X_null = np.column_stack([x_curr, np.ones(n)])
    bn, *_ = np.linalg.lstsq(X_null, x_next, rcond=None)
    rss_null = float(np.sum((x_next - X_null @ bn) ** 2))
    df_resid = n - 3
    if df_resid <= 0 or rss_full <= 0:
        return float("nan"), float("nan"), df_resid
    F = ((rss_null - rss_full) / 1) / (rss_full / df_resid)
    p = float(f_dist.sf(F, 1, df_resid))
    return float(F), p, df_resid


def run_e1(data, K=3):
    xs_tr = data["xs"][:data["n_train"]]
    ys_tr = data["ys"][:data["n_train"]]
    rs_tr = data["rs"][:data["n_train"]]

    fit_raw = fit_supervised(rs_tr, xs_tr, ys_tr, K=K, q=1, s=1, constraint=None)
    rows = []
    for k in range(K):
        B = fit_raw["B_list"][k]
        F, p, dfr = fisher_B_zero(xs_tr, ys_tr, rs_tr, k)
        n_k = int(np.sum(rs_tr[1:] == k))
        rows.append({
            "k":       k,
            "regime":  REGIME_NAMES[k],
            "n_k":     n_k,
            "B":       float(np.atleast_2d(B).ravel()[0]),
            "B_fro":   float(np.linalg.norm(B, ord="fro")),
            "F":       F,
            "p_value": p,
        })
    return {"K": K, "n_train": int(rs_tr.size), "rows": rows}


# ---------------------------------------------------------------------------
# E2 — filter comparison
# ---------------------------------------------------------------------------
@dataclass
class FilterScore:
    name:        str
    log_lik:     float
    nll_per_obs: float
    mse_x:       float
    time_s:      float


def run_filter(name, filt, xs_te, ys_te, store=None):
    t0 = time.perf_counter()
    N = len(ys_te); ll = 0.0; sse = 0.0
    x_hat = np.empty(N)
    pi_n  = []
    for i, y in enumerate(ys_te):
        r = filt.step(np.asarray(y, dtype=float).reshape(-1, 1))
        x_hat[i] = float(r.E_x.ravel()[0])
        sse += (xs_te[i, 0] - x_hat[i]) ** 2
        ll += r.log_lik
        pi_n.append(r.pi if hasattr(r, "pi") else None)
    dt = time.perf_counter() - t0
    if store is not None:
        store[f"{name}_x_hat"] = x_hat
        if pi_n[0] is not None:
            store[f"{name}_pi"] = np.asarray([np.asarray(p) for p in pi_n])
    return FilterScore(name, ll, -ll / N, sse / N, dt)


def run_e2(data, K=3):
    n_tr = data["n_train"]
    xs_tr, ys_tr, rs_tr = (data["xs"][:n_tr], data["ys"][:n_tr],
                            data["rs"][:n_tr])
    xs_te, ys_te = data["xs"][n_tr:], data["ys"][n_tr:]

    fit_raw = fit_supervised(rs_tr, xs_tr, ys_tr, K=K, q=1, s=1, constraint=None)
    fit_h5  = fit_supervised(rs_tr, xs_tr, ys_tr, K=K, q=1, s=1, constraint="b")
    p_raw = params_from_dict(fit_raw)
    p_h5  = params_from_dict(fit_h5)

    traces = {}
    scores = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scores.append(run_filter("h5_exact_h5fit", GSSFilter(p_h5,  mode="h5_exact"),
                                  xs_te, ys_te, traces))
    scores.append(run_filter("imm_general_olsfit",
                              GSSFilter(p_raw, mode="imm_general"),
                              xs_te, ys_te, traces))
    scores.append(run_filter("imm_general_h5fit",
                              GSSFilter(p_h5,  mode="imm_general"),
                              xs_te, ys_te, traces))
    kf = SingleKalmanFilter.from_regressed(xs_tr, ys_tr)
    scores.append(run_filter("kalman_k1", kf, xs_te, ys_te, traces))

    return {"K": K, "scores": [asdict(s) for s in scores], "traces": traces,
            "fit_raw": fit_raw, "fit_h5": fit_h5}


# ---------------------------------------------------------------------------
# E3 — semi-supervised EM variants
# ---------------------------------------------------------------------------
EM_VARIANTS = [
    ("V0_unconstrained", dict(constraint=None, constraint_each_iter=False)),
    ("V1_posthoc_B",     dict(constraint="b",  constraint_each_iter=False)),
    ("V2_posthoc_A",     dict(constraint="a",  constraint_each_iter=False)),
    ("V3_GEM_B",         dict(constraint="b",  constraint_each_iter=True)),
]


def best_perm_acc_ari(r_hat, r_true, K):
    best_acc = -1.0; best_perm_arr = None
    for p in permutations(range(K)):
        pa = np.array(p)
        a = float(np.mean(pa[r_hat] == r_true))
        if a > best_acc:
            best_acc, best_perm_arr = a, pa
    ari = float(adjusted_rand_score(r_true, r_hat))
    return float(best_acc), ari, best_perm_arr


def run_e3(data, K=3, n_inits=5, max_iter=50, seed=42):
    n_tr = data["n_train"]
    xs_tr, ys_tr = data["xs"][:n_tr], data["ys"][:n_tr]
    xs_te, ys_te, rs_te = (data["xs"][n_tr:], data["ys"][n_tr:],
                            data["rs"][n_tr:])

    out = []
    pi_test_per_variant = {}
    for name, cfg in EM_VARIANTS:
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            params, info = fit_semi_supervised(
                xs_tr, ys_tr, K=K,
                constraint=cfg["constraint"],
                constraint_each_iter=cfg["constraint_each_iter"],
                n_inits=n_inits, max_iter=max_iter, seed=seed, verbose=False,
            )
        dt = time.perf_counter() - t0

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f = GSSFilter(params_from_dict(params), mode="imm_general")
            ll = 0.0; sse = 0.0
            pi_n = np.empty((len(ys_te), K))
            for i, y in enumerate(ys_te):
                r = f.step(np.asarray(y, dtype=float).reshape(-1, 1))
                pi_n[i] = r.pi
                xi = float(r.E_x.ravel()[0])
                sse += (xs_te[i, 0] - xi) ** 2
                ll += r.log_lik

        r_hat = np.argmax(pi_n, axis=1)
        acc, ari, _ = best_perm_acc_ari(r_hat, rs_te, K)
        pi_test_per_variant[name] = pi_n

        out.append({
            "name":      name,
            "train_LL":  float(info["best_log_lik"]),
            "test_LL":   float(ll),
            "test_NLL_per_obs": float(-ll / len(ys_te)),
            "test_MSE":  float(sse / len(ys_te)),
            "acc":       acc,
            "ARI":       ari,
            "n_iter":    int(info["best_n_iter"]),
            "converged": bool(info["best_converged"]),
            "time_s":    dt,
            "all_train_LLs": [float(x) for x in info["all_log_liks"]],
        })
    return {"K": K, "n_inits": n_inits, "max_iter": max_iter, "seed": seed,
            "variants": out, "pi_test": pi_test_per_variant}


# ---------------------------------------------------------------------------
# LaTeX emitters
# ---------------------------------------------------------------------------
def _write_table(path, caption, label, col_spec, header, rows):
    """Write a self-contained \\begin{table}...\\end{table} block."""
    lines = [
        r"\begin{table}[ht]",
        r"  \caption{" + caption + "}",
        r"  \label{" + label + "}",
        r"  \centering\small",
        r"  \renewcommand{\arraystretch}{1.2}",
        r"  \begin{tabular}{@{}" + col_spec + r"@{}}",
        r"    \toprule",
        "    " + header + r" \\",
        r"    \midrule",
    ] + ["    " + r for r in rows] + [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def emit_e1_tex(res, path):
    def fp(p):
        if np.isnan(p):
            return "n/a"
        if p < 1e-4:
            return r"$<\!10^{-4}$"
        return f"{p:.3f}"
    rows = []
    for r in res["rows"]:
        rows.append(f"{r['regime']} & {r['n_k']} & {r['B']:+.4f} & "
                    f"{r['B_fro']:.4f} & {fp(r['p_value'])} \\\\")
    cap = (
        r"Empirical test of (H5) on the ENSO training period"
        r" (1950--2010, 732 months). $B(k)$ is the unconstrained OLS"
        r" estimate within regime~$k$, $\|B(k)\|_F$ its Frobenius norm,"
        r" and $p$-value is for $H_0:B(k)=0$ via a nested-model"
        r" Fisher $F$-test."
    )
    _write_table(path, cap, "tab:enso_h5_test", "lrrrr",
                 r"Regime & $n_k$ & $B(k)$ & $\|B(k)\|_F$ & $p$-value",
                 rows)


def emit_e2_tex(res, path):
    label = {
        "h5_exact_h5fit":      "H5-exact (H5 fit)",
        "imm_general_olsfit":  "IMM-general (OLS fit)",
        "imm_general_h5fit":   "IMM-general (H5 fit)",
        "kalman_k1":           "Kalman $K=1$",
    }
    rows = []
    for s in res["scores"]:
        rows.append(f"{label[s['name']]} & {s['log_lik']:+.2f} & "
                    f"{s['nll_per_obs']:+.4f} & {s['mse_x']:.4f} & "
                    f"{s['time_s']:.2f} \\\\")
    cap = (
        r"Filter comparison on the ENSO test period"
        r" (2011-01 to 2026-02, 182 months)."
        r" Parameters $\boldsymbol{\theta}$ are fit by supervised OLS"
        r" on the training period; the H5-fit projects on $\tau=B$, the"
        r" OLS-fit is unconstrained. $\log\hat L$ is the joint test"
        r" log-likelihood, NLL/obs $= -\log\hat L / N_{\rm test}$;"
        r" MSE is computed on $X$ (Niño~1+2)."
    )
    _write_table(path, cap, "tab:enso_filter", "lrrrr",
                 r"Filter & $\log\hat L$ & NLL/obs & MSE $X$ & time (s)",
                 rows)


def emit_e3_tex(res, path):
    label = {
        "V0_unconstrained": "V0 unconstrained",
        "V1_posthoc_B":     r"V1 post-hoc $\tau=B$",
        "V2_posthoc_A":     r"V2 post-hoc $\tau=A^\dagger$",
        "V3_GEM_B":         r"V3 GEM $\tau=B$",
    }
    rows = []
    for v in res["variants"]:
        nll = v["test_NLL_per_obs"]
        mse = v["test_MSE"]
        if not np.isfinite(nll) or abs(nll) > 100 or mse > 100:
            mse_str = r"\text{---}$^\dagger$"
            nll_str = r"\text{---}$^\dagger$"
        else:
            mse_str = f"{mse:.4f}"
            nll_str = f"{nll:+.4f}"
        rows.append(f"{label[v['name']]} & {v['train_LL']:+.1f} & "
                    f"{nll_str} & {mse_str} & "
                    f"{v['acc']:.3f} & {v['ARI']:+.3f} \\\\")
    cap = (
        r"Semi-supervised EM variants on ENSO"
        r" ($K=3$, $n_{\mathrm{init}}=5$, $I_{\max}=50$)."
        r" Train log-lik is on the joint $(X,Y)$ training period;"
        r" test metrics use the IMM-general filter on EM-fitted"
        r" parameters. Acc and ARI are computed on the ground-truth"
        r" ONI regime after optimal regime-permutation alignment."
        r" $^\dagger$:~$\tau=A^\dagger$ projection numerically unstable"
        r" ($G=PM^{-1}Q-\Delta^T$ ill-conditioned), as documented"
        r" already in the simulation study (Table~\ref{tab:supervised_M1})."
    )
    _write_table(path, cap, "tab:enso_em", "lrrrrr",
                 r"Variant & train $\log\hat L$ & test NLL/obs & "
                 r"test MSE $X$ & acc($R$) & ARI($R$)",
                 rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description="Run §7 real-data experiments (ENSO).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR,
                   help="Where to write LaTeX tables for paper inclusion.")
    p.add_argument("--K", type=int, default=3)
    p.add_argument("--n-inits", type=int, default=5)
    p.add_argument("--max-iter", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip", choices=["e1", "e2", "e3"], action="append",
                   default=[])
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading: {args.csv}")
    data = load_enso(args.csv)
    print(f"  n_train = {data['n_train']}, n_test = {len(data['xs']) - data['n_train']}")
    print(f"  X = std Niño 1+2  Y = std Niño 3.4")

    # ------------- E1 -------------
    if "e1" not in args.skip:
        print("\n=== E1 — Empirical (H5) test ===")
        e1 = run_e1(data, K=args.K)
        for r in e1["rows"]:
            p_str = f"{r['p_value']:.3g}" if not np.isnan(r['p_value']) else "n/a"
            print(f"  {r['regime']:<8s} n={r['n_k']:>4d}  "
                  f"B={r['B']:+.4f}  ||B||={r['B_fro']:.4f}  "
                  f"F={r['F']:>6.2f}  p={p_str}")
        (args.out_dir / "e1_table.json").write_text(
            json.dumps(e1, indent=2), encoding="utf-8")
        emit_e1_tex(e1, args.fig_dir / "tab_enso_h5_test.tex")
        print(f"  saved {args.fig_dir/'tab_enso_h5_test.tex'}")

    # ------------- E2 -------------
    if "e2" not in args.skip:
        print("\n=== E2 — Filter comparison ===")
        e2 = run_e2(data, K=args.K)
        for s in e2["scores"]:
            print(f"  {s['name']:25s}  logL={s['log_lik']:>+9.2f}  "
                  f"NLL/obs={s['nll_per_obs']:+.4f}  MSE={s['mse_x']:.4f}  "
                  f"t={s['time_s']:.2f}s")
        # Persist
        e2_serial = {"K": e2["K"], "scores": e2["scores"]}
        (args.out_dir / "e2_table.json").write_text(
            json.dumps(e2_serial, indent=2), encoding="utf-8")
        emit_e2_tex(e2, args.fig_dir / "tab_enso_filter.tex")
        print(f"  saved {args.fig_dir/'tab_enso_filter.tex'}")

    # ------------- E3 -------------
    if "e3" not in args.skip:
        print(f"\n=== E3 — Semi-supervised EM (n_inits={args.n_inits}, "
              f"max_iter={args.max_iter}) ===")
        e3 = run_e3(data, K=args.K, n_inits=args.n_inits,
                    max_iter=args.max_iter, seed=args.seed)
        for v in e3["variants"]:
            mse_str = (f"{v['test_MSE']:.4f}"
                       if abs(v['test_MSE']) < 1e6 else f"{v['test_MSE']:.2e}")
            print(f"  {v['name']:18s}  train_LL={v['train_LL']:>+9.1f}  "
                  f"test_LL={v['test_LL']:>+9.2f}  "
                  f"MSE={mse_str:>12s}  "
                  f"acc={v['acc']:.3f}  ARI={v['ARI']:+.3f}  "
                  f"t={v['time_s']:.1f}s")

        e3_serial = {k: v for k, v in e3.items() if k != "pi_test"}
        (args.out_dir / "e3_table.json").write_text(
            json.dumps(e3_serial, indent=2), encoding="utf-8")
        emit_e3_tex(e3, args.fig_dir / "tab_enso_em.tex")
        print(f"  saved {args.fig_dir/'tab_enso_em.tex'}")

        # Regime trace CSV (only for finite-MSE variants)
        n_tr = data["n_train"]
        test_dates = data["df"].index[n_tr:]
        rows = {"date": test_dates,
                "x_true": data["xs"][n_tr:, 0],
                "y_obs":  data["ys"][n_tr:, 0],
                "r_true": data["rs"][n_tr:]}
        for name, pi in e3["pi_test"].items():
            for k in range(args.K):
                rows[f"{name}_pi{k}"] = pi[:, k]
        pd.DataFrame(rows).to_csv(args.out_dir / "regime_trace.csv", index=False)
        print(f"  saved {args.out_dir/'regime_trace.csv'}")

    print("\nAll done.")


if __name__ == "__main__":
    sys.exit(main() or 0)
