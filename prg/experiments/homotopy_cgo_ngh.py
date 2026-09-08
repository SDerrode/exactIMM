#!/usr/bin/env python3
"""CGO -> NGH homotopy: which exact filter degrades less off its family?

A one-parameter path of Gaussian pairwise switching models joins a genuine
CGO-MSM (epsilon = 0: C = 0, state memory A_0) to a genuine NGH-MSM
(epsilon = 1: C = C_1, A = M C_1, B = M D), all other blocks (D, Sigma_W,
hence M and Gamma) being shared:

    C(eps) = eps C_1,   A(eps) = (1-eps) A_0 + eps M C_1,   B(eps) = (1-eps) B_0 + eps M D.

Data are simulated with the model at epsilon.  Two exact filters are run with
FROZEN parameters, each on its own family's model:

* the CGO exact filter -- the pairwise IMM fed the epsilon = 0 model (exact on
  that family by Theorem 1 of the exactness paper);
* the NGH constant-gain filter -- fed the epsilon = 1 model.

Both are therefore misspecified in A, B, C by an amount linear in their
distance to their endpoint.  The Bayes-optimal filter at epsilon (exact K^N
mixture, short horizon) gives the irreducible error; GPB2 with the TRUE
parameters is the collapsed reference, exact at both ends.

Metric.  By the orthogonality principle, for any estimator f(y) of X and the
conditional mean e(y) = E[X | y],   E(f - X)^2 = E(e - X)^2 + E(f - e)^2:
the excess mean-square error of f over the Bayes filter is exactly the
mean-square distance between f and e, and needs no simulated state to be
estimated (no Monte-Carlo noise from the state noise).  We report
    excess(eps) = mean_n E(f_n - e_n)^2 / mean_n Var[X_n | y_{1:n}]
(relative excess MSE, the denominator being the exact posterior variance),
zero for each frozen filter at its own endpoint and comparable across
filters.  Raw RMSEs against the simulated state and the normalized sup-norm
gap to the exact filter are reported as well.

Run:  python -m prg.experiments.homotopy_cgo_ngh [--seeds 200] [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

from prg.classes.FMatrix import FMatrix
from prg.classes.GSSParams import GSSParams
from prg.classes.NoiseCovariance import GSSNoiseCovariance
from prg.experiments.reference_filters import (
    exact_mixture_filter,
    gpb2_filter,
    imm_filter,
    with_stationary_init,
)
from prg.experiments.study import _run, _simulate

N_STEPS = 10
SEED0 = 2000

# shared blocks ------------------------------------------------------------
SV = [0.20, 0.60]            # observation noise (regime identifiable)
M = [0.6, -0.5]              # slaved gain Delta / Sigma_V
D = [0.50, 0.50]
GAM = [0.25, 0.30]           # residual state noise
A0 = [0.7, 0.4]              # CGO endpoint: state memory
B0 = [0.10, 0.10]
C1 = [0.5, 0.5]              # NGH endpoint: active channel
P_SWITCH = 0.10


def path_blocks(eps: float) -> dict:
    """Blocks of the model at epsilon (K=2, q=s=1)."""
    A = [(1 - eps) * A0[k] + eps * M[k] * C1[k] for k in range(2)]
    B = [(1 - eps) * B0[k] + eps * M[k] * D[k] for k in range(2)]
    C = [eps * C1[k] for k in range(2)]
    Dt = [M[k] * SV[k] for k in range(2)]
    SU = [GAM[k] + M[k] * SV[k] * M[k] for k in range(2)]
    return dict(A=A, B=B, C=C, D=list(D), SU=SU, Dt=Dt, SV=list(SV))


def model(eps: float) -> GSSParams:
    b = path_blocks(eps)
    K, q, s = 2, 1, 1
    P = np.array([[1 - P_SWITCH, P_SWITCH], [P_SWITCH, 1 - P_SWITCH]])
    as_mat = lambda v: [np.array([[x]], dtype=float) for x in v]
    fm = FMatrix(K, q, s, as_mat(b["A"]), as_mat(b["B"]), as_mat(b["C"]), as_mat(b["D"]))
    nc = GSSNoiseCovariance(K, q, s, as_mat(b["SU"]), as_mat(b["Dt"]), as_mat(b["SV"]))
    p = GSSParams(
        K=K, q=q, s=s, P=P, f_matrix=fm, noise_cov=nc, pi0=None,
        mu_z0_list=[np.zeros((q + s, 1)) for _ in range(K)],
        Sigma_z0_list=[np.eye(q + s) for _ in range(K)],
    )
    rho = max(max(abs(np.linalg.eigvals(fm.F(k)))) for k in range(K))
    assert rho < 1.0, f"unstable model at eps={eps} (rho={rho:.3f})"
    return with_stationary_init(p)


def spectral_radius(eps: float) -> float:
    b = path_blocks(eps)
    return max(
        max(abs(np.linalg.eigvals(np.array([[b["A"][k], b["B"][k]], [b["C"][k], b["D"][k]]]))))
        for k in range(2)
    )


def _rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a - b)) / (np.max(np.abs(b)) + 1e-12))


def run_eps(eps: float, p0: GSSParams, p1: GSSParams, n_seeds: int) -> dict:
    """Pooled RMSE (vs simulated state) and median gap-to-exact for each filter."""
    p = model(eps)
    se = {k: 0.0 for k in ("exact", "cgo", "ngh", "gpb2")}
    d2 = {k: 0.0 for k in ("cgo", "ngh", "gpb2")}     # sum (f - e)^2
    cross = {k: 0.0 for k in ("cgo", "ngh", "gpb2")}  # sum (f - e)(e - x): orthogonality check
    var_e = 0.0                                       # sum of exact posterior variances
    gap = {k: [] for k in ("cgo", "ngh", "gpb2")}
    n_tot = 0
    for sd in range(n_seeds):
        _, xs, ys = _simulate(p, N_STEPS, seed=SEED0 + sd)
        x = xs[:, 0]
        ex_e, var_x, _ = exact_mixture_filter(p, ys)
        var_e += float(np.sum(np.asarray(var_x).reshape(len(x), -1)[:, 0]))
        ex_c, _, _ = imm_filter(p0, ys)[:3]          # CGO exact filter, frozen at eps=0
        ex_n, _, _ = _run(p1, ys, "ngh_kf")           # NGH constant gain, frozen at eps=1
        ex_g, _, _ = gpb2_filter(p, ys)[:3]           # GPB2, true parameters
        est = {"exact": ex_e[:, 0], "cgo": ex_c[:, 0], "ngh": ex_n[:, 0], "gpb2": ex_g[:, 0]}
        for k, v in est.items():
            se[k] += float(np.sum((v - x) ** 2))
        for k in ("cgo", "ngh", "gpb2"):
            gap[k].append(_rel(est[k], est["exact"]))
            d2[k] += float(np.sum((est[k] - est["exact"]) ** 2))
            cross[k] += float(np.sum((est[k] - est["exact"]) * (est["exact"] - x)))
        n_tot += len(x)
    rmse = {k: float(np.sqrt(v / n_tot)) for k, v in se.items()}
    post_var = var_e / n_tot
    out = {"eps": eps, "rho": spectral_radius(eps), "rmse": rmse, "posterior_var": post_var,
           "excess_mse": {k: d2[k] / n_tot for k in d2},
           "excess": {k: (d2[k] / n_tot) / post_var for k in d2},
           "cross_term": {k: cross[k] / n_tot for k in cross},
           "excess_rmse_raw": {k: rmse[k] / rmse["exact"] - 1.0 for k in ("cgo", "ngh", "gpb2")},
           "gap_median": {k: float(np.median(v)) for k, v in gap.items()}}
    return out


def crossing(rows, key_a="cgo", key_b="ngh", field="excess"):
    """Linear-interpolated epsilon where excess[key_a] == excess[key_b]."""
    e = np.array([r["eps"] for r in rows])
    d = np.array([r[field][key_a] - r[field][key_b] for r in rows])
    for i in range(len(e) - 1):
        if d[i] == 0:
            return float(e[i])
        if d[i] * d[i + 1] < 0:
            return float(e[i] - d[i] * (e[i + 1] - e[i]) / (d[i + 1] - d[i]))
    return float("nan")


def plot(results: Path, figdir: Path) -> None:
    """Two panels: relative excess MSE (linear) and gap-to-exact (log)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = json.loads(Path(results).read_text())
    rows = res["rows"]
    eps = np.array([r["eps"] for r in rows])
    figdir.mkdir(parents=True, exist_ok=True)
    col = {"cgo": "#1f5f8b", "ngh": "#b8451f", "gpb2": "#5f7d3a"}
    lab = {"cgo": r"CGO exact filter, frozen at $\varepsilon=0$",
           "ngh": r"NGH constant-gain filter, frozen at $\varepsilon=1$",
           "gpb2": "GPB2, true parameters (reference)"}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 3.9))
    for k in ("cgo", "ngh", "gpb2"):
        a1.plot(eps, [r["excess"][k] for r in rows], "o-", ms=4, color=col[k], label=lab[k])
        a2.semilogy(eps, [max(r["gap_median"][k], 1e-16) for r in rows], "o-", ms=4, color=col[k])
    xc = crossing(rows)
    if np.isfinite(xc):
        a1.axvline(xc, color="k", ls=":", lw=1)
        a1.text(xc + 0.02, 0.5 * a1.get_ylim()[1], rf"crossing $\varepsilon^\ast\approx{xc:.2f}$", ha="left", va="center", fontsize=8)
    a1.set_xlabel(r"$\varepsilon$  (0 = CGO model, 1 = NGH model)")
    a1.set_ylabel(r"relative excess MSE  $\mathbb{E}(f-e)^2\,/\,\mathrm{Var}[X\mid y]$")
    a1.set_title("(a) excess over the Bayes filter", fontsize=10)
    a1.grid(alpha=.3); a1.legend(fontsize=7.5, loc="upper left")
    a2.set_xlabel(r"$\varepsilon$"); a2.set_ylabel("median normalized sup-norm gap to exact")
    a2.set_title("(b) gap to the exact filter (log)", fontsize=10); a2.grid(alpha=.3, which="both")
    fig.tight_layout()
    out = figdir / "homotopy_cgo_ngh.pdf"
    fig.savefig(out); fig.savefig(out.with_suffix(".png"), dpi=160)
    print(f"[homotopy] figure -> {out}   crossing eps* = {xc:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=200)
    ap.add_argument("--out", type=Path, default=Path("data/experiments/homotopy_cgo_ngh"))
    ap.add_argument("--grid", type=str, default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    ap.add_argument("--plot-only", type=Path, default=None,
                    help="skip the sweep; plot this results.json into --figdir")
    ap.add_argument("--figdir", type=Path, default=None)
    args = ap.parse_args()
    if args.plot_only is not None:
        plot(args.plot_only, args.figdir or args.plot_only.parent)
        return
    args.out.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings("ignore")
    p0, p1 = model(0.0), model(1.0)
    print(f"[homotopy] N={N_STEPS}, {args.seeds} seeds; rho(F) along the path: "
          + ", ".join(f"{spectral_radius(e):.3f}" for e in (0, 0.25, 0.5, 0.75, 1)))
    print(f"{'eps':>5} {'rmse_ex':>8} {'postvar':>8} | {'xs_cgo':>8} {'xs_ngh':>8} {'xs_gpb2':>8} | "
          f"{'gap_cgo':>8} {'gap_ngh':>8} {'gap_gpb2':>8} | {'cross_c':>8} {'cross_n':>8}")
    rows = []
    for eps in [float(e) for e in args.grid.split(",")]:
        r = run_eps(eps, p0, p1, args.seeds)
        rows.append(r)
        rm, xs, gp, cr = r["rmse"], r["excess"], r["gap_median"], r["cross_term"]
        print(f"{eps:5.2f} {rm['exact']:8.4f} {r['posterior_var']:8.4f} | "
              f"{xs['cgo']:8.4f} {xs['ngh']:8.4f} {xs['gpb2']:8.1e} | "
              f"{gp['cgo']:8.1e} {gp['ngh']:8.1e} {gp['gpb2']:8.1e} | {cr['cgo']:8.1e} {cr['ngh']:8.1e}", flush=True)
    (args.out / "results.json").write_text(json.dumps(
        {"N": N_STEPS, "seeds": args.seeds, "blocks": {"A0": A0, "B0": B0, "C1": C1, "D": D,
         "SV": SV, "M": M, "Gamma": GAM, "p_switch": P_SWITCH}, "rows": rows}, indent=1))
    print(f"[homotopy] results -> {args.out/'results.json'}")
    plot(args.out / "results.json", args.figdir or args.out)


if __name__ == "__main__":
    main()
