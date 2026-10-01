#!/usr/bin/env python3
"""CGO -> NGH homotopy: which exact filter degrades less off its family?

A one-parameter path of Gaussian pairwise switching models joins a genuine
CGO-MSM (epsilon = 0: C = 0, state memory A_0) to a genuine NGH-MSM
(epsilon = 1: C = C_1, A = M C_1, B = M D), all other blocks (D, Sigma_W,
hence M and Gamma) being shared:

    C(eps) = eps C_1,   A(eps) = (1-eps) A_0 + eps M C_1,   B(eps) = (1-eps) B_0 + eps M D.

Two further paths share the endpoints and separate the two effects at play:
"c_first"  -- the channel opens on [0, 1/2] (C: 0 -> C_1, A = A_0, B = B_0 kept),
              then the state row slaves itself on [1/2, 1] (A -> M C_1, B -> M D);
"ab_first" -- the state row slaves itself on [0, 1/2] with the channel still
              closed (the model stays a CGO model with C = 0), then the channel
              opens on [1/2, 1].

Data are simulated with the model at epsilon.  Two exact filters are run with
FROZEN parameters, each on its own family's model:

* the CGO exact filter -- the pairwise IMM fed the epsilon = 0 model (exact on
  that family by Theorem 1 of the exactness paper);
* the NGH constant-gain filter -- fed the epsilon = 1 model.

Both are therefore misspecified in A, B, C by an amount linear in their
distance to their endpoint.  Two other ways of "filtering with a family" off
the family are run alongside:

* block projection ("blk") -- each filter is fed the TRUE blocks of the model
  at epsilon wherever its family allows, and only the forbidden blocks are
  replaced: the CGO filter gets A(eps), B(eps), D, Sigma_W with C := 0; the NGH
  filter gets C(eps), D, Sigma_W with A := M C(eps), B := M D.  Each filter is
  then wrong only in what its family cannot represent.
* KL projection ("kl") -- the member of each family closest to the true model
  in the Kullback-Leibler sense: the transition kernel N(F z + b, Sigma_W) of
  arrival regime k is approximated by N(F' z + b, Sigma_W) with F' in the
  family's linear subspace (C' = 0 for CGO; A' = M C', B' = M D' for NGH),
  minimizing E[(F - F') z]^T Sigma_W^-1 [(F - F') z] under the stationary law
  of the true model (departure regimes weighted by pi_j p_jk).  This is the
  weighted least-squares regression that Petetin & Desbouvries' criterion
  reduces to for Gaussian kernels with a common covariance.  The Bayes-optimal filter at epsilon (exact K^N
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
SV = [0.20, 0.60]  # observation noise (regime identifiable)
M = [0.6, -0.5]  # slaved gain Delta / Sigma_V
D = [0.50, 0.50]
GAM = [0.25, 0.30]  # residual state noise
A0 = [0.7, 0.4]  # CGO endpoint: state memory
B0 = [0.10, 0.10]
C1 = [0.5, 0.5]  # NGH endpoint: active channel
P_SWITCH = 0.10


PATHS = ("straight", "c_first", "ab_first")


def _stage(eps: float, kind: str) -> tuple[float, float]:
    """(fraction of the channel opened, fraction of the state row slaved)."""
    if kind == "straight":
        return eps, eps
    if kind == "c_first":
        return min(1.0, 2 * eps), max(0.0, 2 * eps - 1)
    if kind == "ab_first":
        return max(0.0, 2 * eps - 1), min(1.0, 2 * eps)
    raise ValueError(kind)


def path_blocks(eps: float, kind: str = "straight") -> dict:
    """Blocks of the model at epsilon on the given path (K=2, q=s=1)."""
    c, a = _stage(eps, kind)
    A = [(1 - a) * A0[k] + a * M[k] * C1[k] for k in range(2)]
    B = [(1 - a) * B0[k] + a * M[k] * D[k] for k in range(2)]
    C = [c * C1[k] for k in range(2)]
    Dt = [M[k] * SV[k] for k in range(2)]
    SU = [GAM[k] + M[k] * SV[k] * M[k] for k in range(2)]
    return dict(A=A, B=B, C=C, D=list(D), SU=SU, Dt=Dt, SV=list(SV))


def _build(b: dict) -> GSSParams:
    """GSSParams (stationary init) from a block dict of K=2, q=s=1 scalars."""
    K, q, s = 2, 1, 1
    P = np.array([[1 - P_SWITCH, P_SWITCH], [P_SWITCH, 1 - P_SWITCH]])
    as_mat = lambda v: [np.array([[x]], dtype=float) for x in v]
    fm = FMatrix(K, q, s, as_mat(b["A"]), as_mat(b["B"]), as_mat(b["C"]), as_mat(b["D"]))
    nc = GSSNoiseCovariance(K, q, s, as_mat(b["SU"]), as_mat(b["Dt"]), as_mat(b["SV"]))
    p = GSSParams(
        K=K,
        q=q,
        s=s,
        P=P,
        f_matrix=fm,
        noise_cov=nc,
        pi0=None,
        mu_z0_list=[np.zeros((q + s, 1)) for _ in range(K)],
        Sigma_z0_list=[np.eye(q + s) for _ in range(K)],
    )
    rho = max(max(abs(np.linalg.eigvals(fm.F(k)))) for k in range(K))
    assert rho < 1.0, f"unstable model (rho={rho:.3f}): {b}"
    return with_stationary_init(p)


def model(eps: float, kind: str = "straight") -> GSSParams:
    return _build(path_blocks(eps, kind))


def project_blocks(eps: float, kind: str, family: str) -> GSSParams:
    """Block projection of the model at epsilon onto a family."""
    b = path_blocks(eps, kind)
    if family == "cgo":
        b["C"] = [0.0, 0.0]
    elif family == "ngh":
        b["A"] = [M[k] * b["C"][k] for k in range(2)]
        b["B"] = [M[k] * b["D"][k] for k in range(2)]
    else:
        raise ValueError(family)
    return _build(b)


def project_kl(eps: float, kind: str, family: str) -> GSSParams:
    """KL projection of the model at epsilon onto a family (see module doc)."""
    from prg.experiments.reference_filters import stationary_moments

    p = model(eps, kind)
    b = path_blocks(eps, kind)
    mu, Sig = stationary_moments(p)
    S_j = [Sig[j] + mu[j] @ mu[j].T for j in range(2)]  # E[z z^T | r=j]
    pi = np.asarray(p.pi0).ravel()
    P = np.asarray(p.P)
    new = {k: list(v) for k, v in b.items()}
    for k in range(2):
        w = pi * P[:, k]
        w = w / w.sum()
        S = sum(w[j] * S_j[j] for j in range(2))  # departure law reaching k
        F = np.array([[b["A"][k], b["B"][k]], [b["C"][k], b["D"][k]]])
        W = np.linalg.inv(np.array([[b["SU"][k], b["Dt"][k]], [b["Dt"][k], b["SV"][k]]]))
        if family == "cgo":  # F' = a E1 + bb E2 + d E3, C' = 0
            basis = [
                np.array([[1, 0], [0, 0]]),
                np.array([[0, 1], [0, 0]]),
                np.array([[0, 0], [0, 1]]),
            ]
        elif family == "ngh":  # F' = c [[M,0],[1,0]] + d [[0,M],[0,1]]
            basis = [np.array([[M[k], 0], [1, 0]]), np.array([[0, M[k]], [0, 1]])]
        else:
            raise ValueError(family)
        G = np.array([[np.trace(W @ Ei @ S @ El.T) for El in basis] for Ei in basis])
        r = np.array([np.trace(W @ F @ S @ Ei.T) for Ei in basis])
        theta = np.linalg.solve(G, r)
        Fp = sum(th * Ei for th, Ei in zip(theta, basis))
        new["A"][k], new["B"][k], new["C"][k], new["D"][k] = Fp[0, 0], Fp[0, 1], Fp[1, 0], Fp[1, 1]
    return _build(new)


def spectral_radius(eps: float, kind: str = "straight") -> float:
    b = path_blocks(eps, kind)
    return max(
        max(abs(np.linalg.eigvals(np.array([[b["A"][k], b["B"][k]], [b["C"][k], b["D"][k]]]))))
        for k in range(2)
    )


def _rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a - b)) / (np.max(np.abs(b)) + 1e-12))


def run_eps(eps: float, p0: GSSParams, p1: GSSParams, n_seeds: int, kind: str = "straight") -> dict:
    """Pooled RMSE (vs simulated state) and median gap-to-exact for each filter."""
    p = model(eps, kind)
    pc_blk, pn_blk = project_blocks(eps, kind, "cgo"), project_blocks(eps, kind, "ngh")
    pc_kl, pn_kl = project_kl(eps, kind, "cgo"), project_kl(eps, kind, "ngh")
    keys = ("cgo", "ngh", "cgo_blk", "ngh_blk", "cgo_kl", "ngh_kl", "gpb2", "imm")
    se = {k: 0.0 for k in ("exact",) + keys}
    d2 = {k: 0.0 for k in keys}  # sum (f - e)^2
    cross = {k: 0.0 for k in keys}  # sum (f - e)(e - x): orthogonality check
    var_e = 0.0  # sum of exact posterior variances
    gap = {k: [] for k in keys}
    n_tot = 0
    for sd in range(n_seeds):
        _, xs, ys = _simulate(p, N_STEPS, seed=SEED0 + sd)
        x = xs[:, 0]
        ex_e, var_x, _ = exact_mixture_filter(p, ys)
        var_e += float(np.sum(np.asarray(var_x).reshape(len(x), -1)[:, 0]))
        ex_c, _, _ = imm_filter(p0, ys)[:3]  # CGO exact filter, frozen at eps=0
        ex_n, _, _ = _run(p1, ys, "ngh_kf")  # NGH constant gain, frozen at eps=1
        ex_g, _, _ = gpb2_filter(p, ys)[:3]  # GPB2, true parameters
        ex_i, _, _ = imm_filter(p, ys)[:3]  # pairwise IMM, true parameters (approximate off C=0)
        est = {
            "exact": ex_e[:, 0],
            "cgo": ex_c[:, 0],
            "ngh": ex_n[:, 0],
            "gpb2": ex_g[:, 0],
            "imm": ex_i[:, 0],
            "cgo_blk": imm_filter(pc_blk, ys)[0][:, 0],
            "ngh_blk": _run(pn_blk, ys, "ngh_kf")[0][:, 0],
            "cgo_kl": imm_filter(pc_kl, ys)[0][:, 0],
            "ngh_kl": _run(pn_kl, ys, "ngh_kf")[0][:, 0],
        }
        for k, v in est.items():
            se[k] += float(np.sum((v - x) ** 2))
        for k in keys:
            gap[k].append(_rel(est[k], est["exact"]))
            d2[k] += float(np.sum((est[k] - est["exact"]) ** 2))
            cross[k] += float(np.sum((est[k] - est["exact"]) * (est["exact"] - x)))
        n_tot += len(x)
    rmse = {k: float(np.sqrt(v / n_tot)) for k, v in se.items()}
    post_var = var_e / n_tot
    out = {
        "eps": eps,
        "path": kind,
        "rho": spectral_radius(eps, kind),
        "rmse": rmse,
        "posterior_var": post_var,
        "excess_mse": {k: d2[k] / n_tot for k in d2},
        "excess": {k: (d2[k] / n_tot) / post_var for k in d2},
        "cross_term": {k: cross[k] / n_tot for k in cross},
        "excess_rmse_raw": {k: rmse[k] / rmse["exact"] - 1.0 for k in keys},
        "gap_median": {k: float(np.median(v)) for k, v in gap.items()},
    }
    out["kl_blocks"] = {"cgo": _blocks_of(pc_kl), "ngh": _blocks_of(pn_kl)}
    return out


def _blocks_of(p: GSSParams) -> dict:
    fm = p.f_matrix
    return {
        name: [float(getattr(fm, name)(k)[0, 0]) for k in range(p.K)]
        for name in ("A", "B", "C", "D")
    }


def endpoint_sensitivity(
    eps0: float, kind: str = "straight", delta: float = 0.02, n_seeds: int = 200
) -> float:
    """Relative mean-square sensitivity s^2 of the Bayes filter to the path at eps0.

    A filter frozen at endpoint eps0 and run under P_eps satisfies
    f - e_eps = -(eps - eps0) d_eps e + O((eps-eps0)^2), so its relative excess is
    (eps-eps0)^2 s^2 to leading order, with s^2 = E[(d_eps e)^2] / Var[X|y] under
    P_eps0.  Hence the crossing of the two frozen curves is, to this order,
    eps* = s1 / (s0 + s1).  The Bayes filter is tangent to GPB2 at the endpoints
    (GPB2 - e = O(eps^3) by the cubic law), so d_eps e is taken as the central
    finite difference of the GPB2 recursion along the path.
    """
    p = model(eps0, kind)
    lo, hi = max(0.0, eps0 - delta), min(1.0, eps0 + delta)
    plo, phi = model(lo, kind), model(hi, kind)
    num = den = 0.0
    for sd in range(n_seeds):
        _, _, ys = _simulate(p, N_STEPS, seed=SEED0 + sd)
        d = (gpb2_filter(phi, ys)[0][:, 0] - gpb2_filter(plo, ys)[0][:, 0]) / (hi - lo)
        _, var, _ = exact_mixture_filter(p, ys)
        num += float(np.sum(d**2))
        den += float(np.sum(np.asarray(var).reshape(len(d), -1)[:, 0]))
    return num / den


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
    col = {"cgo": "#1f5f8b", "ngh": "#b8451f", "gpb2": "#5f7d3a", "imm": "#8a6d1f"}
    lab = {
        "cgo": r"CGO exact filter, frozen at $\varepsilon=0$",
        "ngh": r"NGH constant-gain filter, frozen at $\varepsilon=1$",
        "gpb2": "GPB2, true parameters (reference)",
        "imm": "IMM, true parameters",
    }
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 3.9))
    for k in ("cgo", "ngh", "gpb2", "imm"):
        if k == "imm" and "imm" not in rows[0]["excess"]:
            continue
        st = "s--" if k == "imm" else "o-"
        a1.plot(eps, [r["excess"][k] for r in rows], st, ms=4, color=col[k], label=lab[k])
        a2.semilogy(eps, [max(r["gap_median"][k], 1e-16) for r in rows], st, ms=4, color=col[k])
    xc = crossing(rows)
    if np.isfinite(xc):
        a1.axvline(xc, color="k", ls=":", lw=1)
        a1.text(
            xc + 0.02,
            0.5 * a1.get_ylim()[1],
            rf"crossing $\varepsilon^\ast\approx{xc:.2f}$",
            ha="left",
            va="center",
            fontsize=8,
        )
    a1.set_xlabel(r"$\varepsilon$  (0 = CGO model, 1 = NGH model)")
    a1.set_ylabel(r"relative excess MSE  $\mathbb{E}(f-e)^2\,/\,\mathrm{Var}[X\mid y]$")
    a1.set_title("(a) excess over the Bayes filter", fontsize=10)
    a1.grid(alpha=0.3)
    a1.legend(fontsize=7.5, loc="upper left")
    a2.axhspan(1e-17, 1e-12, color="0.88", zorder=0)
    a2.text(
        0.5,
        3e-15,
        "round-off floor (double precision)",
        ha="center",
        va="center",
        fontsize=7.5,
        color="0.35",
    )
    a2.set_ylim(1e-17, 3.0)
    a2.set_xlabel(r"$\varepsilon$")
    a2.set_ylabel("median normalized sup-norm gap to exact")
    a2.set_title("(b) gap to the exact filter (log)", fontsize=10)
    a2.grid(alpha=0.3, which="both")
    fig.tight_layout()
    out = figdir / "homotopy_cgo_ngh.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=160)
    print(f"[homotopy] figure -> {out}   crossing eps* = {xc:.3f}")


def plot_compare(results_list, figdir: Path) -> None:
    """One panel per path: relative excess MSE of the two frozen filters."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figdir.mkdir(parents=True, exist_ok=True)
    col = {"cgo": "#1f5f8b", "ngh": "#b8451f", "gpb2": "#5f7d3a"}
    title = {
        "straight": "(a) straight path: $C$, $A$, $B$ move together",
        "c_first": "(b) channel first, then slaving",
        "ab_first": "(c) slaving first, then channel",
    }
    fig, axes = plt.subplots(
        1, len(results_list), figsize=(3.6 * len(results_list), 3.7), sharey=True
    )
    axes = np.atleast_1d(axes)
    style = {"": "o-", "_blk": "s--", "_kl": "^:"}
    vlab = {"": "frozen endpoint", "_blk": "block projection", "_kl": "KL projection"}
    for ax, rp in zip(axes, results_list):
        res = json.loads(Path(rp).read_text())
        rows = res["rows"]
        kind = res.get("path", "straight")
        eps = [r["eps"] for r in rows]
        for fam, name in (("cgo", "CGO"), ("ngh", "NGH")):
            for suf, st in style.items():
                key = fam + suf
                if key not in rows[0]["excess"]:
                    continue
                ax.plot(
                    eps,
                    [r["excess"][key] for r in rows],
                    st,
                    ms=3.2,
                    lw=1.2,
                    color=col[fam],
                    label=f"{name} filter, {vlab[suf]}",
                )
        ax.plot(
            eps,
            [r["excess"]["gpb2"] for r in rows],
            "o-",
            ms=3,
            color=col["gpb2"],
            label="GPB2, true model",
        )
        xc = crossing(rows)
        if np.isfinite(xc):
            ax.axvline(xc, color="k", ls=":", lw=1)
            ax.text(
                xc + 0.02,
                0.55 * ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 0.05,
                rf"$\varepsilon^\ast\approx{xc:.2f}$",
                fontsize=8,
                ha="left",
            )
        if kind != "straight":
            ax.axvline(0.5, color="gray", lw=0.8, alpha=0.6)
        ax.set_title(title.get(kind, kind), fontsize=9.5)
        ax.grid(alpha=0.3)
        ax.set_xlabel(r"$\varepsilon$")
    axes[0].set_ylabel(r"relative excess MSE  $\mathbb{E}(f-e)^2/\mathrm{Var}[X\mid y]$")
    axes[0].legend(fontsize=6.5, loc="upper left")
    fig.tight_layout()
    out = figdir / "homotopy_paths.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=160)
    print(f"[homotopy] comparison figure -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=200)
    ap.add_argument("--out", type=Path, default=Path("data/experiments/homotopy_cgo_ngh"))
    ap.add_argument("--grid", type=str, default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    ap.add_argument("--path", choices=PATHS, default="straight")
    ap.add_argument(
        "--plot-only",
        type=Path,
        default=None,
        help="skip the sweep; plot this results.json into --figdir",
    )
    ap.add_argument(
        "--compare",
        type=Path,
        nargs="*",
        default=None,
        help="plot several results.json side by side into --figdir",
    )
    ap.add_argument("--figdir", type=Path, default=None)
    args = ap.parse_args()
    if args.plot_only is not None:
        plot(args.plot_only, args.figdir or args.plot_only.parent)
        return
    if args.compare:
        plot_compare(args.compare, args.figdir or args.compare[0].parent)
        return
    kind = args.path
    if kind != "straight" and args.out == Path("data/experiments/homotopy_cgo_ngh"):
        args.out = args.out / kind
    args.out.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings("ignore")
    p0, p1 = model(0.0), model(1.0)
    print(
        f"[homotopy] path={kind} N={N_STEPS}, {args.seeds} seeds; rho(F) along the path: "
        + ", ".join(f"{spectral_radius(e, kind):.3f}" for e in (0, 0.25, 0.5, 0.75, 1))
    )
    print(
        f"{'eps':>5} {'postvar':>8} | {'cgo':>8} {'cgo_blk':>8} {'cgo_kl':>8} | "
        f"{'ngh':>8} {'ngh_blk':>8} {'ngh_kl':>8} | {'gpb2':>8} {'imm':>8}"
    )
    rows = []
    for eps in [float(e) for e in args.grid.split(",")]:
        r = run_eps(eps, p0, p1, args.seeds, kind)
        rows.append(r)
        xs = r["excess"]
        print(
            f"{eps:5.2f} {r['posterior_var']:8.4f} | {xs['cgo']:8.4f} {xs['cgo_blk']:8.4f} {xs['cgo_kl']:8.4f} | "
            f"{xs['ngh']:8.4f} {xs['ngh_blk']:8.4f} {xs['ngh_kl']:8.4f} | {xs['gpb2']:8.1e} {xs['imm']:8.1e}",
            flush=True,
        )
    (args.out / "results.json").write_text(
        json.dumps(
            {
                "N": N_STEPS,
                "seeds": args.seeds,
                "path": kind,
                "blocks": {
                    "A0": A0,
                    "B0": B0,
                    "C1": C1,
                    "D": D,
                    "SV": SV,
                    "M": M,
                    "Gamma": GAM,
                    "p_switch": P_SWITCH,
                },
                "rows": rows,
            },
            indent=1,
        )
    )
    print(f"[homotopy] results -> {args.out / 'results.json'}")
    plot(args.out / "results.json", args.figdir or args.out)


if __name__ == "__main__":
    main()
