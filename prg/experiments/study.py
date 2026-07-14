#!/usr/bin/env python3
"""
prg/experiments/study.py
========================
Simulation study for "On Fast Optimal Filtering in Gaussian Switching Systems".

Runs ten jump-filtering experiments (E1–E9, including E3′) and writes one
vector figure per experiment + a ``results.json`` to an output directory.

    python -m prg.experiments.study docs/rapport_experimental

E1  Exactness        h5_exact == brute-force Bayesian filter (all Kᴺ histories)
E2  Speed            wall-time vs N: linear, as fast as a single pairwise Kalman filter
E3  Value            RMSE / regime accuracy vs naive baselines and the oracle
E3′ Value sweep      filtering gain across a value-/regime-contrast grid
E4  Multivariate     q=s=2 tracking with regime posterior
E5  Closed form      Γ_k constant in n (no Riccati); X slaved to Y
E6  Robustness       bias of h5_exact off the AB manifold vs the general IMM
E7  Rank-deficient C C with s<q (rank-deficient observation coupling)
E8  C influence      role of C as the regime-identification channel
E9  C mismatch       filtering C≠0 data with a C=0 (CGOMSM) filter
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np

from prg.classes.GSSParams import GSSParams
from prg.classes.GSSSimulator import GSSSimulator
from prg.experiments.models_paper import get_params
from prg.experiments.reference_filters import (
    exact_mixture_filter,
    gpb2_filter,
    imm_filter,
    oracle_filter,
    rbpf_filter,
    single_kalman_filter,
    with_stationary_init,
)
from prg.experiments.run_simulations import _params_from_dict
from prg.filter.gss_filter import GSSFilter

plt = None  # matplotlib.pyplot — imported lazily by _setup_mpl()


def _setup_mpl():
    """Import matplotlib lazily, only when figures are actually generated.

    Keeping matplotlib out of import time lets the model builders below be
    imported without the optional plotting dependency (e.g. from the test
    suite, where matplotlib is not installed)."""
    global plt
    if plt is None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as _plt

        _plt.rcParams.update(
            {
                "font.size": 10,
                "axes.grid": True,
                "grid.alpha": 0.3,
                "figure.dpi": 120,
                "savefig.bbox": "tight",
            }
        )
        plt = _plt
    return plt


_C = {"h5": "#1f77b4", "imm": "#ff7f0e", "kal": "#2ca02c", "exact": "#000000", "oracle": "#9467bd"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _build(name: str) -> GSSParams:
    """Stationary-init GSSParams for M1/M2/M3 (so h5_exact's prior is the true one)."""
    return with_stationary_init(_params_from_dict(get_params(name)))


def contrasted_model(p_switch: float = 0.04, m: float = 0.8) -> GSSParams:
    """A K=2 'quiet / volatile' regime-switching model where the switching genuinely
    matters: the observation reveals the regime (very different Y-volatility/dynamics)
    and the X–Y coupling flips sign across regimes (M_0=+m, M_1=-m, default m=0.8),
    so a regime-blind filter that averages the coupling estimates X≈0. The coupling
    magnitude m is the 'regime contrast' knob swept in E3'.

    Built from the free blocks (NGH-MSM valid by construction); stationary init.
    """
    from prg.classes.FMatrix import FMatrix
    from prg.classes.NoiseCovariance import GSSNoiseCovariance
    from prg.utils.h5_constraint import compute_AB

    K, q, s = 2, 1, 1
    P = np.array([[1 - p_switch, p_switch], [p_switch, 1 - p_switch]])
    M = [m, -m]
    SV = [np.array([[0.05]]), np.array([[0.60]])]  # quiet vs volatile observation
    D = [np.array([[0.30]]), np.array([[0.85]])]  # fast vs persistent Y
    C = [np.array([[0.50]]), np.array([[0.50]])]
    SU = [np.array([[0.10]]), np.array([[0.70]])]
    Dt = [M[k] * SV[k] for k in range(K)]  # Δ = M Σ_V  ⇒  M_k = Δ_k Σ_V_k⁻¹
    A_list, B_list = [], []
    for k in range(K):
        a, b = compute_AB(C[k], D[k], Dt[k], SV[k])
        A_list.append(a)
        B_list.append(b)
    fm = FMatrix(K, q, s, A_list, B_list, C, D)
    nc = GSSNoiseCovariance(K, q, s, SU, Dt, SV)
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
    return with_stationary_init(p)


def c_influence_model(C: float, p_switch: float = 0.02) -> GSSParams:
    """A K=2, q=s=1 AB-constrained model used to sweep the observation coupling C
    (E8). The two regimes differ only in the sign of the state/observation cross
    covariance (Delta_0=+d, Delta_1=-d), so the slaved gain M_r=Delta_r Sigma_V^-1
    flips sign with the regime (the regime matters for the state) while D, Sigma_V,
    Sigma_U are matched. At C=0 the observation is conditionally autonomous
    (Y_{n+1}=D Y_n+V, identical across regimes): the regime is hidden and the model
    is exactly a CGOMSM (the old condition (H4), C=0). For C!=0 the observation
    depends on the state, which reveals the regime. AB keeps it fast-exact for every
    C; stationary init.
    """
    from prg.classes.FMatrix import FMatrix
    from prg.classes.NoiseCovariance import GSSNoiseCovariance
    from prg.utils.h5_constraint import compute_AB

    K, q, s = 2, 1, 1
    P = np.array([[1 - p_switch, p_switch], [p_switch, 1 - p_switch]])
    d = 0.30
    SV = [np.array([[0.50]]), np.array([[0.50]])]
    SU = [np.array([[0.50]]), np.array([[0.50]])]
    D = [np.array([[0.50]]), np.array([[0.50]])]
    Dt = [np.array([[+d]]), np.array([[-d]])]  # Delta flips sign -> M_r flips sign
    Cm = [np.array([[C]]), np.array([[C]])]
    A_list, B_list = [], []
    for k in range(K):
        a, b = compute_AB(Cm[k], D[k], Dt[k], SV[k])
        A_list.append(a)
        B_list.append(b)
    fm = FMatrix(K, q, s, A_list, B_list, Cm, D)
    nc = GSSNoiseCovariance(K, q, s, SU, Dt, SV)
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
    return with_stationary_init(p)


def c_mismatch_model(C: float, p_switch: float = 0.02) -> GSSParams:
    """E9 model: K=2, q=s=1 NGH-MSM whose regimes differ in observation volatility
    (Sigma_V quiet vs volatile) -- so the regime is identifiable even at C=0 -- and
    in the sign of the slaved gain M_r=Delta_r Sigma_V^-1. The observation coupling
    C is the swept parameter. Used to filter C!=0 data with an 'old' C=0 (CGOMSM)
    model and quantify the cost of ignoring the X->Y coupling. AB-constrained;
    stationary init."""
    from prg.classes.FMatrix import FMatrix
    from prg.classes.NoiseCovariance import GSSNoiseCovariance
    from prg.utils.h5_constraint import compute_AB

    K, q, s = 2, 1, 1
    P = np.array([[1 - p_switch, p_switch], [p_switch, 1 - p_switch]])
    SV = [np.array([[0.20]]), np.array([[0.60]])]  # quiet / volatile observation
    SU = [np.array([[0.25]]), np.array([[0.30]])]
    D = [np.array([[0.50]]), np.array([[0.50]])]
    M = [0.6, -0.5]  # slaved gain M_r = Delta_r / Sigma_V (flips sign)
    Dt = [M[k] * SV[k] for k in range(K)]  # Delta = M Sigma_V
    Cm = [np.array([[C]]), np.array([[C]])]
    A_list, B_list = [], []
    for k in range(K):
        a, b = compute_AB(Cm[k], D[k], Dt[k], SV[k])
        A_list.append(a)
        B_list.append(b)
    fm = FMatrix(K, q, s, A_list, B_list, Cm, D)
    nc = GSSNoiseCovariance(K, q, s, SU, Dt, SV)
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
    return with_stationary_init(p)


def rank_deficient_model(p_switch: float = 0.05) -> GSSParams:
    """A K=2, q=2, s=1 NGH-MSM with a *rank-deficient* observation map: each C_r is
    a 1×2 matrix of rank 1 < q, so the two-dimensional hidden state is observed
    through a single scalar at every step. Full column rank of C is NOT required
    for the AB constraint to be the NSC (reformulated Prop. 2): the model is a
    valid NGH-MSM and the fast exact filter stays exact on it (E7).

    Built from the free blocks (A,B derived via the AB constraint); stationary init.
    """
    from prg.classes.FMatrix import FMatrix
    from prg.classes.NoiseCovariance import GSSNoiseCovariance
    from prg.utils.h5_constraint import compute_AB

    K, q, s = 2, 2, 1
    P = np.array([[1 - p_switch, p_switch], [p_switch, 1 - p_switch]])
    C = [np.array([[0.5, 0.2]]), np.array([[0.3, 0.6]])]  # 1×2, rank 1 < q = 2
    D = [np.array([[0.5]]), np.array([[0.6]])]  # 1×1, invertible
    SV = [np.array([[0.25]]), np.array([[0.60]])]  # 1×1, SPD
    Dt = [np.array([[0.10], [0.03]]), np.array([[0.18], [0.10]])]  # Δ : 2×1
    SU = [
        np.array([[0.40, 0.05], [0.05, 0.30]]),
        np.array([[0.70, 0.10], [0.10, 0.60]]),
    ]
    A_list, B_list = [], []
    for k in range(K):
        a, b = compute_AB(C[k], D[k], Dt[k], SV[k])
        A_list.append(a)
        B_list.append(b)
    fm = FMatrix(K, q, s, A_list, B_list, C, D)
    nc = GSSNoiseCovariance(K, q, s, SU, Dt, SV)
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
    return with_stationary_init(p)


def _simulate(params, N, seed):
    sim = GSSSimulator(params, N=N, seed=seed)
    rs, xs, ys = [], [], []
    for _, r, x, y in sim:
        rs.append(int(r))
        xs.append(np.asarray(x, dtype=float).ravel())
        ys.append(np.asarray(y, dtype=float).ravel())
    return np.array(rs), np.array(xs), np.array(ys)


def _run(params, ys, mode):
    """Return (E_x (N,q), pi (N,K), Var_x (N,q,q)) for a GSSFilter mode."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        filt = GSSFilter(params, mode=mode)
        Ex, Pi, Var = [], [], []
        s = params.s
        for y in ys:
            r = filt.step(np.asarray(y, dtype=float).reshape(s, 1))
            Ex.append(r.E_x.ravel())
            Pi.append(np.asarray(r.pi).ravel())
            Var.append(np.asarray(r.Var_x))
    return np.array(Ex), np.array(Pi), np.array(Var)


def _rmse(est, truth):
    return float(np.sqrt(np.mean((est - truth) ** 2)))


# ---------------------------------------------------------------------------
# E1 — Exactness
# ---------------------------------------------------------------------------
def exp_exactness(outdir: Path) -> dict:
    Ns = [3, 4, 5, 6, 7, 8, 9, 10]
    res = {}
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for name, mark in [("M1", "o"), ("M2", "s"), ("M3", "^")]:
        params = _build(name)
        Ns_m = Ns if params.K == 2 else [n for n in Ns if n <= 8]
        errsE, errsP = [], []
        for N in Ns_m:
            _, _, ys = _simulate(params, N, seed=11)
            ExH, PiH, _ = _run(params, ys, "h5_exact")
            ExE, _, PiE = exact_mixture_filter(params, ys)
            errsE.append(float(np.abs(ExH - ExE).max()))
            errsP.append(float(np.abs(PiH - PiE).max()))
        res[name] = {"N": Ns_m, "max_dEx": errsE, "max_dpi": errsP}
        ax.semilogy(Ns_m, errsE, mark + "-", label=f"{name}: $|\\Delta E[X|y]|$", ms=4)
        ax.semilogy(Ns_m, errsP, mark + "--", color=ax.lines[-1].get_color(), alpha=0.6, ms=4)
    ax.axhline(1e-10, color="grey", ls=":", lw=1)
    ax.set_xlabel("sequence length $N$")
    ax.set_ylabel("max abs. difference  (NGH-MSM-KF vs exact)")
    ax.set_title("E1 — NGH-MSM-KF equals the exact Bayesian filter")
    ax.legend(fontsize=7, ncol=1)
    fig.savefig(outdir / "figures" / "e1_exactness.pdf")
    plt.close(fig)
    res["worst_overall"] = max(
        max(res[m]["max_dEx"] + res[m]["max_dpi"]) for m in ("M1", "M2", "M3")
    )
    return res


# ---------------------------------------------------------------------------
# E2 — Speed
# ---------------------------------------------------------------------------
def _time_filter(params, ys, mode, repeats=1):
    best = np.inf
    for _ in range(repeats):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            t0 = time.perf_counter()
            filt = GSSFilter(params, mode=mode)
            for y in ys:
                filt.step(y.reshape(-1, 1))
            best = min(best, time.perf_counter() - t0)
    return best


def exp_speed(outdir: Path) -> dict:
    params = _build("M1")
    Ns = [200, 500, 1000, 2000, 5000, 10000, 20000, 50000]
    _, _, ys_full = _simulate(params, max(Ns), seed=7)
    series = {"h5_exact": [], "imm_general": [], "single_kalman": []}
    for N in Ns:
        ys = ys_full[:N]
        series["h5_exact"].append(_time_filter(params, ys, "h5_exact", repeats=2))
        series["imm_general"].append(_time_filter(params, ys, "imm_general", repeats=2))
        t0 = time.perf_counter()
        single_kalman_filter(params, ys)
        series["single_kalman"].append(time.perf_counter() - t0)

    # Exact mixture: exponential in N (tiny N only)
    Ns_ex = [4, 6, 8, 10, 12, 14]
    t_ex = []
    for N in Ns_ex:
        _, _, ys = _simulate(params, N, seed=7)
        t0 = time.perf_counter()
        exact_mixture_filter(params, ys)
        t_ex.append(time.perf_counter() - t0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.4))
    ax1.loglog(Ns, series["h5_exact"], "o-", color=_C["h5"], label="NGH-MSM-KF (proposed)", ms=4)
    ax1.loglog(
        Ns, series["imm_general"], "s-", color=_C["imm"], label="pairwise IMM (order 1)", ms=4
    )
    ax1.loglog(
        Ns,
        series["single_kalman"],
        "^-",
        color=_C["kal"],
        label="pairwise Kalman (no switching)",
        ms=4,
    )
    ax1.set_xlabel("$N$")
    ax1.set_ylabel("wall time [s]")
    ax1.set_title("linear in $N$, as fast as Kalman")
    ax1.legend(fontsize=7)
    ax2.semilogy(Ns_ex, t_ex, "D-", color=_C["exact"], label="exact mixture ($\\sim K^N$)", ms=4)
    ax2.set_xlabel("$N$")
    ax2.set_ylabel("wall time [s]")
    ax2.set_title("exact enumeration is intractable")
    ax2.legend(fontsize=7)
    fig.savefig(outdir / "figures" / "e2_speed.pdf")
    plt.close(fig)

    us = {m: 1e6 * series[m][-1] / Ns[-1] for m in series}  # µs/step at largest N
    return {"N": Ns, "times": series, "N_exact": Ns_ex, "time_exact": t_ex, "us_per_step": us}


# ---------------------------------------------------------------------------
# E3 — Value vs baselines + regime accuracy
# ---------------------------------------------------------------------------
def exp_value(outdir: Path) -> dict:
    params = contrasted_model()
    K = params.K
    N, seeds = 600, list(range(40))
    rmse = {"h5_exact": [], "single_kalman": [], "zero": [], "oracle": []}
    acc = []
    conf = np.zeros((K, K))
    for sd in seeds:
        rs, xs, ys = _simulate(params, N, seed=100 + sd)
        ExH, PiH, _ = _run(params, ys, "h5_exact")
        ExK, _ = single_kalman_filter(params, ys)
        ExO, _ = oracle_filter(params, rs, ys)
        rmse["h5_exact"].append(_rmse(ExH, xs))
        rmse["single_kalman"].append(_rmse(ExK, xs))
        rmse["zero"].append(_rmse(np.zeros_like(xs), xs))
        rmse["oracle"].append(_rmse(ExO, xs))
        pred = PiH.argmax(axis=1)
        acc.append(float(np.mean(pred == rs)))
        for t in range(N):
            conf[rs[t], pred[t]] += 1
    conf = conf / conf.sum(axis=1, keepdims=True)
    means = {m: float(np.mean(v)) for m, v in rmse.items()}
    stds = {m: float(np.std(v)) for m, v in rmse.items()}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.4))
    order = ["zero", "single_kalman", "h5_exact", "oracle"]
    labels = [
        "zero\npredictor",
        "pairwise\nKalman",
        "NGH-MSM-KF\n(proposed)",
        "oracle\n(regimes known)",
    ]
    cols = ["#bbbbbb", _C["kal"], _C["h5"], _C["oracle"]]
    ax1.bar(
        range(4), [means[m] for m in order], yerr=[stds[m] for m in order], color=cols, capsize=3
    )
    ax1.set_xticks(range(4))
    ax1.set_xticklabels(labels, fontsize=7)
    ax1.set_ylabel("state RMSE")
    ax1.set_title(f"E3a — quiet/volatile model, RMSE ({len(seeds)} runs)")
    im = ax2.imshow(conf, vmin=0, vmax=1, cmap="Blues")
    ax2.set_xticks(range(K))
    ax2.set_yticks(range(K))
    ax2.set_xlabel("predicted regime")
    ax2.set_ylabel("true regime")
    ax2.set_title(f"E3b — regime confusion (acc={np.mean(acc):.2f})")
    for i in range(K):
        for j in range(K):
            ax2.text(
                j,
                i,
                f"{conf[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if conf[i, j] > 0.5 else "black",
                fontsize=8,
            )
    fig.colorbar(im, ax=ax2, fraction=0.046)
    fig.savefig(outdir / "figures" / "e3_value.pdf")
    plt.close(fig)
    return {
        "rmse_mean": means,
        "rmse_std": stds,
        "regime_acc": float(np.mean(acc)),
        "confusion": conf.tolist(),
        "n_seeds": len(seeds),
        "N": N,
    }


# ---------------------------------------------------------------------------
# E3' — Value vs the regime contrast (sweep)
# ---------------------------------------------------------------------------
def exp_value_sweep(outdir: Path) -> dict:
    """E3' — when does modelling the switching pay off? Sweep the regime contrast,
    i.e. the coupling magnitude m of the quiet/volatile model (M_0=+m, M_1=-m, the
    observation volatilities held fixed). At m=0 the two regimes act identically on
    X and the switching is irrelevant (Kalman = h5_exact = oracle); as m grows the
    regime-blind Kalman filter must average +m and -m and degrades, while h5_exact
    tracks the regime-aware oracle. The RMSE reduction over Kalman grows with m.
    """
    ms = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    N, seeds = 500, list(range(25))
    rmse = {"single_kalman": [], "h5_exact": [], "oracle": []}
    rstd = {"single_kalman": [], "h5_exact": [], "oracle": []}
    for m in ms:
        params = contrasted_model(m=m)
        eh, ek, eo = [], [], []
        for sd in seeds:
            rs, xs, ys = _simulate(params, N, seed=300 + sd)
            ExH, _, _ = _run(params, ys, "h5_exact")
            ExK, _ = single_kalman_filter(params, ys)
            ExO, _ = oracle_filter(params, rs, ys)
            eh.append(_rmse(ExH, xs))
            ek.append(_rmse(ExK, xs))
            eo.append(_rmse(ExO, xs))
        rmse["h5_exact"].append(float(np.mean(eh)))
        rstd["h5_exact"].append(float(np.std(eh)))
        rmse["single_kalman"].append(float(np.mean(ek)))
        rstd["single_kalman"].append(float(np.std(ek)))
        rmse["oracle"].append(float(np.mean(eo)))
        rstd["oracle"].append(float(np.std(eo)))
    gain = [
        (rmse["single_kalman"][i] - rmse["h5_exact"][i]) / rmse["single_kalman"][i]
        for i in range(len(ms))
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.4))
    ax1.errorbar(
        ms,
        rmse["single_kalman"],
        yerr=rstd["single_kalman"],
        fmt="^-",
        color=_C["kal"],
        capsize=2,
        label="pairwise Kalman",
    )
    ax1.errorbar(
        ms,
        rmse["h5_exact"],
        yerr=rstd["h5_exact"],
        fmt="o-",
        color=_C["h5"],
        capsize=2,
        label="NGH-MSM-KF (proposed)",
    )
    ax1.errorbar(
        ms,
        rmse["oracle"],
        yerr=rstd["oracle"],
        fmt="s-",
        color=_C["oracle"],
        capsize=2,
        label="oracle (regimes known)",
    )
    ax1.set_xlabel("regime contrast: coupling magnitude $m$  ($M_0={+}m,\\ M_1={-}m$)")
    ax1.set_ylabel("state RMSE")
    ax1.set_title("E3$'$a — RMSE vs regime contrast")
    ax1.legend(fontsize=7)
    ax2.plot(ms, [100 * g for g in gain], "o-", color=_C["h5"])
    ax2.set_xlabel("coupling magnitude $m$")
    ax2.set_ylabel("RMSE reduction over Kalman [%]")
    ax2.set_title("E3$'$b — value grows with the contrast")
    fig.savefig(outdir / "figures" / "e3plus_value_sweep.pdf")
    plt.close(fig)
    return {
        "m": ms,
        "rmse_mean": rmse,
        "rmse_std": rstd,
        "gain_over_kalman": gain,
        "N": N,
        "n_seeds": len(seeds),
    }


# ---------------------------------------------------------------------------
# E4 — Multivariate showcase (M2)
# ---------------------------------------------------------------------------
def exp_multivariate(outdir: Path) -> dict:
    params = _build("M2")
    N = 200
    rs, xs, ys = _simulate(params, N, seed=5)
    ExH, PiH, VarH = _run(params, ys, "h5_exact")
    t = np.arange(N)
    fig, axes = plt.subplots(
        3, 1, figsize=(8.0, 6.0), sharex=True, gridspec_kw={"height_ratios": [3, 3, 1]}
    )
    for i, ax in enumerate(axes[:2]):
        sd = np.sqrt(np.clip(VarH[:, i, i], 0, None))
        ax.fill_between(
            t,
            ExH[:, i] - 2 * sd,
            ExH[:, i] + 2 * sd,
            color=_C["h5"],
            alpha=0.2,
            label="NGH-MSM-KF $\\pm 2\\sigma$",
        )
        ax.plot(t, xs[:, i], color="k", lw=1, label="true $X$")
        ax.plot(t, ExH[:, i], color=_C["h5"], lw=1.2, label="$E[X|y]$")
        ax.set_ylabel(f"$X_{{{i + 1}}}$")
        if i == 0:
            ax.legend(fontsize=7, ncol=3, loc="upper right")
    axes[2].imshow(
        PiH.T,
        aspect="auto",
        cmap="Greys",
        extent=[0, N, -0.5, params.K - 0.5],
        vmin=0,
        vmax=1,
        origin="lower",
    )
    axes[2].plot(t, rs, ".", color="#d62728", ms=3, label="true regime")
    axes[2].set_yticks(range(params.K))
    axes[2].set_ylabel("regime")
    axes[2].set_xlabel("time $n$")
    axes[2].legend(fontsize=7, loc="upper right")
    fig.suptitle("E4 — M2 ($q=s=2$): exact multivariate tracking", y=0.995)
    fig.savefig(outdir / "figures" / "e4_multivariate.pdf")
    plt.close(fig)
    rmse_h5 = _rmse(ExH, xs)
    _, xs2, _ = rs, xs, ys
    return {"N": N, "rmse_h5": rmse_h5, "rmse_zero": _rmse(np.zeros_like(xs), xs)}


# ---------------------------------------------------------------------------
# E5 — Closed form: Γ_k constant; X slaved to Y
# ---------------------------------------------------------------------------
def exp_closed_form(outdir: Path) -> dict:
    params = _build("M1")
    K, q = params.K, params.q
    N = 4000
    rs, xs, ys = _simulate(params, N, seed=2)

    # (a) imm_general per-regime predicted X-variance vs n. Although imm_general
    #     PROPAGATES a Riccati recursion, under AB it sits at its fixed point from
    #     n=1 — the propagation is redundant (this is what h5_exact exploits).
    M = [params.noise_cov.M(k) for k in range(K)]
    Gam = [params.noise_cov.Gamma(k) for k in range(K)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        filt = GSSFilter(params, mode="imm_general")
        traces = {k: [] for k in range(K)}
        for y in ys[:60]:
            filt.step(y.reshape(-1, 1))
            for k in range(K):
                Sig = filt._P_z[k] - filt._mu[k] @ filt._mu[k].T
                traces[k].append(float(Sig[:q, :q].trace()))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.4))
    for k in range(K):
        ax1.plot(traces[k], "-", lw=1.4, label=f"regime {k}")
    ax1.set_xlabel("time $n$")
    ax1.set_ylabel("per-regime $\\mathrm{tr}\\,\\mathrm{Var}[X_n|r_n]$")
    ax1.set_title("E5a — covariance at its fixed point from $n=1$")
    ax1.legend(fontsize=8, title="imm_general")

    # (b) X_n vs M_{r_n} y_n  (slaving): points on the diagonal, spread = Γ_{r_n}.
    pred = np.array([float((M[rs[n]] @ ys[n]).ravel()[0]) for n in range(N)])
    for k in range(K):
        m = rs == k
        ax2.plot(pred[m], xs[m, 0], ".", ms=2, alpha=0.4, label=f"regime {k}")
    lo, hi = pred.min(), pred.max()
    ax2.plot([lo, hi], [lo, hi], "k-", lw=1, label="$X = M_r\\,y$")
    ax2.set_xlabel("$M_{r_n} y_n$")
    ax2.set_ylabel("$X_n$")
    ax2.set_title("E5b — $X$ slaved to current $Y$")
    ax2.legend(fontsize=7)
    fig.savefig(outdir / "figures" / "e5_closed_form.pdf")
    plt.close(fig)

    flat = {k: float(np.std(traces[k][1:])) for k in range(K)}  # ~0 ⇒ constant
    return {
        "M": [m.tolist() for m in M],
        "Gamma": [g.tolist() for g in Gam],
        "trace_std_after_n1": flat,
    }


# ---------------------------------------------------------------------------
# E6 — Robustness off the AB manifold
# ---------------------------------------------------------------------------
def _perturb_A(params, eps):
    K, q, s = params.K, params.q, params.s
    from prg.classes.FMatrix import FMatrix

    A = [params.f_matrix.A(k) + eps * np.ones((q, q)) for k in range(K)]
    fm = FMatrix(
        K,
        q,
        s,
        A_list=A,
        B_list=[params.f_matrix.B(k) for k in range(K)],
        C_list=[params.f_matrix.C(k) for k in range(K)],
        D_list=[params.f_matrix.D(k) for k in range(K)],
    )
    return GSSParams(
        K=K,
        q=q,
        s=s,
        P=params.P,
        f_matrix=fm,
        noise_cov=params.noise_cov,
        pi0=params.pi0,
        mu_z0_list=[params.mu_z0(k) for k in range(K)],
        Sigma_z0_list=[params.Sigma_z0(k) for k in range(K)],
        b_list=[params.b(k) for k in range(K)],
    )


def exp_robustness(outdir: Path) -> dict:
    base = _build("M1")
    epss = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    seeds = list(range(30))
    N = 9
    rmse_h5, rmse_imm = [], []
    used = []
    for eps in epss:
        p = _perturb_A(base, eps)
        rho = max(float(np.max(np.abs(np.linalg.eigvals(p.f_matrix.F(k))))) for k in range(p.K))
        if rho >= 0.999:
            continue
        used.append(eps)
        eh, ei = [], []
        for sd in seeds:
            _, _, ys = _simulate(p, N, seed=200 + sd)
            ExX, _, _ = exact_mixture_filter(p, ys)  # ground truth on perturbed model
            ExH, _, _ = _run(p, ys, "h5_exact")
            ExI, _, _ = _run(p, ys, "imm_general")
            eh.append(_rmse(ExH, ExX))
            ei.append(_rmse(ExI, ExX))
        rmse_h5.append(float(np.mean(eh)))
        rmse_imm.append(float(np.mean(ei)))

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.plot(used, rmse_h5, "o-", color=_C["h5"], label="NGH-MSM-KF (assumes AB)")
    ax.plot(used, rmse_imm, "s-", color=_C["imm"], label="pairwise IMM (order 1)")
    ax.set_xlabel("AB perturbation $\\varepsilon$ (added to $A$)")
    ax.set_ylabel("RMSE vs exact filter")
    ax.set_title("E6 — NGH-MSM-KF degrades gracefully off the AB family")
    ax.legend(fontsize=8)
    fig.savefig(outdir / "figures" / "e6_robustness.pdf")
    plt.close(fig)
    return {"eps": used, "rmse_h5": rmse_h5, "rmse_imm": rmse_imm, "N": N, "n_seeds": len(seeds)}


# ---------------------------------------------------------------------------
# E7 — Exact filtering with a rank-deficient observation map (s < q)
# ---------------------------------------------------------------------------
def exp_rank_deficient(outdir: Path) -> dict:
    """E7 — the fast exact filter does not need C to have full column rank. On a
    K=2, q=2, s=1 NGH-MSM (each C_r is 1×2, rank 1 < q), h5_exact still equals the
    brute-force K^N Bayesian filter, and recovers *both* hidden components from the
    single observation. This is the experimental face of dropping the rank
    hypothesis (reformulated Prop. 2)."""
    from prg.utils.h5_constraint import validate_ngh_msm

    params = rank_deficient_model()
    issues = validate_ngh_msm(params)  # [] ⇒ accepted by the relaxed (C≠0) gate
    rank_C = [int(np.linalg.matrix_rank(params.f_matrix.C(k))) for k in range(params.K)]

    # (a) exactness vs the brute-force K^N filter across N
    Ns = [3, 4, 5, 6, 7, 8, 9, 10]
    errsE, errsP = [], []
    for N in Ns:
        _, _, ys = _simulate(params, N, seed=11)
        ExH, PiH, _ = _run(params, ys, "h5_exact")
        ExE, _, PiE = exact_mixture_filter(params, ys)
        errsE.append(float(np.abs(ExH - ExE).max()))
        errsP.append(float(np.abs(PiH - PiE).max()))
    worst = max(max(errsE), max(errsP))

    # (b) one longer run: both hidden components recovered from the single scalar Y
    N = 200
    rs, xs, ys = _simulate(params, N, seed=4)
    ExH, PiH, VarH = _run(params, ys, "h5_exact")

    fig, (axA, ax0, ax1) = plt.subplots(
        3, 1, figsize=(8.0, 6.4), gridspec_kw={"height_ratios": [2.4, 2.4, 2.4]}
    )
    axA.semilogy(Ns, errsE, "o-", color=_C["h5"], ms=4, label="$|\\Delta E[X|y]|$")
    axA.semilogy(Ns, errsP, "s--", color=_C["imm"], ms=4, label="$|\\Delta p(r|y)|$")
    axA.axhline(1e-10, color="grey", ls=":", lw=1)
    axA.set_xlabel("sequence length $N$")
    axA.set_ylabel("NGH-MSM-KF vs exact")
    axA.set_title(
        f"E7a — exact despite rank-deficient $C$ "
        f"($\\mathrm{{rank}}\\,C_r={rank_C}$, $q={params.q}>s={params.s}$)"
    )
    axA.legend(fontsize=7)
    t = np.arange(N)
    for i, ax in enumerate((ax0, ax1)):
        sd = np.sqrt(np.clip(VarH[:, i, i], 0, None))
        ax.fill_between(
            t,
            ExH[:, i] - 2 * sd,
            ExH[:, i] + 2 * sd,
            color=_C["h5"],
            alpha=0.2,
            label="NGH-MSM-KF $\\pm 2\\sigma$",
        )
        ax.plot(t, xs[:, i], "k", lw=1, label="true $X$")
        ax.plot(t, ExH[:, i], color=_C["h5"], lw=1.2, label="$E[X|y]$")
        ax.set_ylabel(f"$X_{{{i + 1}}}$")
        if i == 0:
            ax.legend(fontsize=7, ncol=3, loc="upper right")
    ax1.set_xlabel("time $n$")
    fig.suptitle("E7 — two hidden states recovered exactly from one observation", y=0.995)
    fig.savefig(outdir / "figures" / "e7_rank_deficient.pdf")
    plt.close(fig)

    return {
        "q": params.q,
        "s": params.s,
        "rank_C": rank_C,
        "validate_issues": issues,
        "worst_exactness": worst,
        "N_exact": Ns,
        "max_dEx": errsE,
        "max_dpi": errsP,
        "rmse_h5": _rmse(ExH, xs),
    }


# ---------------------------------------------------------------------------
# E8 — Influence of C : from CGOMSM (C=0) to NGH-MSM (C != 0)
# ---------------------------------------------------------------------------
def exp_c_influence(outdir: Path) -> dict:
    """E8 — sweep the observation coupling C. At C=0 the model is a CGOMSM (the old
    condition (H4)): the observation is conditionally autonomous and the regime --
    here carrying the sign of the slaved state -- is hidden, so the exact filter can
    only average the two opposite couplings (state estimate ~ 0). As C grows the
    observation measures the state, the regime becomes identifiable, and the exact
    filter recovers the regime-dependent state, approaching the regime-aware oracle.
    The regime-conditional state law M_r/Gamma_r is itself C-independent (slaved):
    C acts purely through regime identifiability."""
    Cs = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    N, seeds = 500, list(range(40))
    rmse = {"h5_exact": [], "single_kalman": [], "imm": [], "oracle": [], "zero": []}
    rstd = {"h5_exact": [], "single_kalman": [], "imm": [], "oracle": [], "zero": []}
    acc = []
    for C in Cs:
        eh, ek, ei, eo, ez, ac = [], [], [], [], [], []
        params = c_influence_model(C)
        for sd in seeds:
            rs, xs, ys = _simulate(params, N, seed=500 + sd)
            ExH, PiH, _ = _run(params, ys, "h5_exact")
            ExK, _ = single_kalman_filter(params, ys)
            ExI = imm_filter(params, ys)[0]
            ExO, _ = oracle_filter(params, rs, ys)
            eh.append(_rmse(ExH, xs))
            ek.append(_rmse(ExK, xs))
            ei.append(_rmse(ExI, xs))
            eo.append(_rmse(ExO, xs))
            ez.append(_rmse(np.zeros_like(xs), xs))
            ac.append(float(np.mean(PiH.argmax(axis=1) == rs)))
        for key, vals in (
            ("h5_exact", eh),
            ("single_kalman", ek),
            ("imm", ei),
            ("oracle", eo),
            ("zero", ez),
        ):
            rmse[key].append(float(np.mean(vals)))
            rstd[key].append(float(np.std(vals)))
        acc.append(float(np.mean(ac)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.4))
    ax1.plot(Cs, acc, "o-", color=_C["h5"], ms=4)  # NGH-MSM-KF regime accuracy (its usual colour)
    ax1.axhline(0.5, color="black", ls=":", lw=1)
    ax1.text(Cs[-1], 0.508, "chance (0.5)", fontsize=7, color="black", ha="right", va="bottom")
    ax1.set_xlabel("observation coupling $C$")
    ax1.set_ylabel("regime accuracy")
    ax1.set_title("$C$ opens the regime channel")
    ax1.annotate(
        "CGOMSM ($C{=}0$)",
        xy=(0.0, acc[0]),
        xytext=(0.16, 0.585),
        fontsize=7,
        ha="left",
        va="center",
        color="black",
        arrowprops=dict(arrowstyle="-|>", lw=1.2, color="black", mutation_scale=18, shrinkB=4),
    )
    ax2.errorbar(
        Cs,
        rmse["single_kalman"],
        yerr=rstd["single_kalman"],
        fmt="^-",
        color=_C["kal"],
        capsize=2,
        label="pairwise Kalman",
    )
    # IMM drawn first (underneath) with a larger hollow marker and wider caps, so the
    # proposed filter -- plotted on top -- stays visible while the IMM ring and its
    # error-bar caps still peek out, showing the two coincide (both keep error bars).
    ax2.errorbar(
        Cs,
        rmse["imm"],
        yerr=rstd["imm"],
        fmt="D--",
        color=_C["imm"],
        mfc="none",
        ms=8,
        capsize=3,
        label="pairwise IMM (order 1)",
    )
    ax2.errorbar(
        Cs,
        rmse["h5_exact"],
        yerr=rstd["h5_exact"],
        fmt="o-",
        color=_C["h5"],
        ms=5,
        capsize=2,
        label="NGH-MSM-KF (proposed)",
    )
    ax2.errorbar(
        Cs,
        rmse["oracle"],
        yerr=rstd["oracle"],
        fmt="s-",
        color=_C["oracle"],
        capsize=2,
        label="oracle",
    )
    ax2.set_xlabel("observation coupling $C$")
    ax2.set_ylabel("state RMSE")
    ax2.set_title("state recovered as $C$ grows")
    ax2.legend(fontsize=7)
    fig.savefig(outdir / "figures" / "e8_c_influence.pdf")
    plt.close(fig)
    recovered = [
        (rmse["zero"][i] - rmse["h5_exact"][i]) / (rmse["zero"][i] - rmse["oracle"][i])
        if rmse["zero"][i] > rmse["oracle"][i]
        else 0.0
        for i in range(len(Cs))
    ]
    return {
        "C": Cs,
        "regime_acc": acc,
        "rmse_mean": rmse,
        "rmse_std": rstd,
        "recovered": recovered,
        "N": N,
        "n_seeds": len(seeds),
    }


# ---------------------------------------------------------------------------
# E9 — Filtering C != 0 data with the old C = 0 (CGOMSM) filter
# ---------------------------------------------------------------------------
def exp_c_mismatch(outdir: Path) -> dict:
    """E9 — what the old family costs. We generate data from a true NGH-MSM with
    C != 0 (the observation measures the state) and filter it two ways: with the
    correct filter (true C), and with the 'old' filter that assumes C = 0, i.e. the
    exact filter of the CGOMSM obtained by zeroing C (same regimes, volatilities
    and slaved gains, but an autonomous observation). The regimes also differ in
    observation volatility, so the C = 0 filter is not blind -- it still tracks the
    regime from the volatility -- which makes the comparison non-trivial: the gap is
    exactly the information carried by the X->Y coupling that the old model discards.
    """
    Cs = [0.0, 0.15, 0.3, 0.45, 0.6, 0.7]
    N, seeds = 500, list(range(40))
    rmse = {"correct": [], "c0_old": [], "imm": [], "oracle": []}
    rstd = {"correct": [], "c0_old": [], "imm": [], "oracle": []}
    old = c_mismatch_model(0.0)  # the old CGOMSM model (C = 0)
    for C in Cs:
        true = c_mismatch_model(C)
        ec, eold, ei, eor = [], [], [], []
        for sd in seeds:
            rs, xs, ys = _simulate(true, N, seed=600 + sd)
            ExC, _, _ = _run(true, ys, "h5_exact")  # correct filter (knows C)
            ExOld, _, _ = _run(old, ys, "h5_exact")  # old C = 0 filter on the same data
            ExI = imm_filter(true, ys)[0]  # general IMM on the true model
            ExOr, _ = oracle_filter(true, rs, ys)
            ec.append(_rmse(ExC, xs))
            eold.append(_rmse(ExOld, xs))
            ei.append(_rmse(ExI, xs))
            eor.append(_rmse(ExOr, xs))
        for key, vals in (("correct", ec), ("c0_old", eold), ("imm", ei), ("oracle", eor)):
            rmse[key].append(float(np.mean(vals)))
            rstd[key].append(float(np.std(vals)))
    penalty = [rmse["c0_old"][i] - rmse["correct"][i] for i in range(len(Cs))]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.4))
    ax1.errorbar(
        Cs,
        rmse["c0_old"],
        yerr=rstd["c0_old"],
        fmt="s-",
        color="#d62728",
        capsize=2,
        label="CGOMSM filter",
    )
    # IMM drawn first (underneath) with a larger hollow marker and wider caps, so the
    # correct filter -- plotted on top -- stays visible while the IMM ring and its
    # error-bar caps still peek out, showing the two coincide (both keep error bars).
    ax1.errorbar(
        Cs,
        rmse["imm"],
        yerr=rstd["imm"],
        fmt="D--",
        color=_C["imm"],
        mfc="none",
        ms=8,
        capsize=3,
        label="pairwise IMM (order 1)",
    )
    ax1.errorbar(
        Cs,
        rmse["correct"],
        yerr=rstd["correct"],
        fmt="o-",
        color=_C["h5"],
        ms=5,
        capsize=2,
        label="NGH-MSM-KF",
    )
    ax1.errorbar(
        Cs,
        rmse["oracle"],
        yerr=rstd["oracle"],
        fmt="^-",
        color=_C["oracle"],
        capsize=2,
        label="oracle",
    )
    ax1.set_xlabel("true observation coupling $C$")
    ax1.set_ylabel("state RMSE")
    ax1.set_title("CGOMSM filter on $C{\\neq}0$ data")
    ax1.legend(fontsize=7)
    ax2.plot(Cs, penalty, "o-", color="#d62728")
    ax2.set_xlabel("true observation coupling $C$")
    ax2.set_ylabel("RMSE penalty of assuming $C{=}0$")
    ax2.set_title("cost of assuming $C{=}0$")
    fig.savefig(outdir / "figures" / "e9_c_mismatch.pdf")
    plt.close(fig)
    return {
        "C": Cs,
        "rmse_mean": rmse,
        "rmse_std": rstd,
        "penalty": penalty,
        "N": N,
        "n_seeds": len(seeds),
    }


# ---------------------------------------------------------------------------
# E11 — Which approximate switching filter is EXACT on the NGH-MSM:
#       GPB2 (order-2 collapse) vs the order-1 Blom-Bar-Shalom IMM, both on the couple Z
# ---------------------------------------------------------------------------
def exp_imm_exactness(outdir: Path) -> dict:
    """Exactness vs coupling C. Both approximate switching filters run on the couple
    Z=[X;Y] (pairwise Kalman sub-filters); they differ ONLY in the mode-collapse ORDER.
    The order-1 IMM (Blom-Bar-Shalom) collapses the K mode priors into one Gaussian per
    mode BEFORE the likelihood (O(K)/step), whereas GPB2 (order 2) keeps the K^2 pairwise
    (r_{n-1},r_n) likelihoods and collapses only afterwards (O(K^2)/step). NB: GPB2 is
    NOT a "pairwise/coupled IMM" -- that name conflates the pairwise sub-filter (shared
    by both) with the collapse order (what actually differs).

    Against the exact K^N filter (short sequences), GPB2 and the proposed constant-gain
    filter are exact to machine precision for every C, while the order-1 IMM is exact
    only at C=0 and departs monotonically as C grows: the error is BORN in the regime
    posterior p(r_n|y_{1:n}) (the AB slaving keeps the per-regime read-out X=M_r y exact
    regardless of the mixed prior) and then propagates into the state estimate. Reports
    the seed-averaged max per-step deviation of the state read-out and of the regime
    posterior."""
    Cs = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    N, seeds = 12, list(range(20))
    dev = {k: {"ex": [], "pi": []} for k in ("imm", "gpb2", "h5")}
    for C in Cs:
        params = with_stationary_init(c_influence_model(C))
        di = {"imm": [], "gpb2": [], "h5": []}
        dp = {"imm": [], "gpb2": [], "h5": []}
        for sd in seeds:
            _, _, ys = _simulate(params, N, seed=sd)
            ExE, _, PiE = exact_mixture_filter(params, ys)
            ExI, _, PiI, _ = imm_filter(params, ys)
            ExG, _, PiG, _ = gpb2_filter(params, ys)
            ExH, PiH, _ = _run(params, ys, "h5_exact")
            for key, Ex, Pi in (("imm", ExI, PiI), ("gpb2", ExG, PiG), ("h5", ExH, PiH)):
                di[key].append(float(np.max(np.abs(np.asarray(Ex).reshape(N, -1) - ExE.reshape(N, -1)))))
                dp[key].append(float(np.max(np.abs(np.asarray(Pi) - PiE))))
        for key in di:
            dev[key]["ex"].append(float(np.mean(di[key])))
            dev[key]["pi"].append(float(np.mean(dp[key])))

    floor = 1e-16
    clip = lambda v: np.maximum(v, floor)
    gpb2col = "#2ca02c"
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.4, 3.4))
    a1.semilogy(Cs, clip(dev["imm"]["ex"]), "D--", color=_C["imm"], mfc="none", ms=7, label="pairwise IMM (order 1)")
    a1.semilogy(Cs, clip(dev["gpb2"]["ex"]), "s-", color=gpb2col, ms=5, label="GPB2 (order 2)")
    a1.semilogy(Cs, clip(dev["h5"]["ex"]), "o-", color=_C["h5"], ms=4, label="NGH-MSM-KF (proposed)")
    a1.set_xlabel("observation coupling $C$")
    a1.set_ylabel(r"max$_n\,|\widehat X_n-\widehat X_n^{\mathrm{exact}}|$")
    a1.set_title("state read-out: exactness vs $C$")
    a1.legend(fontsize=7)
    a1.grid(alpha=0.3, which="both")
    a2.semilogy(Cs, clip(dev["imm"]["pi"]), "D--", color=_C["imm"], mfc="none", ms=7, label="pairwise IMM (order 1)")
    a2.semilogy(Cs, clip(dev["gpb2"]["pi"]), "s-", color=gpb2col, ms=5, label="GPB2 (order 2)")
    a2.set_xlabel("observation coupling $C$")
    a2.set_ylabel(r"max$_n\,\|\pi_n-\pi_n^{\mathrm{exact}}\|_\infty$")
    a2.set_title("regime posterior: where the error is born")
    a2.legend(fontsize=7)
    a2.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(outdir / "figures" / "imm_exactness.pdf")
    plt.close(fig)
    return {"C": Cs, "dev": dev, "N": N, "n_seeds": len(seeds)}


# ---------------------------------------------------------------------------
# E12 — Robustness to AB violation: the constant-gain filter assumes AB (marginal
#       (R,Y)-Markovianity); GPB2 uses the correct model. How do they degrade off-AB?
# ---------------------------------------------------------------------------
def _quasi_ab_model(C: float, eps: float, p_switch: float = 0.02) -> GSSParams:
    """The AB model c_influence_model(C) pushed OFF the AB constraint by eps: the state
    block A_r is set to (M_r C_r) + eps, so the AB residual max_r|A_r - M_r C_r| equals
    eps exactly (eps=0 recovers the AB model). Everything else (noise blocks Delta_r,
    Sigma_V/U, D, C, transition) is unchanged, so the perturbation is a controlled
    departure from the AB manifold. Used by exp_ab_robustness (E12)."""
    from prg.classes.FMatrix import FMatrix
    from prg.classes.NoiseCovariance import GSSNoiseCovariance
    from prg.utils.h5_constraint import compute_AB

    K, q, s = 2, 1, 1
    P = np.array([[1 - p_switch, p_switch], [p_switch, 1 - p_switch]])
    d = 0.30
    SV = [np.array([[0.50]]), np.array([[0.50]])]
    SU = [np.array([[0.50]]), np.array([[0.50]])]
    D = [np.array([[0.50]]), np.array([[0.50]])]
    Dt = [np.array([[+d]]), np.array([[-d]])]  # Delta flips sign -> M_r flips sign
    Cm = [np.array([[C]]), np.array([[C]])]
    A_list, B_list = [], []
    for k in range(K):
        a, b = compute_AB(Cm[k], D[k], Dt[k], SV[k])
        A_list.append(a + eps)  # <-- break AB by eps in the A block
        B_list.append(b)
    fm = FMatrix(K, q, s, A_list, B_list, Cm, D)
    nc = GSSNoiseCovariance(K, q, s, SU, Dt, SV)
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
    return with_stationary_init(p)


def exp_ab_robustness(outdir: Path) -> dict:
    """E12 -- robustness to AB violation. The exact constant-gain filter (NGH-MSM-KF)
    is exact only under the AB constraint, i.e. the marginal (R,Y)-Markovianity (H5).
    We push the model off AB by eps (residual |A_r - M_r C_r| = eps) and score three
    filters against the exact K^N filter of the PERTURBED model: NGH-MSM-KF (assumes
    AB/H5), GPB2 (order 2, correct model, depth-2 mode collapse), and the order-1 IMM.

    Finding (adversarially verified): NGH-MSM-KF is machine-exact on AB but degrades
    ~linearly in eps (its (H5) collapse is misspecified off-AB); GPB2 is the most
    robust by 1.5-2.5 decades (it uses the correct perturbed dynamics and only
    approximates by depth-2 collapse); the order-1 IMM has a nonzero on-AB floor but
    grows slowly. Off-AB the ranking flips to GPB2 < IMM < NGH-MSM-KF -- the AB-exact
    filter becomes the least accurate. Seed-averaged max per-step state deviation."""
    import warnings

    C = 0.4
    epss = [0.0, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
    N, seeds = 12, list(range(20))
    dev = {"h5": [], "gpb2": [], "imm": []}
    for eps in epss:
        params = _quasi_ab_model(C, eps)
        dh, dg, di = [], [], []
        for sd in seeds:
            _, _, ys = _simulate(params, N, seed=sd)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ExE, _, _ = exact_mixture_filter(params, ys)
                ExH, _, _ = _run(params, ys, "h5_exact")
                ExG, _, _, _ = gpb2_filter(params, ys)
                ExI, _, _, _ = imm_filter(params, ys)
            f = lambda E: float(np.max(np.abs(np.asarray(E).reshape(N, -1) - ExE.reshape(N, -1))))
            dh.append(f(ExH))
            dg.append(f(ExG))
            di.append(f(ExI))
        dev["h5"].append(float(np.mean(dh)))
        dev["gpb2"].append(float(np.mean(dg)))
        dev["imm"].append(float(np.mean(di)))

    floor = 1e-16
    clip = lambda v: np.maximum(v, floor)
    gpb2col = "#2ca02c"
    fig, ax = plt.subplots(figsize=(5.6, 3.7))
    ax.semilogy(epss, clip(dev["h5"]), "o-", color=_C["h5"], ms=5, label="NGH-MSM-KF (assumes AB)")
    ax.semilogy(epss, clip(dev["gpb2"]), "s-", color=gpb2col, ms=5, label="GPB2 (order 2)")
    ax.semilogy(epss, clip(dev["imm"]), "D--", color=_C["imm"], mfc="none", ms=6, label="pairwise IMM (order 1)")
    ax.set_xlabel(r"AB-constraint violation $\epsilon=\max_r|A_r-M_rC_r|$")
    ax.set_ylabel(r"max$_n\,|\widehat X_n-\widehat X_n^{\mathrm{exact}}|$")
    ax.set_title("robustness to AB violation")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(outdir / "figures" / "ab_robustness.pdf")
    plt.close(fig)
    return {"C": C, "eps": epss, "dev": dev, "N": N, "n_seeds": len(seeds)}


# ---------------------------------------------------------------------------
# E10 — Standard approximate switching filters become exact under the AB constraint
# ---------------------------------------------------------------------------
def exp_approx_exactness(outdir: Path) -> dict:
    """Under the AB constraint the regime-conditional law p(x_n|r_n,y_{1:n}) is a
    single Gaussian depending only on the current regime (Prop. 4), so collapsing the
    regime HISTORY over the previous regime is lossless -- PROVIDED the pairwise
    (r_{n-1},r_n) likelihood is retained. GPB2 and the RBPF retain it and are therefore
    EXACT (to machine precision). The classical Blom-Bar-Shalom IMM instead moment-
    matches the K mode priors into one Gaussian per mode BEFORE scoring the likelihood,
    so its regime posterior is only APPROXIMATE: near-exact for slow switching / weak
    coupling, and departing as the coupling C grows (quantified in exp_imm_exactness).
    The proposed constant-gain filter equals GPB2's exact recursion in closed form."""
    models = [("M1", 9), ("M2", 9), ("M3", 7)]
    rbpf_M = 4000
    res: dict = {}
    for name, N in models:
        params = _build(name)
        _, _, ys = _simulate(params, N, seed=11)
        ExE, _VarE, PiE = exact_mixture_filter(params, ys)
        ExH, PiH, _ = _run(params, ys, "h5_exact")
        Eg, _, Pg, llg = gpb2_filter(params, ys)
        Ei, _, Pim, lli = imm_filter(params, ys)
        Er, _, Pr, llr = rbpf_filter(params, ys, n_particles=rbpf_M, seed=0)
        res[name] = {
            "N": N,
            "h5": {"dEx": float(np.abs(ExH - ExE).max()), "dpi": float(np.abs(PiH - PiE).max())},
            "gpb2": {
                "dEx": float(np.abs(Eg - ExE).max()),
                "dpi": float(np.abs(Pg - PiE).max()),
                "loglik": llg,
            },
            "imm": {
                "dEx": float(np.abs(Ei - ExE).max()),
                "dpi": float(np.abs(Pim - PiE).max()),
                "loglik": lli,
            },
            "rbpf": {
                "dEx": float(np.abs(Er - ExE).max()),
                "dpi": float(np.abs(Pr - PiE).max()),
                "loglik": llr,
                "n_particles": rbpf_M,
            },
        }

    # RBPF convergence vs particle count (M1)
    pM = _build("M1")
    _, _, ysM = _simulate(pM, 9, seed=11)
    ExE1, _, _ = exact_mixture_filter(pM, ysM)
    Ms = [100, 300, 1000, 3000, 10000, 30000]
    conv_seeds = list(range(20))  # average the Monte-Carlo error over many particle seeds
    rb_err, rb_std = [], []
    for M in Ms:
        e = [
            float(np.abs(rbpf_filter(pM, ysM, n_particles=M, seed=sd)[0] - ExE1).max())
            for sd in conv_seeds
        ]
        rb_err.append(float(np.mean(e)))
        rb_std.append(float(np.std(e)))
    floor = res["M1"]["gpb2"]["dEx"]
    res["rbpf_convergence"] = {
        "n_particles": Ms,
        "max_dEx": rb_err,
        "max_dEx_std": rb_std,
        "n_seeds": len(conv_seeds),
        "floor": floor,
    }

    # per-step wall-time on M1 (reference implementations; see caption)
    Nt = 200
    pT = _build("M1")
    _, _, ysT = _simulate(pT, Nt, seed=11)

    def _us(fn):
        ts = []
        for _ in range(3):
            t0 = time.perf_counter()
            fn()
            ts.append(time.perf_counter() - t0)
        return 1e6 * min(ts) / Nt

    cost = {
        "N": Nt,
        "h5_exact": _us(lambda: _run(pT, ysT, "h5_exact")),
        "imm": _us(lambda: imm_filter(pT, ysT)),
        "gpb2": _us(lambda: gpb2_filter(pT, ysT)),
        "rbpf": _us(lambda: rbpf_filter(pT, ysT, n_particles=rbpf_M, seed=0)),
    }
    res["cost_us_per_step"] = cost

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12.2, 3.4))
    keys = [
        ("h5", "NGH-MSM-KF", _C["h5"]),
        ("gpb2", "GPB2", _C["exact"]),
        ("imm", "IMM", _C["imm"]),
        ("rbpf", f"RBPF ({rbpf_M})", _C["oracle"]),
    ]
    x = np.arange(len(models))
    w = 0.2
    for i, (fk, fl, col) in enumerate(keys):
        vals = [max(res[name][fk]["dEx"], 1e-17) for name, _ in models]
        ax1.bar(x + (i - 1.5) * w, vals, w, label=fl, color=col)
    ax1.set_yscale("log")
    ax1.set_ylim(1e-18, 1e-1)
    ax1.set_xticks(x)
    ax1.set_xticklabels([m for m, _ in models])
    ax1.axhline(1e-12, color="grey", ls=":", lw=1)
    ax1.set_ylabel(r"$\max_n\,|\Delta E[X_n|y]|$ vs exact")
    ax1.set_title("(a) Agreement with the exact $K^N$ filter")
    ax1.legend(fontsize=7, ncol=2)

    ax2.errorbar(
        Ms,
        rb_err,
        yerr=rb_std,
        fmt="o-",
        color=_C["oracle"],
        ms=4,
        capsize=2,
        label="RBPF (M1, mean $\\pm$ s.d., 20 seeds)",
    )
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.axhline(max(floor, 1e-17), color="black", ls="--", lw=1, label="GPB2 / h5 floor")
    guide = rb_err[0] * np.sqrt(Ms[0] / np.array(Ms, dtype=float))
    ax2.plot(Ms, guide, ":", color="grey", lw=1, label=r"$\propto 1/\sqrt{M}$")
    ax2.set_xlabel("RBPF particles $M$")
    ax2.set_ylabel(r"$\max_n\,|\Delta E[X_n|y]|$ vs exact")
    ax2.set_title("(b) RBPF converges to the exact filter")
    ax2.legend(fontsize=7)

    cvals = [cost["h5_exact"], cost["imm"], cost["gpb2"], cost["rbpf"]]
    ccols = [_C["h5"], _C["imm"], _C["exact"], _C["oracle"]]
    ax3.bar(range(4), cvals, color=ccols)
    ax3.set_yscale("log")
    ax3.set_xticks(range(4))
    ax3.set_xticklabels(["NGH-MSM-KF", "IMM", "GPB2", "RBPF"], fontsize=7)
    ax3.set_ylabel(r"time / step [$\mu$s]")
    ax3.set_title(r"(c) Per-step cost (M1, $N=200$)")
    fig.tight_layout()
    fig.savefig(outdir / "figures" / "e10_approx_exactness.pdf")
    plt.close(fig)
    return res


# ---------------------------------------------------------------------------
# E7-fair — Fair comparison of the old (CGOMSM) and new (NGH-MSM) families
# ---------------------------------------------------------------------------
def exp_fair_comparison(outdir: Path) -> dict:
    """Fair head-to-head of the old (CGOMSM) and new (NGH-MSM) approximations on
    a model that lies in *neither* family (scalar, symmetric in x and y).

    (a) Statistical parity: the families are exact time-reversal mirrors, so a
        causal *forward* filter spuriously favours the new one, while a
        time-symmetric (smoothing) metric makes them exactly equal.
    (b) The real difference is computational: the new family is exactly,
        linearly filterable, whereas the old family's exact filter is the
        exponential K^N mixture, and the history-collapse the standard
        approximations perform is lossless only on the new family.
    """
    from prg.classes.FMatrix import FMatrix
    from prg.classes.NoiseCovariance import GSSNoiseCovariance
    from prg.utils.h5_constraint import apply_AB_constraint

    # ---- (a) scalar symmetric moment model: Sigma over (x1,y1,x2,y2) ----
    def _sig4(c, t, d):
        return np.array([[1, c, t, d], [c, 1, d, t], [t, d, 1, c], [d, t, c, 1]], float)

    def _bigcov(M, L):  # length-L joint covariance of the implied stationary VAR(1)
        sz = M[:2, :2]
        F = M[2:, :2] @ np.linalg.inv(M[:2, :2])
        B = np.zeros((2 * L, 2 * L))
        for i in range(L):
            for j in range(L):
                k = j - i
                B[2 * i : 2 * i + 2, 2 * j : 2 * j + 2] = (
                    sz @ np.linalg.matrix_power(F.T, k)
                    if k >= 0
                    else np.linalg.matrix_power(F, -k) @ sz
                )
        return B

    def _mse(Bm, Bt, tgt, W):  # estimator gain from model Bm, error under truth Bt
        g = (Bm[np.ix_([tgt], W)] @ np.linalg.inv(Bm[np.ix_(W, W)])).ravel()
        return Bt[tgt, tgt] - 2 * g @ Bt[tgt, W] + g @ Bt[np.ix_(W, W)] @ g

    c, t, L, nmid = 0.5, 0.5, 25, 12
    xn = 2 * nmid
    w_fwd = [2 * k + 1 for k in range(nmid + 1)]  # forward observations Y_1..Y_n
    w_all = [2 * k + 1 for k in range(L)]  # all observations (smoothing)
    deltas = np.linspace(0.0, 0.72, 25)
    fwd_old, fwd_new, sm_old, sm_new = [], [], [], []
    for delta in deltas:
        St = _sig4(c, t, c * t + delta)
        Mn = St.copy()
        Mn[0, 3] = Mn[3, 0] = c * t  # new: replace Cov(X1,Y2) by the product
        Mo = St.copy()
        Mo[1, 2] = Mo[2, 1] = c * t  # old: replace Cov(Y1,X2) by the product
        Bt, Bn, Bo = _bigcov(St, L), _bigcov(Mn, L), _bigcov(Mo, L)
        fwd_old.append(_mse(Bo, Bt, xn, w_fwd))
        fwd_new.append(_mse(Bn, Bt, xn, w_fwd))
        sm_old.append(_mse(Bo, Bt, xn, w_all))
        sm_new.append(_mse(Bn, Bt, xn, w_all))
    tie_gap = float(np.max(np.abs(np.array(sm_old) - np.array(sm_new))))

    # ---- (b) K=2 switching: NEW (NGH-MSM) exact-linear vs OLD (classical) K^N ----
    def _m(x):
        return np.array([[float(x)]])

    def _build(A, B, C, D, SU, Dl, SV, b_zero=False):
        fm = FMatrix(
            K=2,
            q=1,
            s=1,
            A_list=[_m(A[r]) for r in range(2)],
            B_list=[_m(0.0 if b_zero else B[r]) for r in range(2)],
            C_list=[_m(C[r]) for r in range(2)],
            D_list=[_m(D[r]) for r in range(2)],
        )
        nc = GSSNoiseCovariance(
            K=2,
            q=1,
            s=1,
            Sigma_U_list=[_m(SU[r]) for r in range(2)],
            Delta_list=[_m(Dl[r]) for r in range(2)],
            Sigma_V_list=[_m(SV[r]) for r in range(2)],
        )
        p = GSSParams(
            K=2,
            q=1,
            s=1,
            P=np.array([[0.9, 0.1], [0.1, 0.9]]),
            f_matrix=fm,
            noise_cov=nc,
            pi0=np.array([0.5, 0.5]),
            mu_z0_list=[np.zeros((2, 1))] * 2,
            Sigma_z0_list=[np.eye(2)] * 2,
            b_list=[np.zeros((2, 1))] * 2,
        )
        return with_stationary_init(p)

    A, B, SU, Dl, SV = [0.6, 0.4], [0.2, 0.3], [0.5, 0.6], [0.2, -0.15], [0.5, 0.6]
    C, D = B[:], A[:]  # x<->y symmetric per regime (C=B, D=A)
    true = _build(A, B, C, D, SU, Dl, SV)
    m_new = apply_AB_constraint(true)  # NEW family projection (NGH-MSM)
    m_old = _build(A, B, C, D, SU, Dl, SV, b_zero=True)  # OLD: X autonomous (classical)

    def _resid(p):  # max disagreement of IMM / GPB2 with the exact K^N filter
        ys = np.array([float(np.ravel(y)[0]) for _, _, _, y in GSSSimulator(p, N=9, seed=1)])
        ex = exact_mixture_filter(p, ys)[0].ravel()
        return (
            float(np.max(np.abs(imm_filter(p, ys)[0].ravel() - ex))),
            float(np.max(np.abs(gpb2_filter(p, ys)[0].ravel() - ex))),
        )

    imm_new, gpb2_new = _resid(m_new)
    imm_old, gpb2_old = _resid(m_old)

    # ---- figure ----
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 3.4))
    ax1.plot(deltas, fwd_old, color=_C["kal"], lw=2, label="CGOMSM")
    ax1.plot(deltas, fwd_new, color=_C["h5"], lw=2, label="NGH-MSM")
    ax1.set_title("(a1) naive metric: causal forward filtering")
    ax1.set_xlabel(r"departure $\delta$")
    ax1.set_ylabel("state MSE")
    ax1.legend(fontsize=8)
    ax2.plot(deltas, sm_old, color=_C["kal"], lw=3, label="CGOMSM")
    ax2.plot(deltas, sm_new, "--", color=_C["h5"], lw=2, label="NGH-MSM")
    ax2.set_title("(a2) fair metric: time-symmetric (smoothing)")
    ax2.set_xlabel(r"departure $\delta$")
    ax2.set_ylabel("state MSE")
    ax2.legend(fontsize=8)
    ax2.text(0.30, 0.62, "exact tie", fontsize=8, color="0.4")
    xb = np.arange(2)
    bw = 0.35
    ax3.bar(xb - bw / 2, [imm_new, imm_old], bw, color=_C["imm"], label="IMM")
    ax3.bar(xb + bw / 2, [gpb2_new, gpb2_old], bw, color=_C["oracle"], label="GPB2")
    ax3.set_yscale("log")
    ax3.set_xticks(xb)
    ax3.set_xticklabels(["NGH-MSM", "CGOMSM"])
    ax3.axhline(1e-12, color="black", ls=":", lw=1)
    ax3.text(1.45, 2e-12, "round-off", fontsize=7, ha="right")
    ax3.set_title("(b) history-collapse lossless only on NGH-MSM")
    ax3.set_ylabel("max abs. difference vs exact")
    ax3.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "figures" / "efair_fair_comparison.pdf")
    plt.close(fig)

    return {
        "operating_point": {"c": c, "t": t},
        "smoothing_tie_gap": tie_gap,
        "forward_gap_at_delta0p4": float(
            abs(np.interp(0.4, deltas, fwd_old) - np.interp(0.4, deltas, fwd_new))
        ),
        "collapse_residual": {
            "imm_new": imm_new,
            "gpb2_new": gpb2_new,
            "imm_old": imm_old,
            "gpb2_old": gpb2_old,
        },
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# E_consigne — exogenous input ("consigne"): the exact filter stays exact and
# gains the N_r u_{n-1} read-out; a regime-blind filter cannot exploit a
# sign-flipping input gain.
# ---------------------------------------------------------------------------
def consigne_model(g: float = 1.2, p_switch: float = 0.04, m: float = 0.8) -> GSSParams:
    """K=2 'quiet/volatile' NGH-MSM with a 1-D exogenous input whose hidden-state
    gain flips sign across regimes (G^X = +g in regime 0, -g in regime 1; G^Y = 0),
    so the known input u drives X in opposite directions depending on the regime.
    The slaving read-out gain is then N_r = G^X_r = ±g: only a regime-aware filter
    recovers the input contribution; a regime-blind filter averages the gain to ~0.
    The regime stays identifiable from Y (quiet vs volatile). Built from free blocks
    (NGH-MSM valid by construction), stationary init."""
    from prg.classes.FMatrix import FMatrix
    from prg.classes.NoiseCovariance import GSSNoiseCovariance
    from prg.utils.h5_constraint import compute_AB

    K, q, s = 2, 1, 1
    P = np.array([[1 - p_switch, p_switch], [p_switch, 1 - p_switch]])
    M = [m, -m]
    SV = [np.array([[0.05]]), np.array([[0.60]])]
    D = [np.array([[0.30]]), np.array([[0.85]])]
    C = [np.array([[0.50]]), np.array([[0.50]])]
    SU = [np.array([[0.10]]), np.array([[0.70]])]
    Dt = [M[k] * SV[k] for k in range(K)]
    A_list, B_list = [], []
    for k in range(K):
        a, b = compute_AB(C[k], D[k], Dt[k], SV[k])
        A_list.append(a)
        B_list.append(b)
    fm = FMatrix(K, q, s, A_list, B_list, C, D)
    nc = GSSNoiseCovariance(K, q, s, SU, Dt, SV)
    G_list = [np.array([[g], [0.0]]), np.array([[-g], [0.0]])]  # G^X = ±g, G^Y = 0
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
        G_list=G_list,
    )
    return with_stationary_init(p)


def _sim_u(params, N, seed, u):
    """Simulate (rs, xs, ys) with an exogenous input u of shape (N, p)."""
    rs, xs, ys = [], [], []
    for _, r, x, y in GSSSimulator(params, N=N, seed=seed, u=u):
        rs.append(int(r))
        xs.append(np.ravel(np.asarray(x, dtype=float)))
        ys.append(np.ravel(np.asarray(y, dtype=float)))
    return np.array(rs), np.array(xs), np.array(ys)


def _run_h5_u(params, ys, u):
    """h5_exact E_x with (u not None) or without (u None) the consigne."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        f = GSSFilter(params, mode="h5_exact")
        s = params.s
        ex = []
        for n, y in enumerate(ys):
            un = u[n] if u is not None else None
            ex.append(f.step(np.asarray(y, dtype=float).reshape(s, 1), u=un).E_x.ravel())
    return np.array(ex)


def exp_consigne(outdir: Path) -> dict:
    from prg.utils.input_signal import make_input

    params = consigne_model()
    p_in = params.p
    N, seeds = 600, list(range(40))
    u = make_input("square(140)", N, p_in) * 2.0  # slow ±2 square wave

    rmse = {"pairwise_blind": [], "h5_blind": [], "h5_consigne": [], "oracle_consigne": []}
    for sd in seeds:
        rs, xs, ys = _sim_u(params, N, 100 + sd, u)
        rmse["pairwise_blind"].append(_rmse(single_kalman_filter(params, ys)[0], xs))
        rmse["h5_blind"].append(_rmse(_run_h5_u(params, ys, None), xs))
        rmse["h5_consigne"].append(_rmse(_run_h5_u(params, ys, u), xs))
        rmse["oracle_consigne"].append(_rmse(oracle_filter(params, rs, ys, us=u)[0], xs))
    means = {m: float(np.mean(v)) for m, v in rmse.items()}
    stds = {m: float(np.std(v)) for m, v in rmse.items()}

    # Exactness WITH the input: h5_exact == brute-force Kᴺ (short N)
    Ns = 11
    u_s = make_input("gaussian", Ns, p_in, seed=0)
    _, _, ys_s = _sim_u(params, Ns, 7, u_s)
    resid = float(
        np.max(np.abs(_run_h5_u(params, ys_s, u_s) - exact_mixture_filter(params, ys_s, us=u_s)[0]))
    )

    # Figure: (a) one trajectory segment, (b) RMSE bars
    _, xs0, ys0 = _sim_u(params, N, 100, u)
    ex_u, ex_no = _run_h5_u(params, ys0, u), _run_h5_u(params, ys0, None)
    seg = slice(120, 360)
    nn = np.arange(N)[seg]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.8, 3.4))
    ax1.plot(nn, xs0[seg, 0], color="black", lw=1.3, label="true $X_n$")
    ax1.plot(nn, ex_u[seg, 0], color=_C["h5"], lw=1.1, label=r"NGH-MSM-KF (consigne)")
    ax1.plot(
        nn, ex_no[seg, 0], color=_C["imm"], lw=1.0, ls="--", label=r"NGH-MSM-KF (blind to $u$)"
    )
    ax1.set_xlabel("$n$")
    ax1.set_ylabel("state $X_n$")
    ax1.set_title(r"E$_c$a — sign-flipping input drives $X$")
    ax1.legend(fontsize=7, loc="upper right")
    order = ["pairwise_blind", "h5_blind", "h5_consigne", "oracle_consigne"]
    labels = [
        "pairwise\nKalman\n(blind)",
        "NGH-MSM-KF\n(blind to $u$)",
        "NGH-MSM-KF\n(consigne)",
        "oracle\n(regimes+$u$)",
    ]
    cols = [_C["kal"], _C["imm"], _C["h5"], _C["oracle"]]
    ax2.bar(
        range(4), [means[m] for m in order], yerr=[stds[m] for m in order], color=cols, capsize=3
    )
    ax2.set_xticks(range(4))
    ax2.set_xticklabels(labels, fontsize=7)
    ax2.set_ylabel("state RMSE")
    ax2.set_title(rf"E$_c$b — RMSE ({len(seeds)} runs)")
    fig.savefig(outdir / "figures" / "econsigne.pdf")
    plt.close(fig)
    return {
        "rmse_mean": means,
        "rmse_std": stds,
        "h5_vs_bruteforce_max_abs": resid,
        "N_r": [float(params.N(k).ravel()[0]) for k in range(params.K)],
        "N": N,
        "n_seeds": len(seeds),
        "p": p_in,
    }


def main(outdir: str | Path) -> dict:
    _setup_mpl()
    outdir = Path(outdir)
    (outdir / "figures").mkdir(parents=True, exist_ok=True)
    results = {}
    for key, fn in [
        ("E1_exactness", exp_exactness),
        ("E2_speed", exp_speed),
        ("E3_value", exp_value),
        ("E3p_value_sweep", exp_value_sweep),
        ("E4_multivariate", exp_multivariate),
        ("E5_closed_form", exp_closed_form),
        ("E6_robustness", exp_robustness),
        ("E7_rank_deficient", exp_rank_deficient),
        ("E8_c_influence", exp_c_influence),
        ("E9_c_mismatch", exp_c_mismatch),
        ("E10_approx_exactness", exp_approx_exactness),
        ("Efair_fair_comparison", exp_fair_comparison),
        ("Econsigne", exp_consigne),
    ]:
        print(f"running {key} …", flush=True)
        results[key] = fn(outdir)
    with open(outdir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("done →", outdir)
    return results


if __name__ == "__main__":
    import sys

    main(sys.argv[1] if len(sys.argv) > 1 else "docs/rapport_experimental")
