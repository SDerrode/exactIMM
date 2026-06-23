#!/usr/bin/env python3
"""
prg/experiments/study.py
========================
Simulation study for "On Fast Optimal Filtering in Gaussian Switching Systems".

Runs ten jump-filtering experiments (E1–E9, including E3′) and writes one
vector figure per experiment + a ``results.json`` to an output directory.

    python -m prg.experiments.study docs/wojciech/article_vWojciech_tex

E1  Exactness        h5_exact == brute-force Bayesian filter (all Kᴺ histories)
E2  Speed            wall-time vs N: linear, as fast as a single Kalman filter
E3  Value            RMSE / regime accuracy vs naive baselines and the oracle
E3′ Value sweep      filtering gain across a value-/regime-contrast grid
E4  Multivariate     q=s=2 tracking with regime posterior
E5  Closed form      Γ_k constant in n (no Riccati); X slaved to Y
E6  Robustness       bias of h5_exact off the AB manifold vs the general IMM
E7  Rank-deficient C C with s<q (rank-deficient observation coupling)
E8  C influence      role of C as the regime-identification channel
E9  C mismatch       filtering C≠0 data with a C=0 (CMS-HLM) filter
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
    is exactly a CMS-HLM (the old condition (H4), C=0). For C!=0 the observation
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
    C is the swept parameter. Used to filter C!=0 data with an 'old' C=0 (CMS-HLM)
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
    ax.set_ylabel("max abs. difference  (h5_exact vs exact)")
    ax.set_title("E1 — h5_exact equals the exact Bayesian filter")
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
    ax1.loglog(Ns, series["h5_exact"], "o-", color=_C["h5"], label="h5_exact (proposed)", ms=4)
    ax1.loglog(
        Ns, series["imm_general"], "s-", color=_C["imm"], label="imm_general (Riccati/step)", ms=4
    )
    ax1.loglog(
        Ns,
        series["single_kalman"],
        "^-",
        color=_C["kal"],
        label="single Kalman (no switching)",
        ms=4,
    )
    ax1.set_xlabel("$N$")
    ax1.set_ylabel("wall time [s]")
    ax1.set_title("E2a — linear in $N$, as fast as Kalman")
    ax1.legend(fontsize=7)
    ax2.semilogy(Ns_ex, t_ex, "D-", color=_C["exact"], label="exact mixture ($\\sim K^N$)", ms=4)
    ax2.set_xlabel("$N$")
    ax2.set_ylabel("wall time [s]")
    ax2.set_title("E2b — exact enumeration is intractable")
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
        "single\nKalman",
        "h5_exact\n(proposed)",
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
        label="single Kalman",
    )
    ax1.errorbar(
        ms,
        rmse["h5_exact"],
        yerr=rstd["h5_exact"],
        fmt="o-",
        color=_C["h5"],
        capsize=2,
        label="h5_exact (proposed)",
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
            label="h5_exact $\\pm 2\\sigma$",
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
    ax.plot(used, rmse_h5, "o-", color=_C["h5"], label="h5_exact (assumes AB)")
    ax.plot(used, rmse_imm, "s-", color=_C["imm"], label="imm_general (general)")
    ax.set_xlabel("AB perturbation $\\varepsilon$ (added to $A$)")
    ax.set_ylabel("RMSE vs exact filter")
    ax.set_title("E6 — h5_exact degrades gracefully off the AB family")
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
    axA.set_ylabel("h5_exact vs exact")
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
            label="h5_exact $\\pm 2\\sigma$",
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
# E8 — Influence of C : from CMS-HLM (C=0) to NGH-MSM (C != 0)
# ---------------------------------------------------------------------------
def exp_c_influence(outdir: Path) -> dict:
    """E8 — sweep the observation coupling C. At C=0 the model is a CMS-HLM (the old
    condition (H4)): the observation is conditionally autonomous and the regime --
    here carrying the sign of the slaved state -- is hidden, so the exact filter can
    only average the two opposite couplings (state estimate ~ 0). As C grows the
    observation measures the state, the regime becomes identifiable, and the exact
    filter recovers the regime-dependent state, approaching the regime-aware oracle.
    The regime-conditional state law M_r/Gamma_r is itself C-independent (slaved):
    C acts purely through regime identifiability."""
    Cs = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    N, seeds = 500, list(range(40))
    rmse = {"h5_exact": [], "single_kalman": [], "oracle": [], "zero": []}
    acc = []
    for C in Cs:
        eh, ek, eo, ez, ac = [], [], [], [], []
        params = c_influence_model(C)
        for sd in seeds:
            rs, xs, ys = _simulate(params, N, seed=500 + sd)
            ExH, PiH, _ = _run(params, ys, "h5_exact")
            ExK, _ = single_kalman_filter(params, ys)
            ExO, _ = oracle_filter(params, rs, ys)
            eh.append(_rmse(ExH, xs))
            ek.append(_rmse(ExK, xs))
            eo.append(_rmse(ExO, xs))
            ez.append(_rmse(np.zeros_like(xs), xs))
            ac.append(float(np.mean(PiH.argmax(axis=1) == rs)))
        rmse["h5_exact"].append(float(np.mean(eh)))
        rmse["single_kalman"].append(float(np.mean(ek)))
        rmse["oracle"].append(float(np.mean(eo)))
        rmse["zero"].append(float(np.mean(ez)))
        acc.append(float(np.mean(ac)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.4))
    ax1.plot(Cs, acc, "o-", color=_C["h5"])
    ax1.axhline(0.5, color="grey", ls=":", lw=1, label="chance")
    ax1.set_xlabel("observation coupling $C$")
    ax1.set_ylabel("regime accuracy")
    ax1.set_title("E8a — $C$ opens the regime channel")
    ax1.legend(fontsize=7)
    ax1.annotate(
        "CMS-HLM ($C=0$)",
        xy=(0.0, acc[0]),
        xytext=(0.1, 0.62),
        fontsize=7,
        arrowprops=dict(arrowstyle="->", lw=0.6),
    )
    ax2.plot(Cs, rmse["zero"], "--", color="#bbbbbb", label="zero")
    ax2.plot(Cs, rmse["single_kalman"], "^-", color=_C["kal"], label="single Kalman")
    ax2.plot(Cs, rmse["h5_exact"], "o-", color=_C["h5"], label="h5_exact (proposed)")
    ax2.plot(Cs, rmse["oracle"], "s-", color=_C["oracle"], label="oracle")
    ax2.set_xlabel("observation coupling $C$")
    ax2.set_ylabel("state RMSE")
    ax2.set_title("E8b — state recovered as $C$ grows")
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
        "recovered": recovered,
        "N": N,
        "n_seeds": len(seeds),
    }


# ---------------------------------------------------------------------------
# E9 — Filtering C != 0 data with the old C = 0 (CMS-HLM) filter
# ---------------------------------------------------------------------------
def exp_c_mismatch(outdir: Path) -> dict:
    """E9 — what the old family costs. We generate data from a true NGH-MSM with
    C != 0 (the observation measures the state) and filter it two ways: with the
    correct filter (true C), and with the 'old' filter that assumes C = 0, i.e. the
    exact filter of the CMS-HLM obtained by zeroing C (same regimes, volatilities
    and slaved gains, but an autonomous observation). The regimes also differ in
    observation volatility, so the C = 0 filter is not blind -- it still tracks the
    regime from the volatility -- which makes the comparison non-trivial: the gap is
    exactly the information carried by the X->Y coupling that the old model discards.
    """
    Cs = [0.0, 0.15, 0.3, 0.45, 0.6, 0.7]
    N, seeds = 500, list(range(40))
    rmse = {"correct": [], "c0_old": [], "oracle": []}
    old = c_mismatch_model(0.0)  # the old CMS-HLM model (C = 0)
    for C in Cs:
        true = c_mismatch_model(C)
        ec, eold, eor = [], [], []
        for sd in seeds:
            rs, xs, ys = _simulate(true, N, seed=600 + sd)
            ExC, _, _ = _run(true, ys, "h5_exact")  # correct filter (knows C)
            ExOld, _, _ = _run(old, ys, "h5_exact")  # old C = 0 filter on the same data
            ExOr, _ = oracle_filter(true, rs, ys)
            ec.append(_rmse(ExC, xs))
            eold.append(_rmse(ExOld, xs))
            eor.append(_rmse(ExOr, xs))
        rmse["correct"].append(float(np.mean(ec)))
        rmse["c0_old"].append(float(np.mean(eold)))
        rmse["oracle"].append(float(np.mean(eor)))
    penalty = [rmse["c0_old"][i] - rmse["correct"][i] for i in range(len(Cs))]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.4))
    ax1.plot(Cs, rmse["c0_old"], "s-", color=_C["imm"], label="old $C{=}0$ filter (CMS-HLM)")
    ax1.plot(Cs, rmse["correct"], "o-", color=_C["h5"], label="correct filter (NGH-MSM)")
    ax1.plot(Cs, rmse["oracle"], "^-", color=_C["oracle"], label="oracle")
    ax1.set_xlabel("true observation coupling $C$")
    ax1.set_ylabel("state RMSE")
    ax1.set_title("E9a — old $C{=}0$ model on $C{\\neq}0$ data")
    ax1.legend(fontsize=7)
    ax2.plot(Cs, penalty, "o-", color=_C["imm"])
    ax2.set_xlabel("true observation coupling $C$")
    ax2.set_ylabel("RMSE penalty of assuming $C{=}0$")
    ax2.set_title("E9b — cost of the old assumption")
    fig.savefig(outdir / "figures" / "e9_c_mismatch.pdf")
    plt.close(fig)
    return {"C": Cs, "rmse_mean": rmse, "penalty": penalty, "N": N, "n_seeds": len(seeds)}


# ---------------------------------------------------------------------------
# E10 — Standard approximate switching filters become exact under the AB constraint
# ---------------------------------------------------------------------------
def exp_approx_exactness(outdir: Path) -> dict:
    """The standard approximate switching filters (IMM, GPB2, RBPF) coincide with
    — or converge to — the exact K^N filter on the AB-constrained NGH-MSM, because
    (Prop. 4) the regime-conditional law p(x_n|r_n,y_{1:n}) depends only on the
    current regime: collapsing the regime history (what GPB2/IMM do) is lossless."""
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
    rb_err = [
        float(np.abs(rbpf_filter(pM, ysM, n_particles=M, seed=0)[0] - ExE1).max()) for M in Ms
    ]
    floor = res["M1"]["gpb2"]["dEx"]
    res["rbpf_convergence"] = {"n_particles": Ms, "max_dEx": rb_err, "floor": floor}

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
        ("h5", "h5_exact", _C["h5"]),
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

    ax2.loglog(Ms, rb_err, "o-", color=_C["oracle"], ms=4, label="RBPF (M1)")
    ax2.axhline(max(floor, 1e-17), color="black", ls="--", lw=1, label="GPB2 / h5 floor")
    guide = rb_err[0] * np.sqrt(Ms[0] / np.array(Ms, dtype=float))
    ax2.loglog(Ms, guide, ":", color="grey", lw=1, label=r"$\propto 1/\sqrt{M}$")
    ax2.set_xlabel("RBPF particles $M$")
    ax2.set_ylabel(r"$\max_n\,|\Delta E[X_n|y]|$ vs exact")
    ax2.set_title("(b) RBPF converges to the exact filter")
    ax2.legend(fontsize=7)

    cvals = [cost["h5_exact"], cost["imm"], cost["gpb2"], cost["rbpf"]]
    ccols = [_C["h5"], _C["imm"], _C["exact"], _C["oracle"]]
    ax3.bar(range(4), cvals, color=ccols)
    ax3.set_yscale("log")
    ax3.set_xticks(range(4))
    ax3.set_xticklabels(["h5_exact", "IMM", "GPB2", "RBPF"], fontsize=7)
    ax3.set_ylabel(r"time / step [$\mu$s]")
    ax3.set_title(r"(c) Per-step cost (M1, $N=200$)")
    fig.tight_layout()
    fig.savefig(outdir / "figures" / "e10_approx_exactness.pdf")
    plt.close(fig)
    return res


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
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
    ]:
        print(f"running {key} …", flush=True)
        results[key] = fn(outdir)
    with open(outdir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("done →", outdir)
    return results


if __name__ == "__main__":
    import sys

    main(sys.argv[1] if len(sys.argv) > 1 else "docs/wojciech/article_vWojciech_tex")
