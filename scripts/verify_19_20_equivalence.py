#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/verify_19_20_equivalence.py
====================================
Numerical verification of the equivalence between two methods for computing
the per-regime conditional moments (19)-(20) of the (H5) GSS paper:

  (19)  E[X_{n+1} | r_{n+1}, y_{n+1}]
  (20)  E[X_{n+1} X_{n+1}^T | r_{n+1}, y_{n+1}]

under the (H5)-compatible AB constraint

    A(r) = Δ(r) Σ_V(r)⁻¹ C(r),       B(r) = Δ(r) Σ_V(r)⁻¹ D(r).

Question
--------
Sous la contrainte AB, on peut démontrer par récurrence que

    E [X_{n+1} | r_{n+1}, y_{n+1}]    = Δ(r_{n+1}) Σ_V(r_{n+1})⁻¹ y_{n+1}
    Var[X_{n+1} | r_{n+1}, y_{n+1}]  = Σ_U(r_{n+1}) − Δ(r_{n+1}) Σ_V(r_{n+1})⁻¹ Δ(r_{n+1})ᵀ

ce qui donnerait un calcul direct de (19)-(20) à partir des seuls
paramètres de la covariance du bruit, sans passer par la récursion
(16)-(17) de Wojciech. Ce script vérifie numériquement l'équivalence
entre :

  Méthode (a)  Recursion (16)-(17) jusqu'au point fixe → moments
               stationnaires E[Z_n Z_n^T | r_n=k] ; puis conditionnement
               gaussien (21)-(22).

  Méthode (b)  Formules directes ci-dessus.

Pour rester dans le cadre exact des formules (21)-(22) du papier, qui
n'incluent pas de biais (X̄, Ȳ), les biais b(k) sont forcés à 0 dans
le modèle de test (sinon il faut ajouter le terme correctif
b_X − Δ Σ_V⁻¹ b_Y à la moyenne conditionnelle de (b)).

La preuve analytique sous-jacente : à partir de la dynamique
  X_{n+1} = A x_n + B y_n + U_{n+1}, Y_{n+1} = C x_n + D y_n + V_{n+1},
on a, sous la contrainte AB,
  X_{n+1} = Δ Σ_V⁻¹ Y_{n+1} + W',
avec  W' := U_{n+1} − Δ Σ_V⁻¹ V_{n+1}  indépendant de (Y_{n+1}, x_n, y_n)
sachant r_{n+1}  (la covariance Cov(W', V_{n+1}|r) = Δ − Δ Σ_V⁻¹ Σ_V = 0).
Donc X_{n+1} | (r_{n+1}, y_{n+1}) ~ N(Δ Σ_V⁻¹ y_{n+1}, Σ_U − Δ Σ_V⁻¹ Δᵀ),
indépendamment du passé.

Usage
-----
    # Tirages aléatoires (par défaut : 20 modèles K=3, q=2, s=2) :
    python scripts/verify_19_20_equivalence.py

    # Plus de tirages, dimensions différentes :
    python scripts/verify_19_20_equivalence.py -K 4 -q 3 -s 2 -n 50

    # À partir d'un modèle existant (les biais sont remis à zéro) :
    python scripts/verify_19_20_equivalence.py --model model_gss_K2_q1_s1

    # Trajectoire simulée — comparer E[X|r,y] pour des y observés :
    python scripts/verify_19_20_equivalence.py -N 500 -v

    # Contrôle négatif : sans la contrainte AB, (a) et (b) divergent d'ordre 1 :
    python scripts/verify_19_20_equivalence.py --no-constraint -n 10
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import numpy as np

# Project root on sys.path so we can import prg.* regardless of CWD.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prg.classes.FMatrix import FMatrix  # noqa: E402
from prg.classes.GSSParams import GSSParams  # noqa: E402
from prg.classes.GSSSimulator import GSSSimulator  # noqa: E402
from prg.classes.NoiseCovariance import GSSNoiseCovariance  # noqa: E402
from prg.utils.h5_constraint import apply_AB_constraint, compute_AB  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers — random model construction
# ---------------------------------------------------------------------------
def _random_spd(size: int, rng: np.random.Generator, scale: float = 0.4) -> np.ndarray:
    """Random symmetric positive-definite matrix (size × size)."""
    L = rng.standard_normal((size, size)) * scale
    return L @ L.T + np.eye(size) * 0.3


def _random_transition(K: int, rng: np.random.Generator) -> np.ndarray:
    """Random row-stochastic K × K matrix (Dirichlet-like)."""
    M = rng.uniform(0.2, 1.0, (K, K))
    M /= M.sum(axis=1, keepdims=True)
    return M


def _stabilise(M: np.ndarray, max_rho: float = 0.85) -> np.ndarray:
    """Rescale M so that its spectral radius is below `max_rho`."""
    rho = float(np.max(np.abs(np.linalg.eigvals(M))))
    if rho > max_rho:
        M = M * (max_rho / rho)
    return M


def make_random_AB_params(
    K: int, q: int, s: int, rng: np.random.Generator, ab_constraint: bool = True,
) -> GSSParams:
    """
    Build a random GSSParams with zero biases.

    Σ_W(k) is drawn as a single SPD matrix and split into (Σ_U, Δ, Σ_V).
    C, D are random with D's spectral radius bounded for stability.

    If ``ab_constraint=True`` (default), A, B are determined by the
    closed-form AB constraint A = Δ Σ_V⁻¹ C, B = Δ Σ_V⁻¹ D — the model
    then satisfies (H5).

    If ``ab_constraint=False``, A, B are independently drawn as random
    Gaussian matrices (with spectral radius bounded). The resulting
    model violates (H5), and methods (a) and (b) should disagree.
    """
    P = _random_transition(K, rng)
    A_list, B_list, C_list, D_list = [], [], [], []
    SU_list, Dt_list, SV_list = [], [], []
    mu0, S0 = [], []
    for _ in range(K):
        SW = _random_spd(q + s, rng, scale=0.5)
        SU = SW[:q, :q].copy()
        Dt = SW[:q, q:].copy()
        SV = SW[q:, q:].copy()
        C = rng.standard_normal((s, q)) * 0.3
        D = rng.standard_normal((s, s)) * 0.3
        if ab_constraint:
            # Under AB, F_k = [Δ Σ_V⁻¹; I] · [C, D] has the same non-zero
            # eigenvalues as C·Δ·Σ_V⁻¹ + D (an s × s matrix). Rescale
            # (C, D) so its spectral radius stays well inside the unit
            # disk — otherwise the second-moment recursion (17) diverges.
            M_eff = C @ np.linalg.solve(SV, Dt.T).T + D  # C Δ Σ_V⁻¹ + D
            rho = float(np.max(np.abs(np.linalg.eigvals(M_eff))))
            if rho > 0.85:
                scale = 0.85 / rho
                C = C * scale
                D = D * scale
            A, B = compute_AB(C, D, Dt, SV)
        else:
            # Negative control: random A, B independent of (C, D, Δ, Σ_V).
            D = _stabilise(D)
            A = _stabilise(rng.standard_normal((q, q)) * 0.4)
            B = rng.standard_normal((q, s)) * 0.4
        A_list.append(A); B_list.append(B); C_list.append(C); D_list.append(D)
        SU_list.append(SU); Dt_list.append(Dt); SV_list.append(SV)
        mu0.append(np.zeros((q + s, 1)))
        S0.append(np.eye(q + s) * 0.5)

    f_matrix = FMatrix(K=K, q=q, s=s,
                       A_list=A_list, B_list=B_list,
                       C_list=C_list, D_list=D_list)
    noise_cov = GSSNoiseCovariance(K=K, q=q, s=s,
                                   Sigma_U_list=SU_list,
                                   Delta_list=Dt_list,
                                   Sigma_V_list=SV_list)
    return GSSParams(K=K, q=q, s=s, P=P,
                     f_matrix=f_matrix, noise_cov=noise_cov,
                     pi0=None,
                     mu_z0_list=mu0, Sigma_z0_list=S0,
                     b_list=None)  # zero biases


def load_named_model(name: str) -> GSSParams:
    """
    Load a model from `prg.models.<name>`, apply the AB constraint, and
    force the biases to zero so that the comparison stays inside the
    paper's (21)-(22) framework.
    """
    module = importlib.import_module(f"prg.models.{name}")
    # Pick the first BaseGSSModel subclass found in the module
    from prg.models.base_gss_model import BaseGSSModel
    model_cls = None
    for attr_name in dir(module):
        obj = getattr(module, attr_name)
        if isinstance(obj, type) and issubclass(obj, BaseGSSModel) and obj is not BaseGSSModel:
            model_cls = obj
            break
    if model_cls is None:
        raise ValueError(f"No BaseGSSModel subclass found in prg.models.{name}.")
    params = GSSParams.from_model(model_cls())
    params = apply_AB_constraint(params)
    # Rebuild with zero biases (so that paper's formulas (21)-(22) apply
    # without a bias-correction term)
    return GSSParams(
        K=params.K, q=params.q, s=params.s, P=params.P,
        f_matrix=params.f_matrix, noise_cov=params.noise_cov,
        pi0=params.pi0,
        mu_z0_list=[params.mu_z0(k) for k in range(params.K)],
        Sigma_z0_list=[params.Sigma_z0(k) for k in range(params.K)],
        b_list=None,
    )


# ---------------------------------------------------------------------------
# Method (a) — Wojciech's recursion (16)-(17) + Gaussian conditioning
# ---------------------------------------------------------------------------
def stationary_second_moments(
    params: GSSParams,
    max_iter: int = 5000,
    tol: float = 1e-14,
) -> tuple[list[np.ndarray], int, float]:
    """
    Fixed-point iteration of (17) for the per-regime uncentred second moment
        Π(k) := E[Z_n Z_n^T | r_n = k].

    With zero biases and stationarity, E[Z_n | r_n=k] = 0 (verified
    afterwards). The recursion is

        Π(k) = Σ_j p_rev[j,k] · F_k · Π(j) · F_k^T  +  Σ_W(k),

    where  p_rev[j,k] = π_∞[j] · P[j,k] / π_∞[k]  is the time-reversed
    transition.

    Returns
    -------
    Pi : list of K arrays (q+s, q+s)  — uncentred 2nd moment per regime.
    n_iter : int  — number of iterations to convergence.
    final_diff : float  — max absolute change at the last step.
    """
    K = params.K
    pi_inf = params.stationary_distribution()
    # p_rev[j, k] = p(r_n = j | r_{n+1} = k)
    joint = pi_inf[:, None] * params.P                       # (K, K)
    marg = joint.sum(axis=0)                                 # = π_∞
    safe = np.where(marg > 0.0, marg, 1.0)
    p_rev = joint / safe[None, :]

    # Initial value: Σ_z0(k) + μ_z0 μ_z0^T (uncentred), or identity if zero.
    Pi = [
        params.Sigma_z0(k) + params.mu_z0(k) @ params.mu_z0(k).T
        for k in range(K)
    ]

    diff = np.inf
    for it in range(1, max_iter + 1):
        Pi_new: list[np.ndarray] = []
        for k in range(K):
            F = params.f_matrix.F(k)
            S_W = params.noise_cov.Sigma_W(k)
            acc = sum(p_rev[j, k] * Pi[j] for j in range(K))
            Pi_new.append(0.5 * ((F @ acc @ F.T + S_W) + (F @ acc @ F.T + S_W).T))
        diff = max(float(np.abs(Pi_new[k] - Pi[k]).max()) for k in range(K))
        Pi = Pi_new
        if diff < tol:
            return Pi, it, diff
    return Pi, max_iter, diff


def method_a_moments(
    params: GSSParams,
    Pi: list[np.ndarray],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Apply (21)-(22) to the stationary uncentred second moment Π(k).

        M(k)   = Π_XY(k) · Π_YY(k)⁻¹                                  (21)
        Γ(k)   = Π_XX(k) − Π_XY(k) · Π_YY(k)⁻¹ · Π_YX(k)              (22)

    Returns
    -------
    M_a : list of K arrays (q, s)   — regression coefficient (mean = M y).
    G_a : list of K arrays (q, q)   — conditional covariance.
    """
    K, q = params.K, params.q
    M_a, G_a = [], []
    for k in range(K):
        Pi_XX = Pi[k][:q, :q]
        Pi_XY = Pi[k][:q, q:]
        Pi_YY = Pi[k][q:, q:]
        # M = Pi_XY · Pi_YY⁻¹  (solved as Pi_YY^T x^T = Pi_XY^T)
        M = np.linalg.solve(Pi_YY.T, Pi_XY.T).T
        G = Pi_XX - M @ Pi_YY @ M.T
        # Symmetrise for numerical hygiene
        G_a.append(0.5 * (G + G.T))
        M_a.append(M)
    return M_a, G_a


# ---------------------------------------------------------------------------
# Method (b) — Frédéric's closed-form formula
# ---------------------------------------------------------------------------
def method_b_moments(
    params: GSSParams,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Direct (H5)-AB conditional moments (no recursion, no propagation):

        M(k) = Δ(k) Σ_V(k)⁻¹
        Γ(k) = Σ_U(k) − Δ(k) Σ_V(k)⁻¹ Δ(k)ᵀ
    """
    K = params.K
    M_b, G_b = [], []
    for k in range(K):
        Dt = params.noise_cov.Delta(k)
        SU = params.noise_cov.Sigma_U(k)
        SV = params.noise_cov.Sigma_V(k)
        SV_inv = np.linalg.inv(SV)
        M = Dt @ SV_inv
        G = SU - M @ Dt.T
        M_b.append(M)
        G_b.append(0.5 * (G + G.T))
    return M_b, G_b


# ---------------------------------------------------------------------------
# Comparison driver
# ---------------------------------------------------------------------------
def compare_one_model(
    params: GSSParams,
    N_traj: int = 0,
    seed: int = 0,
    verbose: bool = False,
) -> dict[str, float]:
    """
    Compute methods (a) and (b) for `params`, return summary statistics
    over the K regimes, plus optional trajectory check.
    """
    Pi, n_iter, final_diff = stationary_second_moments(params)
    M_a, G_a = method_a_moments(params, Pi)
    M_b, G_b = method_b_moments(params)

    K, q = params.K, params.q
    dM = [float(np.linalg.norm(M_a[k] - M_b[k], "fro")) for k in range(K)]
    dG = [float(np.linalg.norm(G_a[k] - G_b[k], "fro")) for k in range(K)]

    # Stationary mean check: with zero biases, μ_Z(k) = 0 should hold.
    # We did not propagate the means explicitly because they are zero at
    # the fixed point; verify it via a few iterations of the mean recursion.
    pi_inf = params.stationary_distribution()
    joint = pi_inf[:, None] * params.P
    marg = joint.sum(axis=0)
    p_rev = joint / np.where(marg > 0.0, marg, 1.0)[None, :]
    mu = [params.mu_z0(k).copy() for k in range(K)]
    for _ in range(1000):
        mu_new = [
            params.f_matrix.F(k) @ sum(p_rev[j, k] * mu[j] for j in range(K))
            for k in range(K)
        ]
        if max(float(np.abs(mu_new[k] - mu[k]).max()) for k in range(K)) < 1e-15:
            mu = mu_new
            break
        mu = mu_new
    max_mu = max(float(np.abs(mu[k]).max()) for k in range(K))

    if verbose:
        print(f"  Recursion (17): converged in {n_iter} iter "
              f"(final Δ = {final_diff:.2e})")
        print(f"  Stationary mean ‖μ_Z(k)‖∞      = {max_mu:.2e}  "
              f"(expected 0 with zero biases)")
        for k in range(K):
            print(f"    k={k}  ‖M_a − M_b‖_F = {dM[k]:.3e}   "
                  f"‖Γ_a − Γ_b‖_F = {dG[k]:.3e}")

    out = {
        "max_dM": max(dM),
        "max_dG": max(dG),
        "max_mu": max_mu,
        "n_iter": float(n_iter),
        "final_diff": final_diff,
    }

    # Trajectory check: simulate, evaluate E[X|r,y] via both methods, compare.
    if N_traj > 0:
        sim = GSSSimulator(params, N=N_traj, seed=seed)
        max_diff_Ex = 0.0
        max_diff_Var = 0.0
        for _, r, _x, y in sim:
            r = int(r)
            # Method (a): with zero stationary mean, E[X|r,y] = M_a · y
            Ex_a = M_a[r] @ y
            Ex_b = M_b[r] @ y
            max_diff_Ex = max(max_diff_Ex, float(np.abs(Ex_a - Ex_b).max()))
            max_diff_Var = max(max_diff_Var, float(np.abs(G_a[r] - G_b[r]).max()))
        out["max_diff_Ex_traj"] = max_diff_Ex
        out["max_diff_Var_traj"] = max_diff_Var
        if verbose:
            print(f"  Trajectory (N={N_traj}): "
                  f"max |E_a − E_b|∞ = {max_diff_Ex:.3e}, "
                  f"max |Γ_a − Γ_b|∞ = {max_diff_Var:.3e}")

    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Verify equivalence of methods (a) and (b) for "
                    "computing the (H5) conditional moments (19)-(20) "
                    "under the AB constraint.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("-K", type=int, default=3, help="Regimes (default 3).")
    ap.add_argument("-q", type=int, default=2, help="Dim of X (default 2).")
    ap.add_argument("-s", type=int, default=2, help="Dim of Y (default 2).")
    ap.add_argument("-n", "--n-draws", type=int, default=20,
                    help="Independent random AB-constrained models "
                         "(default 20). Ignored if --model is set.")
    ap.add_argument("--model", type=str, default=None,
                    help="Name of a model in prg/models/ (e.g. "
                         "'model_gss_K2_q2_s2'). Biases are reset to zero "
                         "and the AB constraint is applied before testing.")
    ap.add_argument("-N", "--n-traj", type=int, default=0,
                    help="Simulate N steps and compare E[X|r,y] on actual y "
                         "values (0 → skip, default 0).")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed (default 0).")
    ap.add_argument("--tol", type=float, default=1e-9,
                    help="Pass tolerance on max ‖M_a−M_b‖_F and ‖Γ_a−Γ_b‖_F "
                         "(default 1e-9).")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="Print per-trial and per-regime details.")
    ap.add_argument("--no-constraint", action="store_true",
                    help="Negative control: draw A, B independently of "
                         "(C, D, Δ, Σ_V). Method (b) should then DIFFER "
                         "from method (a) by Ω(1).")
    args = ap.parse_args()

    print("=" * 72)
    print("Verify (19)-(20) — Wojciech's (16)-(17)+(21)-(22)  vs  Frédéric's")
    print("                   direct closed-form, under the (H5) AB constraint.")
    print("=" * 72)

    rng = np.random.default_rng(args.seed)

    if args.model is not None:
        print(f"Loading model: prg.models.{args.model}  "
              f"(biases zeroed, AB constraint applied)\n")
        params = load_named_model(args.model)
        params.summary() if args.verbose else None
        results = [compare_one_model(params, N_traj=args.n_traj,
                                     seed=args.seed, verbose=True)]
    else:
        constraint_str = ("AB constraint" if not args.no_constraint
                          else "NO constraint  [negative control]")
        print(f"Random models — K={args.K}, q={args.q}, s={args.s}, "
              f"n_draws={args.n_draws}, seed={args.seed}  ({constraint_str})\n")
        results = []
        for trial in range(args.n_draws):
            verbose_trial = args.verbose and (trial == 0)
            if verbose_trial:
                print(f"--- Trial 0 ---")
            params = make_random_AB_params(
                args.K, args.q, args.s, rng,
                ab_constraint=not args.no_constraint,
            )
            results.append(compare_one_model(
                params, N_traj=args.n_traj, seed=args.seed + trial,
                verbose=verbose_trial,
            ))
            if verbose_trial:
                print()

    # Aggregate
    dM = np.array([r["max_dM"] for r in results])
    dG = np.array([r["max_dG"] for r in results])
    mu = np.array([r["max_mu"] for r in results])
    n_iter = np.array([r["n_iter"] for r in results])

    print(f"\n{'Quantity':<48} {'min':>10} {'median':>10} {'max':>10}")
    print("-" * 80)
    print(f"{'‖M_a − M_b‖_F  (regression coeff, eq. 21)':<48} "
          f"{dM.min():10.3e} {np.median(dM):10.3e} {dM.max():10.3e}")
    print(f"{'‖Γ_a − Γ_b‖_F  (cond. covariance, eq. 22)':<48} "
          f"{dG.min():10.3e} {np.median(dG):10.3e} {dG.max():10.3e}")
    print(f"{'‖μ_Z(k)‖∞  (zero-bias stationary mean)':<48} "
          f"{mu.min():10.3e} {np.median(mu):10.3e} {mu.max():10.3e}")
    print(f"{'recursion (17) iterations to fixed point':<48} "
          f"{int(n_iter.min()):>10d} {int(np.median(n_iter)):>10d} "
          f"{int(n_iter.max()):>10d}")

    if args.n_traj > 0:
        dE = np.array([r["max_diff_Ex_traj"] for r in results])
        dV = np.array([r["max_diff_Var_traj"] for r in results])
        print(f"{'|E_a[X|r,y] − E_b[X|r,y]|∞   (sim trajectory)':<48} "
              f"{dE.min():10.3e} {np.median(dE):10.3e} {dE.max():10.3e}")
        print(f"{'|Γ_a − Γ_b|∞                  (sim trajectory)':<48} "
              f"{dV.min():10.3e} {np.median(dV):10.3e} {dV.max():10.3e}")

    print()

    if args.no_constraint:
        # Negative control: expect Ω(1) disagreement.
        pass_neg = dM.min() > args.tol
        if pass_neg:
            print(f"OK  : without the AB constraint, methods (a) and (b) "
                  f"differ by {dM.min():.2e} … {dM.max():.2e} "
                  f"(> tol = {args.tol:.0e}).")
            print(f"      → The closed-form (b) requires (H5) AB to hold.")
            return 0
        print(f"WARN: without AB, min ‖M_a − M_b‖_F = {dM.min():.3e}  "
              f"(expected > {args.tol:.0e}).")
        return 1

    pass_M = dM.max() < args.tol
    pass_G = dG.max() < args.tol
    if pass_M and pass_G:
        print(f"OK  : methods (a) and (b) agree to within {args.tol:.0e} for "
              f"every trial.")
        print(f"      → Under the AB constraint, the moments (19)-(20) can be")
        print(f"        computed directly from (Δ, Σ_U, Σ_V) without the "
              f"recursion (16)-(17).")
        return 0
    print(f"FAIL: max ‖M_a − M_b‖_F = {dM.max():.3e}  (tol = {args.tol:.0e})")
    print(f"      max ‖Γ_a − Γ_b‖_F = {dG.max():.3e}  (tol = {args.tol:.0e})")
    return 1


if __name__ == "__main__":
    sys.exit(main())
