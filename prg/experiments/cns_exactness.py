"""Machine-precision checks of the IMM/GPB2 exactness domains (TSP paper).

Ten experiments back the numbers quoted in the exactness-domains paper
(docs/CNS-exactness):

E1  IMM realizes the CGO-MSM exact filter: on a genuine CGO-MSM (C=0, state
    memory A_r != 0, regimes identifiable), IMM and GPB2 match the exact
    mixture filter to ~1e-15 -- mean, covariance and regime posterior.
E2  Slaving-(A)-only models (A=MC, C!=0, B!=MD): GPB2 stays exact (~1e-15)
    while the order-1 IMM drifts, with an error growing with the coupling C.
E3  Outside {C==0} U {A==MC} (perturb A = MC + eta at C!=0): GPB2 itself is
    biased, with an error growing as roughly eta^3.
E4  Time profiles on fast-mixing gauges: off-domain errors settle after a
    short transient (stationary/transient ratios < 1) -- the degradation is
    uniform in time, not accumulating.
E5  Mixed-branch models (paper indices: regime 1 has C=0 with memory A_1,
    regime 2 has C_2 != 0 with A_2 = M_2 C_2; the code uses the 0-indexed
    names A0/C1 for the same blocks): each regime passes the per-regime
    disjunction, yet GPB2 is biased as soon as the memory is nonzero -- the
    exactness domain is the union of the two *uniform* families.
E6  The constant-gain (NGH-MSM) filter: exact on AB, loses half the signal
    one step off AB (B = MD + 0.15) while GPB2 stays at machine precision.
E7  Beyond the scalar two-regime case: a q=2 instance where A=MC genuinely
    constrains the rank, and a K=3 mixed-branch instance of the uniformity
    requirement.
E8  Sharpness of the non-degeneracy assumption (G2): regime-free observation
    row + i.i.d. regime -> the IMM is exact although C != 0.
E9  (G3) and the initialization quantifier: a model outside the union whose
    second-order blocks (A, C, SU, Delta, SV) do not switch. Under a
    regime-common initialization every per-history covariance is equal and
    the gain step of the necessity proof is empty -- GPB2 is nevertheless
    measurably biased (the likelihood step carries a constraint of its own);
    with regime-dependent initial covariances (Lemma 'Reachable covariances'
    of the paper) the covariances spread and the bias grows.
E10 The matrix case (q=2, s=1): cross-annihilation C_k (A_j - M_j C_j) = 0
    for all regime pairs, with every memory A_j - M_j C_j singular and
    nonzero and every C_k nonzero -- outside {C==0} U {A==MC}, yet GPB2 is
    exact (the memory lives in the common kernel of the observation rows);
    rotating the observation rows out of that kernel (same memories)
    restores a measurable bias. The GPB2 domain is the cross-annihilation
    family, which reduces to the union of the two uniform families at q=1.

Numbering follows the order of the paper's Section VI (E1-E7), Remark
'(G2) is sharp' (E8) and Remark '(G3) and the initialization quantifier' (E9); before 2026-09-03 the time profiles were E8 and
E4-E7 were shifted down by one.

Ground truth is the exact K^N mixture filter on short horizons; errors are
normalized sup-norm deviations of the state mean, state variance and regime
posterior (median [min,max] over seeds off the domains, max on them).  All
models are stable (rho(F_r) < 1, asserted).

Run:  python -m prg.experiments.cns_exactness
"""

from __future__ import annotations

import numpy as np

from prg.classes.FMatrix import FMatrix
from prg.classes.GSSParams import GSSParams
from prg.classes.NoiseCovariance import GSSNoiseCovariance
from prg.experiments.reference_filters import (
    exact_mixture_filter,
    gpb2_filter,
    imm_filter,
)
from prg.experiments.study import _simulate, with_stationary_init

N_STEPS = 10  # exact filter cost is K^N -- keep short
N_SEEDS = 100
SEED0 = 100


# ---------------------------------------------------------------------------
# model builders (K=2, q=s=1 throughout; scalar blocks wrapped as 1x1)
# ---------------------------------------------------------------------------
def _params(A, B, C, D, SU, Dt, SV, p_switch):
    """Assemble a K=2, q=s=1 GSSParams from per-regime scalar block lists."""
    K, q, s = 2, 1, 1
    P = np.array([[1 - p_switch, p_switch], [p_switch, 1 - p_switch]])
    as_mat = lambda v: [np.array([[x]], dtype=float) for x in v]
    fm = FMatrix(K, q, s, as_mat(A), as_mat(B), as_mat(C), as_mat(D))
    nc = GSSNoiseCovariance(K, q, s, as_mat(SU), as_mat(Dt), as_mat(SV))
    p = GSSParams(
        K=K, q=q, s=s, P=P, f_matrix=fm, noise_cov=nc, pi0=None,
        mu_z0_list=[np.zeros((q + s, 1)) for _ in range(K)],
        Sigma_z0_list=[np.eye(q + s) for _ in range(K)],
    )
    rho = max(max(abs(np.linalg.eigvals(fm.F(k)))) for k in range(K))
    assert rho < 1.0, f"unstable model (rho={rho:.3f}); comparison meaningless"
    return with_stationary_init(p)


def cgo_memory_model():
    """E1: genuine CGO-MSM -- C=0, state memory A_r != 0 (so outside AB),
    regimes identifiable through the observation volatility (SV 0.20/0.60).

    The volatility contrast matters: with SV=0.30/0.40 the exact regime
    posterior barely leaves its prior (MAP hit-rate ~0.51, i.e. chance), and
    the posterior comparison would be close to vacuous.
    """
    return _params(
        A=[0.7, 0.4], B=[0.10, 0.10], C=[0.0, 0.0], D=[0.50, 0.50],
        SU=[0.40, 0.35], Dt=[0.15, -0.20], SV=[0.20, 0.60], p_switch=0.10,
    )


def slaving_A_model(C, dB=0.15, p_switch=0.02):
    """E2: slaving (A) at both regimes (A=MC), coupling C swept, volet B
    broken (B = MD + dB) so the model is *only* in {A==MC}; regimes
    identifiable (SV 0.20/0.60, slaved gains M = +0.6/-0.5)."""
    SV, M = [0.20, 0.60], [0.6, -0.5]
    D = [0.50, 0.50]
    Dt = [M[k] * SV[k] for k in range(2)]
    A = [M[k] * C for k in range(2)]           # condition (A)
    B = [M[k] * D[k] + dB for k in range(2)]   # condition (B) broken
    return _params(A=A, B=B, C=[C, C], D=D, SU=[0.25, 0.30], Dt=Dt, SV=SV,
                   p_switch=p_switch)


def off_union_model(C, eps, p_switch=0.02):
    """E3: outside the union -- start from slaving_A_model(C) and perturb the
    state dynamics, A = MC + eps, so that C != 0 *and* A != MC."""
    SV, M = [0.20, 0.60], [0.6, -0.5]
    D = [0.50, 0.50]
    Dt = [M[k] * SV[k] for k in range(2)]
    A = [M[k] * C + eps for k in range(2)]     # violates (A) by eps
    B = [M[k] * D[k] for k in range(2)]
    return _params(A=A, B=B, C=[C, C], D=D, SU=[0.25, 0.30], Dt=Dt, SV=SV,
                   p_switch=p_switch)


def mixed_branch_model(A0, C1):
    """E5: regime 0 satisfies C=0 (with memory A0, so (A) fails there as soon
    as A0 != 0); regime 1 satisfies (A) with an active channel (C1 != 0,
    A1 = M1*C1, volet B broken). Per-regime disjunction holds for any A0."""
    SV, M = [0.30, 0.40], [0.50, -0.50]
    D = [0.50, 0.50]
    Dt = [M[k] * SV[k] for k in range(2)]
    A1 = M[1] * C1
    B1 = M[1] * D[1] + 0.20
    return _params(A=[A0, A1], B=[0.10, B1], C=[0.0, C1], D=D,
                   SU=[0.40, 0.35], Dt=Dt, SV=SV, p_switch=0.10)


# ---------------------------------------------------------------------------
# error metric
# ---------------------------------------------------------------------------
def ab_model(C, dB=0.0):
    """E6: same blocks as slaving_A_model but with condition (B) controlled by
    dB: dB=0 gives a full-AB model (the constant-gain filter is exact there),
    dB!=0 leaves the model in {A==MC} but outside AB."""
    return slaving_A_model(C, dB=dB)


def matrix_slaving_model(C_row=(0.5, 0.2), p_switch=0.02):
    """E2-bis: q=2, s=1 instance of {A==MC}. Here A_r = M_r C_r is a genuine
    structural constraint (rank(A_r) <= s = 1), which the scalar case cannot
    impose: at q=s=1 any (A,C) with C!=0 satisfies A=MC for M=A/C."""
    from prg.classes.FMatrix import FMatrix as _FM
    from prg.classes.NoiseCovariance import GSSNoiseCovariance as _NC

    K, q, s = 2, 2, 1
    P = np.array([[1 - p_switch, p_switch], [p_switch, 1 - p_switch]])
    SV = [np.array([[0.20]]), np.array([[0.60]])]
    M = [np.array([[0.6], [-0.3]]), np.array([[-0.5], [0.4]])]  # q x s
    D = [np.array([[0.50]]), np.array([[0.50]])]
    C = [np.array([list(C_row)]), np.array([list(C_row)])]      # s x q
    A = [M[k] @ C[k] for k in range(K)]                          # rank <= s
    B = [M[k] @ D[k] + 0.15 for k in range(K)]                   # (B) broken
    Dt = [M[k] @ SV[k] for k in range(K)]                        # q x s
    Gam = [np.diag([0.25, 0.30]), np.diag([0.30, 0.20])]
    SU = [Gam[k] + M[k] @ SV[k] @ M[k].T for k in range(K)]
    fm = _FM(K, q, s, A, B, C, D)
    nc = _NC(K, q, s, SU, Dt, SV)
    p = GSSParams(
        K=K, q=q, s=s, P=P, f_matrix=fm, noise_cov=nc, pi0=None,
        mu_z0_list=[np.zeros((q + s, 1)) for _ in range(K)],
        Sigma_z0_list=[np.eye(q + s) for _ in range(K)],
    )
    rho = max(max(abs(np.linalg.eigvals(fm.F(k)))) for k in range(K))
    assert rho < 1.0, f"unstable model (rho={rho:.3f})"
    return with_stationary_init(p)


def three_regime_mixed_model(A0, C1):
    """E5-bis: K=3 mixed-branch model -- regime 0 has C=0 with memory A0, the
    other two satisfy A_r = M_r C_r with C_r != 0. Shows the uniformity
    requirement is not a two-regime artifact."""
    from prg.classes.FMatrix import FMatrix as _FM
    from prg.classes.NoiseCovariance import GSSNoiseCovariance as _NC

    K, q, s = 3, 1, 1
    q_sw = 0.10
    P = np.full((K, K), q_sw / (K - 1))
    np.fill_diagonal(P, 1.0 - q_sw)
    SV = [0.30, 0.40, 0.25]
    M = [0.50, -0.50, 0.35]
    D = [0.50, 0.50, 0.50]
    C = [0.0, C1, 0.6 * C1]
    A = [A0] + [M[k] * C[k] for k in (1, 2)]
    B = [0.10, M[1] * D[1] + 0.20, M[2] * D[2] + 0.20]
    Dt = [M[k] * SV[k] for k in range(K)]
    SU = [0.40, 0.35, 0.30]
    m = lambda v: [np.array([[x]], float) for x in v]
    fm = _FM(K, q, s, m(A), m(B), m(C), m(D))
    nc = _NC(K, q, s, m(SU), m(Dt), m(SV))
    p = GSSParams(
        K=K, q=q, s=s, P=P, f_matrix=fm, noise_cov=nc, pi0=None,
        mu_z0_list=[np.zeros((q + s, 1)) for _ in range(K)],
        Sigma_z0_list=[np.eye(q + s) for _ in range(K)],
    )
    rho = max(max(abs(np.linalg.eigvals(fm.F(k)))) for k in range(K))
    assert rho < 1.0, f"unstable model (rho={rho:.3f})"
    return with_stationary_init(p)


def g2_degenerate_model(C=0.5):
    """E8 (sharpness of (G2), Remark 'Assumption (G2) is sharp'): the whole
    observation row is regime-free (common C, D, b^Y, SV) and the regime is
    i.i.d. (identical transition rows), so the one-step predictive law of Y
    does not depend on the arrival regime -- (G2) fails.  Condition (A) holds
    at both regimes (A_r = M_r C), so the state gain is component-independent;
    the IMM is then exact with C != 0, showing the 'only if' of Theorem 1
    cannot survive without (G2)."""
    SV = [0.30, 0.30]                    # common observation noise
    M = [0.6, -0.5]
    D = [0.50, 0.50]                     # common D
    Dt = [M[k] * SV[k] for k in range(2)]
    A = [M[k] * C for k in range(2)]     # condition (A) at both regimes
    B = [0.10, 0.40]                     # state rows DO differ across regimes
    p = _params(A=A, B=B, C=[C, C], D=D, SU=[0.25, 0.30], Dt=Dt, SV=SV,
                p_switch=0.5)            # identical rows: i.i.d. regime
    return p


def constant_second_order_model(p_switch=0.10):
    """E9: outside the union (C=0.4, A=0.5 != M*C = 0.4/3) with *constant*
    second-order blocks A, C, SU, Delta, SV -- only B and D switch (the regime
    is identifiable through D). Under a regime-common initialization every
    per-history covariance is the same and the gain step of the necessity
    proof is empty; GPB2 is biased nonetheless (likelihood step)."""
    return _params(A=[0.5, 0.5], B=[0.10, 0.30], C=[0.4, 0.4], D=[0.2, 0.6],
                   SU=[0.30, 0.30], Dt=[0.10, 0.10], SV=[0.30, 0.30],
                   p_switch=p_switch)


def cross_annihilation_model(annihilate=True, p_switch=0.10):
    """E10: q=2, s=1, K=2. Memories N_j = A_j - M_j C_j = e_2 v_j^T (rank 1,
    nonzero) at both regimes. With ``annihilate=True`` the observation rows
    are C_k = [c_k, 0], so C_k N_j = 0 for every pair (j, k): the model is
    outside {C==0} U {A==MC} but inside the cross-annihilation family. With
    ``annihilate=False`` the rows become C_k = [c_k, 0.3] (same memories):
    C_k N_j != 0 and GPB2 is biased."""
    from prg.classes.FMatrix import FMatrix as _FM
    from prg.classes.NoiseCovariance import GSSNoiseCovariance as _NC

    K, q, s = 2, 2, 1
    P = np.array([[1 - p_switch, p_switch], [p_switch, 1 - p_switch]])
    SV = [np.array([[0.20]]), np.array([[0.60]])]
    M = [np.array([[0.6], [-0.3]]), np.array([[-0.5], [0.4]])]
    D = [np.array([[0.50]]), np.array([[0.50]])]
    c2 = 0.0 if annihilate else 0.3
    C = [np.array([[0.5, c2]]), np.array([[0.3, c2]])]
    N = [np.array([[0.0, 0.0], [0.4, 0.5]]), np.array([[0.0, 0.0], [-0.3, 0.6]])]
    A = [M[k] @ C[k] + N[k] for k in range(K)]
    B = [M[k] @ D[k] + 0.15 for k in range(K)]
    Dt = [M[k] @ SV[k] for k in range(K)]
    Gam = [np.diag([0.25, 0.30]), np.diag([0.30, 0.20])]
    SU = [Gam[k] + M[k] @ SV[k] @ M[k].T for k in range(K)]
    fm = _FM(K, q, s, A, B, C, D)
    nc = _NC(K, q, s, SU, Dt, SV)
    p = GSSParams(
        K=K, q=q, s=s, P=P, f_matrix=fm, noise_cov=nc, pi0=None,
        mu_z0_list=[np.zeros((q + s, 1)) for _ in range(K)],
        Sigma_z0_list=[np.eye(q + s) for _ in range(K)],
    )
    rho = max(max(abs(np.linalg.eigvals(fm.F(k)))) for k in range(K))
    assert rho < 1.0, f"unstable model (rho={rho:.3f})"
    return with_stationary_init(p)


def _with_init(params, mu_list, Sigma_list):
    """Copy of ``params`` with the given per-regime initial laws for Z_1."""
    K = params.K
    return GSSParams(
        K=K, q=params.q, s=params.s, P=params.P, f_matrix=params.f_matrix,
        noise_cov=params.noise_cov, pi0=params.pi0,
        mu_z0_list=[np.array(m, dtype=float) for m in mu_list],
        Sigma_z0_list=[0.5 * (np.array(S) + np.array(S).T) for S in Sigma_list],
        b_list=[params.b(k) for k in range(K)],
        G_list=[params.G(k) for k in range(K)] if params.p > 0 else None,
    )


# ---------------------------------------------------------------------------
# error metric
# ---------------------------------------------------------------------------
def _rel(a, b):
    """Normalized sup-norm deviation, eq. (metric) of the paper."""
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a - b)) / (np.max(np.abs(b)) + 1e-12))


def _run_cg(params, ys):
    """Constant-gain (NGH-MSM) filter, the third filter of the map."""
    from prg.experiments.study import _run
    ex, pi, var = _run(params, ys, "ngh_kf")
    return ex, var, pi


class Gap:
    """Per-seed gaps for one (filter, quantity), summarized as median [min, max].

    The maximum alone is an order statistic on a small sample and grows with
    the number of runs; the median with its observed range is what a reader
    can reproduce. On the exactness domains all three coincide at machine
    precision, and we then quote the maximum as the conservative reading.
    """

    __slots__ = ("v",)

    def __init__(self, values):
        self.v = np.sort(np.asarray(values, dtype=float))

    @property
    def med(self):
        return float(np.median(self.v))

    @property
    def lo(self):
        return float(self.v[0])

    @property
    def hi(self):
        return float(self.v[-1])

    def __format__(self, spec):
        # machine precision: one number is enough
        if self.hi < 1e-10:
            return f"{self.hi:.1e}"
        return f"{self.med:.1e} [{self.lo:.0e},{self.hi:.0e}]"

    def __str__(self):
        return format(self, "")


def gaps(params, n_steps=N_STEPS, with_cg=False, n_seeds=N_SEEDS):
    """Normalized sup-norm deviations to the exact mixture filter, per seed.

    Returns {filter: (Gap_mean, Gap_var, Gap_posterior)}. The constant-gain
    filter is included only on request: its gains are the fixed point of a
    moment recursion, so its floor is ~1e-12 rather than ~1e-15.
    """
    names = ["imm", "gpb2"] + (["cg"] if with_cg else [])
    acc = {k: [[], [], []] for k in names}
    for sd in range(n_seeds):
        _, _, ys = _simulate(params, n_steps, seed=SEED0 + sd)
        ex_e, v_e, pi_e = exact_mixture_filter(params, ys)
        runs = {"imm": lambda: imm_filter(params, ys)[:3],
                "gpb2": lambda: gpb2_filter(params, ys)[:3],
                "cg": lambda: _run_cg(params, ys)}
        for name in names:
            ex, v, pi = runs[name]()
            for i, g in enumerate((_rel(ex, ex_e), _rel(v, v_e),
                                   _rel(pi, pi_e))):
                acc[name][i].append(g)
    return {k: tuple(Gap(c) for c in cols) for k, cols in acc.items()}


# ---------------------------------------------------------------------------
# experiments
# ---------------------------------------------------------------------------
def exp1_imm_realizes_cgo():
    print("E1  CGO-MSM with state memory (C=0, A=(0.7,0.4)):")
    g = gaps(cgo_memory_model())
    for name in ("imm", "gpb2"):
        e, v, p = g[name]
        print(f"    {name:4s}  E_x {e}   Var {v}   posterior {p}")
    return g


def exp2_slaving_A(Cs=(0.0, 0.2, 0.4, 0.55)):
    print("E2  slaving (A) only (A=MC, B!=MD), coupling C swept:")
    out = {}
    for C in Cs:
        g = gaps(slaving_A_model(C))
        out[C] = g
        print(f"    C={C:<5} IMM  E_x {g['imm'][0]:<22} post {g['imm'][2]:<22}"
              f"| GPB2 E_x {g['gpb2'][0]:<12} post {g['gpb2'][2]}")
    return out


def exp3_off_union(C=0.4, epss=(0.05, 0.2, 0.5)):
    print(f"E3  outside the union (C={C}, A=MC+eps):")
    out = {}
    for eps in epss:
        g = gaps(off_union_model(C, eps))
        out[eps] = g
        print(f"    eta={eps:<5} GPB2 E_x {g['gpb2'][0]:<22} post {g['gpb2'][2]:<22}"
              f"| IMM E_x {g['imm'][0]}")
    return out


def exp5_mixed_branch(A0s=(0.0, 0.4, 0.8), C1s=(0.3, 0.7)):
    print("E5  mixed branches (r0: C=0 memory A0; r1: C1!=0, A1=M1*C1):")
    out = {}
    for A0 in A0s:
        for C1 in C1s:
            g = gaps(mixed_branch_model(A0, C1))
            out[(A0, C1)] = g
            print(f"    A0={A0:<4} C1={C1:<4} GPB2 E_x {g['gpb2'][0]:<22}"
                  f" post {g['gpb2'][2]:<22}| IMM E_x {g['imm'][0]}")
    return out


def exp6_constant_gain(Cs=(0.4, 0.55)):
    """The third filter of the map: exact on AB, biased on {A==MC}\\AB."""
    print("E6  constant-gain filter (AB row: dB=0; off-AB row: dB=0.15):")
    out = {}
    for C in Cs:
        for dB in (0.0, 0.15):
            g = gaps(ab_model(C, dB=dB), with_cg=True)
            out[(C, dB)] = g
            tag = "AB    " if dB == 0.0 else "{A=MC}"
            print(f"    C={C:<5} {tag} CG {g['cg'][0]:<22}"
                  f"| GPB2 {g['gpb2'][0]:<12} | IMM {g['imm'][0]}")
    return out


def exp7_beyond_scalar():
    """Dimension and regime-count generalizations of E2 and E5."""
    print("E7  beyond the scalar two-regime case:")
    g = gaps(matrix_slaving_model())
    print(f"    q=2,s=1 {{A=MC}} (rank(A)<=s)  GPB2 {g['gpb2'][0]:<12}"
          f"| IMM {g['imm'][0]}")
    out = {"q2": g}
    for A0 in (0.0, 0.8):
        g3 = gaps(three_regime_mixed_model(A0, 0.7), n_steps=8)
        out[f"K3_A0={A0}"] = g3
        print(f"    K=3,N=8 mixed A0={A0:<4}        GPB2 {g3['gpb2'][0]:<22}"
              f"| IMM {g3['imm'][0]}")
    return out


def exp8_g2_sharpness():
    """Sharpness of (G2): with the observation row regime-free and an i.i.d.
    regime, the IMM is exact although C != 0 (Remark 'G2 is sharp')."""
    print("E8  sharpness of (G2) (regime-free observation row, i.i.d. regime,"
          " C=0.5):")
    g = gaps(g2_degenerate_model())
    print(f"    IMM  E_x {g['imm'][0]}   post {g['imm'][2]}"
          f"   (GPB2 E_x {g['gpb2'][0]})")
    return g


def exp4_time_profile(n_steps=13, n_seeds=25, p_fast=0.2):
    """E4: per-time error profiles on fast-mixing gauges -- does the error
    accumulate over time, or settle after the transient?  Reported: the ratio
    of the second-half to first-half median error levels (ratio < 1 means the
    stationary level sits below the transient peak: no accumulation)."""
    from prg.experiments.study import _run

    def profile(params, runner):
        prof = []
        for sd in range(n_seeds):
            _, _, ys = _simulate(params, n_steps, seed=SEED0 + sd)
            ex_e, _, _ = exact_mixture_filter(params, ys)
            ex = runner(params, ys)
            sc = np.max(np.abs(ex_e)) + 1e-12
            prof.append(np.max(np.abs(ex - ex_e), axis=1) / sc)
        med = np.median(np.array(prof), axis=0)
        h1 = float(np.median(med[3:8])); h2 = float(np.median(med[8:]))
        return h1, h2, h2 / h1

    r_gpb2 = lambda p, ys: gpb2_filter(p, ys)[0]
    r_imm = lambda p, ys: imm_filter(p, ys)[0]
    r_cg = lambda p, ys: _run(p, ys, "ngh_kf")[0]

    print(f"E4  time profiles (fast mixing p={p_fast}, N={n_steps}):"
          " stationary/transient ratios")
    out = {}
    for eta in (0.1, 0.3):
        p = off_union_model(0.4, eta, p_switch=p_fast)
        for name, r in (("GPB2", r_gpb2), ("IMM", r_imm)):
            h1, h2, ratio = profile(p, r)
            out[(name, eta)] = ratio
            print(f"    {name:<4} eta={eta:<4} transient {h1:.2e}"
                  f"  stationary {h2:.2e}  ratio {ratio:.2f}")
    for dB in (0.1, 0.3):
        p = slaving_A_model(0.4, dB=dB, p_switch=p_fast)
        h1, h2, ratio = profile(p, r_cg)
        out[("CG", dB)] = ratio
        print(f"    CG   dB={dB:<4} transient {h1:.2e}"
              f"  stationary {h2:.2e}  ratio {ratio:.2f}")
    return out


def exp9_g3_initialization():
    """E9: (G3) versus the quantifier over admissible initializations.
    (a) regime-common initial law (stationary moments averaged over regimes):
        all per-history covariances equal, gain step empty, yet GPB2 is not
        exact (necessity carried by the likelihood step);
    (b) same means, initial covariances scaled by 0.5 / 2.0 per regime: the
        covariances spread (Lemma 'Reachable covariances'), the bias grows."""
    print("E9  (G3) and the initialization quantifier (constant 2nd-order"
          " blocks, C=0.4, A=0.5 != MC):")
    p = constant_second_order_model()
    K = p.K
    mu0 = sum(p.mu_z0(k) for k in range(K)) / K
    S0 = sum(p.Sigma_z0(k) for k in range(K)) / K
    out = {}
    pa = _with_init(p, [mu0] * K, [S0] * K)
    out["common"] = gaps(pa)
    print(f"    (a) common init, gain step empty  GPB2 E_x {out['common']['gpb2'][0]}"
          f"  post {out['common']['gpb2'][2]}| IMM E_x {out['common']['imm'][0]}")
    pb = _with_init(p, [mu0] * K, [0.5 * S0, 2.0 * S0])
    out["spread"] = gaps(pb)
    print(f"    (b) Sigma0(r) x0.5/x2, spread  GPB2 E_x {out['spread']['gpb2'][0]}"
          f"  post {out['spread']['gpb2'][2]}| IMM E_x {out['spread']['imm'][0]}")
    return out


def exp10_cross_annihilation():
    """E10: the matrix case -- GPB2 exact on the cross-annihilation family
    outside the union, biased once the observation rows leave the kernel of
    the memories."""
    print("E10 matrix case q=2: cross-annihilation C_k N_j = 0 (all memories"
          " singular, nonzero; C_k != 0):")
    out = {}
    for tag, ann in (("C_k N_j = 0 ", True), ("C_k N_j != 0", False)):
        g = gaps(cross_annihilation_model(annihilate=ann))
        out[ann] = g
        print(f"    {tag}  GPB2 E_x {g['gpb2'][0]:<22} var {g['gpb2'][1]:<22}"
              f" post {g['gpb2'][2]:<22}| IMM E_x {g['imm'][0]}")
    return out


def main():
    print(f"Exactness-domain checks -- ground truth: exact K^N mixture filter "
          f"(N={N_STEPS}, {N_SEEDS} seeds), normalized sup-norm gaps:\nmedian [min,max] off the domains; a single number = machine precision (max over all runs).\n")
    exp1_imm_realizes_cgo()
    print()
    exp2_slaving_A()
    print()
    exp3_off_union()
    print()
    exp4_time_profile()
    print()
    exp5_mixed_branch()
    print()
    exp6_constant_gain()
    print()
    exp7_beyond_scalar()
    print()
    exp8_g2_sharpness()
    print()
    exp9_g3_initialization()
    print()
    exp10_cross_annihilation()


if __name__ == "__main__":
    main()
