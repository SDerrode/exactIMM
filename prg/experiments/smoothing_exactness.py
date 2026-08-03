"""Machine-precision checks of the exact fixed-interval smoothers (smoothing paper).

Six experiments back the numbers quoted in the exact-smoothing paper
(docs/exact-smoothing):

S1  Smoothed regime posterior, exact and O(N K^2) on both families: the HMM
    forward-backward on (R,Y) -- kernel Q0 on {C==0}, kernel Q on AB --
    matches the exact smoothed regime posterior to ~1e-15.
S2  {C==0}: the reweighting smoother (pairwise-IMM filtered per-regime
    summaries recombined by the smoothed regime posterior) matches the exact
    smoothed mean AND variance to ~1e-15; the same formula under AB fails at
    O(1) -- the two branches need different smoothers.
S3  AB: the constant-gain smoother (affine read-out + one-step xi correction,
    precomputed gains W_jk, pair-smoothed weights) matches the exact smoothed
    mean and variance to ~1e-15, with no covariance recursion of any kind.
S4  The trap: the naive smoothed read-out sum_k gamma_n(k)(M_k y_n + c_k),
    even fed the EXACT smoothed regime posterior, misses the smoothed mean at
    the order of the entire smoothing gain |filtered - smoothed|.
S5  {A==MC} \\ AB: the lag-1 smoother (read-out M_j y_n + E_j y_{n-1} + c_j,
    same gains W_jk, forward-backward on the lag-1 (R,(Y_{n-1},Y_n)) chain)
    stays exact, mean and variance; on AB it reduces to S3.
S6  Off the union {C==0} u {A==MC}, the collapsed smoothed regime posterior
    (GPB2 forward + pair-likelihood backward) degrades at FIRST order in the
    violation eta (slope ~1.15) -- against third order for the filtered
    quantities (E3 of cns_exactness): smoothing pays structural
    misspecification at first order.
S7  The pair-RTS form and its gain degeneracy: the mode-matched RTS smoother
    run on the pair (GPB2 forward bank, backward mixing over the arrival
    regime with smoothed pair weights) has gains G^{jk} whose x-rows satisfy,
    identically, G_x == 0 and G_y == W_jk on {A==MC} (the backward state
    input is multiplied by zero: RTS collapses to the one-step correction),
    and G_y == -G_x M_k on {C==0} (the correction against pair-conditional
    backward means annihilates: RTS collapses to the reweighting).  The
    naive mode-matched RTS -- which feeds the k-conditional backward mean
    where the pair-conditional one is required -- is therefore exact on the
    whole slaved branch but BIASED on {C==0} with state memory.
S8  Off the union, the three formulations (pair-RTS, one-step correction,
    reweighting) genuinely separate; their pairwise separations vanish with
    the violation eta and are computable online (no ground truth): a
    measurable diagnostic of the structural violation.

Ground truth is the exact K^N path smoother: one conditional pair Kalman
filter + RTS smoother per regime path, the paths weighted by their exact
posterior p(path | y_{1:N}).  Errors are normalized sup-norm deviations,
reported as median and max over seeds.

Run:  python -m prg.experiments.smoothing_exactness
"""

from __future__ import annotations

from itertools import product

import numpy as np

from prg.experiments.cns_exactness import (
    N_STEPS,
    SEED0,
    _rel,
    ab_model,
    cgo_memory_model,
    off_union_model,
)
from prg.experiments.reference_filters import (
    _kalman_exact_y_update,
    _obs_matrix,
    exact_mixture_filter,
)
from prg.experiments.study import _simulate

N_SEEDS = 15  # exact path smoother costs K^N per seed -- keep moderate


# ---------------------------------------------------------------------------
# ground truth: exact K^N path smoother (mean, variance, regime posterior)
# ---------------------------------------------------------------------------
def exact_path_smoother(params, ys):
    """Exact fixed-interval smoother by full path enumeration.

    One conditional pair Kalman filter + RTS smoother per regime path,
    weighted by p(path | y_{1:N}).  Returns the smoothed state mean (N, q),
    state covariance (N, q, q) and regime posterior (N, K).
    """
    K, q, s = params.K, params.q, params.s
    dim = q + s
    ys = np.asarray(ys, float).reshape(-1, s)
    N = ys.shape[0]
    H = _obs_matrix(q, s)
    F = [params.f_matrix.F(k) for k in range(K)]
    b = [params.b(k).reshape(dim, 1) for k in range(K)]
    SW = [params.noise_cov.Sigma_W(k) for k in range(K)]
    mu0 = [params.mu_z0(k).reshape(dim, 1) for k in range(K)]
    S0 = [params.Sigma_z0(k) for k in range(K)]
    logws, store = [], []
    for path in product(range(K), repeat=N):
        k0 = path[0]
        z, P, ll = _kalman_exact_y_update(mu0[k0], S0[k0],
                                          ys[0].reshape(s, 1), H)
        lw = float(np.log(params.pi0[k0] + 1e-300) + ll)
        zf, Pf, zp, Pp = [z], [P], [None], [None]
        for n in range(1, N):
            k = path[n]
            zpr = F[k] @ zf[-1] + b[k]
            Ppr = F[k] @ Pf[-1] @ F[k].T + SW[k]
            z, P, ll = _kalman_exact_y_update(zpr, Ppr, ys[n].reshape(s, 1), H)
            lw += float(np.log(params.P[path[n - 1], k] + 1e-300)) + ll
            zf.append(z)
            Pf.append(P)
            zp.append(zpr)
            Pp.append(Ppr)
        # RTS backward pass along the (conditionally linear-Gaussian) pair
        zs = [None] * N
        Ps_ = [None] * N
        zs[N - 1], Ps_[N - 1] = zf[N - 1], Pf[N - 1]
        for n in range(N - 2, -1, -1):
            Fk = F[path[n + 1]]
            G = Pf[n] @ Fk.T @ np.linalg.inv(Pp[n + 1])
            zs[n] = zf[n] + G @ (zs[n + 1] - zp[n + 1])
            Ps_[n] = Pf[n] + G @ (Ps_[n + 1] - Pp[n + 1]) @ G.T
        logws.append(lw)
        store.append((path, zs, Ps_))
    logws = np.array(logws)
    w = np.exp(logws - logws.max())
    w /= w.sum()
    Ex = np.zeros((N, q))
    Vx = np.zeros((N, q, q))
    gam = np.zeros((N, K))
    for wi, (path, zs, _) in zip(w, store):
        for n in range(N):
            Ex[n] += wi * zs[n][:q, 0]
            gam[n, path[n]] += wi
    for wi, (_, zs, Ps_) in zip(w, store):
        for n in range(N):
            d = zs[n][:q, 0] - Ex[n]
            Vx[n] += wi * (Ps_[n][:q, :q] + np.outer(d, d))
    return Ex, Vx, gam


# ---------------------------------------------------------------------------
# shared machinery: slaving objects, kernels, forward-backward on (R,Y)
# ---------------------------------------------------------------------------
def _slaving(params):
    """Per-regime slaving objects (M_k, Gamma_k, c_k) from the blocks."""
    K, q, s = params.K, params.q, params.s
    M, Gam, c = [], [], []
    for k in range(K):
        SV = params.noise_cov.Sigma_V(k)
        Mk = params.noise_cov.Delta(k) @ np.linalg.inv(SV)
        bk = params.b(k).reshape(q + s, 1)
        M.append(Mk)
        Gam.append(params.noise_cov.Sigma_U(k) - Mk @ SV @ Mk.T)
        c.append(bk[:q] - Mk @ bk[q:])
    return M, Gam, c


def _gauss(x, m, S):
    r = np.atleast_1d(np.asarray(x, float).ravel() - np.asarray(m).ravel())
    S = np.atleast_2d(S)
    return float(np.exp(-0.5 * r @ np.linalg.solve(S, r))
                 / np.sqrt(np.linalg.det(2 * np.pi * S)))


def _alpha1(params, ys):
    """Initial regime posterior p(r_1 | y_1) from the per-regime laws of Y_1."""
    K, q, s = params.K, params.q, params.s
    H = _obs_matrix(q, s)
    y1 = np.asarray(ys, float).reshape(-1, s)[0]
    la = np.zeros(K)
    for k in range(K):
        mY = (H @ params.mu_z0(k).reshape(q + s, 1)).ravel()
        SY = H @ params.Sigma_z0(k) @ H.T
        r = y1 - mY
        la[k] = (np.log(params.pi0[k] + 1e-300)
                 - 0.5 * (r @ np.linalg.solve(SY, r)
                          + np.log(np.linalg.det(2 * np.pi * SY))))
    a = np.exp(la - la.max())
    return a / a.sum()


def _forward_backward(alpha1, Ker):
    """HMM forward-backward for a time-varying kernel Ker (N-1, K, K).

    Returns the smoothed regime marginals gam (N, K) and the smoothed pair
    marginals gampair (N-1, K, K).
    """
    N, K = Ker.shape[0] + 1, alpha1.shape[0]
    alpha = np.zeros((N, K))
    alpha[0] = alpha1
    for n in range(N - 1):
        alpha[n + 1] = alpha[n] @ Ker[n]
        alpha[n + 1] /= alpha[n + 1].sum()
    beta = np.zeros((N, K))
    beta[N - 1] = 1.0
    for n in range(N - 2, -1, -1):
        beta[n] = Ker[n] @ beta[n + 1]
        beta[n] /= beta[n].max()
    gam = alpha * beta
    gam /= gam.sum(axis=1, keepdims=True)
    gampair = np.zeros((N - 1, K, K))
    for n in range(N - 1):
        x = (alpha[n][:, None] * Ker[n]) * beta[n + 1][None, :]
        gampair[n] = x / x.sum()
    return gam, gampair


def kernel_Q0(params, ys):
    """{C==0} kernel Q0(j->k) = p_jk N(y_{n+1}; D_k y_n + bY_k, SV_k)."""
    K, q, s = params.K, params.q, params.s
    ys = np.asarray(ys, float).reshape(-1, s)
    N = ys.shape[0]
    D = [params.f_matrix.D(k) for k in range(K)]
    SV = [params.noise_cov.Sigma_V(k) for k in range(K)]
    bY = [params.b(k).reshape(q + s, 1)[q:] for k in range(K)]
    Ker = np.zeros((N - 1, K, K))
    for n in range(N - 1):
        for k in range(K):
            m = D[k] @ ys[n].reshape(s, 1) + bY[k]
            g = _gauss(ys[n + 1], m, SV[k])
            for j in range(K):
                Ker[n, j, k] = params.P[j, k] * g
    return Ker


def kernel_Qab(params, ys):
    """AB kernel Q(j->k) with mean (C_k M_j + D_k) y_n + C_k c_j + bY_k and
    covariance S_jk = C_k Gam_j C_k^T + SV_k (eq. (kernAB) of the paper)."""
    K, q, s = params.K, params.q, params.s
    ys = np.asarray(ys, float).reshape(-1, s)
    N = ys.shape[0]
    M, Gam, c = _slaving(params)
    C = [params.f_matrix.C(k) for k in range(K)]
    D = [params.f_matrix.D(k) for k in range(K)]
    SV = [params.noise_cov.Sigma_V(k) for k in range(K)]
    bY = [params.b(k).reshape(q + s, 1)[q:] for k in range(K)]
    Ker = np.zeros((N - 1, K, K))
    for n in range(N - 1):
        y0 = ys[n].reshape(s, 1)
        for j in range(K):
            for k in range(K):
                m = (C[k] @ M[j] + D[k]) @ y0 + C[k] @ c[j] + bY[k]
                S = C[k] @ Gam[j] @ C[k].T + SV[k]
                Ker[n, j, k] = params.P[j, k] * _gauss(ys[n + 1], m, S)
    return Ker


# ---------------------------------------------------------------------------
# the three smoothers
# ---------------------------------------------------------------------------
def pairwise_imm_bank(params, ys):
    """Pairwise IMM keeping the per-regime filtered summaries (x-block).

    Returns xs (N, K, q) and Pxs (N, K, q, q) -- exact per-regime filtered
    moments on {C==0} by Theorem 1 of the exactness paper.
    """
    K, q, s = params.K, params.q, params.s
    dim = q + s
    ys = np.asarray(ys, float).reshape(-1, s)
    N = ys.shape[0]
    H = _obs_matrix(q, s)
    F = [params.f_matrix.F(k) for k in range(K)]
    b = [params.b(k).reshape(dim, 1) for k in range(K)]
    SW = [params.noise_cov.Sigma_W(k) for k in range(K)]
    zs, Ps, la = [], [], np.zeros(K)
    for k in range(K):
        z, P, ll = _kalman_exact_y_update(
            params.mu_z0(k).reshape(dim, 1), params.Sigma_z0(k),
            ys[0].reshape(s, 1), H)
        zs.append(z)
        Ps.append(P)
        la[k] = np.log(params.pi0[k] + 1e-300) + ll
    al = np.exp(la - la.max())
    al /= al.sum()
    xs = np.zeros((N, K, q))
    Pxs = np.zeros((N, K, q, q))
    for k in range(K):
        xs[0, k] = zs[k][:q, 0]
        Pxs[0, k] = Ps[k][:q, :q]
    for n in range(1, N):
        y = ys[n].reshape(s, 1)
        nzs, nPs, nla = [], [], np.zeros(K)
        for k in range(K):
            w = np.array([params.P[j, k] * al[j] for j in range(K)])
            cbar = w.sum()
            w = w / cbar
            z0 = sum(w[j] * zs[j] for j in range(K))
            P0 = sum(w[j] * (Ps[j] + (zs[j] - z0) @ (zs[j] - z0).T)
                     for j in range(K))
            zpr = F[k] @ z0 + b[k]
            Ppr = F[k] @ P0 @ F[k].T + SW[k]
            z, P, ll = _kalman_exact_y_update(zpr, Ppr, y, H)
            nzs.append(z)
            nPs.append(P)
            nla[k] = np.log(cbar + 1e-300) + ll
        zs, Ps = nzs, nPs
        al = np.exp(nla - nla.max())
        al /= al.sum()
        for k in range(K):
            xs[n, k] = zs[k][:q, 0]
            Pxs[n, k] = Ps[k][:q, :q]
    return xs, Pxs


def reweight_smoother(params, ys, Ker):
    """{C==0} smoother: smoothed-regime reweighting of the FILTERED per-regime
    summaries (Theorem 1 of the smoothing paper).  Returns (Ex, Vx)."""
    xs, Pxs = pairwise_imm_bank(params, ys)
    gam, _ = _forward_backward(_alpha1(params, ys), Ker)
    N, K, q = xs.shape
    Ex = np.einsum("nk,nkq->nq", gam, xs)
    Vx = np.zeros((N, q, q))
    for n in range(N):
        for k in range(K):
            d = xs[n, k] - Ex[n]
            Vx[n] += gam[n, k] * (Pxs[n, k] + np.outer(d, d))
    return Ex, Vx


def cg_smoother(params, ys):
    """AB constant-gain smoother (Theorem 2 of the smoothing paper).

    X_{n|N} = sum_{j,k} gampair_n(j,k) [ M_j y_n + c_j
              + W_jk (y_{n+1} - m_jk(y_n)) ],   n < N,
    with precomputed gains W_jk = Gam_j C_k^T S_jk^{-1}, per-pair conditional
    covariance Gam_j - W_jk C_k Gam_j, and the filtered read-out at n = N.
    Returns (Ex, Vx); cost O(N K^2), no covariance recursion.
    """
    K, q, s = params.K, params.q, params.s
    ys_ = np.asarray(ys, float).reshape(-1, s)
    N = ys_.shape[0]
    M, Gam, c = _slaving(params)
    C = [params.f_matrix.C(k) for k in range(K)]
    D = [params.f_matrix.D(k) for k in range(K)]
    SV = [params.noise_cov.Sigma_V(k) for k in range(K)]
    bY = [params.b(k).reshape(q + s, 1)[q:] for k in range(K)]
    W = [[Gam[j] @ C[k].T @ np.linalg.inv(C[k] @ Gam[j] @ C[k].T + SV[k])
          for k in range(K)] for j in range(K)]
    Vpair = [[Gam[j] - W[j][k] @ C[k] @ Gam[j] for k in range(K)]
             for j in range(K)]
    gam, gampair = _forward_backward(_alpha1(params, ys_),
                                     kernel_Qab(params, ys_))
    Ex = np.zeros((N, q))
    Vx = np.zeros((N, q, q))
    for n in range(N - 1):
        y0 = ys_[n].reshape(s, 1)
        y1 = ys_[n + 1].reshape(s, 1)
        means = np.zeros((K, K, q))
        for j in range(K):
            for k in range(K):
                m_jk = (C[k] @ M[j] + D[k]) @ y0 + C[k] @ c[j] + bY[k]
                means[j, k] = (M[j] @ y0 + c[j]
                               + W[j][k] @ (y1 - m_jk)).ravel()
                Ex[n] += gampair[n, j, k] * means[j, k]
        for j in range(K):
            for k in range(K):
                d = means[j, k] - Ex[n]
                Vx[n] += gampair[n, j, k] * (Vpair[j][k] + np.outer(d, d))
    yN = ys_[N - 1].reshape(s, 1)
    mN = np.array([(M[j] @ yN + c[j]).ravel() for j in range(K)])
    Ex[N - 1] = gam[N - 1] @ mN
    for j in range(K):
        d = mN[j] - Ex[N - 1]
        Vx[N - 1] += gam[N - 1, j] * (Gam[j] + np.outer(d, d))
    return Ex, Vx


def lag1_smoother(params, ys):
    """{A==MC} lag-1 smoother (Theorem 3 of the smoothing paper).

    Read-out M_j y_n + E_j y_{n-1} + c_j (E_j = B_j - M_j D_j), same gains
    W_jk, forward-backward on the lag-1 (R, (Y_{n-1}, Y_n)) chain; at n = 1
    the read-out comes from the initial law.  Reduces to cg_smoother on AB.
    Returns (Ex, Vx).
    """
    K, q, s = params.K, params.q, params.s
    ys_ = np.asarray(ys, float).reshape(-1, s)
    N = ys_.shape[0]
    dim = q + s
    M, Gam, c = _slaving(params)
    B = [params.f_matrix.B(k) for k in range(K)]
    C = [params.f_matrix.C(k) for k in range(K)]
    D = [params.f_matrix.D(k) for k in range(K)]
    SV = [params.noise_cov.Sigma_V(k) for k in range(K)]
    bY = [params.b(k).reshape(q + s, 1)[q:] for k in range(K)]
    H = _obs_matrix(q, s)

    # per-(n, j) read-out mean/cov: init-based at n=0, closed form after
    m_read = np.zeros((N, K, q, 1))
    P_read = np.zeros((N, K, q, q))
    for j in range(K):
        z, P, _ = _kalman_exact_y_update(
            params.mu_z0(j).reshape(dim, 1), params.Sigma_z0(j),
            ys_[0].reshape(s, 1), H)
        m_read[0, j] = z[:q]
        P_read[0, j] = P[:q, :q]
    for n in range(1, N):
        y0 = ys_[n - 1].reshape(s, 1)
        y1 = ys_[n].reshape(s, 1)
        for j in range(K):
            m_read[n, j] = M[j] @ y1 + (B[j] - M[j] @ D[j]) @ y0 + c[j]
            P_read[n, j] = Gam[j]
    # time-varying lag-1 kernel
    Ker = np.zeros((N - 1, K, K))
    for n in range(N - 1):
        y1 = ys_[n].reshape(s, 1)
        for j in range(K):
            for k in range(K):
                mm = C[k] @ m_read[n, j] + D[k] @ y1 + bY[k]
                SS = C[k] @ P_read[n, j] @ C[k].T + SV[k]
                Ker[n, j, k] = params.P[j, k] * _gauss(ys_[n + 1], mm, SS)
    gam, gampair = _forward_backward(_alpha1(params, ys_), Ker)
    Ex = np.zeros((N, q))
    Vx = np.zeros((N, q, q))
    for n in range(N - 1):
        y1 = ys_[n].reshape(s, 1)
        y2 = ys_[n + 1].reshape(s, 1)
        means = np.zeros((K, K, q))
        covs = np.zeros((K, K, q, q))
        for j in range(K):
            for k in range(K):
                mm = C[k] @ m_read[n, j] + D[k] @ y1 + bY[k]
                SS = C[k] @ P_read[n, j] @ C[k].T + SV[k]
                Wjk = P_read[n, j] @ C[k].T @ np.linalg.inv(SS)
                means[j, k] = (m_read[n, j] + Wjk @ (y2 - mm)).ravel()
                covs[j, k] = P_read[n, j] - Wjk @ C[k] @ P_read[n, j]
                Ex[n] += gampair[n, j, k] * means[j, k]
        for j in range(K):
            for k in range(K):
                d = means[j, k] - Ex[n]
                Vx[n] += gampair[n, j, k] * (covs[j, k] + np.outer(d, d))
    Ex[N - 1] = sum(gam[N - 1, j] * m_read[N - 1, j].ravel() for j in range(K))
    for j in range(K):
        d = m_read[N - 1, j].ravel() - Ex[N - 1]
        Vx[N - 1] += gam[N - 1, j] * (P_read[N - 1, j] + np.outer(d, d))
    return Ex, Vx


# ---------------------------------------------------------------------------
# S6: collapsed smoothed regime posterior off the union (GPB2 forward +
# pair-likelihood backward), and its light exact reference
# ---------------------------------------------------------------------------
def exact_regime_smoother(params, ys):
    """Exact smoothed regime posterior by path enumeration (no RTS needed)."""
    K, q, s = params.K, params.q, params.s
    dim = q + s
    ys = np.asarray(ys, float).reshape(-1, s)
    N = ys.shape[0]
    H = _obs_matrix(q, s)
    F = [params.f_matrix.F(k) for k in range(K)]
    b = [params.b(k).reshape(dim, 1) for k in range(K)]
    SW = [params.noise_cov.Sigma_W(k) for k in range(K)]
    lws, paths = [], []
    for path in product(range(K), repeat=N):
        k0 = path[0]
        z, P, ll = _kalman_exact_y_update(
            params.mu_z0(k0).reshape(dim, 1), params.Sigma_z0(k0),
            ys[0].reshape(s, 1), H)
        lw = float(np.log(params.pi0[k0] + 1e-300) + ll)
        for n in range(1, N):
            k = path[n]
            zpr = F[k] @ z + b[k]
            Ppr = F[k] @ P @ F[k].T + SW[k]
            z, P, ll = _kalman_exact_y_update(zpr, Ppr, ys[n].reshape(s, 1), H)
            lw += float(np.log(params.P[path[n - 1], k] + 1e-300)) + ll
        lws.append(lw)
        paths.append(path)
    lws = np.array(lws)
    w = np.exp(lws - lws.max())
    w /= w.sum()
    gam = np.zeros((N, K))
    for wi, path in zip(w, paths):
        for n in range(N):
            gam[n, path[n]] += wi
    return gam


def gpb2_bank(params, ys):
    """GPB2 forward pass keeping the per-regime merged PAIR summaries.

    Returns zs (N, K, dim, 1), Ps (N, K, dim, dim), alpha (N, K) and the
    backward-kernel matrices A_n (N-1, K, K) with a_n(j,k) = p_jk Lam^{jk}.
    """
    K, q, s = params.K, params.q, params.s
    dim = q + s
    ys = np.asarray(ys, float).reshape(-1, s)
    N = ys.shape[0]
    H = _obs_matrix(q, s)
    F = [params.f_matrix.F(k) for k in range(K)]
    b = [params.b(k).reshape(dim, 1) for k in range(K)]
    SW = [params.noise_cov.Sigma_W(k) for k in range(K)]
    zs = np.zeros((N, K, dim, 1))
    Ps = np.zeros((N, K, dim, dim))
    alpha = np.zeros((N, K))
    la = np.zeros(K)
    for k in range(K):
        z, P, ll = _kalman_exact_y_update(
            params.mu_z0(k).reshape(dim, 1), params.Sigma_z0(k),
            ys[0].reshape(s, 1), H)
        zs[0, k], Ps[0, k] = z, P
        la[k] = np.log(params.pi0[k] + 1e-300) + ll
    alpha[0] = np.exp(la - la.max())
    alpha[0] /= alpha[0].sum()
    A_n = np.zeros((N - 1, K, K))  # a_n(j,k) = p_jk Lam^{jk}
    for n in range(1, N):
        y = ys[n].reshape(s, 1)
        for k in range(K):
            upd = []
            for j in range(K):
                zpr = F[k] @ zs[n - 1, j] + b[k]
                Ppr = F[k] @ Ps[n - 1, j] @ F[k].T + SW[k]
                z, P, ll = _kalman_exact_y_update(zpr, Ppr, y, H)
                A_n[n - 1, j, k] = params.P[j, k] * np.exp(ll)
                upd.append((z, P))
            w = alpha[n - 1] * A_n[n - 1, :, k]
            wsum = w.sum()
            w = w / wsum if wsum > 0 else np.full(K, 1.0 / K)
            zbar = sum(w[j] * upd[j][0] for j in range(K))
            Pbar = sum(w[j] * (upd[j][1]
                               + (upd[j][0] - zbar) @ (upd[j][0] - zbar).T)
                       for j in range(K))
            zs[n, k], Ps[n, k] = zbar, Pbar
        un = alpha[n - 1] @ A_n[n - 1]
        alpha[n] = un / un.sum()
    return zs, Ps, alpha, A_n


def gpb2_regime_smoother(params, ys):
    """Collapsed smoothed regime posterior: GPB2 forward pass (per-regime
    collapsed summaries, pair likelihoods Lam^{jk}) + backward recursion on
    a_n(j,k) = p_jk Lam^{jk}.  Exact on the union, biased off it (S6)."""
    _, _, alpha, A_n = gpb2_bank(params, ys)
    N = alpha.shape[0]
    beta = np.zeros_like(alpha)
    beta[N - 1] = 1.0
    for n in range(N - 2, -1, -1):
        beta[n] = A_n[n] @ beta[n + 1]
        beta[n] /= beta[n].max()
    gam = alpha * beta
    gam /= gam.sum(axis=1, keepdims=True)
    return gam


# ---------------------------------------------------------------------------
# S7/S8: the pair-RTS form -- gain degeneracy and off-domain separation
# ---------------------------------------------------------------------------
def pair_rts_smoother(params, ys, with_residuals=False):
    """Mode-matched pair-RTS smoother run on the pair (RTS-IMM style).

    Forward: GPB2 bank (per-regime merged pair summaries).  Backward: the
    per-regime smoothed pair means are mixed over the ARRIVAL regime with the
    smoothed pair weights, with the classical RTS pair gains
        G^{jk} = P_n(j) F_k^T (F_k P_n(j) F_k^T + SW_k)^{-1} .
    This is the naive collapse: the backward input is the k-conditional
    smoothed summary where the exact recursion calls for the pair-conditional
    one.  On {A==MC} the x-rows of G^{jk} are [0, W_jk] identically, so the
    mismatch is multiplied by zero and the smoother is exact; on {C==0} with
    state memory it is genuinely biased (S7).

    Returns the smoothed state mean (N, q); with_residuals=True also returns
    the max residuals of the two gain identities along the run:
    ('gx', |G_x| / |G_y|)   -- zero on {A==MC}, where G == [0, W_jk];
    ('w',  |G_y - W_jk| / |W_jk|)          -- same identity, gain part;
    ('c0', |G_y + G_x M_k| / |G_x M_k|)    -- zero on {C==0}.
    """
    K, q, s = params.K, params.q, params.s
    dim = q + s
    ys_ = np.asarray(ys, float).reshape(-1, s)
    N = ys_.shape[0]
    F = [params.f_matrix.F(k) for k in range(K)]
    b = [params.b(k).reshape(dim, 1) for k in range(K)]
    SW = [params.noise_cov.Sigma_W(k) for k in range(K)]
    M, Gam, _ = _slaving(params)
    C = [params.f_matrix.C(k) for k in range(K)]
    SV = [params.noise_cov.Sigma_V(k) for k in range(K)]
    W = [[Gam[j] @ C[k].T @ np.linalg.inv(C[k] @ Gam[j] @ C[k].T + SV[k])
          for k in range(K)] for j in range(K)]
    zs, Ps, alpha, A_n = gpb2_bank(params, ys_)
    beta = np.zeros((N, K))
    beta[N - 1] = 1.0
    for n in range(N - 2, -1, -1):
        beta[n] = A_n[n] @ beta[n + 1]
        beta[n] /= beta[n].max()
    gam = alpha * beta
    gam /= gam.sum(axis=1, keepdims=True)

    res = {"gx": 0.0, "w": 0.0, "c0": 0.0}
    Ex = np.zeros((N, q))
    zsm = zs[N - 1].copy()  # (K, dim, 1) smoothed per-regime pair means
    Ex[N - 1] = gam[N - 1] @ zsm[:, :q, 0]
    for n in range(N - 2, -1, -1):
        xi = (alpha[n][:, None] * A_n[n]) * beta[n + 1][None, :]
        xi /= xi.sum()
        new = np.zeros_like(zsm)
        for j in range(K):
            wkj = xi[j, :] / max(xi[j, :].sum(), 1e-300)
            corr = np.zeros((dim, 1))
            for k in range(K):
                Ppred = F[k] @ Ps[n, j] @ F[k].T + SW[k]
                G = Ps[n, j] @ F[k].T @ np.linalg.inv(Ppred)
                zpred = F[k] @ zs[n, j] + b[k]
                corr += wkj[k] * (G @ (zsm[k] - zpred))
                if with_residuals:
                    Gx, Gy = G[:q, :q], G[:q, q:]
                    nGy = np.max(np.abs(Gy)) + 1e-300
                    res["gx"] = max(res["gx"], np.max(np.abs(Gx)) / nGy)
                    if n >= 1:
                        # at n=0 the bank covariance is the (updated) initial
                        # law, not Gamma_j, so G_y = W_jk(P_0) != W_jk there
                        res["w"] = max(res["w"],
                                       np.max(np.abs(Gy - W[j][k])) / nGy)
                    GxM = Gx @ M[k]
                    nGxM = np.max(np.abs(GxM)) + 1e-300
                    res["c0"] = max(res["c0"],
                                    np.max(np.abs(Gy + GxM)) / nGxM)
            new[j] = zs[n, j] + corr
        zsm = new
        Ex[n] = gam[n] @ zsm[:, :q, 0]
    return (Ex, res) if with_residuals else Ex


# ---------------------------------------------------------------------------
# experiments
# ---------------------------------------------------------------------------
def _med_max(vals):
    return f"median {np.median(vals):.1e}  max {np.max(vals):.1e}"


def exp_ab_gauge():
    """S1(AB), S3, S4 and the S2 discriminant, on the AB gauge (C=0.4)."""
    p = ab_model(0.4, dB=0.0)
    M, Gam, c = _slaving(p)
    g_reg, g_cgm, g_cgv, g_naive, g_gain, g_rw = [], [], [], [], [], []
    exact_by_seed = {}
    for sd in range(N_SEEDS):
        _, _, ys = _simulate(p, N_STEPS, seed=SEED0 + sd)
        Ex_s, Vx_s, gam_ex = exact_path_smoother(p, ys)
        exact_by_seed[sd] = (ys, Ex_s, Vx_s, gam_ex)
        # S1(AB): smoothed regime posterior from the (R,Y) forward-backward
        gam_h, _ = _forward_backward(_alpha1(p, ys), kernel_Qab(p, ys))
        g_reg.append(_rel(gam_h, gam_ex))
        # S3: constant-gain smoother, mean and variance
        Ex_cg, Vx_cg = cg_smoother(p, ys)
        g_cgm.append(_rel(Ex_cg, Ex_s))
        g_cgv.append(_rel(Vx_cg, Vx_s))
        # S4: naive smoothed read-out fed the EXACT smoothed regime posterior
        yv = np.asarray(ys, float).reshape(-1, p.s)
        Ex_naive = np.array([
            sum(gam_ex[n, k] * (M[k] @ yv[n].reshape(p.s, 1) + c[k]).ravel()
                for k in range(p.K)) for n in range(N_STEPS)])
        g_naive.append(_rel(Ex_naive, Ex_s))
        Ex_f, _, _ = exact_mixture_filter(p, ys)
        g_gain.append(_rel(Ex_f, Ex_s))
        # S2 discriminant: the {C==0} reweighting formula on the wrong branch
        Ex_rw, _ = reweight_smoother(p, ys, kernel_Qab(p, ys))
        g_rw.append(_rel(Ex_rw, Ex_s))
    print("AB gauge (C=0.4, full AB):")
    print(f"  S1  (R,Y) FB regime smoother, kernel Q : {_med_max(g_reg)}")
    print(f"  S3  constant-gain smoother, mean       : {_med_max(g_cgm)}")
    print(f"  S3  constant-gain smoother, variance   : {_med_max(g_cgv)}")
    print(f"  S4  naive smoothed read-out (trap)     : median"
          f" {np.median(g_naive):.2f}"
          f"  (smoothing gain scale {np.median(g_gain):.2f})")
    print(f"  S2d reweighting formula under AB       : median"
          f" {np.median(g_rw):.2f}  (expected O(1) failure)")
    return exact_by_seed


def exp_c0_gauge():
    """S1({C==0}) and S2, on the CGO-MSM gauge with state memory."""
    p = cgo_memory_model()
    g_reg, g_m, g_v = [], [], []
    exact_by_seed = {}
    for sd in range(N_SEEDS):
        _, _, ys = _simulate(p, N_STEPS, seed=SEED0 + sd)
        Ex_s, Vx_s, gam_ex = exact_path_smoother(p, ys)
        exact_by_seed[sd] = (ys, Ex_s, Vx_s, gam_ex)
        Ker = kernel_Q0(p, ys)
        gam_h, _ = _forward_backward(_alpha1(p, ys), Ker)
        g_reg.append(_rel(gam_h, gam_ex))
        Ex_r, Vx_r = reweight_smoother(p, ys, Ker)
        g_m.append(_rel(Ex_r, Ex_s))
        g_v.append(_rel(Vx_r, Vx_s))
    print("{C==0} gauge (memory A=(0.7,0.4)):")
    print(f"  S1  (R,Y) FB regime smoother, kernel Q0: {_med_max(g_reg)}")
    print(f"  S2  reweighting smoother, mean         : {_med_max(g_m)}")
    print(f"  S2  reweighting smoother, variance     : {_med_max(g_v)}")
    return exact_by_seed


def exp_lag1(exact_ab):
    """S5: lag-1 smoother on {A==MC} \\ AB, and its reduction on AB."""
    pA = ab_model(0.4, dB=0.15)
    pab = ab_model(0.4, dB=0.0)
    g_m, g_v, g_abm, g_abv = [], [], [], []
    exact_by_seed = {}
    for sd in range(N_SEEDS):
        _, _, ys = _simulate(pA, N_STEPS, seed=SEED0 + sd)
        Ex_s, Vx_s, gam_ex = exact_path_smoother(pA, ys)
        exact_by_seed[sd] = (ys, Ex_s, Vx_s, gam_ex)
        Ex_l, Vx_l = lag1_smoother(pA, ys)
        g_m.append(_rel(Ex_l, Ex_s))
        g_v.append(_rel(Vx_l, Vx_s))
        ys2, Ex_s2, Vx_s2, _ = exact_ab[sd]
        Ex_l2, Vx_l2 = lag1_smoother(pab, ys2)
        g_abm.append(_rel(Ex_l2, Ex_s2))
        g_abv.append(_rel(Vx_l2, Vx_s2))
    print("{A==MC} \\ AB gauge (C=0.4, B=MD+0.15):")
    print(f"  S5  lag-1 smoother, mean               : {_med_max(g_m)}")
    print(f"  S5  lag-1 smoother, variance           : {_med_max(g_v)}")
    print(f"  S5  reduction on AB, mean              : {_med_max(g_abm)}")
    print(f"  S5  reduction on AB, variance          : {_med_max(g_abv)}")
    return exact_by_seed


def exp_rts(exact_ab, exact_c0, exact_A):
    """S7: pair-RTS gain degeneracy and asymmetric exactness."""
    print("S7  pair-RTS (mode-matched, GPB2 forward bank):")
    gauges = [
        ("AB gauge          ", ab_model(0.4, dB=0.0), exact_ab, "slaved"),
        ("{A==MC} \\ AB gauge", ab_model(0.4, dB=0.15), exact_A, "slaved"),
        ("{C==0} gauge      ", cgo_memory_model(), exact_c0, "c0"),
    ]
    for label, p, exact, branch in gauges:
        errs, r_gx, r_w, r_c0 = [], [], [], []
        for sd in range(N_SEEDS):
            ys, Ex_s, _, _ = exact[sd]
            Ex_r, res = pair_rts_smoother(p, ys, with_residuals=True)
            errs.append(_rel(Ex_r, Ex_s))
            r_gx.append(res["gx"])
            r_w.append(res["w"])
            r_c0.append(res["c0"])
        if branch == "slaved":
            print(f"  {label}: mean {_med_max(errs)}")
            print(f"      gain identity G==[0,W_jk]: |G_x|/|G_y| max"
                  f" {np.max(r_gx):.1e}, |G_y-W|/|W| max {np.max(r_w):.1e}")
        else:
            print(f"  {label}: mean median {np.median(errs):.2e}"
                  f"  [{np.min(errs):.0e},{np.max(errs):.0e}]  <- BIASED")
            print(f"      gain identity G_y==-G_x M_k: residual max"
                  f" {np.max(r_c0):.1e}  (the gain is degenerate; the bias"
                  " is the k-conditional backward input)")


def exp_separation(etas=(0.02, 0.08, 0.3)):
    """S8: off the union, the three formulations separate; the pairwise
    separations vanish with eta and need no ground truth (diagnostic)."""
    print("S8  off the union (C=0.4, A=MC+eta): separation of formulations")
    sep_rc, sep_rw, sep_cw = [], [], []
    for eta in etas:
        p = off_union_model(0.4, eta)
        e_rts, e_cg, e_rw, s_rc, s_rw_, s_cw = [], [], [], [], [], []
        for sd in range(N_SEEDS):
            _, _, ys = _simulate(p, N_STEPS, seed=SEED0 + sd)
            Ex_s, _, _ = exact_path_smoother(p, ys)
            scale = np.max(np.abs(Ex_s)) + 1e-12
            E_rts = pair_rts_smoother(p, ys)
            E_cg, _ = cg_smoother(p, ys)
            E_rw, _ = reweight_smoother(p, ys, kernel_Qab(p, ys))
            e_rts.append(_rel(E_rts, Ex_s))
            e_cg.append(_rel(E_cg, Ex_s))
            e_rw.append(_rel(E_rw, Ex_s))
            s_rc.append(float(np.max(np.abs(E_rts - E_cg)) / scale))
            s_rw_.append(float(np.max(np.abs(E_rts - E_rw)) / scale))
            s_cw.append(float(np.max(np.abs(E_cg - E_rw)) / scale))
        sep_rc.append(np.median(s_rc))
        sep_rw.append(np.median(s_rw_))
        sep_cw.append(np.median(s_cw))
        print(f"  eta={eta:<5} err: RTS {np.median(e_rts):.2e}"
              f"  one-step {np.median(e_cg):.2e}"
              f"  reweight {np.median(e_rw):.2e}"
              f" | sep: RTS/1step {sep_rc[-1]:.2e}"
              f"  RTS/rw {sep_rw[-1]:.2e}  1step/rw {sep_cw[-1]:.2e}")
    x = np.log10(etas)
    sl = [np.polyfit(x, np.log10(v), 1)[0] for v in (sep_rc, sep_rw, sep_cw)]
    print(f"  separation slopes: RTS/1step {sl[0]:.2f}"
          f"  RTS/rw {sl[1]:.2f}  1step/rw {sl[2]:.2f}")


def exp_off_domain(etas=(0.02, 0.08, 0.3), n_seeds=20):
    """S6: first-order degradation of the collapsed smoothed regime posterior
    off the union (C=0.4, A=MC+eta), against the exact smoothed regime."""
    print("off the union (C=0.4, A=MC+eta): collapsed smoothed regime")
    meds = []
    for eta in etas:
        p = off_union_model(0.4, eta)
        errs = []
        for sd in range(n_seeds):
            _, _, ys = _simulate(p, N_STEPS, seed=SEED0 + sd)
            errs.append(_rel(gpb2_regime_smoother(p, ys),
                             exact_regime_smoother(p, ys)))
        meds.append(np.median(errs))
        print(f"  S6  eta={eta:<5} median {meds[-1]:.2e}")
    slope = np.polyfit(np.log10(etas), np.log10(meds), 1)[0]
    print(f"  S6  fitted slope: {slope:.2f}  (first order; filtered E3 is ~3)")
    return meds, slope


def main():
    print(f"[smoothing_exactness] N={N_STEPS}, {N_SEEDS} seeds,"
          f" ground truth K^N path smoother\n")
    exact_ab = exp_ab_gauge()
    print()
    exact_c0 = exp_c0_gauge()
    print()
    exact_A = exp_lag1(exact_ab)
    print()
    exp_off_domain()
    print()
    exp_rts(exact_ab, exact_c0, exact_A)
    print()
    exp_separation()


if __name__ == "__main__":
    main()
