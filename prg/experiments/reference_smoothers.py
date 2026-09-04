#!/usr/bin/env python3
"""
prg/experiments/reference_smoothers.py
======================================
Ground-truth fixed-interval smoother, mirroring
:func:`prg.experiments.reference_filters.exact_mixture_filter`.

``exact_mixture_smoother(params, ys)`` enumerates all Kᴺ regime paths; each
path runs a conditional Kalman filter plus an RTS backward pass on the couple
Z = [X; Y] (exact read-out ``Y_n = H Z_n``), and the per-path smoothed moments
are recombined with the posterior path weights p(r_{1:N} | y_{1:N}). This is
the reference against which the O(N K²) smoothers of
:mod:`prg.filter.gss_smoother` are validated to machine precision on their
exactness domains. Cost grows as Kᴺ — keep N short (N ≲ 12 for K = 2).
"""

from __future__ import annotations

from itertools import product

import numpy as np

from prg.experiments.reference_filters import _kalman_exact_y_update, _obs_matrix

__all__ = ["exact_mixture_smoother"]


def exact_mixture_smoother(params, ys):
    """Exact smoothed posterior over all Kᴺ regime paths (no pruning).

    Returns ``(E_x, Var_x, pi, pi_pair)`` with shapes ``(N, q)``,
    ``(N, q, q)``, ``(N, K)`` and ``(N−1, K, K)``: ``E[X_n | y_{1:N}]``,
    ``Var[X_n | y_{1:N}]``, ``p(r_n | y_{1:N})`` and
    ``p(r_n, r_{n+1} | y_{1:N})``.
    """
    K, q, s = params.K, params.q, params.s
    dim = q + s
    ys = np.asarray(ys, dtype=float).reshape(-1, s)
    N = ys.shape[0]
    H = _obs_matrix(q, s)
    P = params.P
    pi0 = params.pi0
    pi0 = (
        np.asarray(pi0, dtype=float).ravel()
        if pi0 is not None
        else params.stationary_distribution()
    )
    F = [params.f_matrix.F(k) for k in range(K)]
    b = [params.b(k).reshape(dim, 1) for k in range(K)]
    SW = [params.noise_cov.Sigma_W(k) for k in range(K)]
    mu0 = [params.mu_z0(k).reshape(dim, 1) for k in range(K)]
    Sig0 = [params.Sigma_z0(k) for k in range(K)]

    logws, store = [], []
    for path in product(range(K), repeat=N):
        k0 = path[0]
        z, Pc, ll = _kalman_exact_y_update(mu0[k0], Sig0[k0], ys[0].reshape(s, 1), H)
        lw = float(np.log(pi0[k0] + 1e-300) + ll)
        zf, Pf, zp, Pp = [z], [Pc], [None], [None]
        for n in range(1, N):
            k = path[n]
            z_pred = F[k] @ zf[-1] + b[k]
            P_pred = F[k] @ Pf[-1] @ F[k].T + SW[k]
            z, Pc, ll = _kalman_exact_y_update(z_pred, P_pred, ys[n].reshape(s, 1), H)
            lw += float(np.log(P[path[n - 1], k] + 1e-300)) + ll
            zf.append(z)
            Pf.append(Pc)
            zp.append(z_pred)
            Pp.append(P_pred)
        # RTS backward pass on the couple, conditional on the path
        zs = [None] * N
        Ps = [None] * N
        zs[N - 1], Ps[N - 1] = zf[N - 1], Pf[N - 1]
        for n in range(N - 2, -1, -1):
            Fk = F[path[n + 1]]
            G = Pf[n] @ Fk.T @ np.linalg.inv(Pp[n + 1])
            zs[n] = zf[n] + G @ (zs[n + 1] - zp[n + 1])
            Ps[n] = Pf[n] + G @ (Ps[n + 1] - Pp[n + 1]) @ G.T
        logws.append(lw)
        store.append((path, zs, Ps))

    logws = np.array(logws)
    w = np.exp(logws - logws.max())
    w /= w.sum()

    E_x = np.zeros((N, q))
    Var_x = np.zeros((N, q, q))
    pi = np.zeros((N, K))
    pi_pair = np.zeros((N - 1, K, K))
    for wi, (path, zs, _Ps) in zip(w, store):
        for n in range(N):
            E_x[n] += wi * zs[n][:q, 0]
            pi[n, path[n]] += wi
        for n in range(N - 1):
            pi_pair[n, path[n], path[n + 1]] += wi
    for wi, (_path, zs, Ps) in zip(w, store):
        for n in range(N):
            d = zs[n][:q, 0] - E_x[n]
            Var_x[n] += wi * (Ps[n][:q, :q] + np.outer(d, d))
    return E_x, Var_x, pi, pi_pair
