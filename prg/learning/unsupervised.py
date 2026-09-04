#!/usr/bin/env python3
"""
prg/learning/unsupervised.py
============================
Unsupervised (Y-only) EM for the ``{C≡0}`` family — the *exact* EM of the
smoothing companion paper.

Setting
-------
Only the observations ``Y_{1:N}`` are available — neither the regimes ``R``
nor the hidden state ``X`` (contrast with :mod:`prg.learning.semi_supervised`,
where the full couple ``Z = [X; Y]`` is observed). Under ``C≡0`` the Y-chain
is an autonomous switching vector AR(1):

    Y_{n+1} = D_{r_{n+1}} Y_n + b^Y_{r_{n+1}} + V_{n+1},   V ~ N(0, Σ_V(r_{n+1})),

so the identifiable parameters are exactly those of the observed (R, Y)
chain, θ = (P, D_r, b^Y_r, Σ_V(r)); the whole state block (A, B, b^X, Σ_U, Δ)
is a strict nuisance — the Y-likelihood is flat in it.

Exactness
---------
The E-step needs only the smoothed regime pair posteriors, computed *exactly*
in O(N K²) by the forward-backward on the observed chain (the same machinery
as :func:`prg.filter.gss_smoother.regime_smoother` with the ``"c0"`` kernel).
Consequently this EM **is** the exact EM: its iterates coincide with those of
the EM whose E-step enumerates all Kᴺ regime paths (guarded by the tests).
The M-step is a closed-form weighted affine regression per arrival regime.

Conventions
-----------
The initial law of the chain is the per-regime *stationary* law implied by
the current θ (recomputed at every iteration, matching the repository-wide
stationary-init convention); its θ-dependence is dropped in the M-step, the
standard simplification for switching AR estimation — monotonicity of the
likelihood is asserted empirically in the tests.
"""

from __future__ import annotations

import dataclasses
import logging

import numpy as np
from scipy.cluster.vq import kmeans2

from prg.classes.FMatrix import FMatrix
from prg.classes.GSSParams import GSSParams
from prg.classes.NoiseCovariance import GSSNoiseCovariance
from prg.filter.gss_smoother import _forward_backward, _logsumexp
from prg.learning.semi_supervised import _log_mvn_batch
from prg.learning.supervised import _nearest_spd

__all__ = ["fit_em_c0", "c0_chain_to_gss", "EMC0Result"]

_log = logging.getLogger("exactIMM.learning.unsupervised")

_WEIGHT_FLOOR = 1e-12  # below this total weight a regime is left untouched


@dataclasses.dataclass(frozen=True)
class EMC0Result:
    """Fitted ``{C≡0}`` chain parameters and diagnostics.

    Attributes
    ----------
    params : GSSParams
        A valid GSS model carrying the fitted chain (P, D_r, b^Y_r, Σ_V(r))
        with a *declared-nuisance* state block (A = B = 0, b^X = 0, Σ_U = I,
        Δ = 0) and the stationary initial law — ready for ``GSSFilter``, the
        smoothers and the LRT.
    P, D, bY, SV : ndarray / lists
        The chain parameters themselves (transition matrix and per-regime
        AR blocks).
    log_liks : list of float
        Observed-chain log-likelihood at the *start* of each iteration.
    n_iter : int
        Number of EM iterations performed.
    converged : bool
        True when the last increment fell below ``tol``.
    """

    params: GSSParams
    P: np.ndarray
    D: list
    bY: list
    SV: list
    log_liks: list
    n_iter: int
    converged: bool


# ---------------------------------------------------------------------------
# Stationary law of the autonomous Y-chain
# ---------------------------------------------------------------------------
def _stationary_pi(P: np.ndarray) -> np.ndarray:
    K = P.shape[0]
    A = np.vstack([P.T - np.eye(K), np.ones((1, K))])
    b = np.zeros(K + 1)
    b[-1] = 1.0
    pi, *_ = np.linalg.lstsq(A, b, rcond=None)
    pi = np.clip(pi, 0.0, None)
    return pi / pi.sum()


def _chain_stationary(P, D, bY, SV, n_iter: int = 1000, tol: float = 1e-13):
    """Per-regime stationary moments of the switching AR(1) Y-chain.

    Returns ``(pi (K,), muY (K,s,1), SigY (K,s,s))`` — the fixed point of the
    time-reversed moment recursion (weights p(r_n = j | r_{n+1} = k)).
    """
    K = P.shape[0]
    s = D[0].shape[0]
    pi = _stationary_pi(P)
    W = np.zeros((K, K))  # W[j, k] = p(r_n = j | r_{n+1} = k)
    for k in range(K):
        col = pi * P[:, k]
        tot = col.sum()
        W[:, k] = col / tot if tot > 0 else pi
    m = np.zeros((K, s, 1))
    S = np.stack([SV[k].copy() for k in range(K)])  # uncentred second moments
    for _ in range(n_iter):
        m_new = np.zeros_like(m)
        S_new = np.zeros_like(S)
        for k in range(K):
            mbar = sum(W[j, k] * m[j] for j in range(K))
            Sbar = sum(W[j, k] * S[j] for j in range(K))
            m_new[k] = D[k] @ mbar + bY[k]
            S_new[k] = (
                D[k] @ Sbar @ D[k].T
                + D[k] @ mbar @ bY[k].T
                + bY[k] @ mbar.T @ D[k].T
                + bY[k] @ bY[k].T
                + SV[k]
            )
        delta = max(np.max(np.abs(m_new - m)), np.max(np.abs(S_new - S)))
        m, S = m_new, S_new
        if delta < tol:
            break
    else:
        _log.warning("chain stationary moments: fixed point not reached (delta=%.2e)", delta)
    SigY = np.stack([0.5 * ((S[k] - m[k] @ m[k].T) + (S[k] - m[k] @ m[k].T).T) for k in range(K)])
    return pi, m, SigY


# ---------------------------------------------------------------------------
# E- and M-steps
# ---------------------------------------------------------------------------
def _estep(ys_, P, D, bY, SV):
    """Exact posteriors of the observed chain: (gamma, xi, log_lik)."""
    N, s = ys_.shape
    K = P.shape[0]
    pi, muY, SigY = _chain_stationary(P, D, bY, SV)
    la = np.zeros(K)
    for k in range(K):
        la[k] = float(
            np.log(pi[k] + 1e-300) + _log_mvn_batch(ys_[0][None, :], muY[k].ravel(), SigY[k])[0]
        )
    if N == 1:
        gamma = np.exp(la - _logsumexp(la))[None, :]
        return gamma / gamma.sum(), np.zeros((0, K, K)), _logsumexp(la)
    emis = np.zeros((N - 1, K))
    for k in range(K):
        resid = ys_[1:] - ys_[:-1] @ D[k].T - bY[k].ravel()[None, :]
        emis[:, k] = _log_mvn_batch(resid, np.zeros(s), SV[k])
    log_ker = np.log(P + 1e-300)[None, :, :] + emis[:, None, :]
    return _forward_backward(la, log_ker)


def _mstep(ys_, gamma, xi, prev):
    """Closed-form weighted update of (P, D_r, b^Y_r, Σ_V(r))."""
    P_prev, D_prev, bY_prev, SV_prev = prev
    K = gamma.shape[1]
    s = ys_.shape[1]
    P = np.array(P_prev, dtype=float, copy=True)
    counts = xi.sum(axis=0)
    for j in range(K):
        row = counts[j]
        tot = row.sum()
        if tot > _WEIGHT_FLOOR:
            P[j] = row / tot
    D = [np.array(D_prev[k], dtype=float, copy=True) for k in range(K)]
    bY = [np.array(bY_prev[k], dtype=float, copy=True) for k in range(K)]
    SV = [np.array(SV_prev[k], dtype=float, copy=True) for k in range(K)]
    X = np.hstack([ys_[:-1], np.ones((ys_.shape[0] - 1, 1))])  # (N-1, s+1)
    for k in range(K):
        w = xi[:, :, k].sum(axis=1)  # arrival weight of regime k at n+1
        tot = float(w.sum())
        if tot <= _WEIGHT_FLOOR:
            continue
        sw = np.sqrt(w)[:, None]
        Theta, *_ = np.linalg.lstsq(sw * X, sw * ys_[1:], rcond=None)  # (s+1, s)
        D[k] = Theta[:s].T
        bY[k] = Theta[s].reshape(s, 1)
        resid = ys_[1:] - X @ Theta
        SV[k] = _nearest_spd((resid * w[:, None]).T @ resid / tot)
    return P, D, bY, SV


# ---------------------------------------------------------------------------
# Initialisation and public API
# ---------------------------------------------------------------------------
def _kmeans_init(ys_, K, seed):
    """Hard-assignment start: k-means on the standardized lag pairs
    (y_n, y_{n+1}), then one M-step on the one-hot pair posteriors."""
    N = ys_.shape[0]
    feats = np.hstack([ys_[:-1], ys_[1:]])
    feats = (feats - feats.mean(axis=0)) / (feats.std(axis=0) + 1e-12)
    for attempt in range(5):
        _, labels = kmeans2(feats, K, minit="++", seed=seed + attempt)
        if len(np.unique(labels)) == K:
            break
    labels = np.concatenate([labels, labels[-1:]])  # length N
    gamma = np.zeros((N, K))
    gamma[np.arange(N), labels] = 1.0
    xi = np.zeros((N - 1, K, K))
    xi[np.arange(N - 1), labels[:-1], labels[1:]] = 1.0
    s = ys_.shape[1]
    prev = (
        np.full((K, K), 1.0 / K),
        [np.zeros((s, s)) for _ in range(K)],
        [np.zeros((s, 1)) for _ in range(K)],
        [np.eye(s) for _ in range(K)],
    )
    P, D, bY, SV = _mstep(ys_, gamma, xi, prev)
    P = 0.9 * P + 0.1 / K  # soften the hard counts (no zero transitions)
    P /= P.sum(axis=1, keepdims=True)
    return P, D, bY, SV


def c0_chain_to_gss(P, D, bY, SV, q: int = 1) -> GSSParams:
    """Embed fitted ``{C≡0}`` chain parameters into a valid :class:`GSSParams`.

    The state block is a **declared nuisance** (A = B = 0, b^X = 0, Σ_U = I_q,
    Δ = 0): any choice leaves the Y-likelihood unchanged. The initial law is
    the stationary one (X ⊥ Y at the nuisance block).
    """
    K = P.shape[0]
    s = D[0].shape[0]
    pi, muY, SigY = _chain_stationary(P, D, bY, SV)
    fm = FMatrix(
        K,
        q,
        s,
        [np.zeros((q, q)) for _ in range(K)],
        [np.zeros((q, s)) for _ in range(K)],
        [np.zeros((s, q)) for _ in range(K)],
        [np.asarray(D[k], dtype=float) for k in range(K)],
    )
    nc = GSSNoiseCovariance(
        K,
        q,
        s,
        [np.eye(q) for _ in range(K)],
        [np.zeros((q, s)) for _ in range(K)],
        [np.asarray(SV[k], dtype=float) for k in range(K)],
    )
    mu_z0 = [np.vstack([np.zeros((q, 1)), muY[k]]) for k in range(K)]
    Sig_z0 = [
        np.block([[np.eye(q), np.zeros((q, s))], [np.zeros((s, q)), SigY[k]]]) for k in range(K)
    ]
    b_list = [np.vstack([np.zeros((q, 1)), np.asarray(bY[k], dtype=float)]) for k in range(K)]
    return GSSParams(
        K=K,
        q=q,
        s=s,
        P=np.asarray(P, dtype=float),
        f_matrix=fm,
        noise_cov=nc,
        pi0=None,
        mu_z0_list=mu_z0,
        Sigma_z0_list=Sig_z0,
        b_list=b_list,
    )


def fit_em_c0(
    ys,
    K: int,
    *,
    theta0=None,
    max_iter: int = 100,
    tol: float = 1e-6,
    seed: int = 0,
) -> EMC0Result:
    """Exact EM for the ``{C≡0}`` family from Y-only data, O(N K²) per pass.

    Parameters
    ----------
    ys : array-like (N, s)
        Observations.
    K : int
        Number of regimes.
    theta0 : tuple (P, D, bY, SV), optional
        Starting chain parameters (lists of per-regime arrays). Defaults to a
        k-means hard-assignment start on the lag pairs.
    max_iter, tol : EM stopping rule on the log-likelihood increment.
    seed : int
        Seed of the k-means start (unused when ``theta0`` is given).

    Returns
    -------
    EMC0Result
        Fitted chain parameters, an embedding ``GSSParams`` and diagnostics.
    """
    ys_ = np.asarray(ys, dtype=float)
    ys_ = ys_.reshape(ys_.shape[0], -1)
    theta = theta0 if theta0 is not None else _kmeans_init(ys_, K, seed)
    log_liks: list[float] = []
    converged = False
    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        gamma, xi, ll = _estep(ys_, *theta)
        log_liks.append(float(ll))
        theta = _mstep(ys_, gamma, xi, theta)
        if len(log_liks) >= 2 and abs(log_liks[-1] - log_liks[-2]) < tol:
            converged = True
            break
    P, D, bY, SV = theta
    return EMC0Result(
        params=c0_chain_to_gss(P, D, bY, SV),
        P=np.asarray(P, dtype=float),
        D=[np.asarray(d, dtype=float) for d in D],
        bY=[np.asarray(b, dtype=float) for b in bY],
        SV=[np.asarray(v, dtype=float) for v in SV],
        log_liks=log_liks,
        n_iter=n_iter,
        converged=converged,
    )
