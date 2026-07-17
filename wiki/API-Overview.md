# API Overview

Module map and the role of each component.

## Top-level layout

```
prg/
├── classes/        # Data containers (parameters, simulator, F matrix)
├── filter/         # Optimal filter (ngh_kf and gpb2 modes)
├── learning/       # Supervised + semi-supervised estimators
├── models/         # Built-in BaseGSSModel subclasses
├── experiments/    # Paper-reproduction pipelines (§6, §7)
├── gui/            # PyQt6 interface (optional)
├── utils/          # Errors, matrix checks, AB constraint closed form
├── simulate.py     # CLI: simulator
└── filter/main.py  # CLI: filter
```

## Core classes

### `GSSParams` (`prg/classes/GSSParams.py`)

Aggregates all GSS model parameters and validates them. Construct from
a `BaseGSSModel`:

```python
params = GSSParams.from_model(MyModel())
print(params.K, params.q, params.s)
print(params.stationary_distribution())   # π_∞ from P
```

Mutable in tests, immutable in normal use. Build instances with
`from_model`; it exposes `K`, `q`, `s`, the `f_matrix`/`noise_cov`
blocks and `stationary_distribution()`.

### `GSSSimulator` (`prg/classes/GSSSimulator.py`)

Iterator: yields `(n, r, x, y)` tuples one step at a time.

```python
sim = GSSSimulator(params, seed=42)
for n, r, x, y in sim:
    ...                                    # n is 1-based, length unbounded
sim_df = sim.to_dataframe(N=1000)         # convenience: collect as DataFrame
```

### `FMatrix`, `NoiseCovariance` (`prg/classes/`)

Block representations of \(F(k)\) and \(\Sigma_W(k)\) with cached
spectral radius, conditioning, etc.

## Filtering

### `GSSFilter` (`prg/filter/gss_filter.py`)

```python
from prg.filter import GSSFilter

filt = GSSFilter(params, mode="ngh_kf")  # or "gpb2"
res  = filt.step(y)                         # one observation
print(res.E_x, res.Var_x, res.pi, res.log_lik, res.innovation)
```

| `mode` | Cost per step | Assumption |
|---|---|---|
| `ngh_kf` | constant Kalman gain (precomputed) | requires AB |
| `gpb2` | full IMM mixing | works for any GSS |

Both modes use the exact pair-conditional predictive covariance, so on
AB-constrained params they produce numerically identical output (checked
in `tests/test_filter_modes.py`); `gpb2` is the fallback for
unconstrained models.

Note that `ngh_kf` issues a `RuntimeWarning` if the params do not
satisfy AB — you can suppress this with `warnings.catch_warnings()`
when you knowingly want to feed unconstrained params (as in the §7
fairness comparison).

## Learning

### `fit_supervised(rs, xs, ys, K, q, s, constraint=None, …)`

Per-regime weighted OLS in closed form. `constraint='ab'` applies a
post-hoc AB projection. Returns a parameter dict
ready for `GSSParams`.

### `fit_semi_supervised(xs, ys, K, n_inits=10, max_iter=100, constraint=None, constraint_each_iter=False, …)`

Baum-Welch EM with k-means initialisation and multi-start. Returns
`(params_dict, info_dict)` where `info_dict` contains
`best_log_lik`, `all_log_liks`, `log_lik_history`, `best_n_iter`,
`best_converged`.

Set `constraint_each_iter=True` to obtain the **GEM** variant of
the paper (projection at every M-step, log-lik no longer monotone but
constraint holds throughout).

## Utilities

### `apply_AB_constraint(params)`

Recompute \(A(k)\) and \(B(k)\) for every regime from
\((C, D, \Delta, \Sigma_V)\) via the closed-form AB constraint
\(A = \Delta\,\Sigma_V^{-1}\,C\), \(B = \Delta\,\Sigma_V^{-1}\,D\).

### `compute_AB(C, D, Delta, Sv)`

Same, for a single regime. Returns the tuple `(A, B)`.

### `compute_ab_residual(A, B, C, D, Su, Delta, Sv)`

Frobenius-norm residual of the **same-regime** \((k, k)\) AB algebraic
identity
\(\Delta^T A^T + \Sigma_V B^T - P M^{-1}(Q A^T + R B^T + \Delta^T)\).
Returns a `(s, q)` array. \(\|F\|_F = 0\) is **necessary but not
sufficient** for AB: when regimes have different joint covariances,
AB also constrains the cross pairs \((j, k)\), \(j \neq k\).

### `compute_ab_pair_residual(A_k, B_k, C_k, D_k, Delta_k, Sv_k, Su_j, Delta_j, Sv_j)`

Pairwise residual \(\beta_1(j, k)\) (shape `(q, s)`) — the loading on
\(Y_n\) of the regression of \(X_{n+1}\) on \((Y_n, Y_{n+1})\) given
\(r_n = j,\ r_{n+1} = k\). \(\|\beta_1(j, k)\| = 0\) for **all** \(K^2\)
ordered pairs \(\iff\) AB holds.

### `ab_residual_max(params, *, relative=True)`

The **complete** AB check. Returns `(max_resid, (j, k))`: the largest
pairwise residual over all \(K^2\) ordered pairs (normalised by the
regression scale when `relative=True`). `max_resid == 0` \(\iff\) the
model is fully AB-constrained. This is the test `GSSFilter` uses to
decide whether to emit the `ngh_kf` warning.

### Error hierarchy

```
GSSError
├── GSSConfigError        # bad parameters at construction time
├── GSSStateError         # invalid runtime state (e.g. NaN posterior)
└── GSSConstraintError    # AB not satisfied / cannot be enforced
```

## Reproducibility helpers

The `prg/experiments/` package is structured so each script can be
invoked as a module *or* called from Python:

```python
from prg.experiments.run_real_data import (
    load_enso, run_e1, run_e2, run_e3,
)

data = load_enso(Path("data/real/enso_sst.csv"))
e2   = run_e2(data, K=3)
print([s for s in e2["scores"]])
```

This makes it trivial to re-run an experiment with different seeds or
hyper-parameters from a notebook.

## See also

- [Tutorial](Tutorial) — end-to-end example
- [Paper-Reproduce](Paper-Reproduce) — paper experiments
