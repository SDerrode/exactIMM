# API Overview

Module map and the role of each component.

## Top-level layout

```
prg/
├── classes/        # Data containers (parameters, simulator, F matrix)
├── filter/         # Optimal filter (h5_exact and imm_general modes)
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

Mutable in tests, immutable in normal use. Provides `to_dict()` for
serialisation and `from_dict()` for round-trips.

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

filt = GSSFilter(params, mode="h5_exact")  # or "imm_general"
res  = filt.step(y)                         # one observation
print(res.E_x, res.Var_x, res.pi, res.log_lik, res.innovation)
```

| `mode` | Cost per step | Assumption |
|---|---|---|
| `h5_exact` | constant Kalman gain (precomputed) | requires (H5) |
| `imm_general` | full IMM mixing | works for any GSS |

Note that `h5_exact` issues a `RuntimeWarning` if the params do not
satisfy (H5) — you can suppress this with `warnings.catch_warnings()`
when you knowingly want to feed unconstrained params (as in the §7
fairness comparison).

## Learning

### `fit_supervised(rs, xs, ys, K, q, s, constraint=None, …)`

Per-regime weighted OLS in closed form. `constraint='ab'` applies a
post-hoc (H5)-compatible AB projection. Returns a parameter dict
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

### `compute_h5_residual(A, B, C, D, Su, Delta, Sv)`

Frobenius-norm residual of the (H5) algebraic identity
\(\Delta^T A^T + \Sigma_V B^T - P M^{-1}(Q A^T + R B^T + \Delta^T)\).
Returns a `(s, q)` array; \(\|F\|_F = 0\) iff (H5) holds.

### Error hierarchy

```
GSSError
├── GSSConfigError        # bad parameters at construction time
├── GSSStateError         # invalid runtime state (e.g. NaN posterior)
└── GSSConstraintError    # H5 not satisfied / cannot be enforced
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
