# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.13.0] — 2026-05-06

### Changed

- **Renamed the (H5)-compatible (A, B) parametrisation** introduced in
  v0.12.0 to **"AB constraint"** in the public API. The mathematical
  content is unchanged; only the spelling differs (the personal name
  attached to the original derivation is kept only in the math note
  ``docs/wojciech/H5_isolation_difficulty.tex``). Public renames:

  | v0.12.0                              | v0.13.0                  |
  | ------------------------------------ | ------------------------ |
  | ``compute_AB_lehmann``               | ``compute_AB``           |
  | ``apply_lehmann_constraint``         | ``apply_AB_constraint``  |
  | ``--constraint lehmann`` (CLI)       | ``--constraint ab``      |
  | ``constraint='lehmann'`` (Python)    | ``constraint='ab'``      |
  | "Lehmann constraint on (A(k), B(k))" | "AB constraint on (A(k), B(k))" |

  Internal GUI methods follow the same convention
  (``is_AB_constraint_active``, ``_recompute_AB``, ``_restore_AB``,
  ``_on_AB_toggled``, ``_on_apply_AB_all``, ``_constraint_AB_check``).

## [0.12.0] — 2026-05-06

### Added

- **Lehmann's (H5)-compatible parametrisation** as the canonical
  closed-form constraint on (A, B):

      A(k) = Δ(k) Σ_V(k)⁻¹ C(k),
      B(k) = Δ(k) Σ_V(k)⁻¹ D(k).

  Derived by F. Lehmann (handwritten note, 6 May 2026) from the
  requirement that (H5) holds *uniformly* in the joint covariance
  Σ(r₁), this is the unique parametrisation that makes the K²
  regime-pair equations of (H5) trivially satisfied.

- ``prg.utils.h5_constraint.compute_AB_lehmann(C, D, Δ, Σ_V) → (A, B)``
  — closed-form helper.
- ``prg.utils.h5_constraint.apply_lehmann_constraint(params)`` —
  return a new ``GSSParams`` with each regime's (A, B) replaced by the
  Lehmann form.
- ``scripts/verify_h5_compat.py`` — numerical verification of the
  parametrisation across regime pairs.
- 4 new pytest tests on Lehmann recovery and idempotency.

### Changed

- **GUI** ``ParamPanel``: the four mutually-exclusive H5 checkboxes
  (Constraint on A / B / C / Σ_U) and the independent Δ=0 checkbox are
  replaced by a **single** checkbox per regime, "Lehmann constraint on
  (A(k), B(k))". When checked, both A and B blocks of F(k) are read-only
  and recomputed from (C, D, Δ, Σ_V) on every edit. Unchecked by default
  on every newly built / loaded tab; previous A and B values are
  restored on toggle off.
- **Learning CLI** (``supervised``, ``semi_supervised``):
  ``--constraint {a,b,su}`` → ``--constraint lehmann``. The
  ``constraint='a'/'b'/'su'`` Python API is replaced by
  ``constraint='lehmann'``.
- **Reference paper models** ``M1``, ``M2``, ``M3``: A and B are now
  *both* computed from Lehmann (the previous hand-picked A values are
  superseded). The free blocks are now ``(C, D, Σ_U, Σ_V, Δ)``.

### Fixed

- LHS of the (H5) algebraic constraint corrected from
  ``Δᵀ A + Σ_V Bᵀ`` to ``Δᵀ Aᵀ + Σ_V Bᵀ`` in
  ``prg/utils/h5_constraint.py``, ``prg/filter/gss_filter.py``,
  ``prg/experiments/{models_paper,run_supervised}.py``,
  ``tests/test_h5_constraint.py``, and the paper sources
  (``paper/sections/04_constraint.tex``,
  ``paper/appendix/{B_h5_derivation,C_projections}.tex``,
  ``paper/sections/06_experiments.tex``). Internally inconsistent before
  the fix: ``compute_A_from_h5`` already used the correct convention
  while sister functions used the typo'd one — tests passed by
  bug-with-bug round-trip self-consistency.
- ``compute_SU_from_h5`` (since removed) used the non-exact
  rearrangement ``M Z = P W`` valid only when M and P commute. The
  pre-removal Lehmann-only refactor uses the equivalent exact form
  ``W = M P⁻¹ Z`` when computing Σ_U analytically.

### Removed

- Per-matrix H5 projection helpers ``compute_A_from_h5``,
  ``compute_B_from_h5``, ``compute_SU_from_h5``, ``compute_C_from_h5``,
  and ``apply_h5_constraint`` from ``prg.utils.h5_constraint``.
  The Lehmann parametrisation supersedes all four.
- All four corresponding GUI checkboxes (A / B / C / Σ_U) and the
  independent Δ=0 checkbox.

### Migrating from v0.11.x

| Old (v0.11)                                         | New (v0.12)                                       |
| --------------------------------------------------- | ------------------------------------------------- |
| ``apply_h5_constraint(params)``                     | ``apply_lehmann_constraint(params)``              |
| ``compute_B_from_h5(A, C, D, SU, Dt, SV)``          | ``_, B = compute_AB_lehmann(C, D, Dt, SV)``       |
| ``compute_A_from_h5(B, C, D, SU, Dt, SV)``          | ``A, _ = compute_AB_lehmann(C, D, Dt, SV)``       |
| ``compute_SU_from_h5(A, B, C, D, Dt, SV)``          | (no longer applicable — Σ_U is fully free)        |
| ``compute_C_from_h5(A, B, D, SU, Dt, SV)``          | (no longer applicable — C is fully free)          |
| ``--constraint a`` / ``b`` / ``su`` (CLI)           | ``--constraint lehmann``                          |
| ``constraint='a'/'b'/'su'`` (Python)                | ``constraint='lehmann'``                          |
| GUI checkboxes "Constraint on A / B / C / Σ_U"      | Single GUI checkbox "Lehmann constraint on (A, B)"|
| GUI checkbox "Δ = 0"                                | Edit the off-diagonal block of Σ_W manually       |

**Behavioural notes:**

- The GUI checkbox is **always unchecked** on a freshly built or loaded
  tab — users must opt in explicitly.
- Reference paper models ``M1``, ``M2``, ``M3`` now compute ``A`` from
  Lehmann; the previous hand-picked ``A`` values are superseded.
  Numerical Monte-Carlo results may shift accordingly.
- ``dof_h5(K, q, s)`` (free-parameter count) was corrected: under
  Lehmann both ``A`` and ``B`` contribute zero free parameters; ``D``
  (previously omitted by mistake) now contributes ``s²``. Values for
  ``q == s`` are unchanged; values for ``q ≠ s`` differ from v0.11.

## [0.11.0] — 2026-05-04

### Changed

- **Project renamed** from ``fofgss`` to ``exactIMM`` for consistency
  with the GitHub repository. The Python distribution name
  (``pyproject.toml``), all logger names (``logging.getLogger("exactIMM…")``),
  the GUI ``QSettings`` namespace, the session file extension
  (``.exactIMM``), and all branding in docs/wiki/CITATION have been
  updated. The GitLab remote was renamed accordingly to
  ``gitlab.ec-lyon.fr/sderrode/exactIMM``.

## [0.10.1] — 2026-04-20

### Added

- **Two filter modes** in ``GSSFilter``, selected by the new ``mode``
  constructor argument:
  - ``mode="imm_general"`` (new **default**) — IMM recursion with
    per-step moment propagation; no (H5) requirement. Matches the
    ``exactIMM ≤ v0.9.0`` implementation; correct for models with
    ``B(k) ≠ 0``.
  - ``mode="h5_exact"`` — exact IMM under hypothesis (H5), with
    stationary pre-computed regime moments (the v0.10.0 default).
    Requires ``B(k) = 0`` for all ``k``; emits a ``RuntimeWarning``
    at construction when (H5) is violated.
- ``GSSFilter.mode`` read-only property.
- **GUI**: filter-mode combo box ("IMM general" / "Exact IMM under
  (H5)") with QSettings persistence and tooltip. The Joseph form
  checkbox is automatically grayed out unless ``h5_exact`` is
  selected (Joseph is only meaningful in that mode). The session
  summary now shows ``filter=IMM-general`` or ``filter=H5-exact``
  (and appends ``cov=Joseph``/``cov=short`` only when h5_exact).
- ``TestFilterModes`` in ``tests/test_gss_filter.py``:
  default-mode, explicit-h5, warning on non-(H5) model, no-warning
  for imm_general, ``ValueError`` on unknown mode, and a regression
  check that ``imm_general`` strictly beats ``h5_exact`` in MSE on
  non-(H5) models.

### Changed

- **Default filter mode is ``imm_general``**, not ``h5_exact``. This
  reverses the v0.10.0 default: all shipped models (including the
  canonical ``model_gss_K2_q1_s1``) have ``B(k) ≠ 0`` and therefore
  gave biased filter outputs under v0.10.0's default. Users who
  specifically want the paper-exact h5 recursion must now opt in
  with ``mode="h5_exact"``.
- ``_FilterWorker`` (GUI) now takes a ``mode`` kwarg (default
  ``"imm_general"``) forwarded to ``GSSFilter``.
- ``TestJosephForm`` and ``TestStationaryMoments`` updated to
  construct the filter with an explicit ``mode="h5_exact"`` (both
  classes now carry a ``@pytest.mark.filterwarnings`` to silence the
  expected (H5)-violation warning when the default fixture model
  has ``B ≠ 0``).

### Fixed

- Filter now produces correct results on all shipped models (and
  user-custom non-(H5) models) by default. Regression check on
  ``model_gss_K2_q1_s1_custom1``: MSE drops from ≈ 34.8 (h5_exact
  default) to ≈ 1.5 (imm_general default) — a 20× improvement.

## [0.10.0] — 2026-04-20

### Added

- **`--constraint-each-iter` flag** for the semi-supervised EM
  estimator (`prg.learning.semi_supervised`).  When set, the H5
  projection (and `--delta-zero`) is applied at *every* M-step
  (Generalized-EM, log-likelihood may not be monotone).
- New tests in `tests/test_semi_supervised.py`:
  `test_post_hoc_keeps_log_lik_monotone`,
  `test_constraint_each_iter_b`,
  `test_post_hoc_vs_each_iter_differ`,
  `test_constraint_each_iter_passes`,
  `test_constraint_each_iter_flag` (CLI).
- **Joseph form for the mode-conditional posterior covariance
  update** in `prg/filter/gss_filter.py`: optional, controlled by
  `joseph: bool = False` at the `GSSFilter` constructor.
  Mathematically equivalent to the short form under stationarity but
  numerically more stable (symmetric and PSD by construction).
- **GUI checkbox "Joseph form (covariance update)"** in the main
  window, with QSettings persistence and tooltip.  The session
  summary now appends `cov=Joseph` or `cov=short`.
- New tests in `tests/test_gss_filter.py`: `TestJosephForm`
  (constructor flag, equivalence with the short form, end-to-end
  filter equivalence, PSD check) and `TestStationaryMoments`
  (`π_∞ P = π_∞`, fixed-point equation for `μ(k)`).
- New model files in `prg/models/`:
  - `model_em_K2_q1_s1.py` — parameters estimated by Baum-Welch EM
    (semi-supervised) on a synthetic K=2, q=s=1 trajectory of N=3000.
  - `model_gss_K2_q1_s1_contrast.py` — model designed to produce
    visually contrasted X⁰ and Y⁰ signals (decoupled cross-terms,
    swapped fixed points across regimes).
  - `model_gss_K2_q1_s1_custom.py` — placeholder template generated
    by the GUI for user-edited parameters.

### Changed

- **Default constraint timing in semi-supervised EM is now post-hoc.**
  `--constraint` (and `--delta-zero`) are no longer applied during
  EM iterations — instead, EM converges on the unconstrained
  log-likelihood and the projection is applied **once** to the
  best run's converged parameters.  This restores standard EM
  monotonicity by default and matches the supervised estimator's
  behaviour.  To recover the previous behaviour, pass
  `--constraint-each-iter` (or `constraint_each_iter=True` in the
  Python API).
- New keyword argument `constraint_each_iter: bool = False` on
  `fit_semi_supervised(...)` and the internal `_em_run(...)`.
- README updated: dedicated "Constraint timing" caveat, CLI table
  entry, and Python-API example.
- **Filter recursion rewritten as exact IMM under (H5)** in
  `prg/filter/gss_filter.py`.  The reverse-time regime transition
  now uses the **stationary** marginal `π_∞` (instead of the
  filtered `π_n`), aligning the implementation with the IEEE paper
  (§3, eq. *reverse_P*).  Under stationarity all per-regime Kalman
  quantities (gain `K_gain[k]`, posterior covariance `P_post[k]`)
  and pair-conditional likelihood quantities (`μ_Y(j,k)`, `M_t(j,k)`,
  `Γ(j,k)`) become constants in `n`: they are now pre-computed once
  in `_precompute()` at filter construction by fixed-point iteration
  on `μ(k), P(k)` (max 1000 iters, tol 1e-12) and stored on the
  filter instance.  The per-step `_update_step()` is correspondingly
  much shorter.
- `GSSFilter` now exposes `joseph` and `stationary_distribution`
  read-only properties.
- `tests/test_gss_filter.py::test_regime_probabilities_not_degenerate`
  relaxed: trajectories must visit both regimes (regime 0 preferred
  at some steps, regime 1 at others) instead of bounding the
  trajectory-mean of `p_r_0` between 0.2 and 0.8.  The new exact-IMM
  formulation is more confident in regime decisions when (H5) does
  not hold for the test model.
- **Build system**: `pyproject.toml` switched from the deprecated
  `setuptools.backends.legacy:build` backend to the standard
  `setuptools.build_meta`.
- `.gitignore` now excludes the `paper/` directory (LaTeX sources
  for the IEEE submission, kept out of the repo).
- `docs/CS_FinaleBis.tex` / `.pdf` updated with the new derivations
  (stationary regime moments, Joseph remark, time-reversed
  stationary transition).

## [0.9.0] — 2026-04-19

### Added

- **`prg/learning/semi_supervised.py`** — Baum-Welch EM estimator for
  the case where (X, Y) are observed but the regime sequence R is
  *hidden*.  The model is treated as an HMM with continuous emissions
  N(F(k) Z_n + b(k), Σ_W(k)) and Markov transitions P:
  - **E-step**: forward / backward in log-domain
    (`scipy.special.logsumexp`); Cholesky-based vectorised emission
    log-pdf (`_log_mvn_batch`); posteriors γ_n(k) and ξ_n(j,k)
  - **M-step**: closed-form weighted updates of P, π₀, μ_z0(k),
    Σ_z0(k); weighted OLS for F(k), b(k); weighted MLE for Σ_W(k);
    optional H5 projection on A / B / Σ_U *at every M-step* (Generalized
    EM — log-likelihood not guaranteed to be monotone)
  - **Initialisation**: k-means on the first differences ΔZ_n
    (`scipy.cluster.vq.kmeans2`), seeded by reproducible RNG draws
  - **Multi-start**: `n_inits` independent EM runs (default 10), best
    log-likelihood retained; failed runs are skipped
  - **Label-switching mitigation**: regimes reordered by A[0,0]
    descending after convergence
  - **`fit_semi_supervised(xs, ys, K, …)`** — public API; returns
    `(params, info)` where `info` contains `best_log_lik`,
    `best_init_seed`, `log_lik_history`, and `all_log_liks`
  - **CLI** (`python -m prg.learning.semi_supervised`): `-K`,
    `--constraint`, `--delta-zero`, `--n-inits`, `--max-iter`, `--tol`,
    `--seed`, `--output`, `--model-name`, `-v`
- **`tests/test_semi_supervised.py`** — 25 pytest tests:
  `_log_mvn_batch` against scipy reference, forward/backward against
  brute-force enumeration over K^N sequences, γ/ξ marginal consistency,
  k-means init validity & reproducibility, weighted-fit reduction to
  OLS, log-likelihood monotonicity (no constraint), constraint-b
  satisfaction post-EM, multi-start ranking, statistical recovery
  (A within 0.30 of truth at N=2000, K=2, 5 starts), regime reordering,
  CLI smoke tests

### Changed

- **CLI stem precedence** in both `supervised.py` and `semi_supervised.py`:
  the generated class name is now derived as `--model-name` if given,
  otherwise from the `--output` file stem if given, otherwise from the
  auto-generated stem (previously the auto stem was used whenever
  `--model-name` was omitted, yielding a class name that could mismatch
  the requested output filename)

## [0.8.0] — 2026-04-19

### Added

- **`prg/learning/` package** — new supervised estimation module
- **`prg/learning/supervised.py`** — estimate all GSS parameters from
  fully-observed (R, X, Y) data (CSV produced by `prg.simulate`):
  - `fit_supervised(rs, xs, ys, K, q, s, …)` — public API returning a
    parameter dict; per-regime OLS on pairs (Z_n, Z_{n+1}) for which
    r_{n+1} = k; P estimated from transition counts; initial conditions
    from sample moments per regime
  - `--constraint {a,b,su}` — post-hoc H5 projection: recompute A, B, or
    Σ_U analytically from the other estimated blocks (via
    `compute_A_from_h5`, `compute_B_from_h5`, `compute_SU_from_h5`)
  - `--delta-zero` — force Δ(k) = 0 before the H5 step
  - `_generate_model_code()` — renders a ready-to-use `BaseGSSModel`
    subclass file in the same style as the GUI export
  - CLI entry point (`python -m prg.learning.supervised <csv> [OPTIONS]`)
    with `--output`, `--model-name`, `-v`
- **`tests/test_supervised.py`** — 43 pytest tests covering `_read_csv`
  (valid/invalid inputs), `_nearest_spd`, `_fit_regime` (OLS exactness on
  noise-free data, shapes, `delta_zero`, `constraint='b'`),
  `fit_supervised` (keys, dimensions, row-stochastic P, SPD guarantees,
  error cases, statistical recovery within 0.15), code-generation
  helpers (`_fmt_arr` / `_fmt_list` eval-roundtrip), generated-file
  importability, and full CLI smoke tests

## [0.7.0] — 2026-04-19

### Added

- **`_SessionState` dataclass** (`prg/gui/main_window.py`) — single source of truth
  for all session results (data, params, innovations, mc_xs_all); replaces 5 scattered
  `_last_*` attributes with explicit predicates (`has_data()`, `can_filter()`, …) and
  atomic mutations (`reset()`, `begin_simulation()`, `store_innovations()`, …)
- **Worker cancellation** — new `_cancel_active_workers()` helper disconnects signals
  and calls `requestInterruption()` + `quit()` on all three workers; invoked by
  Reset and `closeEvent()`; each worker checks `isInterruptionRequested()` at fixed
  intervals (256 iterations) and aborts silently without emitting `finished`
- **Stale-signal guards** — every `_on_*_finished` / `_on_*_error` handler now
  verifies `self.sender() is self._<worker>` before touching state, preventing
  queued signals from corrupting a freshly-reset session
- **Param drift indicator** — `⚠ Filter` button label + tooltip when the GUI
  parameters differ from those captured at the last Simulate; the comparison is based
  on a byte-level signature (`_params_signature()`) recomputed on every cell edit;
  `VectorWidget.value_changed` signal added; propagated up via `_StateTab.value_changed`
  → `ParamPanel.value_changed`; status-bar message added at Filter launch
- **Filter enhancements** (`prg/filter/gss_filter.py`):
  - `_safe_solve(A, B)` — `lstsq` fallback when `np.linalg.solve` raises `LinAlgError`
  - `allow_singular=True` on all `multivariate_normal.logpdf` calls
  - NaN guards on π normalisation (fallback to π₀ or marginal)
  - `log_lik: float` field in `FilterResult`; incremental accumulation via `logsumexp`
- **Log-likelihood display** — `log L = … (mean = …/step)` shown in the Filter quality
  frame after each Filter run
- **π_n(k) dedicated subplot** — regime posteriors now occupy their own axis (height
  ratio 0.55) directly below R_n; R_n is a clean step plot again (no twinx)
- **Innovation diagnostics frame** — two-column grid: Ljung-Box whiteness badges
  (per component) + Skew · Kurt badges coloured by moment thresholds
  (|S| < 0.25 ∧ |K| < 0.50 = green; intermediate = amber; otherwise = red);
  tooltip explains that GSS innovations are theoretically a mixture of Gaussians
- **Menu bar** (`File` / `Simulation` / `View`) with keyboard shortcuts:
  Ctrl+S (Save CSV), Ctrl+O (Load CSV), Ctrl+E (Export model), Ctrl+Shift+E
  (Export plots), Ctrl+Q (Quit), Ctrl+R (Simulate), Ctrl+F (Filter),
  Ctrl+Shift+R (Reset), Ctrl+I (Innovation histograms), Ctrl+Shift+X (MC distributions)
- **Status bar** with live session summary `K=·q=·s= | N= | M= | seed= | auto-filter`
- **Auto-filter checkbox** — when checked, Filter runs automatically after each
  single Simulate completes
- **Progress dialog** — indeterminate spinner by default; switches to determinate bar
  with elapsed + ETA during Monte Carlo
- **QSettings persistence** for window geometry, splitter position, M, seed, and
  auto-filter state (N and MC-on intentionally not persisted — always start at
  defaults N = 1000, MC unchecked)
- **Export plots** button and Ctrl+Shift+E — saves the figure to PNG/PDF/SVG
- **Export model code** button and Ctrl+E — generates a ready-to-use `.py` model
  file from the current GUI parameters
- **Innovation histogram dialog** (Ctrl+I) — per-component KDE + Gaussian reference
- **MC X-distribution dialog** (Ctrl+Shift+X) — per-component, per-time-step KDE

### Fixed

- **X-axis not updating when N decreases** (`prg/gui/plot_panel.py`) — after `cla()`
  on multiple `sharex=True` axes, autoscale-x was silently disabled so new data did
  not drive the limits; `_set_shared_xlim()` now re-enables autoscale and pins
  x-limits to the current data range at the end of `update_plots`,
  `update_mc_plots`, and `update_innovations`
- **`Artist.remove()` crash after `cla()`** — `clear_filter_overlay`,
  `clear_innovations`, `_clear_mc_plots` now catch `NotImplementedError` and
  `AttributeError` in addition to `ValueError`
- **Wrong legend axis index** in `clear_filter_overlay` (`1 + i` → `_x_offset + i`)
- **Race condition on Reset** — a late `finished` signal from a cancelled worker
  could overwrite freshly-reset state; fixed by disconnect-before-null in
  `_cancel_active_workers()` combined with `sender()` guards in handlers
- **`_on_filter_finished` crash** when `_state.data` was cleared by a concurrent Reset
- `_on_reset()` now also closes the wait dialog and re-enables the Simulate button
  after cancelling an in-flight MC or Filter

### Changed

- Default N = 1000 at launch (not persisted); Monte Carlo checkbox unchecked at
  launch (not persisted); previous persisted values are cleaned up from QSettings
- Import hygiene (`prg/gui/main_window.py`): `csv`, `jarque_bera as _jb`,
  `GSSFilter`, `PRESETS` moved to module-level; redundant local re-imports of
  `GSSNoiseCovariance` and `FMatrix` inside `_load_model()` removed (the
  module-level names were already available)
- `VectorWidget` (`prg/gui/matrix_widget.py`) — new `value_changed = pyqtSignal()`
  emitted on every cell edit (previously only `validity_changed` was available)

## [0.6.0] — 2026-04-16

### Added
- **H5 constraint (eq. 4.8)** — given A(k), C(k), D(k), Σ_U(k), Δ(k), Σ_V(k), the
  B(k) block is uniquely determined by solving the linear system `L Bᵀ = rhs` where
  L = Σ_V − PM⁻¹R is the Schur complement of M (see `docs/CS_FinaleBis.tex` eqs 4.4–4.8)
- `prg/utils/h5_constraint.py` — new module: `compute_B_from_h5()` (core formula) and
  `apply_h5_constraint(params, *, logger)` returning a new `GSSParams` with corrected
  B(k) blocks; raises `ValueError` on singular or ill-conditioned systems (cond > 1e12)
- `--constraint` CLI flag — added to both `prg/simulate.py` and `prg/filter/main.py`;
  when set, B(k) is recomputed from the other 6 matrices before simulation/filtering
- **GUI constraint checkbox** (`prg/gui/param_panel.py`) — each `_StateTab` now has a
  green "Constraint on F(k)" checkbox; when checked, B(k) is auto-computed in real-time
  from the current A, C, D, Σ_U, Δ, Σ_V values, B cells become read-only (saturated
  green tint), and a status label "✓ B satisfies constraint (4.7)" appears; unchecking
  restores full editability
- **Block colour coding in F(k)** (`prg/gui/matrix_widget.py`) — matrix cells are now
  coloured by block: A=blue (#d6eaf8), B=green (#d5f5e3), C=yellow (#fef9e7),
  D=pink (#fde8e8); computed/locked cells use saturated versions
- `MatrixTableWidget.value_changed` signal — new `pyqtSignal()` that fires on every
  cell edit (complements the existing `validity_changed` which only fires on transitions)
- `tests/test_h5_constraint.py` — 11 new tests: output shape, constraint satisfaction
  (atol=1e-10), singular-M error, idempotency, preservation of A/C/D/noise/bias,
  and `apply_h5_constraint` roundtrip
- `docs/CS_FinaleBis.tex` — derivation section (eqs 4.4–4.8, red): general form,
  fully subscripted form, P/Q/R/M definitions, linear system in Bᵀ, and boxed solution

### Fixed
- **Filter quality label visibility** (`prg/gui/main_window.py`) — the status frame
  (green/amber/red) now uses Bootstrap-style dark foreground colours (#155724, #856404,
  #721c24) so text is readable in both light and dark OS themes

## [0.5.0] — 2026-04-15

### Added
- **Regime-dependent drift bias `b(k)`** — extends the state equation to
  `Z_{n+1} = F(r_{n+1}) Z_n + b(r_{n+1}) + W_{n+1}` (eq. 7bis, magenta)
- `GSSParams` — new optional parameter `b_list` (list of K vectors shape
  `(q+s, 1)`); new accessor `b(k)`; `from_model()` reads `b_list` from dict;
  `summary()` prints bias values; backward-compatible (defaults to zero)
- `GSSSimulator` — uses `params.b(r_n)` in the transition step
- `GSSFilter` — mean propagation (17ter) adds `+ b(r_{n+1})`; conditional
  mean of Y_{n+1} in (13') adds `+ b_Y(r_{n+1})`; second-moment recursion
  corrected to `P_{n+1} = F w_P F^T + F w_µ b^T + b w_µ^T F^T + b b^T + Σ_W`
  (ensures centred covariance `Σ = P − µµ^T` stays PSD for any bias magnitude)
- All five model files — `b_list` attribute (zero by default); exported in
  `get_params()`
- `ParamPanel` / `_StateTab` — `VectorWidget` editable for each `b(k)`;
  `get_b_list()`, `set_state_params(..., b=)` in `ParamPanel`
- `GSSMainWindow` — passes `b_list` to `GSSParams`; exports `b_list` in
  generated model file; loads `b_list` from model via `_load_model()`
- `docs/CS_FinaleBis.tex` — equations (7), (7bis), (17ter), (13') updated
  with `b_{r_{n+1}}` terms in **magenta**; definition block after (7ter)

### Fixed
- `GSSFilter._update_step` — second-moment propagation was incorrect when
  `b ≠ 0` (missing cross-terms `F w_µ b^T + b w_µ^T F^T + b b^T`), causing
  the centred covariance to go indefinite and the filter to diverge; now
  MSE/variance ≈ 0.16 for a strong-bias test case (was ≫ 1 before the fix)

## [0.4.0] — 2026-04-15

### Added
- `prg/filter/__init__.py` — filter package
- `prg/filter/gss_filter.py` — `GSSFilter`: fast optimal filter implementing
  Option B (general non-zero mean, CS_FinaleBis eqs I.1–I.3, 13'–22):
  - iterator interface `step(y_n)` returning a `FilterResult`
  - `run(N, seed, output_dir)` — simulate and filter jointly
  - `run_csv(path)` — filter from an existing simulation CSV
  - `reset()` — restart from n = 0
- `prg/filter/main.py` — CLI entry point (`python -m prg.filter.main`):
  `--model` + `-N`/`--seed` for simulate-and-filter; `--csv` to filter an
  existing file; `--no-save` dry-run; standard `-v`/`--log-level` options
- `tests/test_gss_filter.py` — 31 pytest tests covering construction,
  `FilterResult` shapes and PSD guarantees, multi-step recursion,
  reproducibility, `run()`/`run_csv()` consistency, statistical sanity
  (RMSE < naive baseline), and Option B / zero-mean equivalence
- `docs/CS_FinaleBis.tex` — initialisation section (eqs I.1–I.3) and
  Option B green annotations (eqs 13', 17ter, 21')

### Changed
- `pyproject.toml` — added `scipy>=1.14` to core dependencies
- `README.md` — added Filter section

## [0.3.0] — 2026-04-14

### Added
- `prg/gui/__init__.py` — GUI package
- `prg/gui/matrix_widget.py` — `MatrixTableWidget`: editable QTableWidget with
  block colour coding (A=blue, B=green, C=yellow, D=pink), per-cell float
  validation, optional SPD check for covariance matrices, `validity_changed` signal
- `prg/gui/param_panel.py` — `ParamPanel`: QTabWidget with one `_StateTab` per
  Markov state; each tab holds F(k) and Σ_W(k) widgets side by side
- `prg/gui/plot_panel.py` — `PlotPanel`: embedded matplotlib canvas
  (FigureCanvasQTAgg + NavigationToolbar2QT) with 1 + q + s subplots
  (R_n step, X_i lines, Y_i lines)
- `prg/gui/main_window.py` — `GSSMainWindow`: left panel (ParamPanel + N spinbox +
  seed field + [Simuler] / [Enregistrer CSV] buttons) + right PlotPanel;
  background `_SimWorker` (QThread) + modal `_WaitDialog`; [Simuler] disabled +
  red when any parameter is invalid; CSV auto-saved to `data/simulated/`
- `prg/gui/main.py` — CLI entry point (`python -m prg.gui.main`):
  optional `--model` flag pre-fills tables from any `BaseGSSModel`; `-K/-q/-s`
  for standalone launch without a model file

### Changed
- `pyproject.toml` — added `[gui]` optional-dependency group (PyQt6≥6.6, matplotlib≥3.8)
- `README.md` — added GUI section

## [0.2.0] — 2026-04-14

### Added
- `prg/utils/exceptions.py` — centralised exception hierarchy (`GSSError`,
  `ParamError`, `NumericalError`, `CovarianceError`, `SimulationError`)
- `prg/utils/matrix_checks.py` — diagnostic tools for matrices:
  `CovarianceMatrix` (symmetry + Cholesky) and `StochasticMatrix` (row-sums),
  with `DiagnosticReport` / `CheckResult` / `Status` pattern
- `prg/classes/FMatrix.py` — block transition matrix `F(k) = [[A_k, B_k], [C_k, D_k]]`
  with pre-computed full matrices cached at construction
- `prg/classes/NoiseCovariance.py` — `GSSNoiseCovariance`: block noise covariance
  `Σ_W(k)`, Cholesky factors cached at construction, SPD validation
- `prg/models/base_gss_model.py` — `BaseGSSModel` abstract base class;
  `MODEL_NAME` derived automatically from class name
- `prg/classes/GSSParams.py` — `GSSParams`: aggregates all model parameters,
  validates `P` (row-stochastic), `π₀` (sums to 1), `Σ_{z0}(k)` (SPD);
  `pi0=None` computes the stationary distribution automatically
- `prg/classes/GSSSimulator.py` — `GSSSimulator`: Python iterator yielding
  `(n, r_n, x_n, y_n)` with `x_n` shape `(q, 1)`, `y_n` shape `(s, 1)`;
  `run()` saves results to CSV in `data/simulated/`; `reset()` restarts from `n=0`
- `prg/models/model_gss_K2_q1_s1.py` — example model: K=2 states, q=1, s=1
- `prg/simulate.py` — CLI entry point (`python -m prg.simulate`):
  reads `config.toml` automatically, configures file + console logging,
  loads model dynamically from `prg/models/`
- `tests/` — 71 pytest tests covering matrix diagnostics, parameter validation,
  iterator protocol, reproducibility, CSV output, and statistical sanity

### Changed
- `README.md` — complete rewrite: scientific context, installation via venv,
  CLI usage, model authoring guide, test instructions

## [0.1.0] — 2026-04-13

### Added
- Initial project structure (`prg/`, `data/`, `logs/`, `tests/`)
- Data directory layout: `simulated/`, `output/`, `plot/`, `historyTracker/`
- Python virtual environment (Python 3.14) with numpy and pandas
- `.gitignore` for Python projects
- GNU AGPL v3 license
- `pyproject.toml` project configuration
- `config.toml` runtime configuration (paths, log level, simulation stubs)
