# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
