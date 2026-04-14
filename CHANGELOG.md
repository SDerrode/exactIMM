# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
