# exactIMM — Fast Optimal Filtering in Gaussian Switching Systems

Implementation of the filtering and simulation algorithms described in:

> *On fast optimal filtering in Gaussian switching systems*  
> Stéphane Derrode & Wojciech Pieczynski (preprint, 2026)

The paper introduces an exact constant-gain filter for switching state-space
models satisfying a structural assumption (H5), validated by a Monte-Carlo
simulation study and a real-data experiment on NOAA SST anomalies (ENSO
regime detection). See [Reproducing the paper experiments](#reproducing-the-paper-experiments) below.

---

## Scientific context

The project targets **Gaussian Switching Systems (GSS)**, a class of hidden Markov models where the state $(X_n, Y_n)$ evolves according to a regime $R_n \in \{0,\ldots,K-1\}$:

$$Z_{n+1} = F(R_{n+1})\,Z_n + W_{n+1}, \qquad Z_n = \begin{bmatrix}X_n \\ Y_n\end{bmatrix}$$

with $F(k) = \begin{bmatrix}A_k & B_k \\ C_k & D_k\end{bmatrix}$ and $W_{n+1}\mid R_{n+1}=k \sim \mathcal{N}(0,\Sigma_W(k))$.

The central objective is to compute $\mathbb{E}[X_n \mid Y_{1:n}]$ efficiently (fast optimal filter).

---

## Project structure

```
exactIMM/
├── prg/
│   ├── utils/
│   │   ├── exceptions.py       # GSSError hierarchy
│   │   ├── matrix_checks.py    # CovarianceMatrix, StochasticMatrix diagnostics
│   │   └── h5_constraint.py    # H5 constraint: compute B(k) from A,C,D,Σ_U,Δ,Σ_V
│   ├── models/
│   │   ├── base_gss_model.py   # BaseGSSModel (abstract)
│   │   └── model_gss_K2_q1_s1.py  # Example: K=2, q=1, s=1
│   ├── classes/
│   │   ├── FMatrix.py          # Block transition matrix F(k)
│   │   ├── NoiseCovariance.py  # Block noise covariance Sigma_W(k)
│   │   ├── GSSParams.py        # Aggregates all model parameters
│   │   └── GSSSimulator.py     # Iterator-based simulator
│   ├── filter/
│   │   ├── gss_filter.py       # GSSFilter — fast optimal filter (Option B)
│   │   └── main.py             # CLI entry point for the filter
│   ├── learning/
│   │   ├── supervised.py       # OLS fit when R is observed
│   │   └── semi_supervised.py  # Baum-Welch EM when R is hidden
│   ├── experiments/            # §6 simulation + §7 ENSO pipelines (paper reproducibility)
│   │   ├── run_simulations.py  # §6.2 filter benchmark MC
│   │   ├── run_supervised.py   # §6.3 supervised OLS MC
│   │   ├── run_em.py           # §6.4 semi-supervised EM MC
│   │   ├── run_real_data.py    # §7 ENSO E1+E2+E3
│   │   ├── make_figures.py     # §6 figures + LaTeX tables
│   │   └── make_figures_real.py# §7 figures (regime trace)
│   ├── simulate.py             # CLI entry point for the simulator
│   └── gui/                    # Optional PyQt6 graphical interface
├── scripts/                    # Standalone exploratory scripts (E1/E2/E3 originals,
│   │                             baselines: Hamilton-MSAR, Kalman K=1)
│   ├── baselines/
│   ├── figures/
│   └── e{1,2,3}_*.py
├── tests/                      # 204 pytest tests
├── data/
│   ├── simulated/              # Generated CSV files (gitignored)
│   └── real/                   # Real datasets: ENSO (NOAA SST), SP500/VIX
├── logs/                       # Execution logs (one file per run)
├── config.toml                 # Runtime configuration
└── pyproject.toml
```

---

## Installation

Python 3.14 is required.  
**Use a project virtual environment** — the system-wide homebrew installation of numpy does not load correctly on this platform.

```bash
# Create and activate the venv
python3 -m venv .venv
source .venv/bin/activate

# Core + dev dependencies
pip install -e ".[dev]"

# Optional: GUI dependencies (PyQt6 + matplotlib)
pip install -e ".[gui]"
```

> The `.venv/` directory is excluded from version control (see `.gitignore`).

---

## Running the simulator

```bash
# Basic run: 1 000 steps, reproducible seed
python -m prg.simulate --model model_gss_K2_q1_s1 -N 1000 --seed 42

# Dry run (no CSV written), full debug output
python -m prg.simulate --model model_gss_K2_q1_s1 -N 500 --no-save --log-level DEBUG -v 2

# Enforce H5 constraint: recompute B(k) before simulating
python -m prg.simulate --model model_gss_K2_q1_s1 -N 1000 --seed 42 --constraint

# Custom output directory
python -m prg.simulate --model model_gss_K2_q1_s1 -N 2000 --seed 0 --output my_run.csv
```

Output CSV columns: `n, r, x_0, …, x_{q-1}, y_0, …, y_{s-1}`  
Log files are written automatically to `logs/`.

### CLI options

| Option | Default | Description |
|---|---|---|
| `--model` | — | Model name in `prg/models/` (required) |
| `-N` | — | Number of time steps (required) |
| `--seed` | `None` | Random seed (omit for non-deterministic) |
| `--output` | auto | Output CSV filename |
| `--log-level` | from `config.toml` | `DEBUG`/`INFO`/`WARNING`/`ERROR` |
| `--no-save` | `False` | Skip CSV writing (dry run) |
| `--constraint` | `False` | Enforce H5 constraint (eq. 4.8): recompute B(k) before run |
| `-v` / `--verbose` | `1` | Console verbosity: 0=silent, 1=normal, 2=debug |

---

## Running the filter

The filter computes $\mathbb{E}[X_n \mid Y_{1:n}]$, $\mathbb{E}[X_nX_n^T \mid Y_{1:n}]$, and $p(R_n \mid Y_{1:n})$ recursively.

```bash
# Simulate 1 000 steps and filter in one command
python -m prg.filter.main --model model_gss_K2_q1_s1 -N 1000 --seed 42

# Filter from an existing simulation CSV
python -m prg.filter.main --model model_gss_K2_q1_s1 \
    --csv data/simulated/simulated_model_gss_K2_q1_s1_N1000_seed42.csv

# Enforce H5 constraint before filtering
python -m prg.filter.main --model model_gss_K2_q1_s1 -N 1000 --seed 42 --constraint

# Dry run (no CSV written)
python -m prg.filter.main --model model_gss_K2_q1_s1 -N 500 --no-save -v 2
```

Output CSV columns: `n, E_x_0, …, E_x_{q-1}, V_x_0, …, V_x_{q-1}, p_r_0, …, p_r_{K-1}, sq_err`

| Column | Description |
|---|---|
| `E_x_i` | $\mathbb{E}[X_{n,i} \mid Y_{1:n}]$ — filtered mean |
| `V_x_i` | $\mathrm{Var}(X_{n,i} \mid Y_{1:n})$ — posterior variance (diagonal) |
| `p_r_k` | $p(R_n = k \mid Y_{1:n})$ — regime probability |
| `sq_err` | $\|X_n - \mathbb{E}[X_n \mid Y_{1:n}]\|^2$ — squared error (requires true X) |

### CLI options

| Option | Default | Description |
|---|---|---|
| `--model` | — | Model name in `prg/models/` (required) |
| `--csv` | `None` | Path to an existing simulation CSV |
| `-N` | — | Steps to simulate (required without `--csv`) |
| `--seed` | `None` | Random seed for simulation |
| `--output` | auto | Output CSV filename for filter results |
| `--log-level` | from `config.toml` | `DEBUG`/`INFO`/`WARNING`/`ERROR` |
| `--no-save` | `False` | Skip all CSV writing (dry run) |
| `--constraint` | `False` | Enforce H5 constraint (eq. 4.8): recompute B(k) before run |
| `-v` / `--verbose` | `1` | Console verbosity: 0=silent, 1=normal, 2=debug |

### Python API

```python
from prg.classes.GSSParams import GSSParams
from prg.models.model_gss_K2_q1_s1 import ModelGss_K2_q1_s1
from prg.filter import GSSFilter

params = GSSParams.from_model(ModelGss_K2_q1_s1())
filt   = GSSFilter(params)

# Step by step
for y in observations:          # y shape (s,) or (s, 1)
    res = filt.step(y)
    print(res.E_x)              # E[X_n | y_{1:n}]  shape (q, 1)
    print(res.Var_x)            # Var[X_n | y_{1:n}] shape (q, q)
    print(res.pi)               # p(r_n | y_{1:n})   shape (K,)

# Or in one call
sim_path, df = filt.run(N=1000, seed=42, output_dir="data/simulated")
```

### H5 constraint API

```python
from prg.utils.h5_constraint import apply_h5_constraint, compute_B_from_h5

# Apply to all regimes at once
constrained_params = apply_h5_constraint(params)

# Or compute B for a single regime
B = compute_B_from_h5(A, C, D, SU, Delta, SV)   # returns (q, s) array
```

---

## Learning the model from data

Two estimators are available:

- **Supervised** (`prg.learning.supervised`) — when the regime sequence
  $R_n$ is observed (alongside $X_n$ and $Y_n$): closed-form per-regime
  OLS, fast, deterministic.
- **Semi-supervised** (`prg.learning.semi_supervised`) — when only
  $X_n$ and $Y_n$ are observed and $R_n$ is hidden: Baum-Welch EM with
  k-means initialisation and multi-start.

Both write a ready-to-use `BaseGSSModel` subclass to `prg/models/`.

---

## Supervised learning

The supervised estimator learns all GSS parameters from a fully-observed
$(R_n, X_n, Y_n)$ CSV (such as those produced by the simulator).

```bash
# Estimate from a simulated CSV (no constraint)
python -m prg.learning.supervised data/simulated/simulated_model_gss_K2_q1_s1_N1000_seed42.csv

# Enforce H5 constraint on B (post-hoc) and force Δ = 0
python -m prg.learning.supervised sim.csv --constraint b --delta-zero

# Enforce H5 constraint on A; save to a custom file
python -m prg.learning.supervised sim.csv --constraint a \
    --output prg/models/model_my_estimated.py --model-name model_my_estimated

# Verbose output (per-regime summaries)
python -m prg.learning.supervised sim.csv -v
```

The generated `.py` file is a `BaseGSSModel` subclass and can be used directly:

```bash
python -m prg.simulate --model model_learned_K2_q1_s1 -N 1000 --seed 42
python -m prg.filter.main --model model_learned_K2_q1_s1 -N 1000 --seed 42
```

### CLI options

| Option | Default | Description |
|---|---|---|
| `csv` | — | Path to simulation CSV (required) |
| `--constraint {a,b,su}` | `None` | H5 post-hoc projection on A, B, or Σ_U |
| `--delta-zero` | `False` | Force Δ(k) = 0 before the H5 step |
| `--output PATH` | auto | Destination `.py` file |
| `--model-name NAME` | auto | File/class base name |
| `-v` / `--verbose` | `False` | Print per-regime fit summaries |

### Python API

```python
from prg.learning.supervised import fit_supervised, _read_csv
import pathlib

rs, xs, ys, K, q, s = _read_csv(pathlib.Path("sim.csv"))
params = fit_supervised(rs, xs, ys, K, q, s, constraint="b", delta_zero=True)

# params is a dict ready for GSSParams.from_model() or code generation
from prg.classes.GSSParams import GSSParams
gss_params = GSSParams(
    K=params["K"], q=params["q"], s=params["s"],
    P=params["P"],
    A_list=params["A_list"], B_list=params["B_list"],
    C_list=params["C_list"], D_list=params["D_list"],
    Sigma_U_list=params["Sigma_U_list"],
    Delta_list=params["Delta_list"],
    Sigma_V_list=params["Sigma_V_list"],
    pi0=params["pi0"],
    mu_z0_list=params["mu_z0_list"],
    Sigma_z0_list=params["Sigma_z0_list"],
    b_list=params["b_list"],
)
```

### Estimation approach

For each regime $k$ the model is $Z_{n+1} = F(k)\,Z_n + b(k) + W_{n+1}$.

1. **Free OLS** — collect all pairs $(Z_n, Z_{n+1})$ for which $r_{n+1}=k$,
   augment with a constant column, and solve the least-squares problem.
   The noise covariance $\Sigma_W(k)$ is the MLE sample covariance of residuals.

2. **Δ = 0** *(optional)* — zero out the off-diagonal block of $\Sigma_W(k)$.

3. **H5 projection** *(optional)* — recompute $A$, $B$, or $\Sigma_U$ from the
   remaining estimated blocks using the analytical H5 formula (eq. 4.8).

The Markov matrix $P$ is estimated by transition-frequency counts.
Constraints `--constraint {a,b,su}` are mutually exclusive; `--delta-zero`
is independent and applied before the H5 step.

---

## Semi-supervised learning

When $R_n$ is hidden but $(X_n, Y_n)$ are observed, the regime sequence
is inferred jointly with the parameters by Baum-Welch EM.

```bash
# Estimate K=2 regimes from a CSV (the 'r' column is ignored)
python -m prg.learning.semi_supervised data/simulated/sim.csv -K 2

# 20 random restarts, verbose log-L per iteration
python -m prg.learning.semi_supervised sim.csv -K 2 \
    --n-inits 20 --seed 42 -v

# H5 constraint applied once at the end of EM (default — log-lik monotone)
python -m prg.learning.semi_supervised sim.csv -K 2 \
    --constraint b --delta-zero

# Same constraint, but enforced at every M-step (Generalized EM mode)
python -m prg.learning.semi_supervised sim.csv -K 2 \
    --constraint b --delta-zero --constraint-each-iter
```

### CLI options

| Option | Default | Description |
|---|---|---|
| `csv` | — | Input CSV (the `r` column is ignored if present) |
| `-K`, `--K` | — | Number of regimes (required) |
| `--constraint {a,b,su}` | `None` | H5 projection target (post-hoc by default) |
| `--delta-zero` | `False` | Force Δ(k) = 0 before the H5 step |
| `--constraint-each-iter` | `False` | Apply the constraint at every M-step (GEM); otherwise applied once at the end |
| `--n-inits` | `10` | Number of independent EM restarts |
| `--max-iter` | `100` | Maximum EM iterations per run |
| `--tol` | `1e-5` | Convergence threshold on \|Δ log L\| |
| `--seed` | `None` | Base RNG seed (different k-means seeds derived from it) |
| `--output` / `--model-name` | auto | Same as supervised |
| `-v` | `False` | Print per-iteration log-likelihood |

### Algorithm

For each EM iteration:

1. **E-step** — log-domain forward / backward on the HMM with emissions
   $p(Z_n \mid Z_{n-1}, R_n=k) = \mathcal{N}(F(k) Z_{n-1} + b(k), \Sigma_W(k))$
   yields posteriors $\gamma_n(k)$ and $\xi_n(j,k)$ and the marginal
   log-likelihood.
2. **M-step**:
   - $\hat P(j,k) = \sum_n \xi_n(j,k) / \sum_n \gamma_n(j)$
   - $\hat\pi_0(k) = \gamma_0(k)$
   - $\hat F(k), \hat b(k)$ by **weighted OLS** with weights $\gamma_{n+1}(k)$
   - $\hat\Sigma_W(k)$ = weighted MLE of residual covariance
   - Optional H5 projection on $A$, $B$, or $\Sigma_U$ (only with
     `--constraint-each-iter`; otherwise projected once after EM)
3. **Convergence** — stop when $|\Delta \log L| < \mathrm{tol}$.
4. **Post-hoc projection** — if `--constraint` is set without
   `--constraint-each-iter`, the H5 projection (and $\Delta = 0$) is
   applied **once** to the converged parameters of the best run.

### Notes & caveats

- **Multi-start is essential.** EM is non-convex; the algorithm runs
  `--n-inits` independent k-means initialisations and keeps the
  best-likelihood solution.
- **Label switching.** After convergence, regimes are reordered by
  $A(k)[0,0]$ (descending) for reproducibility. Don't expect the
  estimated regime indices to match a known ground truth without
  post-hoc alignment.
- **Constraint timing.** By default the H5 projection is applied **once
  at the end** of EM, so the iterations remain a standard EM with
  monotone log-likelihood — the constraint behaves exactly like the
  supervised post-hoc projection. Pass `--constraint-each-iter` to
  enforce H5 at every M-step (Generalized EM): the constraint is
  satisfied throughout the optimisation but log-likelihood
  monotonicity is no longer guaranteed (convergence is then monitored
  on $|\Delta \log L|$).
- **Required N.** With $K=2$ and $q=s=1$, $N \ge 2000$ typically gives
  $|\hat A - A_{\text{true}}| < 0.05$.

### Python API

```python
from prg.learning.semi_supervised import fit_semi_supervised

params, info = fit_semi_supervised(
    xs, ys, K=2,
    constraint=None,            # 'a' | 'b' | 'su' | None
    delta_zero=False,
    constraint_each_iter=False, # True → GEM; False (default) → post-hoc
    n_inits=10,
    max_iter=100,
    tol=1e-5,
    seed=42,
)
print(info["best_log_lik"])           # log-likelihood of best run
print(info["all_log_liks"])           # log L of every restart
print(info["log_lik_history"])        # per-iteration log L of best run
```

---

## Graphical interface

The GUI requires `pip install -e ".[gui]"` (PyQt6 + matplotlib).

```bash
# Standalone launch: choose K, q, s interactively
python -m prg.gui.main -K 2 -q 1 -s 1

# Pre-fill from an existing model file
python -m prg.gui.main --model model_gss_K2_q1_s1
```

**Left panel — Parameters**

| Widget | Description |
|---|---|
| Preset selector | Load any model from `prg/models/` in one click; triggers a full window restart when K, q, or s differ |
| Parameter tabs | One tab per Markov state; editable F(k), Σ_W(k), μ_{z0}(k), b(k) with block colour coding (A=blue, B=green, C=yellow, D=pink) |
| H5 constraint checkboxes | Four per tab (A / B / Σ_U / Δ=0); auto-compute B(k) or enforce Δ=0 in real-time; affected cells become read-only |
| Stability badges | Spectral radii ρ(F), ρ(A), ρ(D) shown live below the F(k) table |
| Randomize button 🎲 | Fills F(k) and Σ_W(k) with random stable parameters |
| Transition matrix P | Editable (K×K) row-stochastic matrix; stationary distribution π* shown live |
| N / Seed | Number of steps (default 1 000); optional integer seed |
| Monte Carlo checkbox | Enables M-trajectory mode; M spinbox (default 50) |
| Auto-filter checkbox | Runs Filter automatically after each single Simulate |

**Right panel — Plots** (`2 + q + 2s` subplots, shared x-axis)

| Subplot | Content |
|---|---|
| R_n | Regime sequence (step plot) |
| π_n(k) | Filtered regime posteriors — one line per state, populated after Filter |
| X^i (×q) | Hidden component(s); filter overlay (dashed mean ± 2σ band) added after Filter |
| Y^i (×s) | Observed component(s) |
| ν^i (×s) | Filter innovations, populated after Filter |

**Simulation modes**

| Mode | How |
|---|---|
| Single trajectory | Simulate → optional Filter (or tick Auto-filter) |
| Monte Carlo | Tick Monte Carlo, set M, Simulate — shows mean ± 2σ + median over M runs |
| Load CSV | File → Load CSV — display external data, then run Filter with current params |

**Filter quality frame** (shown after Filter)

| Metric | Description |
|---|---|
| log L | Total log-likelihood and per-step mean |
| MSE / RMSE | Against ground-truth X (only when X is available) |
| Ljung-Box | Whiteness test per innovation component (green/amber/red badge) |
| Skew · Kurt | Skewness and excess kurtosis badges; note that GSS innovations are theoretically a mixture of Gaussians, so non-zero kurtosis is expected |

**Param drift indicator** — if you edit any parameter after Simulate and before Filter, the Filter button shows `⚠ Filter` with a tooltip; the Filter will use the parameters captured at Simulate, not the current widget values. Re-run Simulate to apply the new parameters.

**Keyboard shortcuts**

| Shortcut | Action |
|---|---|
| Ctrl+R | Simulate |
| Ctrl+F | Filter |
| Ctrl+Shift+R | Reset |
| Ctrl+S | Save CSV |
| Ctrl+O | Load CSV |
| Ctrl+E | Export model code (.py) |
| Ctrl+Shift+E | Export plots (PNG/PDF/SVG) |
| Ctrl+I | Innovation histograms |
| Ctrl+Shift+X | MC X distributions |

Window geometry, splitter position, M, seed, and auto-filter state are restored across sessions.

---

## Adding a new model

Create a file `prg/models/model_<name>.py` that subclasses `BaseGSSModel`:

```python
import numpy as np
from prg.models.base_gss_model import BaseGSSModel

class ModelMyGSS(BaseGSSModel):
    K, q, s = 3, 2, 1

    P = np.array(...)          # (K, K) row-stochastic

    A_list = [...]             # list of K arrays, each (q, q)
    B_list = [...]             # list of K arrays, each (q, s)
    C_list = [...]             # list of K arrays, each (s, q)
    D_list = [...]             # list of K arrays, each (s, s)

    Sigma_U_list = [...]       # list of K arrays, each (q, q) — SPD
    Delta_list   = [...]       # list of K arrays, each (q, s)
    Sigma_V_list = [...]       # list of K arrays, each (s, s) — SPD

    pi0 = None                 # None → stationary distribution of P
    mu_z0_list    = [...]      # list of K arrays, each (q+s, 1)
    Sigma_z0_list = [...]      # list of K arrays, each (q+s, q+s) — SPD

    def get_params(self) -> dict:
        return {k: getattr(self, k) for k in (
            "K", "q", "s", "P",
            "A_list", "B_list", "C_list", "D_list",
            "Sigma_U_list", "Delta_list", "Sigma_V_list",
            "pi0", "mu_z0_list", "Sigma_z0_list",
        )}
```

Then run with `--model model_my_gss`.

---

## Reproducing the paper experiments

The full pipeline for both the simulation study (§6) and the ENSO
real-data experiment (§7) is in `prg/experiments/`. Each script is
self-contained, deterministic (fixed seeds), and writes JSON + LaTeX
tables ready for paper inclusion.

### §6 — Simulation study (Monte-Carlo)

Three reference models (M1, M2, M3) all satisfying (H5). Each script
takes a few minutes to ~2h depending on the run.

```bash
# §6.2 filter benchmark   (~30 min, 1800 trials)
python -m prg.experiments.run_simulations

# §6.3 supervised OLS MC  (~15 min)
python -m prg.experiments.run_supervised

# §6.4 semi-supervised EM (~2h, 360 trials)
python -m prg.experiments.run_em

# Generate all PDF figures + LaTeX tables for §6
python -m prg.experiments.make_figures

# Fill numerical \ph{} placeholders in paper/sections/06_experiments.tex
python -m prg.experiments.fill_placeholders
```

Outputs land in `data/experiments/` (CSV results) and
`paper/figures/generated/` (PDFs + `.tex` tables).

### §7 — ENSO real-data experiment

Tests (H5) empirically and applies the framework to NOAA monthly SST
anomalies (Niño 1+2 as state, Niño 3.4 as observation, ONI-derived
3-regime label).

```bash
# Full pipeline E1 (H5 test) + E2 (filter comparison) + E3 (EM variants)
# (~1 minute total)
python -m prg.experiments.run_real_data

# Time-series figures (overview + regime trace)
python -m prg.experiments.make_figures_real
```

Raw NOAA dumps are in `data/real/{nino34,nino12,oni}.txt`; the unified
CSV `data/real/enso_sst.csv` is regenerated by the test harness if
removed (see `scripts/build_enso_csv.py` if you need a clean rebuild).

### Reproducibility checklist

- All scripts accept `--seed` (default 42) and write `*_summary.json`
  alongside `*.tex` for byte-level reproducibility.
- The 204-test suite (`pytest`) runs in under a minute and validates
  the underlying components (filter, EM, projections, H5 constraint)
  on which the paper claims rest.

---

## Testing

```bash
source .venv/bin/activate
pytest
```

204 tests covering matrix diagnostics, parameter validation, iterator
protocol, reproducibility, CSV output, statistical sanity, filter
correctness, H5 constraint computation, supervised OLS estimation, and
Baum-Welch EM (forward/backward, weighted M-step, multi-start,
constraint integration).

---

## Configuration

Edit `config.toml` to change default paths and log level:

```toml
[general]
log_level = "INFO"    # DEBUG | INFO | WARNING | ERROR

[paths]
data_simulated  = "data/simulated"
logs            = "logs"
```

---

## Citing

If you use this code or the (H5) framework in your work, please cite:

```bibtex
@misc{derrode_exactIMM_2026,
  author       = {Derrode, St{\'e}phane and Pieczynski, Wojciech},
  title        = {{On Fast Optimal Filtering in Gaussian Switching Systems}},
  year         = {2026},
  howpublished = {\url{https://github.com/SDerrode/exactIMM}},
  note         = {Software v0.11.0, accompanying paper preprint},
}
```

A `CITATION.cff` file is also provided at the repo root.

---

## Authors

Stéphane Derrode — [stephane.derrode@ec-lyon.fr](mailto:stephane.derrode@ec-lyon.fr)
([École Centrale de Lyon](https://www.ec-lyon.fr/))

Wojciech Pieczynski — [wojciech.pieczynski@telecom-sudparis.eu](mailto:wojciech.pieczynski@telecom-sudparis.eu)
([Télécom SudParis](https://www.telecom-sudparis.eu/))

## License

GNU Affero General Public License v3.0 — see `LICENSE`.
