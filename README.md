# exactIMM — On Fast Optimal Filtering in Gaussian Switching Systems

Simulation and **fast exact jump-filtering** algorithms for the paper:

> *On Fast Optimal Filtering in Gaussian Switching Systems*  
> Stéphane Derrode, Clément Fernandes, Frédéric Lehmann & Wojciech Pieczynski (preprint, 2026)

The paper introduces an exact, **linear-time** optimal filter for Gaussian
switching state-space models satisfying a closed-form structural constraint
(the **AB / (H5) constraint**). The LaTeX sources of the current version live in
`docs/wojciech/article_vWojciech_tex/`;
the commands that regenerate its figures are in
[Reproducing the paper figures](#reproducing-the-paper-figures) below.

This README documents the **jump-filtering** pipeline: define a model →
simulate → filter exactly.

---

## Scientific context

The project targets **Gaussian Switching Systems (GSS)**, a class of hidden Markov models where the state $(X_n, Y_n)$ evolves according to a regime $R_n \in \{0,\ldots,K-1\}$:

$$Z_{n+1} = F(R_{n+1})\,Z_n + W_{n+1}, \qquad Z_n = \begin{bmatrix}X_n \\ Y_n\end{bmatrix}$$

with $F(k) = \begin{bmatrix}A_k & B_k \\ C_k & D_k\end{bmatrix}$ and $W_{n+1}\mid R_{n+1}=k \sim \mathcal{N}(0,\Sigma_W(k))$.

The central objective is to compute $\mathbb{E}[X_n \mid Y_{1:n}]$ efficiently (fast optimal filter).

The (H5) structural assumption translates into an algebraic constraint
between the seven block matrices $(A, B, C, D, \Sigma_U, \Sigma_V, \Delta)$
of each regime. Since **v0.13.0** the constraint is enforced by the
closed-form **AB constraint**:

$$
A(k) = \Delta(k)\,\Sigma_V(k)^{-1}\,C(k),
\qquad
B(k) = \Delta(k)\,\Sigma_V(k)^{-1}\,D(k).
$$

This is the unique parametrisation of $(A, B)$ compatible with (H5)
*uniformly* in the joint covariance $\Sigma(r_1)$, making the $K^2$
regime-pair equations of (H5) trivially satisfied. The free blocks per
regime become $(C, D, \Sigma_U, \Sigma_V, \Delta)$.

```python
from prg.utils.h5_constraint import compute_AB, apply_AB_constraint

# Single regime: closed-form helper
A, B = compute_AB(C, D, Delta, Sigma_V)

# All regimes at once: project an existing GSSParams onto the (H5) manifold
constrained_params = apply_AB_constraint(params)
```

In the GUI, a single "AB constraint on (A(k), B(k))" checkbox per
regime locks both blocks simultaneously. See the
[CHANGELOG](CHANGELOG.md) for details and the v0.11 → v0.13 migration
table.

**Why this matters for filtering.** The AB constraint is exactly the
closed-form condition under which the observation marginal $(R_n, Y_n)$ is
itself a homogeneous Markov chain. This *marginal Markovianity* is what
makes an **exact, linear-time** optimal filter possible: $p(R_n \mid
Y_{1:n})$ is propagated by a forward recursion and $\mathbb{E}[X_n \mid
Y_{1:n}]$ then follows in closed form, with no Riccati covariance
propagation to carry. The condition $C(k) \neq 0$ characterises this new
family, distinguishing it from the classical conditionally-Gaussian
switching models ($C = 0$); it is checked at model-build time.

---

## Project structure

```
exactIMM/
├── prg/
│   ├── utils/
│   │   ├── exceptions.py       # GSSError hierarchy
│   │   ├── matrix_checks.py    # CovarianceMatrix, StochasticMatrix diagnostics
│   │   └── h5_constraint.py    # (H5) AB constraint closed form: A=Δ Σ_V⁻¹ C, B=Δ Σ_V⁻¹ D
│   ├── models/
│   │   ├── base_gss_model.py   # BaseGSSModel (abstract)
│   │   └── model_gss_K2_q1_s1.py  # Example: K=2, q=1, s=1
│   ├── classes/
│   │   ├── FMatrix.py          # Block transition matrix F(k)
│   │   ├── NoiseCovariance.py  # Block noise covariance Sigma_W(k)
│   │   ├── GSSParams.py        # Aggregates all model parameters
│   │   └── GSSSimulator.py     # Iterator-based simulator
│   ├── filter/
│   │   ├── gss_filter.py       # GSSFilter — fast exact optimal filter
│   │   └── main.py             # CLI entry point for the filter
│   ├── experiments/
│   │   └── study.py            # Jump-filtering study (E1–E9) → paper figures
│   ├── simulate.py             # CLI entry point for the simulator
│   └── gui/                    # Optional PyQt6 graphical interface
├── docs/wojciech/
│   └── article_vWojciech_tex/  # LaTeX sources of the current paper version
├── tests/                      # pytest suite
├── data/
│   └── simulated/              # Generated CSV files (gitignored)
├── logs/                       # Execution logs (one file per run)
├── config.toml                 # Runtime configuration
└── pyproject.toml
```

---

## Installation

Python **3.14** is required (declared in `pyproject.toml`).  
**Use a project virtual environment** — the system-wide Homebrew installation of NumPy does not load correctly on this platform.

```bash
# Create and activate the venv (must point to Python 3.14)
python3.14 -m venv .venv
source .venv/bin/activate

# Core + dev dependencies
pip install -e ".[dev]"

# Optional: GUI dependencies (PyQt6 + matplotlib)
pip install -e ".[gui]"
```

> The `.venv/` directory is excluded from version control (see `.gitignore`).

### Recommended: use the `Makefile`

A `Makefile` wraps the canonical workflow and **always invokes the venv
interpreter explicitly** (`.venv/bin/python`), which avoids the silent
"wrong interpreter" failures described below:

```bash
make venv      # create .venv with python3.14
make install   # editable install + dev deps
make test      # run the test suite
make check     # lint + typecheck + tests (CI parity)
make help      # list all targets
```

Override the bootstrap interpreter if needed: `make venv PYTHON_SYS=/path/to/python3.14`.

### Troubleshooting: `Package 'exactimm' requires a different Python`

If `pip install` fails with

```
ERROR: Package 'exactimm' requires a different Python: 3.11.15 not in '>=3.14'
```

it means `pip` is bound to a Python interpreter older than 3.14 (typical
on macOS where Homebrew installs `pip` for `python@3.11`). Check with:

```bash
pip --version
# pip 26.x from /opt/homebrew/lib/python3.11/site-packages/pip (python 3.11)  ← wrong
```

Fix it with **one** of the following:

```bash
# (a) Activate the project venv first — cleanest for development
source .venv/bin/activate && pip install -e ".[dev]"

# (b) Use the venv's pip explicitly, no activation needed
./.venv/bin/pip install -e ".[dev]"

# (c) Bypass the global pip shim and use python 3.14 directly
python3.14 -m pip install -e ".[dev]"

# (d) Use the version-suffixed pip
pip3.14 install -e ".[dev]"
```

After running any of these, `pip --version` from within the venv should report `(python 3.14)`.

---

## Running the simulator

```bash
# Basic run: 1 000 steps, reproducible seed
python -m prg.simulate --model model_gss_K2_q1_s1 -N 1000 --seed 42

# Dry run (no CSV written), full debug output
python -m prg.simulate --model model_gss_K2_q1_s1 -N 500 --no-save --log-level DEBUG -v 2

# Enforce (H5) AB constraint: A, B from (C, D, Δ, Σ_V) before simulating
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
| `--constraint` | `False` | Apply (H5)-compatible AB constraint: A=Δ Σ_V⁻¹ C, B=Δ Σ_V⁻¹ D before run |
| `-v` / `--verbose` | `1` | Console verbosity: 0=silent, 1=normal, 2=debug |

---

## Running the filter

The filter computes $\mathbb{E}[X_n \mid Y_{1:n}]$, $\mathbb{E}[X_nX_n^T \mid Y_{1:n}]$, and $p(R_n \mid Y_{1:n})$ recursively, in time linear in $n$.

```bash
# Simulate 1 000 steps and filter in one command
python -m prg.filter.main --model model_gss_K2_q1_s1 -N 1000 --seed 42

# Filter from an existing simulation CSV
python -m prg.filter.main --model model_gss_K2_q1_s1 \
    --csv data/simulated/simulated_model_gss_K2_q1_s1_N1000_seed42.csv

# Enforce (H5) AB constraint before filtering
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
| `--constraint` | `False` | Apply (H5)-compatible AB constraint: A=Δ Σ_V⁻¹ C, B=Δ Σ_V⁻¹ D before run |
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

### (H5) AB constraint API

```python
from prg.utils.h5_constraint import (
    apply_AB_constraint, compute_AB, compute_h5_residual,
)

# Apply A = Δ Σ_V⁻¹ C, B = Δ Σ_V⁻¹ D to all regimes at once
constrained_params = apply_AB_constraint(params)

# Or compute (A, B) for a single regime from (C, D, Δ, Σ_V)
A, B = compute_AB(C, D, Delta, SV)

# Verify any (A, B, C, D, Σ_U, Δ, Σ_V) satisfies (H5):
F = compute_h5_residual(A, B, C, D, SU, Delta, SV)  # ‖F‖_F = 0 ⇔ (H5) holds
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

## Reproducing the paper figures

The LaTeX sources of the current paper version live in
`docs/wojciech/article_vWojciech_tex/`.
Its figures are produced by the **jump-filtering study harness**
`prg/experiments/study.py` — a single, self-contained, deterministic
(fixed-seed) script that runs a battery of filtering experiments and writes
one vector PDF per experiment plus a `results.json` summary.

```bash
# Regenerate the candidate figures into the paper's figures/ directory
python -m prg.experiments.study docs/wojciech/article_vWojciech_tex
```

This creates `docs/wojciech/article_vWojciech_tex/figures/*.pdf` and
`docs/wojciech/article_vWojciech_tex/results.json`. The candidate
experiments — all on the **exact jump filter** — are:

| Figure | Experiment |
|---|---|
| `e1_exactness.pdf` | E1 — exactness against a brute-force reference filter |
| `e2_speed.pdf` | E2 — linear-time $O(K^2 N)$ scaling |
| `e3_value.pdf`, `e3plus_value_sweep.pdf` | E3 / E3′ — filtering gain vs value/regime contrast |
| `e4_multivariate.pdf` | E4 — multivariate state / observation |
| `e5_closed_form.pdf` | E5 — closed-form AB constraint check |
| `e6_robustness.pdf` | E6 — robustness |
| `e7_rank_deficient.pdf` | E7 — rank-deficient $C$ ($s < q$) |
| `e8_c_influence.pdf` | E8 — influence of $C$ (regime-identification channel) |
| `e9_c_mismatch.pdf` | E9 — filtering $C \neq 0$ data with a $C = 0$ filter |

> **The figure selection is not fixed yet.** The harness above generates the
> full candidate set; once the figures retained for the paper are settled,
> this section will list them — and the exact commands to rebuild just
> those — explicitly.

---

## Testing

```bash
source .venv/bin/activate
pytest                        # full suite, ~45 s
mypy                          # strict typing on prg/utils/h5_constraint.py
```

The suite covers matrix diagnostics, parameter validation, the
iterator-based simulator, reproducibility, CSV output, statistical sanity,
exact-filter correctness, and the (H5) AB constraint computation.

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
  author       = {Derrode, St{\'e}phane and Fernandes, Cl{\'e}ment and Lehmann, Fr{\'e}d{\'e}ric and Pieczynski, Wojciech},
  title        = {{On Fast Optimal Filtering in Gaussian Switching Systems}},
  year         = {2026},
  howpublished = {\url{https://github.com/SDerrode/exactIMM}},
  note         = {Software accompanying paper preprint},
}
```

A `CITATION.cff` file is also provided at the repo root.

---

## Authors

- **Stéphane Derrode** — [stephane.derrode@ec-lyon.fr](mailto:stephane.derrode@ec-lyon.fr) ([École Centrale de Lyon](https://www.ec-lyon.fr/) — LIRIS, CNRS UMR 5205)
- **Clément Fernandes** — [Télécom SudParis](https://www.telecom-sudparis.eu/) (SAMOVAR, Institut Polytechnique de Paris)
- **Frédéric Lehmann** — [Télécom SudParis](https://www.telecom-sudparis.eu/) (SAMOVAR)
- **Wojciech Pieczynski** — [wojciech.pieczynski@telecom-sudparis.eu](mailto:wojciech.pieczynski@telecom-sudparis.eu) ([Télécom SudParis](https://www.telecom-sudparis.eu/) — SAMOVAR)

## License

GNU Affero General Public License v3.0 — see `LICENSE`.
