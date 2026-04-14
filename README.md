# FofGss — Fast Optimal Filtering in Gaussian Switching Systems

Implementation of the filtering and simulation algorithms described in:

> *On fast optimal filtering in Gaussian switching systems*  
> Stéphane Derrode & Wojciech Pieczynski — `docs/CS_Finale.pdf`

---

## Scientific context

The project targets **Gaussian Switching Systems (GSS)**, a class of hidden Markov models where the state $(X_n, Y_n)$ evolves according to a regime $R_n \in \{0,\ldots,K-1\}$:

$$Z_{n+1} = F(R_{n+1})\,Z_n + W_{n+1}, \qquad Z_n = \begin{bmatrix}X_n \\ Y_n\end{bmatrix}$$

with $F(k) = \begin{bmatrix}A_k & B_k \\ C_k & D_k\end{bmatrix}$ and $W_{n+1}\mid R_{n+1}=k \sim \mathcal{N}(0,\Sigma_W(k))$.

The central objective is to compute $\mathbb{E}[X_n \mid Y_{1:n}]$ efficiently (fast optimal filter).

---

## Project structure

```
fofgss/
├── prg/
│   ├── utils/
│   │   ├── exceptions.py       # GSSError hierarchy
│   │   └── matrix_checks.py    # CovarianceMatrix, StochasticMatrix diagnostics
│   ├── models/
│   │   ├── base_gss_model.py   # BaseGSSModel (abstract)
│   │   └── model_gss_K2_q1_s1.py  # Example: K=2, q=1, s=1
│   ├── classes/
│   │   ├── FMatrix.py          # Block transition matrix F(k)
│   │   ├── NoiseCovariance.py  # Block noise covariance Sigma_W(k)
│   │   ├── GSSParams.py        # Aggregates all model parameters
│   │   └── GSSSimulator.py     # Iterator-based simulator
│   └── simulate.py             # CLI entry point
├── tests/                      # 71 pytest tests
├── data/
│   └── simulated/              # Generated CSV files
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
| `-v` / `--verbose` | `1` | Console verbosity: 0=silent, 1=normal, 2=debug |

---

## Graphical interface

The GUI requires `pip install -e ".[gui]"` (PyQt6 + matplotlib).

```bash
# Standalone launch: choose K, q, s interactively
python -m prg.gui.main -K 2 -q 1 -s 1

# Pre-fill from an existing model file
python -m prg.gui.main --model model_gss_K2_q1_s1
```

**Features**

| Panel | Description |
|---|---|
| Parameter tabs | One tab per Markov state; editable F(k) and Σ_W(k) tables with block colour coding |
| Validation | [Simuler] disabled + red text when any matrix is invalid (non-float entry or non-SPD Σ_W) |
| Simulation | Runs in a background thread; modal wait dialog prevents interaction |
| Plot | 1 + q + s subplots: R_n (step), X_i (lines), Y_i (lines); full matplotlib toolbar |
| Save | [Enregistrer CSV] writes `data/simulated/simulated_gui_N{N}_seed{seed}_{ts}.csv` |

Fixed (non-editable) parameters: P = uniform 1/K (or model's P), π₀ = stationary, μ_{z0} = 0, Σ_{z0} = I.

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

## Testing

```bash
source .venv/bin/activate
pytest
```

71 tests covering matrix diagnostics, parameter validation, iterator protocol,
reproducibility, CSV output, and statistical sanity checks.

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

## Authors

Stéphane Derrode — `stephane.derrode@ec-lyon.fr`

## License

GNU Affero General Public License v3.0 — see `LICENSE`.
