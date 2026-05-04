# Installation

## Requirements

- Python **3.14+** (lower versions untested; 3.13 likely works but the
  type annotations target 3.14)
- `pip ≥ 24` (for PEP 660 editable installs)
- A C/Fortran toolchain is *not* required: NumPy, SciPy and pandas
  ship as wheels for all major platforms.

## Quick install (CPU only)

```bash
git clone https://github.com/SDerrode/exactIMM.git
cd exactIMM       # the Python package inside is called `fofgss`

# Create and activate a project virtualenv
python3 -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate

# Editable install + dev tools (pytest)
pip install -e ".[dev]"
```

> **Note on macOS / homebrew users.** Do *not* rely on the system
> NumPy: it can fail to load due to BLAS conflicts. Always work inside
> the project virtualenv created above.

## Optional extras

| Extra | Installs | When to use |
|---|---|---|
| `gui` | PyQt6, matplotlib | Interactive parameter exploration |
| `paper` | matplotlib, statsmodels, hmmlearn, scikit-learn, requests, yfinance | Reproducing §6 and §7 experiments |
| `dev` | pytest, pytest-cov | Running the test suite |

```bash
pip install -e ".[gui]"           # GUI only
pip install -e ".[paper]"         # paper experiments
pip install -e ".[gui,paper,dev]" # everything
```

## Verifying the install

```bash
# Run the test suite (≈ 45 s, 204 tests)
pytest

# Smoke-test the simulator
python -m prg.simulate --model model_gss_K2_q1_s1 -N 100 --seed 0 --no-save

# Smoke-test the filter
python -m prg.filter.main --model model_gss_K2_q1_s1 -N 100 --seed 0 --no-save
```

If `pytest` reports `204 passed` and the two CLI calls finish without
error, the install is good.

## Updating

```bash
git pull
pip install -e ".[dev]" --upgrade
```

If a previous version of `fofgss.egg-info/` lingers and causes
`ModuleNotFoundError`, just delete it: `rm -rf fofgss.egg-info` and
re-run `pip install -e ".[dev]"`.

## Common issues

| Symptom | Fix |
|---|---|
| `ImportError: numpy.core.multiarray failed to import` | Activate the venv (`source .venv/bin/activate`) |
| `Qt platform plugin "cocoa" could not be found` (macOS GUI) | `pip install --upgrade --force-reinstall PyQt6` |
| GUI launches but no plots show | Check that `matplotlib` is installed *inside the venv* |
| `pytest` fails on `test_semi_supervised` only on Apple Silicon | Known stochastic flakiness on `--n-inits 1`; rerun with `pytest --rerun-failures 2` |
