# Tutorial — A complete simulate → filter → learn cycle

This tutorial walks through one full cycle on a built-in 2-regime,
scalar GSS model. Each command takes < 1 s.

## 1. Simulate a trajectory

```bash
python -m prg.simulate --model model_gss_K2_q1_s1 -N 1000 --seed 42
```

This writes
`data/simulated/simulated_model_gss_K2_q1_s1_N1000_seed42.csv` with
columns `n, r, x_0, y_0`.

## 2. Run the filter

```bash
python -m prg.filter.main --model model_gss_K2_q1_s1 -N 1000 --seed 42
```

Writes a sibling CSV with the filtered means
\(\hat x_n = \mathbb{E}[X_n \mid Y_{1:n}]\), posterior variances and
regime probabilities.

## 3. Learn parameters back from the simulation

```bash
# Supervised (R is in the CSV)
python -m prg.learning.supervised \
    data/simulated/simulated_model_gss_K2_q1_s1_N1000_seed42.csv \
    --constraint ab --output prg/models/model_learned_K2.py

# Semi-supervised (ignores the r column, recovers regimes by EM)
python -m prg.learning.semi_supervised \
    data/simulated/simulated_model_gss_K2_q1_s1_N1000_seed42.csv \
    -K 2 --constraint ab --n-inits 5 --output prg/models/model_em_K2.py
```

Both write a Python file containing a `BaseGSSModel` subclass with
the estimated parameters.

## 4. Refilter with the learned model

```bash
python -m prg.filter.main --model model_learned_K2 -N 1000 --seed 42
```

Compare the resulting MSE against the oracle filter (step 2) — this is
exactly the type of comparison §6.3 of the paper does at scale.

## 5. Use the Python API

```python
from prg.classes.GSSParams import GSSParams
from prg.models.model_gss_K2_q1_s1 import ModelGss_K2_q1_s1
from prg.filter import GSSFilter
import numpy as np

params = GSSParams.from_model(ModelGss_K2_q1_s1())
filt   = GSSFilter(params, mode="ngh_kf")     # or "gpb2"

# Process a stream of observations (one at a time)
for y in observations:                            # y shape (s,) or (s,1)
    res = filt.step(y)
    print(res.E_x.ravel(), res.pi)               # E[X_n|Y], P(R_n|Y)
```

## 6. Plot innovations and check whiteness

```python
import numpy as np
from scipy.stats import jarque_bera
from statsmodels.stats.diagnostic import acorr_ljungbox   # in the [paper] extra

# Collect innovations during a run
innov = []
for y in observations:
    res = filt.step(y)
    innov.append(res.innovation)
innov = np.asarray(innov).squeeze()

# Whiteness test (good filter ↔ p > 0.05)
print(acorr_ljungbox(innov, lags=[20], return_df=True))
print(jarque_bera(innov))                          # mixture-of-Gaussians: kurtosis ≠ 0 expected
```

## 7. Try the GUI

```bash
pip install -e ".[gui]"
python -m prg.gui.main -K 2 -q 1 -s 1
```

The GUI lets you tweak the parameters, simulate, filter, and visualise
everything from the previous steps interactively. See
[GUI-Guide](GUI-Guide).

## Next

- [Paper-Reproduce](Paper-Reproduce) — reproduce all paper experiments
- [API-Overview](API-Overview) — module and class reference
