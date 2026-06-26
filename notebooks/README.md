# Usage notebooks — filtering

Worked examples for the package's fast exact jump-filter (NGH-MSM-KF). Filtering
only; parameter learning (EM) is not covered here.

| Notebook | Contents |
|---|---|
| [`01_filtering_quickstart.ipynb`](01_filtering_quickstart.ipynb) | Build a model, simulate, run the filter (`GSSFilter.step`), read off `E[X|y]`, `Var[X|y]`, `p(r|y)`; verify exactness against the brute-force filter. |
| [`02_filters_comparison.ipynb`](02_filters_comparison.ipynb) | Exactness across M1/M2/M3, the value of modelling the switching (vs a pairwise Kalman and an oracle), multivariate tracking, and NGH-MSM-KF vs the general IMM. |

## Running

From the repo root, with the project installed (`pip install -e .`):

```bash
jupyter lab notebooks/        # or: jupyter notebook
```

Use the kernel of the project's environment (the one where `import prg` works).
The notebooks import the public filtering API (`GSSFilter`, `GSSSimulator`) plus
the paper's example models and reference filters in `prg.experiments`.
