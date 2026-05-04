# Reproducing the paper experiments

The paper has two empirical sections:

| Section | Content | Wall time | Script |
|---|---|---|---|
| §6 | Monte-Carlo simulation study (M1, M2, M3 reference models) | ~3 h total | `prg.experiments.run_simulations` + `run_supervised` + `run_em` |
| §7 | ENSO real-data experiment (K=3 regimes, NOAA SST) | ~1 min | `prg.experiments.run_real_data` |

All scripts are deterministic given a fixed `--seed`.

## §6 — Simulation study

### 6.2 — Filter benchmark MC

```bash
python -m prg.experiments.run_simulations
```

- 3 models × 3 sequence lengths × 100 seeds × 2 filter modes = 1800 trials
- Wall time ≈ 30 min on a modern laptop
- Output: `data/experiments/mc_results.csv`

### 6.3 — Supervised OLS MC

```bash
python -m prg.experiments.run_supervised
```

- 4 projection variants × 4 sequence lengths × 100 seeds = 1600 trials
- Wall time ≈ 15 min
- Output: `data/experiments/supervised_results.csv`

### 6.4 — Semi-supervised EM MC

```bash
python -m prg.experiments.run_em
```

- 3 models × 2 sequence lengths × 30 seeds × 2 variants × 5 restarts = 360 EM runs
- Wall time ≈ 2 h (the bulk of the budget)
- Output: `data/experiments/em_results.csv` + `em_ll_history.csv`

### Generating the paper figures and tables

After all three runs above:

```bash
python -m prg.experiments.make_figures
python -m prg.experiments.fill_placeholders
```

`make_figures` creates 3 PDFs and 6 LaTeX tables in
`paper/figures/generated/`.
`fill_placeholders` substitutes numerical `\ph{…}` macros in
`paper/sections/06_experiments.tex` with the values just computed.

## §7 — ENSO real-data experiment

```bash
# (Only needed if data/real/enso_sst.csv is missing or stale)
python scripts/build_enso_csv.py

# Full pipeline E1 + E2 + E3
python -m prg.experiments.run_real_data

# Time-series figures
python -m prg.experiments.make_figures_real
```

`run_real_data` writes:
- `paper/figures/generated/tab_enso_h5_test.tex`  (E1)
- `paper/figures/generated/tab_enso_filter.tex`   (E2)
- `paper/figures/generated/tab_enso_em.tex`       (E3)
- `results/enso/{e1,e2,e3}_table.json`            (raw numbers, JSON)
- `results/enso/regime_trace.csv`                 (filter posteriors on test)

`make_figures_real` writes:
- `paper/figures/generated/fig_enso_overview.pdf`
- `paper/figures/generated/fig_enso_regime_trace.pdf`

## Expected numerical results

If you run the scripts as shipped (default seed 42), you should get
the values printed in the paper, typically reproducible to 3–4
significant digits. The most distinctive results:

### §7 / E1 (H5 empirical test on ENSO)

| Regime | n_k | ‖B‖_F | p-value (H₀: B=0) |
|---|---|---|---|
| La Niña | 198 | 0.0505 | **0.359** |
| Neutral | 336 | 0.0460 | **0.372** |
| El Niño | 197 | 0.0261 | **0.664** |

→ (H5) is *not* rejected on this dataset.

### §7 / E2 (filter comparison)

| Filter | logL test | MSE X |
|---|---|---|
| **H5-exact (H5 fit)** | **−20.56** | 0.69 |
| IMM-general (OLS fit) | −32.38 | 0.54 |
| IMM-general (H5 fit)  | −31.55 | 0.81 |
| Kalman K=1            | −24.26 | 0.47 |

→ H5-exact achieves the best test log-likelihood, as predicted by theory.

### §7 / E3 (EM variants)

| Variant | train logL | test logL | MSE X |
|---|---|---|---|
| V0 unconstrained | −437.4 | −32.21 | 0.42 |
| V1 post-hoc τ=B  | −437.4 | −34.06 | 0.41 |
| V2 post-hoc τ=A† | −437.4 | **diverges** | **diverges** |
| V3 GEM τ=B       | −441.4 | −31.21 | 0.42 |

→ V2 fails for the same numerical reason as in simulation (§6.3): the
G matrix is ill-conditioned. V0/V1/V3 are essentially equivalent,
confirming that the H5 projection is benign on data compatible with (H5).

## Recompiling the paper

The paper sources are *not* in the public repo (gitignored), but with
a local copy:

```bash
cd paper
pdflatex paper && bibtex paper && pdflatex paper && pdflatex paper
```

You should obtain a 27-page PDF.

## Why are some `*.csv` not in the repo?

The MC simulation outputs (`data/experiments/*.csv`, ~1 MB) are
**regenerable** in a few hours and are excluded from the repository
to keep clones small. Only the small ENSO files (`data/real/*`) and
some §7 outputs (`results/enso/regime_trace.csv`) are versioned.
