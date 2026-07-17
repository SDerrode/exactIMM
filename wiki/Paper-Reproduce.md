# Reproducing the paper experiments

The paper's experiments, plus a companion real-data study that is no longer part of it:

| Section | Content | Wall time | Script |
|---|---|---|---|
| §6 | Monte-Carlo simulation study (M1, M2, M3 reference models) | ~20 h total (EM-dominated) | `prg.experiments.run_simulations` + `run_supervised` + `run_em` |
| — | ENSO real-data study (K=3 regimes, NOAA SST) — **companion, not in the paper**: the current paper's real-data section is the vehicle-dynamics study | ~1 min | `prg.experiments.run_real_data` |

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

- 1 model (M1) × 2 sequence lengths {500, 2000} × 100 seeds × 2 variants (PH / GEM) = 400 EM trials, each the best of 5 restarts (k-means init, 50 EM iterations)
- Wall time ≈ 20 h single-threaded — the EM dominates the §6 budget (measured: a 4-seed subset ≈ 50 min on an Apple-silicon laptop, scaling ~25× to the full 100-seed run)
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

## ENSO real-data study (companion, not in the paper)

```bash
# (Only needed if data/real/enso_sst.csv is missing or stale)
python scripts/build_enso_csv.py

# Full pipeline E1 + E2 + E3
python -m prg.experiments.run_real_data

# Time-series figures
python -m prg.experiments.make_figures_real
```

`run_real_data` writes:
- `paper/figures/generated/tab_enso_ab_test.tex`  (E1)
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

### E1 (AB empirical test on ENSO)

| Regime | n_k | ‖B‖_F | p-value (H₀: B=0) |
|---|---|---|---|
| La Niña | 198 | 0.0505 | **0.359** |
| Neutral | 336 | 0.0460 | **0.372** |
| El Niño | 197 | 0.0261 | **0.664** |

→ AB is *not* rejected on this dataset.

### E2 (filter comparison)

Regenerated with v2.0.0; these are the values in `results/enso/e2_table.json`.

| Filter | logL test | MSE X |
|---|---|---|
| Kalman K=1 | **−24.26** | **0.467** |
| GPB2 (OLS fit) | −24.85 | 0.577 |
| GPB2 (AB fit) | −29.80 | 0.645 |
| NGH-MSM-KF (AB fit) | −30.09 | 0.644 |

→ On ENSO the constant-gain NGH-MSM-KF does **not** win, and there is no reason it
should: this dataset is not an NGH-MSM, so the AB constraint the filter assumes is
misspecified, and a plain regime-blind Kalman filter comes out ahead on both test
log-likelihood and MSE. GPB2, which does not assume AB, sits in between — as
expected. The exactness claims of the paper are established on models that *are*
NGH-MSMs (see the paper's §VI); this study probes what happens when they are not.

*(Before v2.0.0 this table reported −20.56 for the constant-gain filter and
concluded that it "achieves the best test log-likelihood, as predicted by theory".
That number matched no reproducible run — the committed `e2_table.json` already
said −30.09 — and the conclusion was the opposite of what the data shows.)*

### E3 (EM variants)

Regenerated with v2.0.0 (`results/enso/e3_table.json`, seed 42, 5 restarts,
50 iterations). The pipeline runs **three** variants, not four.

| Variant | train logL | test logL | MSE X | regime acc. |
|---|---|---|---|---|
| V0 unconstrained | −437.4 | +0.26 | 0.414 | 0.484 |
| V1 post-hoc AB | −437.4 | +0.34 | 1.066 | 0.500 |
| V2 GEM AB | −1482.1 | −21.70 | 0.664 | 0.418 |

→ V0 and V1 reach the same training likelihood, so the *post-hoc* AB projection is
benign on this dataset — consistent with E1, where AB is not rejected. Enforcing AB
at every M-step (V2) is a different matter: it costs ~1000 nats of training
likelihood and degrades every test metric, because the constraint is imposed
throughout the optimisation rather than at the end. None of the three separates the
regimes (accuracy ≈ 0.42–0.50 for K=3, i.e. at or below chance): recovering an
NGH-MSM from data is the open identifiability problem the paper's conclusion flags
as future work, not a solved one.

## Recompiling the paper

The paper sources are *not* in the public repo (gitignored). With a local copy in
`docs/NGH-MSM_V2/`:

```bash
cd docs/NGH-MSM_V2
pdflatex main && biber main && pdflatex main && pdflatex main
```

The figures it uses are regenerated by `python -m prg.experiments.make_paper_figures`.

## Why are some `*.csv` not in the repo?

The MC simulation outputs (`data/experiments/*.csv`, ~1 MB) are
**regenerable** in a few hours and are excluded from the repository
to keep clones small. Only the small ENSO files (`data/real/*`) and
some ENSO outputs (`results/enso/regime_trace.csv`) are versioned.
