# GUI Guide

The optional PyQt6 interface lets you tweak GSS parameters and inspect
the filter output interactively.

## Launching

```bash
pip install -e ".[gui]"
python -m prg.gui.main -K 2 -q 1 -s 1
# or pre-fill from an existing model
python -m prg.gui.main --model model_gss_K2_q1_s1
```

## Layout

The window is split horizontally:
- **Left:** parameter editor (one tab per regime + transition matrix)
- **Right:** plot panel (state, observation, regime, innovations)

## Left panel — parameters

| Widget | Effect |
|---|---|
| Preset selector | Load any built-in model in one click |
| F(k), Σ_W(k) tables | Inline editing with block colour coding (A blue, B green, C yellow, D pink) |
| H5 checkboxes (4 per tab) | Auto-recompute B(k) or zero out Δ in real-time |
| Stability badges | ρ(F), ρ(A), ρ(D) shown live below the F(k) table |
| Randomize 🎲 | Fill F(k), Σ_W(k) with random stable parameters |
| P (transition) table | K×K row-stochastic; π_∞ shown live |
| N / Seed | Sequence length and optional integer seed |
| Monte Carlo checkbox | Run M trajectories at once |
| Auto-filter checkbox | Run Filter automatically after each Simulate |

## Right panel — plots

`2 + q + 2s` subplots, all sharing the x-axis:

| Subplot | Content |
|---|---|
| R_n | Regime sequence (step plot) |
| π_n(k) | Filtered regime posteriors |
| X^i (×q) | Hidden state(s); filter overlay (mean ± 2σ) added after Filter |
| Y^i (×s) | Observation(s) |
| ν^i (×s) | Filter innovations |

## Filter quality frame

After Filter, a frame shows:

| Metric | Meaning |
|---|---|
| log L | Total log-likelihood and per-step mean |
| MSE / RMSE | Against ground-truth X (only when X is available) |
| Ljung-Box | Whiteness test per innovation component (green = pass) |
| Skew · Kurt | Skewness and excess kurtosis (kurtosis ≠ 0 is expected: GSS innovations are a mixture of Gaussians) |

## Workflow patterns

### Compare two parameter settings
1. Set parameters, **Simulate**, **Filter**.
2. Note the `log L` and `RMSE`.
3. Tweak a parameter → **Simulate** again (resampling) → **Filter** again.

The plot panel keeps the previous run visible until the new one
finishes, so you can A/B compare visually.

### Monte Carlo distribution
1. Tick **Monte Carlo**, set **M** (e.g. 50).
2. **Simulate** — the panel now shows mean ± 2σ + median ribbons.
3. **Ctrl+Shift+X** opens the per-component MC X distribution dialog.

### Loading a CSV
1. **File → Load CSV** — display external data.
2. Edit parameters to match the dataset (or use **File → Estimate
   parameters** if the CSV contains the regime column).
3. **Filter** to overlay the model's predictions.

## Keyboard shortcuts

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

## Persistent state

Window geometry, splitter position, M, seed, and auto-filter state are
saved to a per-user QSettings file and restored on next launch.

## Known caveats

- **Param drift indicator.** If you edit a parameter *after* Simulate
  but *before* Filter, the Filter button shows `⚠ Filter`. The Filter
  uses the parameters captured at Simulate time, *not* the current
  widget values. Re-run Simulate to apply the new parameters.
- **Resizing during MC simulation.** Resizing the main window in the
  middle of a Monte-Carlo run can occasionally drop the latest frame.
  Wait for the run to finish before resizing.
