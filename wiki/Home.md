# exactIMM — Wiki

Welcome to the **exactIMM** wiki. This software accompanies the paper

> *On Fast Optimal Filtering in Gaussian Switching Systems*  
> S. Derrode & W. Pieczynski (2026)

and provides a reference implementation of the AB-constrained
Gaussian Switching System framework.

## Where to start

| If you want to… | go to |
|---|---|
| Install and run a first example | [Installation](Installation) |
| Walk through a complete simulate→filter→learn cycle | [Tutorial](Tutorial) |
| Reproduce the paper figures and tables | [Paper-Reproduce](Paper-Reproduce) |
| Understand the codebase architecture | [API-Overview](API-Overview) |
| Use the GUI to explore parameters interactively | [GUI-Guide](GUI-Guide) |

## What is in the package?

- **Filter** (`prg/filter/`) — exact constant-gain optimal filter under
  AB, plus a general IMM mode for unconstrained models.
- **Simulator** (`prg/simulate.py`) — fast iterator-based generator of
  GSS trajectories.
- **Learning** (`prg/learning/`) — supervised OLS and semi-supervised
  Baum-Welch EM with three AB projection variants
  (\(\tau \in \{B, A^\dagger, \Sigma_U\}\)).
- **Experiments** (`prg/experiments/`) — Monte-Carlo benchmarks (§6 of
  the paper) and the ENSO real-data study (§7).
- **GUI** (`prg/gui/`) — interactive PyQt6 interface (optional).

## Citing

See [Citing](Citing) or the `CITATION.cff` at the repo root.

## License

[GNU Affero General Public License v3.0](https://www.gnu.org/licenses/agpl-3.0.html)
