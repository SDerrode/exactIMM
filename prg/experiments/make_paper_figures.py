#!/usr/bin/env python3
"""
prg/experiments/make_paper_figures.py
=====================================
Regenerate, into ``docs/NGH-MSM_V2/figures/``, exactly the figures used in the
IEEE TAC paper *Marginal Markovianity and Exact Filtering in Gaussian Switching
Systems*:

    e2_speed.pdf          Fig. 4   computational cost         study.exp_speed
    e8_c_influence.pdf    Fig. 5 (top)  sweeping C            study.exp_c_influence
    e9_c_mismatch.pdf     Fig. 5 (bot)  mismatched filter     study.exp_c_mismatch
    vehicle_consigne.pdf  Fig. 6   real driving data          make_vehicle_consigne_fig
    imm_exactness.pdf     Supp.    GPB2 (order 2) vs order-1 IMM exactness    study.exp_imm_exactness
    ab_robustness.pdf     Supp.    robustness to AB violation (KF vs GPB2 vs IMM)  study.exp_ab_robustness

The synthetic experiments are deterministic (fixed seeds), so the figures are
bit-for-bit reproducible. The vehicle figure downloads a ~20 MB open dataset once
(cached under ``data/real/vehicle/``).

Run from the repository root, inside the project venv:

    python -m prg.experiments.make_paper_figures
"""

from __future__ import annotations

from pathlib import Path

OUT = Path("docs/NGH-MSM_V2")
FIG = OUT / "figures"


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")

    FIG.mkdir(parents=True, exist_ok=True)
    from prg.experiments import make_vehicle_consigne_fig, study

    study._setup_mpl()  # initialise the module-level pyplot (Agg) used by exp_*

    print("== synthetic figures ==")
    study.exp_speed(OUT)  # -> figures/e2_speed.pdf
    study.exp_c_influence(OUT)  # -> figures/e8_c_influence.pdf
    study.exp_c_mismatch(OUT)  # -> figures/e9_c_mismatch.pdf
    study.exp_imm_exactness(OUT)  # -> figures/imm_exactness.pdf (supplementary)
    study.exp_ab_robustness(OUT)  # -> figures/ab_robustness.pdf (supplementary)

    print("== real-data figure ==")
    make_vehicle_consigne_fig.OUT = FIG / "vehicle_consigne.pdf"
    make_vehicle_consigne_fig.main()

    print("\npaper figures in", FIG)
    for f in sorted(FIG.glob("*.pdf")):
        print("  ", f.name)


if __name__ == "__main__":
    main()
