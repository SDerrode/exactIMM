#!/usr/bin/env python3
"""Regenerate the "Real-data evidence" figure of the experimental report.

Shows the two NGH-MSM ingredients that occur in real data:
  (a) slaving + a regime-dependent read-out  -> ENSO climate indices;
  (b) a sign-flipping read-out               -> S&P 500 vs 10-year Treasury.

Inputs  (committed under data/real/):
    enso_sst.csv     date, nino34, nino12, oni, regime
    stock_bond.csv   Date, stock_ret, dy, bond_ret   (bond_ret = -d(DGS10), FRED)
Output:
    docs/wojciech/rapport_experimental/figures/real_structure.pdf

    python -m prg.experiments.make_real_structure_fig
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("docs/wojciech/rapport_experimental/figures/real_structure.pdf")


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    warnings.filterwarnings("ignore")
    z = lambda a: (a - a.mean()) / a.std()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8))

    # (a) ENSO: slaving + regime-dependent read-out
    e = pd.read_csv("data/real/enso_sst.csv")
    Ys, Xs, R = z(e["nino34"].to_numpy()), z(e["nino12"].to_numpy()), e["regime"].to_numpy(int)
    cols = {0: "#1f77b4", 1: "#7f7f7f", 2: "#d62728"}
    labs = {0: "La Nina", 1: "Neutral", 2: "El Nino"}
    for k in (0, 1, 2):
        msk = R == k
        a1.scatter(Ys[msk], Xs[msk], s=8, c=cols[k], alpha=0.5, label=labs[k])
        b = np.polyfit(Ys[msk], Xs[msk], 1)
        xx = np.array([Ys[msk].min(), Ys[msk].max()])
        a1.plot(xx, np.polyval(b, xx), c=cols[k], lw=2)
    a1.set_title("(a) ENSO climate: slaving holds (obs-history adds < 1%)", fontsize=9)
    a1.set_xlabel("observation  Y = Nino 3.4  (std)")
    a1.set_ylabel("state  X = Nino 1+2  (std)")
    a1.legend(fontsize=7)
    a1.grid(alpha=0.3)

    # (b) stock-bond: sign-flipping read-out (rolling correlation)
    d = pd.read_csv("data/real/stock_bond.csv")
    d["Date"] = pd.to_datetime(d["Date"])
    xb, ys = z(d["bond_ret"].to_numpy()), z(d["stock_ret"].to_numpy())
    roll = pd.Series(xb).rolling(60).corr(pd.Series(ys))
    a2.plot(d["Date"], roll, color="#1f77b4", lw=0.8)
    a2.axhline(0, color="k", lw=0.8)
    a2.fill_between(
        d["Date"],
        roll,
        0,
        where=(roll < 0),
        color="#d62728",
        alpha=0.3,
        label="risk-off  (beta < 0)",
    )
    a2.fill_between(
        d["Date"],
        roll,
        0,
        where=(roll > 0),
        color="#2ca02c",
        alpha=0.25,
        label="risk-on  (beta > 0)",
    )
    a2.set_title("(b) Stock-bond: the read-out flips sign  (beta +0.35 <-> -0.37)", fontsize=9)
    a2.set_ylabel("rolling corr(bond, stock)")
    a2.legend(fontsize=7, loc="lower left")
    a2.grid(alpha=0.3)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    print("saved", OUT)


if __name__ == "__main__":
    main()
