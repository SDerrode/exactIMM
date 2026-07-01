#!/usr/bin/env python3
"""Real-data evidence for the consigne (exogenous-input) extension — vehicle driving.

Downloads the *aggressive* subset of the open dataset "Experimental Sensor Data
from Vehicles for Dynamic Vehicle Models" (Mendeley 10.17632/x7n6jnjh36.3, mirrored
on Figshare; real driving at the Continental test track, Veszprém, Hungary; 100 Hz)
and shows that the **steering input (consigne)** sharply reduces the error of a
slaving read-out for the yaw rate on held-out data:

    yaw_n  ≈  M_r · a_lat_n  +  N_r · steering_n          (regime r = observed speed band)

i.e. the exact NGH-MSM read-out X_n = M_{r_n} y_n + N_{r_n} u_{n-1} with
X = yaw rate, Y = lateral acceleration, u = steering. Four estimators are compared
on held-out files: with/without the consigne, with/without the speed regime.

    python -m prg.experiments.make_vehicle_consigne_fig

Output: docs/rapport_consigne/figures/vehicle_consigne.pdf  (+ printed RMSE).
The ~20 MB download is cached under data/real/vehicle/.
"""

from __future__ import annotations

import glob
import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

FIGSHARE_AGGRESSIVE = "https://ndownloader.figshare.com/files/52533854"  # aggressive_driver.zip
CACHE = Path("data/real/vehicle")
OUT = Path("docs/rapport_consigne/figures/vehicle_consigne.pdf")

YAW = "Yaw rate [deg/s]"
ALAT = "Lateral acceleration [g]"
STR = "Steering angle of first axle [s]"  # degrees (header unit is a typo)
WS = [f"Wheel speed ({w}) [km/h]" for w in ("Front Left", "Front Right", "Rear Right", "Rear Left")]


def _ensure_data() -> list[str]:
    CACHE.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(str(CACHE / "**" / "*.parquet"), recursive=True))
    if files:
        return files
    print("downloading aggressive_driver.zip (~20 MB) from Figshare …")
    raw = urllib.request.urlopen(FIGSHARE_AGGRESSIVE, timeout=120).read()  # noqa: S310
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        zf.extractall(CACHE)
    return sorted(glob.glob(str(CACHE / "**" / "*.parquet"), recursive=True))


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(",", "."), errors="coerce")


def _load(files: list[str]):
    rows = []
    for f in files:
        d = pd.read_parquet(f)
        for c in [YAW, ALAT, STR, *WS]:
            d[c] = _num(d[c])
        d["spd"] = d[WS].mean(axis=1)
        rows.append(d.dropna(subset=[YAW, ALAT, STR, "spd"]))
    d = pd.concat(rows, ignore_index=True)
    return d[YAW].to_numpy(), d[ALAT].to_numpy(), d[STR].to_numpy(), d["spd"].to_numpy()


def _fit(y, feats):
    X = np.column_stack([np.ones(len(y)), *feats])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return b


def _rmse(y, yh):
    return float(np.sqrt(np.mean((y - yh) ** 2)))


def main() -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    files = _ensure_data()
    ytr, atr, str_tr, vtr = _load(files[:80])
    yte, ate, str_te, vte = _load(files[80:110])

    # regime = speed band (tertiles of the TRAIN speed), identifiable from observed speed
    qs = np.quantile(vtr, [0, 1 / 3, 2 / 3, 1.0])
    qs[0], qs[-1] = -1e9, 1e9
    band = lambda v: np.clip(np.digitize(v, qs[1:-1]), 0, 2)
    btr, bte = band(vtr), band(vte)

    # (A) regime + consigne ; (B) regime, input-blind ; (C) blind + consigne ; (D) blind, input-blind
    yhA, yhB = np.zeros_like(yte), np.zeros_like(yte)
    for r in range(3):
        mtr, mte = btr == r, bte == r
        bA = _fit(ytr[mtr], [atr[mtr], str_tr[mtr]])
        yhA[mte] = bA[0] + bA[1] * ate[mte] + bA[2] * str_te[mte]
        bB = _fit(ytr[mtr], [atr[mtr]])
        yhB[mte] = bB[0] + bB[1] * ate[mte]
    bC = _fit(ytr, [atr, str_tr])
    yhC = bC[0] + bC[1] * ate + bC[2] * str_te
    bD = _fit(ytr, [atr])
    yhD = bD[0] + bD[1] * ate
    # (E) consigne + a regime on the |steering| magnitude (captures the nonlinear
    # steering->yaw response): this is the regime that genuinely pays here.
    sq = np.quantile(np.abs(str_tr), [0, 1 / 3, 2 / 3, 1.0])
    sq[0], sq[-1] = -1e18, 1e18
    sband = lambda s: np.clip(np.digitize(np.abs(s), sq[1:-1]), 0, 2)
    sbtr, sbte = sband(str_tr), sband(str_te)
    yhE = np.zeros_like(yte)
    for r in range(3):
        mtr, mte = sbtr == r, sbte == r
        bE = _fit(ytr[mtr], [atr[mtr], str_tr[mtr]])
        yhE[mte] = bE[0] + bE[1] * ate[mte] + bE[2] * str_te[mte]
    # (F) consigne + the vehicle SPEED, via a steering*speed term: the yaw produced
    # by a given steering angle grows with speed, so this also cures the
    # near-standstill bias of the plain read-out (where v~0 but N*steering still
    # predicts a yaw). Preliminary "extra variable" result.
    bF = _fit(ytr, [atr, str_tr * vtr])
    yhF = bF[0] + bF[1] * ate + bF[2] * (str_te * vte)
    rmse = {
        "D_blind_inputblind": _rmse(yte, yhD),
        "C_blind_consigne": _rmse(yte, yhC),
        "B_regime_inputblind": _rmse(yte, yhB),
        "A_regime_speed_consigne": _rmse(yte, yhA),
        "E_steerregime_consigne": _rmse(yte, yhE),
        "F_speed_consigne": _rmse(yte, yhF),
    }

    # figure: (a) yaw on a test segment, (b) RMSE bars
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8))
    seg = slice(98400, 99600)  # representative driving episode (~39 km/h, active turning)
    nn = np.arange(seg.stop - seg.start) / 100.0  # seconds
    a1.plot(nn, yte[seg], color="black", lw=1.2, label="true yaw rate")
    a1.plot(nn, yhC[seg], color="#1f77b4", lw=1.0, label="with steering (consigne)")
    a1.plot(nn, yhD[seg], color="#ff7f0e", lw=0.9, ls="--", label="without steering")
    a1.set_xlabel("time [s]")
    a1.set_ylabel("yaw rate [deg/s]")
    a1.set_title("(a) held-out real driving: the consigne tracks the yaw", fontsize=11)
    a1.legend(fontsize=10, loc="upper right")
    a1.grid(alpha=0.3)
    order = ["D_blind_inputblind", "C_blind_consigne", "E_steerregime_consigne"]
    labels = ["no steering\n(1 regime)", "+ consigne\n(steering)", "+ switching\n(steer-magnitude)"]
    cols = ["#999999", "#17becf", "#1f77b4"]
    a2.bar(range(3), [rmse[k] for k in order], color=cols)
    a2.set_xticks(range(3))
    a2.set_xticklabels(labels, fontsize=10)
    a2.set_ylabel("held-out yaw RMSE [deg/s]")
    a2.set_title("(b) command, then regime: each lowers the error", fontsize=11)
    a2.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    print("saved", OUT)
    for k, v in rmse.items():
        print(f"  {k:26s} RMSE = {v:.3f} deg/s")
    g_cons = 100 * (1 - rmse["C_blind_consigne"] / rmse["D_blind_inputblind"])
    g_sw = 100 * (1 - rmse["E_steerregime_consigne"] / rmse["C_blind_consigne"])
    g_all = 100 * (1 - rmse["E_steerregime_consigne"] / rmse["D_blind_inputblind"])
    print(f"  gains: consigne {g_cons:.0f}%  +switching {g_sw:.0f}%  total {g_all:.0f}%")
    return rmse


if __name__ == "__main__":
    main()
