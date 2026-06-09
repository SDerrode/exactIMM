#!/usr/bin/env python3
"""
scripts/build_enso_csv.py
==========================
Build ``data/real/enso_sst.csv`` from the three NOAA monthly anomaly files
(Niño 3.4, Niño 1+2, ONI). Optionally downloads them first.

Usage
-----
    # Download NOAA files (if missing) and build enso_sst.csv
    python scripts/build_enso_csv.py

    # Force re-download
    python scripts/build_enso_csv.py --force-download

    # Skip download (require local files in data/real/)
    python scripts/build_enso_csv.py --no-download

Inputs (data/real/)
-------------------
    nino34.txt   monthly SST anomaly in Niño 3.4 region (NOAA PSL)
    nino12.txt   monthly SST anomaly in Niño 1+2 region
    oni.txt      Oceanic Niño Index (3-month running mean of Niño 3.4)

Output (data/real/)
-------------------
    enso_sst.csv  columns: date, nino34, nino12, oni, regime
                  where regime ∈ {0=La Niña, 1=Neutral, 2=El Niño}
                  is derived from ONI thresholds (-0.5, +0.5).
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT       = Path(__file__).resolve().parent.parent
DATA_DIR   = ROOT / "data" / "real"
NOAA_BASE  = "https://psl.noaa.gov/data/correlation"
FILES = {
    "nino34.txt": f"{NOAA_BASE}/nina34.anom.data",
    "nino12.txt": f"{NOAA_BASE}/nina1.anom.data",
    "oni.txt":    f"{NOAA_BASE}/oni.data",
}


def download_noaa_files(force: bool = False) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for fname, url in FILES.items():
        target = DATA_DIR / fname
        if target.exists() and not force:
            print(f"  skipped (exists): {fname}")
            continue
        print(f"  downloading {url} → {target}")
        urllib.request.urlretrieve(url, target)


def parse_psl(path: Path) -> pd.DataFrame:
    """Parse a NOAA PSL fixed-width monthly anomaly file."""
    rows = []
    with open(path) as f:
        lines = f.readlines()
    hdr = lines[0].split()
    y0, y1 = int(hdr[0]), int(hdr[1])
    for line in lines[1:]:
        parts = line.split()
        if len(parts) < 13:
            break
        try:
            year = int(parts[0])
        except ValueError:
            break
        if year < y0 or year > y1:
            break
        vals = [float(v) for v in parts[1:13]]
        rows.append((year, vals))
    return pd.DataFrame(rows, columns=["year", "vals"])


def to_monthly(df: pd.DataFrame, name: str) -> pd.Series:
    """Explode (year, [12 values]) rows into a monthly Series with NaN sentinels."""
    out = []
    for _, row in df.iterrows():
        for m, v in enumerate(row["vals"], 1):
            out.append((pd.Timestamp(int(row["year"]), m, 1), v))
    s = pd.Series([v for _, v in out], index=[d for d, _ in out], name=name)
    # NOAA uses -99.99 (Niño 3.4 / 1+2) or -99.90 (ONI) for missing values
    return s.where(s > -50.0, np.nan)


def build_csv() -> Path:
    s34  = to_monthly(parse_psl(DATA_DIR / "nino34.txt"), "nino34")
    s12  = to_monthly(parse_psl(DATA_DIR / "nino12.txt"), "nino12")
    soni = to_monthly(parse_psl(DATA_DIR / "oni.txt"),   "oni")

    df = pd.concat([s34, s12, soni], axis=1).dropna()
    df.index.name = "date"

    # K=3 regime label from ONI thresholds (NOAA convention)
    df["regime"] = 1                                  # neutral
    df.loc[df["oni"] < -0.5, "regime"] = 0           # La Niña
    df.loc[df["oni"] >  0.5, "regime"] = 2           # El Niño

    target = DATA_DIR / "enso_sst.csv"
    df.to_csv(target)

    print(f"\n  built {target} ({len(df)} rows)")
    print(f"  range: {df.index[0].strftime('%Y-%m')} → {df.index[-1].strftime('%Y-%m')}")
    counts = df["regime"].value_counts().sort_index().to_dict()
    print(f"  regimes: La Niña={counts.get(0, 0)}  "
          f"Neutral={counts.get(1, 0)}  El Niño={counts.get(2, 0)}")
    n_trans = int((df["regime"].diff().abs() > 0).sum())
    print(f"  transitions: {n_trans}  (avg run length ≈ {len(df)/max(n_trans,1):.1f} months)")
    return target


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[3],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--no-download", action="store_true",
                    help="Use only local files; fail if any is missing.")
    ap.add_argument("--force-download", action="store_true",
                    help="Re-download even if local files exist.")
    args = ap.parse_args()

    if not args.no_download:
        print("Step 1/2 — NOAA download:")
        download_noaa_files(force=args.force_download)
    else:
        print("Step 1/2 — skipped (--no-download)")
        for fname in FILES:
            if not (DATA_DIR / fname).exists():
                print(f"  ERROR: missing {DATA_DIR / fname}", file=sys.stderr)
                return 2

    print("\nStep 2/2 — Building enso_sst.csv:")
    build_csv()
    return 0


if __name__ == "__main__":
    sys.exit(main())
