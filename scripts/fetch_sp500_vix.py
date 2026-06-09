#!/usr/bin/env python3
"""
scripts/fetch_sp500_vix.py
==========================
Download S&P 500 (Yahoo Finance ^GSPC) and VIX (FRED VIXCLS) daily
closing prices, align them on common trading days, and cache the result
to ``data/real/sp500_vix.csv``.

The resulting CSV has columns:
    date         ISO-8601 (YYYY-MM-DD)
    sp500        Adjusted close of ^GSPC
    vix          Closing value of VIXCLS
    log_return   100 * log(sp500_t / sp500_{t-1})        (X of the paper)
    log_vix      log(vix)                                 (Y of the paper)

Usage
-----
    python scripts/fetch_sp500_vix.py                       # default range
    python scripts/fetch_sp500_vix.py --start 2004-01-01 \\
                                       --end   2024-12-31 \\
                                       --out   data/real/sp500_vix.csv

Dependencies
------------
    pip install ".[paper]"        (installs yfinance, requests, ...)

The downloaded CSV is meant to be committed for reproducibility.
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS"
DEFAULT_START = "2004-01-01"
DEFAULT_END = "2024-12-31"
DEFAULT_OUT = Path("data/real/sp500_vix.csv")


def fetch_sp500(start: str, end: str) -> pd.DataFrame:
    """Download ^GSPC adjusted close from Yahoo Finance."""
    print(f"[sp500] downloading ^GSPC from {start} to {end} ...", flush=True)
    df = yf.download(
        "^GSPC",
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )
    if df.empty:
        raise RuntimeError("Yahoo Finance returned no data for ^GSPC")
    # yfinance may return a MultiIndex on columns; flatten if so.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    out = df[["Close"]].rename(columns={"Close": "sp500"})
    out.index = pd.to_datetime(out.index).tz_localize(None).normalize()
    out.index.name = "date"
    print(f"[sp500] got {len(out)} rows", flush=True)
    return out


def fetch_vix(start: str, end: str) -> pd.DataFrame:
    """Download VIXCLS series from FRED (CSV endpoint, no API key)."""
    print("[vix] downloading VIXCLS from FRED ...", flush=True)
    resp = requests.get(FRED_URL, timeout=30)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    # FRED historically used "DATE" then switched to "observation_date"; handle both.
    date_col = next(
        (c for c in df.columns if c.lower() in {"date", "observation_date"}),
        None,
    )
    if date_col is None:
        raise RuntimeError(f"Unexpected FRED columns: {list(df.columns)}")
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col)
    df.index.name = "date"
    df = df.rename(columns={"VIXCLS": "vix"})
    # FRED marks missing values with '.' — coerce to NaN then drop.
    df["vix"] = pd.to_numeric(df["vix"], errors="coerce")
    df = df.dropna(subset=["vix"])
    df = df.loc[(df.index >= start) & (df.index <= end)]
    print(f"[vix] got {len(df)} rows", flush=True)
    return df[["vix"]]


def merge_and_transform(sp: pd.DataFrame, vix: pd.DataFrame) -> pd.DataFrame:
    """Inner-join on common trading days, compute log-return and log-VIX."""
    merged = sp.join(vix, how="inner").sort_index()
    merged["log_return"] = 100.0 * np.log(merged["sp500"]).diff()
    merged["log_vix"] = np.log(merged["vix"])
    merged = merged.dropna()
    return merged


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[3])
    parser.add_argument("--start", default=DEFAULT_START, help="ISO start date")
    parser.add_argument("--end", default=DEFAULT_END, help="ISO end date")
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Output CSV path",
    )
    args = parser.parse_args()

    sp = fetch_sp500(args.start, args.end)
    vix = fetch_vix(args.start, args.end)
    merged = merge_and_transform(sp, vix)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out, float_format="%.6f")
    print(
        f"[out] wrote {len(merged)} rows to {args.out} "
        f"covering {merged.index.min().date()} .. {merged.index.max().date()}",
        flush=True,
    )
    print(merged.describe().to_string(), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
