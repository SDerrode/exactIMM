#!/usr/bin/env python3
"""
scripts/labels.py
=================
Build proxy regime labels L1 (VIX-median) and L2 (NBER recession) on
the S&P500 / VIX dataset.

Both labels are 0/1 sequences aligned on the CSV's ``date`` index.

L1 — VIX median threshold (learned on *train*)
    label = 0 if log_vix <= median_train(log_vix)  "calm"
    label = 1 otherwise                            "turbulent"

L2 — NBER recession dates (fixed, publicly listed)
    label = 0 outside recession periods             "expansion"
    label = 1 inside                                "recession"

Only training-period information is ever used to set thresholds, so
neither label leaks test-period statistics back into the training set.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

# NBER recession periods covered by the 2004-2024 range.
NBER_RECESSIONS = [
    ("2007-12-01", "2009-06-30"),  # Great Recession
    ("2020-02-01", "2020-04-30"),  # COVID-19
]

TRAIN_END = pd.Timestamp("2018-12-31")


def load_dataset(csv_path: Path) -> pd.DataFrame:
    """Load the sp500_vix CSV, sorted by date."""
    df = pd.read_csv(csv_path, parse_dates=["date"]).set_index("date").sort_index()
    return df


def train_test_split(
    df: pd.DataFrame, train_end: pd.Timestamp = TRAIN_END
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = df.loc[df.index <= train_end].copy()
    test = df.loc[df.index > train_end].copy()
    return train, test


def build_label_L1(df: pd.DataFrame, train: pd.DataFrame, column: str = "log_vix") -> pd.Series:
    """L1: VIX-median threshold learned on the *training* period."""
    threshold = train[column].median()
    labels = (df[column] > threshold).astype(np.int8)
    labels.attrs["threshold"] = float(threshold)
    labels.attrs["kind"] = "L1_vix_median"
    return labels


def build_label_L2(df: pd.DataFrame) -> pd.Series:
    """L2: 1 if date falls in an NBER recession, else 0."""
    labels = pd.Series(0, index=df.index, dtype=np.int8)
    for start, end in NBER_RECESSIONS:
        mask = (df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))
        labels.loc[mask] = 1
    labels.attrs["kind"] = "L2_nber"
    return labels


def summarize_label(labels: pd.Series, name: str) -> str:
    n = len(labels)
    n1 = int(labels.sum())
    return (
        f"{name:12s}  kind={labels.attrs.get('kind', '?'):18s}  "
        f"n={n} , n(r=1)={n1} ({100.0 * n1 / n:5.1f}%)"
    )


def standardize_with_train_stats(
    df: pd.DataFrame, train: pd.DataFrame, cols: tuple[str, ...]
) -> tuple[pd.DataFrame, dict[str, tuple[float, float]]]:
    """Z-score each column using *train-only* mean/std."""
    stats: dict[str, tuple[float, float]] = {}
    out = df.copy()
    for c in cols:
        mu = float(train[c].mean())
        sd = float(train[c].std(ddof=0))
        if sd == 0.0:
            raise RuntimeError(f"Column {c!r} has zero std on training period")
        out[c] = (df[c] - mu) / sd
        stats[c] = (mu, sd)
    return out, stats


LabelKind = Literal["L1", "L2"]


def get_label(df: pd.DataFrame, train: pd.DataFrame, kind: LabelKind) -> pd.Series:
    if kind == "L1":
        return build_label_L1(df, train)
    if kind == "L2":
        return build_label_L2(df)
    raise ValueError(f"Unknown label kind {kind!r}")
