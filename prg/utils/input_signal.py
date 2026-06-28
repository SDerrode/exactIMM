#!/usr/bin/env python3
"""
prg/utils/input_signal.py
=========================
Build exogenous-input ("consigne") sequences u_{1:N} of shape (N, p) for the
GSS simulator, filter, CLI and GUI — so the generator logic lives in one place.

A spec is one of:
  * an ndarray / list already shaped (N, p) — validated and returned as float;
  * a (p,) vector — broadcast to a constant (N, p);
  * a path to an existing CSV file with p columns and >= N rows (first N used;
    a header row is skipped automatically if present);
  * a string keyword, optionally parameterised as ``name(arg, ...)``:
        'zeros'                     u_n = 0
        'ones'                      u_n = 1
        'const(a, b, ...)'          constant vector [a, b, ...] (length p)
        'gaussian' | 'gaussian(std)'   i.i.d. N(0, std^2), std default 1
        'step'  | 'step(n0)'        0, then amplitude 1 from index n0 (def N//2)
        'ramp'  | 'ramp(slope)'     slope * n / N            (slope default 1)
        'sin'   | 'sin(period)'     sin(2*pi*n / period)     (period default N/4)
        'square'| 'square(period)'  +/-1 square wave         (period default N/4)

For the scalar generators the 1-D waveform is broadcast to all p channels.
"""

from __future__ import annotations

import pathlib

import numpy as np

__all__ = ["make_input", "INPUT_GENERATORS", "parse_spec"]

# Generator names exposed to the GUI / CLI (CSV and constant handled separately).
INPUT_GENERATORS = ("zeros", "ones", "gaussian", "step", "ramp", "sin", "square")


def parse_spec(spec: str) -> tuple[str, list[float]]:
    """Split ``'name(a, b)'`` into ``('name', [a, b])`` (``('name', [])`` if no args)."""
    spec = spec.strip()
    if "(" not in spec:
        return spec.lower(), []
    name, _, rest = spec.partition("(")
    rest = rest.rstrip(")")
    args = [float(x) for x in rest.split(",") if x.strip() != ""]
    return name.strip().lower(), args


def _waveform(name: str, args: list[float], N: int, rng: np.random.Generator) -> np.ndarray:
    """Return a 1-D (N,) waveform for a scalar generator name."""
    n = np.arange(N, dtype=float)
    if name == "zeros":
        return np.zeros(N)
    if name == "ones":
        return np.ones(N)
    if name == "gaussian":
        std = args[0] if args else 1.0
        return rng.standard_normal(N) * std
    if name == "step":
        n0 = int(args[0]) if args else N // 2
        return (n >= n0).astype(float)
    if name == "ramp":
        slope = args[0] if args else 1.0
        return slope * n / max(N, 1)
    if name == "sin":
        period = args[0] if args else max(N / 4.0, 1.0)
        return np.sin(2.0 * np.pi * n / period)
    if name == "square":
        period = args[0] if args else max(N / 4.0, 1.0)
        return np.sign(np.sin(2.0 * np.pi * n / period))
    raise ValueError(
        f"Unknown input generator {name!r}. Known: {', '.join(INPUT_GENERATORS)}, "
        f"const(...), or a CSV path."
    )


def _load_csv(path: pathlib.Path, N: int, p: int) -> np.ndarray:
    """Load (N, p) input from a CSV; skip a non-numeric header row if present."""
    try:
        arr = np.loadtxt(path, delimiter=",")
    except ValueError:
        arr = np.loadtxt(path, delimiter=",", skiprows=1)  # header present
    arr = np.atleast_2d(np.asarray(arr, dtype=float))
    if arr.shape[0] == 1 and p != 1:  # a single row read as (1, p)
        arr = arr.reshape(1, -1)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.shape[0] < N:
        raise ValueError(f"CSV {path} has {arr.shape[0]} rows, need at least N={N}.")
    if arr.shape[1] != p:
        raise ValueError(f"CSV {path} has {arr.shape[1]} columns, expected p={p}.")
    return arr[:N]


def make_input(spec, N: int, p: int, *, seed: int | None = None) -> np.ndarray:
    """
    Build an exogenous-input array of shape ``(N, p)`` from ``spec``.

    Parameters
    ----------
    spec : str | ndarray | list | None
        See the module docstring.  ``None`` returns an all-zero (N, p) array.
    N, p : int
        Sequence length and input dimension (``p`` from ``params.p``).
    seed : int or None
        Seed for the ``'gaussian'`` generator.

    Returns
    -------
    ndarray of shape (N, p), dtype float.
    """
    if p <= 0:
        raise ValueError("make_input requires p >= 1 (the model has no input gain).")

    if spec is None:
        return np.zeros((N, p), dtype=float)

    # Already an array-like signal or a constant vector.
    if not isinstance(spec, str):
        arr = np.asarray(spec, dtype=float)
        if arr.shape == (N, p):
            return arr
        if arr.shape == (p,):
            return np.tile(arr.reshape(1, p), (N, 1))
        if arr.size == N * p:
            return arr.reshape(N, p)
        raise ValueError(f"Input array has shape {arr.shape}, expected (N, p)=({N}, {p}) or (p,).")

    # A CSV path?
    path = pathlib.Path(spec)
    if path.suffix.lower() == ".csv" or path.exists():
        return _load_csv(path, N, p)

    rng = np.random.default_rng(seed)
    name, args = parse_spec(spec)

    if name == "const":
        if len(args) != p:
            raise ValueError(f"const(...) needs p={p} values, got {len(args)}.")
        return np.tile(np.asarray(args, dtype=float).reshape(1, p), (N, 1))

    # Scalar generator broadcast to all p channels.
    wave = _waveform(name, args, N, rng)
    return np.tile(wave.reshape(N, 1), (1, p))
