#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prg/learning/supervised.py
==========================
Supervised estimation of a GSS model from fully-observed (R, X, Y) data.

Given a CSV produced by ``prg.simulate`` (columns: n, r, x_0, …, x_{q-1},
y_0, …, y_{s-1}), this module estimates all model parameters by ordinary
least squares (OLS) per regime, then optionally enforces the H5 constraint
post-hoc, and saves the result as a ready-to-use model file in ``prg/models/``.

Estimation approach
-------------------
For each regime k the model is

    Z_{n+1} = F(k) Z_n + b(k) + W_{n+1}   (W_{n+1} ~ N(0, Σ_W(k)))

Step 1 — free OLS
    Collect all pairs (Z_n, Z_{n+1}) for which r_{n+1} = k, augment with
    a constant column, and solve the least-squares problem.  This yields
    F(k), b(k), and the sample noise covariance Σ_W(k).

Step 2 — Δ = 0 (optional)
    Zero out the off-diagonal block Δ(k) of Σ_W(k).

Step 3 — H5 projection (optional)
    Recompute one of A, B, or Σ_U analytically from the other parameters
    using the H5 constraint (eq. 4.8).

The Markov transition matrix P is estimated by maximum-likelihood
(transition frequency counts, then row-normalise).

Usage
-----
    python -m prg.learning.supervised <csv> [OPTIONS]

Options
-------
    csv                    Path to the simulation CSV (required)
    --constraint {a,b,su}  H5 constraint target (mutually exclusive)
    --delta-zero           Force Δ(k)=0 before the H5 step
    --output PATH          Output .py path (default: prg/models/<auto>.py)
    --model-name NAME      File/class base name (default: model_learned_K…)
    -v, --verbose          Print per-regime fit summaries

Examples
--------
    python -m prg.learning.supervised data/simulated/sim.csv
    python -m prg.learning.supervised data/simulated/sim.csv --constraint b -v
    python -m prg.learning.supervised data/simulated/sim.csv \\
        --constraint su --delta-zero --output prg/models/my_model.py
"""

from __future__ import annotations

import argparse
import csv as _csv_mod
import logging
import pathlib
import sys
from datetime import datetime

import numpy as np

__all__ = ["fit_supervised"]

_log = logging.getLogger("fofgss.learning.supervised")


# ---------------------------------------------------------------------------
# CSV reader
# ---------------------------------------------------------------------------


def _read_csv(
    path: pathlib.Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int, int]:
    """
    Read a simulation CSV and return ``(rs, xs, ys, K, q, s)``.

    Expected header columns: ``n, r, x_0, …, x_{q-1}, y_0, …, y_{s-1}``
    (the format produced by :class:`prg.classes.GSSSimulator`).

    Parameters
    ----------
    path : pathlib.Path
        Path to the CSV file.

    Returns
    -------
    rs : ndarray (N,)   int   — regime sequence
    xs : ndarray (N, q) float — hidden state
    ys : ndarray (N, s) float — observed state
    K  : int            — number of regimes (= max(r) + 1)
    q  : int            — hidden dimension
    s  : int            — observed dimension

    Raises
    ------
    ValueError
        If the CSV is empty, required columns are missing, or data cannot
        be parsed.
    """
    with path.open(newline="", encoding="utf-8") as fh:
        reader = _csv_mod.reader(fh)
        header = next(reader)
        header = [h.strip() for h in header]
        rows = list(reader)

    if not rows:
        raise ValueError(f"CSV is empty: {path}")

    # Mandatory columns
    if "n" not in header:
        raise ValueError("Missing required column 'n' in CSV header.")
    if "r" not in header:
        raise ValueError("Missing required column 'r' in CSV header.")

    idx_r = header.index("r")

    # Detect x_* and y_* columns (ordered by numeric suffix)
    x_cols = sorted(
        [h for h in header if h.startswith("x_")],
        key=lambda h: int(h[2:]),
    )
    y_cols = sorted(
        [h for h in header if h.startswith("y_")],
        key=lambda h: int(h[2:]),
    )

    if not x_cols:
        raise ValueError("No 'x_*' columns found in CSV header.")
    if not y_cols:
        raise ValueError("No 'y_*' columns found in CSV header.")

    q = len(x_cols)
    s = len(y_cols)
    x_idx = [header.index(h) for h in x_cols]
    y_idx = [header.index(h) for h in y_cols]

    N = len(rows)
    rs = np.empty(N, dtype=int)
    xs = np.empty((N, q), dtype=float)
    ys = np.empty((N, s), dtype=float)

    for i, row in enumerate(rows):
        try:
            rs[i] = int(float(row[idx_r]))
            for j, xi in enumerate(x_idx):
                xs[i, j] = float(row[xi])
            for j, yi in enumerate(y_idx):
                ys[i, j] = float(row[yi])
        except (ValueError, IndexError) as exc:
            raise ValueError(f"Error parsing CSV row {i + 2}: {exc}") from exc

    K = int(rs.max()) + 1
    return rs, xs, ys, K, q, s


# ---------------------------------------------------------------------------
# Numerical helpers
# ---------------------------------------------------------------------------


def _nearest_spd(M: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Return the nearest symmetric positive-definite matrix to *M*.

    Symmetrises M, then clamps eigenvalues below *eps* to *eps*.
    This avoids degenerate covariance matrices due to limited data or
    near-collinear residuals.
    """
    M = (M + M.T) / 2.0
    vals, vecs = np.linalg.eigh(M)
    vals = np.maximum(vals, eps)
    return (vecs * vals) @ vecs.T


# ---------------------------------------------------------------------------
# Per-regime OLS
# ---------------------------------------------------------------------------


def _fit_regime(
    Z_curr: np.ndarray,   # (N_k, dim_z)  Z_n   for transitions arriving in k
    Z_next: np.ndarray,   # (N_k, dim_z)  Z_{n+1}
    q: int,
    s: int,
    constraint: str | None,
    delta_zero: bool,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray,   # A, B, C, D
    np.ndarray, np.ndarray, np.ndarray,               # Sigma_U, Delta, Sigma_V
    np.ndarray,                                       # b  (dim_z, 1)
]:
    """
    Estimate F(k), b(k), Σ_W(k) for one regime by OLS, then apply
    optional Δ=0 and H5 post-processing.

    OLS model
    ---------
    Z_{n+1} ≈ [Z_n | 1] Θ   with   Θ ∈ ℝ^{(dim_z+1) × dim_z}

    The first *dim_z* rows of Θᵀ give F, the last row gives bᵀ.
    The noise covariance is estimated as the sample covariance of the
    residuals (MLE, divided by N_k).

    Post-processing order
    ---------------------
    1. If *delta_zero*: set Δ block of Σ_W to zero.
    2. Clamp Σ_U, Σ_V to nearest SPD (numerical safety).
    3. If *constraint* ∈ {'a','b','su'}: apply H5 projection.
    """
    N_k, dim_z = Z_curr.shape

    # Augment Z_curr with a constant column for bias estimation
    Z_aug = np.hstack([Z_curr, np.ones((N_k, 1))])   # (N_k, dim_z+1)

    # OLS — minimum-norm solution handles underdetermined cases
    Theta, _, _, _ = np.linalg.lstsq(Z_aug, Z_next, rcond=None)
    # Theta : (dim_z+1, dim_z)

    F_full = Theta[:dim_z, :].T           # (dim_z, dim_z)
    b_full = Theta[dim_z, :].reshape(dim_z, 1)  # (dim_z, 1)

    # Residuals and noise covariance (MLE)
    residuals = Z_next - Z_aug @ Theta    # (N_k, dim_z)
    SigW = (residuals.T @ residuals) / N_k
    SigW = (SigW + SigW.T) / 2.0         # symmetrise

    # Extract blocks
    A  = F_full[:q, :q]   # (q, q)
    B  = F_full[:q, q:]   # (q, s)
    C  = F_full[q:, :q]   # (s, q)
    D  = F_full[q:, q:]   # (s, s)

    SU = SigW[:q, :q]     # (q, q)  Σ_U
    Dt = SigW[:q, q:]     # (q, s)  Δ
    SV = SigW[q:, q:]     # (s, s)  Σ_V

    # --- Step 1: enforce Δ = 0 ---
    if delta_zero:
        Dt = np.zeros((q, s))

    # --- Step 2: clamp covariance blocks to SPD ---
    SU = _nearest_spd(SU)
    SV = _nearest_spd(SV)

    # --- Step 3: H5 projection ---
    if constraint == "b":
        from prg.utils.h5_constraint import compute_B_from_h5
        B = compute_B_from_h5(A, C, D, SU, Dt, SV)
    elif constraint == "a":
        from prg.utils.h5_constraint import compute_A_from_h5
        A = compute_A_from_h5(B, C, D, SU, Dt, SV)
    elif constraint == "su":
        from prg.utils.h5_constraint import compute_SU_from_h5
        SU = compute_SU_from_h5(A, B, C, D, Dt, SV)
        SU = _nearest_spd(SU)

    return A, B, C, D, SU, Dt, SV, b_full


# ---------------------------------------------------------------------------
# Full supervised fit
# ---------------------------------------------------------------------------


def fit_supervised(
    rs: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    K: int,
    q: int,
    s: int,
    constraint: str | None = None,
    delta_zero: bool = False,
    verbose: bool = False,
) -> dict:
    """
    Estimate all GSS parameters from fully observed (R, X, Y) data.

    Parameters
    ----------
    rs : ndarray (N,) int
        Regime sequence.
    xs : ndarray (N, q) float
        Hidden-state sequence.
    ys : ndarray (N, s) float
        Observed-state sequence.
    K, q, s : int
        Model dimensions.
    constraint : None | 'a' | 'b' | 'su'
        Post-hoc H5 constraint target.  ``None`` means no constraint.
    delta_zero : bool
        Force Δ(k) = 0 (zero cross-covariance) for all regimes.
    verbose : bool
        Print per-regime summaries to stdout.

    Returns
    -------
    dict with keys:
        K, q, s, P,
        A_list, B_list, C_list, D_list,
        Sigma_U_list, Delta_list, Sigma_V_list,
        pi0 (always None → stationary distribution),
        mu_z0_list, Sigma_z0_list,
        b_list

    Raises
    ------
    ValueError
        If any regime never appears as a *source* in the transition
        sequence (cannot estimate P row), or never appears as a
        *destination* (cannot estimate F(k) by OLS).
    """
    dim_z = q + s
    N = len(rs)

    # ------------------------------------------------------------------
    # Estimate Markov transition matrix P from frequency counts
    # ------------------------------------------------------------------
    P = np.zeros((K, K))
    for n in range(N - 1):
        P[rs[n], rs[n + 1]] += 1

    row_sums = P.sum(axis=1)
    missing_src = np.where(row_sums == 0)[0]
    if missing_src.size > 0:
        raise ValueError(
            f"Regime(s) {missing_src.tolist()} never appear as a transition source — "
            "cannot estimate the corresponding row(s) of P."
        )
    P = P / row_sums[:, np.newaxis]

    # ------------------------------------------------------------------
    # Stack Z = [X; Y]  — shape (N, dim_z)
    # ------------------------------------------------------------------
    Z = np.hstack([xs, ys])

    # ------------------------------------------------------------------
    # Per-regime OLS
    # ------------------------------------------------------------------
    A_list, B_list, C_list, D_list = [], [], [], []
    SU_list, Dt_list, SV_list, b_list = [], [], [], []
    mu_z0_list, Sigma_z0_list = [], []

    for k in range(K):
        # Indices n such that r_{n+1} = k  (transitions arriving at k)
        mask = rs[1:] == k          # length N-1
        idx  = np.where(mask)[0]   # into 0 … N-2

        if idx.size == 0:
            raise ValueError(
                f"Regime k={k} never appears as a transition destination — "
                "cannot estimate F(k) by OLS."
            )

        Z_curr = Z[idx]       # Z_n
        Z_next = Z[idx + 1]   # Z_{n+1}
        N_k = idx.size

        _log.info("Regime k=%d: %d transitions (OLS)", k, N_k)
        if verbose:
            print(f"  Regime k={k}: {N_k} transitions for OLS")
        if N_k < dim_z + 1:
            _log.warning(
                "Regime k=%d: only %d samples for %d parameters — "
                "OLS solution may be underdetermined.",
                k, N_k, dim_z + 1,
            )

        try:
            A, B, C, D, SU, Dt, SV, b_vec = _fit_regime(
                Z_curr, Z_next, q, s, constraint, delta_zero
            )
        except ValueError as exc:
            raise ValueError(f"Regime k={k}: {exc}") from exc

        A_list.append(A)
        B_list.append(B)
        C_list.append(C)
        D_list.append(D)
        SU_list.append(SU)
        Dt_list.append(Dt)
        SV_list.append(SV)
        b_list.append(b_vec)

        # Initial conditions: sample moments over all Z in regime k
        in_k = rs == k
        n_in_k = in_k.sum()
        if n_in_k >= 2:
            Z_k = Z[in_k]
            mu_z0_list.append(Z_k.mean(axis=0).reshape(dim_z, 1))
            cov_k = np.cov(Z_k, rowvar=False)
            Sigma_z0_list.append(_nearest_spd(np.atleast_2d(cov_k)))
        else:
            _log.warning(
                "Regime k=%d has only %d time-step(s) — "
                "using zero mean and identity covariance for μ_z0, Σ_z0.",
                k, n_in_k,
            )
            mu_z0_list.append(np.zeros((dim_z, 1)))
            Sigma_z0_list.append(np.eye(dim_z))

    return {
        "K": K, "q": q, "s": s,
        "P": P,
        "A_list": A_list,
        "B_list": B_list,
        "C_list": C_list,
        "D_list": D_list,
        "Sigma_U_list": SU_list,
        "Delta_list":   Dt_list,
        "Sigma_V_list": SV_list,
        "pi0": None,
        "mu_z0_list":    mu_z0_list,
        "Sigma_z0_list": Sigma_z0_list,
        "b_list": b_list,
    }


# ---------------------------------------------------------------------------
# Code generation
# ---------------------------------------------------------------------------


def _fmt_arr(arr: np.ndarray) -> str:
    """
    Format a 2-D numpy array as a compact ``np.array(…)`` literal.

    Continuation rows are aligned under the opening bracket, matching the
    style produced by the FofGss GUI.  Values are formatted with ``:.8g``
    (up to 8 significant digits, no trailing zeros).
    """
    prefix = "np.array(["
    align  = " " * len(prefix)
    rows   = []
    for r in range(arr.shape[0]):
        vals = ", ".join(f"{v:.10g}" for v in arr[r])
        rows.append(f"[{vals}]")
    inner = (",\n" + align).join(rows)
    return f"{prefix}{inner}])"


def _fmt_list(arrays: list[np.ndarray], field_indent: int = 4) -> str:
    """Format a list of arrays, one per line, aligned past the ``[``."""
    pad = " " * (field_indent + 1)
    items = [_fmt_arr(a) for a in arrays]
    if len(items) == 1:
        return f"[{items[0]}]"
    joined = (",\n" + pad).join(items)
    return f"[{joined}]"


def _generate_model_code(
    params: dict,
    class_name: str,
    file_stem: str,
    source_csv: str,
    constraint: str | None,
    delta_zero: bool,
) -> str:
    """
    Render the source code for a :class:`BaseGSSModel` subclass.

    The generated file is self-contained and importable with
    ``python -m prg.simulate --model <file_stem> …``.
    """
    K, q, s = params["K"], params["q"], params["s"]
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    constraint_line = (
        f"Constraint : {constraint.upper()}" if constraint else "Constraint : none"
    )
    delta_line = "Delta=0    : yes" if delta_zero else "Delta=0    : no"

    lines: list[str] = [
        "#!/usr/bin/env python3",
        "# -*- coding: utf-8 -*-",
        '"""',
        f"prg/models/{file_stem}.py",
        "=" * (len("prg/models/") + len(file_stem) + 3),
        f"GSS model: K={K} states, q={q} (hidden), s={s} (observed).",
        "",
        "Estimated by supervised OLS from fully-observed (R,X,Y) data.",
        f"Source CSV : {source_csv}",
        f"Estimated  : {timestamp}",
        constraint_line,
        delta_line,
        '"""',
        "",
        "from __future__ import annotations",
        "",
        "import numpy as np",
        "",
        "from prg.models.base_gss_model import BaseGSSModel",
        "",
        f'__all__ = ["{class_name}"]',
        "",
        "",
        f"class {class_name}(BaseGSSModel):",
        f'    """GSS model estimated from {pathlib.Path(source_csv).name} '
        f'(K={K}, q={q}, s={s})."""',
        "",
        f"    K: int = {K}",
        f"    q: int = {q}",
        f"    s: int = {s}",
        "",
        "    # --- Markov chain ---",
        f"    P: np.ndarray = {_fmt_arr(params['P'])}",
        "",
        "    # --- Dynamics: F(k) = [[A_k, B_k], [C_k, D_k]] ---",
        f"    A_list: list[np.ndarray] = {_fmt_list(params['A_list'])}",
        f"    B_list: list[np.ndarray] = {_fmt_list(params['B_list'])}",
        f"    C_list: list[np.ndarray] = {_fmt_list(params['C_list'])}",
        f"    D_list: list[np.ndarray] = {_fmt_list(params['D_list'])}",
        "",
        "    # --- Noise covariances: Σ_W(k) = [[Σ_U, Δ], [Δᵀ, Σ_V]] ---",
        f"    Sigma_U_list: list[np.ndarray] = {_fmt_list(params['Sigma_U_list'])}",
        f"    Delta_list:   list[np.ndarray] = {_fmt_list(params['Delta_list'])}",
        f"    Sigma_V_list: list[np.ndarray] = {_fmt_list(params['Sigma_V_list'])}",
        "",
        "    # --- Drift bias ---",
        f"    b_list: list[np.ndarray] = {_fmt_list(params['b_list'])}",
        "",
        "    # --- Initial conditions ---",
        "    pi0: np.ndarray | None = None   # None → stationary distribution",
        "",
        f"    mu_z0_list:    list[np.ndarray] = {_fmt_list(params['mu_z0_list'])}",
        f"    Sigma_z0_list: list[np.ndarray] = {_fmt_list(params['Sigma_z0_list'])}",
        "",
        "    # ------------------------------------------------------------------",
        "",
        "    def get_params(self) -> dict:",
        "        return {",
        '            "K": self.K, "q": self.q, "s": self.s, "P": self.P,',
        '            "A_list": self.A_list, "B_list": self.B_list,',
        '            "C_list": self.C_list, "D_list": self.D_list,',
        '            "Sigma_U_list": self.Sigma_U_list,',
        '            "Delta_list": self.Delta_list,',
        '            "Sigma_V_list": self.Sigma_V_list,',
        '            "pi0": self.pi0,',
        '            "mu_z0_list": self.mu_z0_list,',
        '            "Sigma_z0_list": self.Sigma_z0_list,',
        '            "b_list": self.b_list,',
        "        }",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------


def _class_name_from_stem(stem: str) -> str:
    """Convert a snake_case file stem to a CamelCase class name."""
    return "".join(word.capitalize() for word in stem.split("_"))


def _auto_model_stem(K: int, q: int, s: int) -> str:
    return f"model_learned_K{K}_q{q}_s{s}"


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m prg.learning.supervised",
        description=(
            "Estimate a GSS model by supervised OLS from fully-observed "
            "(R, X, Y) data and write a ready-to-use model file."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "csv", metavar="CSV",
        help="Path to the simulation CSV (columns: n, r, x_0, …, y_0, …).",
    )
    parser.add_argument(
        "--constraint", choices=["a", "b", "su"], default=None,
        metavar="TARGET",
        help=(
            "Enforce H5 constraint post-hoc: recompute A (a), B (b), "
            "or Σ_U (su) from the other estimated parameters. "
            "Options a / b / su are mutually exclusive."
        ),
    )
    parser.add_argument(
        "--delta-zero", action="store_true",
        help="Force Δ(k) = 0 (zero cross-covariance block) before the H5 step.",
    )
    parser.add_argument(
        "--output", default=None, metavar="PATH",
        help=(
            "Destination .py file.  "
            "Default: prg/models/<auto>.py in the package tree."
        ),
    )
    parser.add_argument(
        "--model-name", default=None, metavar="NAME",
        help=(
            "Base name (file stem) for the generated model "
            "(e.g. model_my_gss).  "
            "Default: model_learned_K<K>_q<q>_s<s>."
        ),
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Print per-regime fit summaries.",
    )
    return parser


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-8s %(name)s — %(message)s",
        stream=sys.stdout,
    )

    csv_path = pathlib.Path(args.csv).resolve()
    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    # --- Read data ---
    if args.verbose:
        print(f"Reading {csv_path} …")
    try:
        rs, xs, ys, K, q, s = _read_csv(csv_path)
    except ValueError as exc:
        print(f"Error reading CSV: {exc}", file=sys.stderr)
        sys.exit(1)

    N = len(rs)
    if args.verbose:
        print(f"Data loaded: N={N}, K={K}, q={q}, s={s}")
        for k in range(K):
            print(f"  Regime k={k}: {(rs == k).sum()} time steps")

    # --- Fit ---
    if args.verbose:
        print(
            f"\nFitting (constraint={args.constraint!r}, "
            f"delta_zero={args.delta_zero}) …"
        )
    try:
        params = fit_supervised(
            rs, xs, ys, K, q, s,
            constraint=args.constraint,
            delta_zero=args.delta_zero,
            verbose=args.verbose,
        )
    except ValueError as exc:
        print(f"Error during fitting: {exc}", file=sys.stderr)
        sys.exit(1)

    # --- Resolve file stem and class name ---
    # Stem precedence: --model-name > --output stem > auto
    if args.model_name is not None:
        stem = pathlib.Path(args.model_name).stem
    elif args.output is not None:
        stem = pathlib.Path(args.output).stem
    else:
        stem = _auto_model_stem(K, q, s)
    class_name = _class_name_from_stem(stem)

    # --- Resolve output path ---
    if args.output is not None:
        out_path = pathlib.Path(args.output).resolve()
    else:
        models_dir = pathlib.Path(__file__).resolve().parent.parent / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        out_path = models_dir / f"{stem}.py"

    # --- Generate and write ---
    code = _generate_model_code(
        params=params,
        class_name=class_name,
        file_stem=stem,
        source_csv=str(csv_path),
        constraint=args.constraint,
        delta_zero=args.delta_zero,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(code, encoding="utf-8")

    print(f"\nModel saved   : {out_path}")
    print(f"Class name    : {class_name}")
    print(f"Use with      : python -m prg.simulate --model {stem} -N 1000 --seed 42")

    if args.verbose:
        print("\nEstimated P:")
        print(params["P"])
        for k in range(K):
            print(f"\nRegime k={k}:")
            print(f"  A =\n{params['A_list'][k]}")
            print(f"  B =\n{params['B_list'][k]}")
            print(f"  C =\n{params['C_list'][k]}")
            print(f"  D =\n{params['D_list'][k]}")
            print(f"  Σ_U =\n{params['Sigma_U_list'][k]}")
            print(f"  Δ   =\n{params['Delta_list'][k]}")
            print(f"  Σ_V =\n{params['Sigma_V_list'][k]}")
            print(f"  b   = {params['b_list'][k].ravel()}")


if __name__ == "__main__":
    main()
