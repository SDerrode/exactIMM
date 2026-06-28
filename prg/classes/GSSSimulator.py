#!/usr/bin/env python3
"""
prg/classes/GSSSimulator.py
===========================
Simulator for the GSS (Gaussian Switching System) model (equation 7).

Algorithm
---------
n = 0  (initial step):
    R_0 ~ Categorical(pi0)
    Z_0 ~ N(mu_z0[R_0], Sigma_z0[R_0])

n = 1, ..., N-1:
    R_n ~ Categorical(P[R_{n-1}, :])
    Z_n ~ N(F(R_n) @ Z_{n-1} + b(R_n) + G(R_n) @ u_{n-1}, Sigma_W(R_n))

The exogenous-input term G(R_n) @ u_{n-1} ("consigne") is present only when the
model defines input gains (params.p > 0) and an input sequence u is supplied;
otherwise the model is autonomous and behaves exactly as before.  The transition
to Z_n is driven by the *previous* input u_{n-1}, consistent with the slaving
read-out X_n = M_{r_n} y_n + N_{r_n} u_{n-1} of the exact filter.

Each step yields (n, r_n, x_n, y_n) where:
    n    : int          — time index in {0, ..., N-1}
    r_n  : int          — state in {0, ..., K-1}
    x_n  : ndarray (q, 1)
    y_n  : ndarray (s, 1)

Usage
-----
    sim = GSSSimulator(params, N=500, seed=42)
    for n, r, x, y in sim:
        ...  # process step n

    # Or run and save to CSV in one call:
    sim.run(output_dir="data/simulated")
"""

from __future__ import annotations

import csv
import logging
import pathlib
import time

import numpy as np

from prg.classes.GSSParams import GSSParams
from prg.utils.exceptions import ParamError, SimulationError

__all__ = ["GSSSimulator"]

logger = logging.getLogger("exactIMM.simulator")


class GSSSimulator:
    """
    Iterator that generates (n, r_n, x_n, y_n) according to the GSS model.

    Parameters
    ----------
    params : GSSParams
        All model parameters.
    N : int
        Total number of time steps to generate (n = 0 to N-1).
        Must be a strictly positive integer.
    seed : int or None, optional
        Random seed for reproducibility.  ``None`` gives a non-deterministic
        sequence (default).
    u : ndarray (N, p), callable, or None, optional
        Exogenous input ("consigne").  If an array, ``u[i]`` is the input
        u_i applied at time i (the transition to Z_n uses u_{n-1}).  If a
        callable, it is invoked as ``u(i) -> (p,)`` or ``(p, 1)``.  If None
        (default) the model is autonomous (no input term).  Requires the
        model to define input gains (``params.p > 0``).

    Raises
    ------
    ParamError
        If ``N`` is not a strictly positive integer, or if ``u`` is given
        with an incompatible shape or for a model without input gains.

    Examples
    --------
    >>> sim = GSSSimulator(params, N=100, seed=0)
    >>> for n, r, x, y in sim:
    ...     pass   # x.shape == (q, 1), y.shape == (s, 1)

    >>> sim.reset(seed=0)   # restart from n=0 with the same seed
    >>> path = sim.run()    # run & save CSV
    """

    def __init__(
        self,
        params: GSSParams,
        N: int,
        seed: int | None = None,
        u: np.ndarray | object | None = None,
    ) -> None:
        if not (isinstance(N, int) and N > 0):
            raise ParamError(f"N must be a strictly positive integer, got {N!r}.")

        self._params = params
        self._N = N
        self._seed = seed
        self._u = self._validate_input(u, params, N)
        self._reset_state(seed)

    @staticmethod
    def _validate_input(u, params: GSSParams, N: int):
        """Normalise the optional exogenous input; return array/callable/None."""
        if u is None:
            return None
        p = params.p
        if p == 0:
            raise ParamError(
                "An exogenous input u was provided but the model has no input "
                "gain (params.p == 0). Add G_list to the model parameters."
            )
        if callable(u):
            return u
        u = np.asarray(u, dtype=float)
        if u.shape != (N, p):
            raise ParamError(f"Input u must have shape (N, p) = ({N}, {p}), got {u.shape}.")
        return u

    def _get_u(self, i: int) -> np.ndarray:
        """Input u_i as a (p, 1) column (zeros when no input is set)."""
        p = self._params.p
        if self._u is None or p == 0 or i < 0:
            return np.zeros((p, 1), dtype=float)
        if callable(self._u):
            return np.asarray(self._u(i), dtype=float).reshape(p, 1)
        return self._u[i].reshape(p, 1)

    # ------------------------------------------------------------------
    # Internal state management
    # ------------------------------------------------------------------

    def _reset_state(self, seed: int | None) -> None:
        """Initialise (or reinitialise) the RNG and iteration counter."""
        self._rng = np.random.default_rng(seed)
        self._n: int = 0  # next step index to yield
        self._r_prev: int | None = None
        self._z_prev: np.ndarray | None = None  # shape (dim_z, 1)

    # ------------------------------------------------------------------
    # Iterator protocol
    # ------------------------------------------------------------------

    def __iter__(self) -> GSSSimulator:
        return self

    def __next__(self) -> tuple[int, int, np.ndarray, np.ndarray]:
        """
        Generate the next sample (n, r_n, x_n, y_n).

        Returns
        -------
        n   : int
        r_n : int in {0, ..., K-1}
        x_n : ndarray of shape (q, 1)
        y_n : ndarray of shape (s, 1)

        Raises
        ------
        StopIteration
            After N steps have been yielded.
        SimulationError
            If the generated values are non-finite.
        """
        if self._n >= self._N:
            raise StopIteration

        params = self._params
        n = self._n

        if n == 0:
            # --- Initial step ---
            r_n = int(self._rng.choice(params.K, p=params.pi0))
            L = params.chol_z0(r_n)
            mu = params.mu_z0(r_n)  # (dim_z, 1)
            noise = self._rng.standard_normal((params.dim_z, 1))
            z_n = mu + L @ noise
        else:
            # --- Transition step ---
            r_n = int(self._rng.choice(params.K, p=params.P[self._r_prev]))
            F = params.f_matrix.F(r_n)  # (dim_z, dim_z)
            L = params.noise_cov.chol_W(r_n)  # (dim_z, dim_z)
            noise = self._rng.standard_normal((params.dim_z, 1))
            z_n = F @ self._z_prev + params.b(r_n) + L @ noise
            # Exogenous input: transition to Z_n is driven by u_{n-1}.
            if params.p > 0 and self._u is not None:
                z_n = z_n + params.G(r_n) @ self._get_u(n - 1)

        if __debug__:
            if not np.all(np.isfinite(z_n)):
                raise SimulationError(f"Non-finite values generated at step n={n}, r={r_n}.")

        self._r_prev = r_n
        self._z_prev = z_n
        self._n += 1

        q = params.q
        x_n = z_n[:q, :]  # (q, 1)
        y_n = z_n[q:, :]  # (s, 1)

        return n, r_n, x_n, y_n

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None) -> None:
        """
        Restart the simulation from n = 0.

        Parameters
        ----------
        seed : int or None
            New seed.  If omitted, the original seed is reused.
        """
        new_seed = seed if seed is not None else self._seed
        self._seed = new_seed
        self._reset_state(new_seed)
        logger.debug("Simulator reset (seed=%s).", new_seed)

    # ------------------------------------------------------------------
    # Run and save
    # ------------------------------------------------------------------

    def run(
        self, output_dir: str | pathlib.Path | None = None, model_name: str = "gss"
    ) -> pathlib.Path:
        """
        Run the full simulation and save the results to a CSV file.

        Parameters
        ----------
        output_dir : str or Path, optional
            Directory where the CSV is written.
            Defaults to ``data/simulated`` relative to the current
            working directory.
        model_name : str
            Used in the output filename.

        Returns
        -------
        pathlib.Path
            Path to the written CSV file.
        """
        if output_dir is None:
            output_dir = pathlib.Path("data") / "simulated"
        output_dir = pathlib.Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        seed_str = str(self._seed) if self._seed is not None else "random"
        filename = f"simulated_{model_name}_N{self._N}_seed{seed_str}.csv"
        filepath = output_dir / filename

        params = self._params
        q, s = params.q, params.s

        # Build CSV header.  Optional input columns u_* go LAST so that the
        # x_/y_ columns keep their fixed positions (backward compat: no u
        # columns when the model is autonomous).
        has_input = self._u is not None and params.p > 0
        header = ["n", "r"]
        header += [f"x_{i}" for i in range(q)]
        header += [f"y_{i}" for i in range(s)]
        if has_input:
            header += [f"u_{i}" for i in range(params.p)]

        logger.info(
            "Starting simulation: model=%s  N=%d  K=%d  q=%d  s=%d  seed=%s",
            model_name,
            self._N,
            params.K,
            q,
            s,
            seed_str,
        )

        self.reset()  # always start from n=0

        t_start = time.perf_counter()
        log_every = max(1, self._N // 10)  # log ~10 progress messages

        rows: list[list] = []
        for n, r, x, y in self:
            row = [n, r] + x.ravel().tolist() + y.ravel().tolist()
            if has_input:
                row += self._get_u(n).ravel().tolist()  # records u_n at row n
            rows.append(row)

            if (n + 1) % log_every == 0 or n + 1 == self._N:
                pct = 100 * (n + 1) / self._N
                logger.debug("Progress: step %d/%d  (%.0f%%)", n + 1, self._N, pct)

        elapsed = time.perf_counter() - t_start

        with filepath.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(header)
            writer.writerows(rows)

        logger.info(
            "Simulation complete in %.3f s.  Saved %d rows to '%s'.",
            elapsed,
            len(rows),
            filepath,
        )
        self._log_empirical_stats(rows, q, s)

        return filepath

    def _log_empirical_stats(self, rows: list[list], q: int, s: int) -> None:
        """Log basic empirical statistics (means, std) at DEBUG level."""
        if not rows:
            return
        data = np.array([[row[i] for i in range(2, 2 + q + s)] for row in rows])
        means = data.mean(axis=0)
        stds = data.std(axis=0)
        x_names = [f"x_{i}" for i in range(q)]
        y_names = [f"y_{i}" for i in range(s)]
        names = x_names + y_names
        stats = "  ".join(f"{n}: mean={m:.3f} std={sd:.3f}" for n, m, sd in zip(names, means, stds))
        logger.debug("Empirical stats — %s", stats)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def N(self) -> int:
        return self._N

    @property
    def params(self) -> GSSParams:
        return self._params

    def __repr__(self) -> str:
        return (
            f"<GSSSimulator(N={self._N}, K={self._params.K}, "
            f"q={self._params.q}, s={self._params.s}, seed={self._seed})>"
        )
