#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prg/gui/main_window.py
======================
GSSMainWindow — top-level application window.

Layout
------
  ┌─────────────────────┬───────────────────────────────────────┐
  │  ParamPanel         │  PlotPanel                            │
  │  (tabs F(k)/Σ_W(k)) │                                       │
  │  ─────────────────  │                                       │
  │  N  [spinbox]       │                                       │
  │  Seed  [lineedit]   │                                       │
  │  [Simuler]          │                                       │
  │  [Enregistrer CSV]  │                                       │
  └─────────────────────┴───────────────────────────────────────┘

Fixed (non-editable) parameters
--------------------------------
  P       = uniform (1/K) or model's P
  pi0     = None → stationary distribution
  mu_z0   = zeros
  Sigma_z0= I_{q+s}
"""

import datetime
import pathlib
import numpy as np

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QSpinBox, QLineEdit, QPushButton,
    QDialog, QMessageBox, QSizePolicy,
)

from prg.classes.FMatrix import FMatrix
from prg.classes.GSSParams import GSSParams
from prg.classes.GSSSimulator import GSSSimulator
from prg.classes.NoiseCovariance import GSSNoiseCovariance
from prg.gui.param_panel import ParamPanel
from prg.gui.plot_panel import PlotPanel


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------

class _SimWorker(QThread):
    """Run GSSSimulator in a background thread."""

    finished = pyqtSignal(list, list, object, object)  # ns, rs, xs, ys
    error = pyqtSignal(str)

    def __init__(self, params: GSSParams, N: int, seed: int | None, parent=None):
        super().__init__(parent)
        self._params = params
        self._N = N
        self._seed = seed

    def run(self) -> None:
        try:
            sim = GSSSimulator(self._params, N=self._N, seed=self._seed)
            ns, rs = [], []
            xs_rows, ys_rows = [], []
            for n, r, x, y in sim:
                ns.append(n)
                rs.append(r)
                xs_rows.append(x.ravel())
                ys_rows.append(y.ravel())
            xs = np.array(xs_rows)
            ys = np.array(ys_rows)
            self.finished.emit(ns, rs, xs, ys)
        except Exception as exc:  # noqa: BLE001
            self.error.emit(str(exc))


# ---------------------------------------------------------------------------
# Wait dialog (modal, no close button)
# ---------------------------------------------------------------------------

class _WaitDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Simulation en cours…")
        self.setModal(True)
        self.setWindowFlag(Qt.WindowType.WindowCloseButtonHint, False)
        layout = QVBoxLayout(self)
        lbl = QLabel("Simulation en cours, veuillez patienter…")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(lbl)
        self.setFixedSize(320, 80)


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class GSSMainWindow(QMainWindow):
    """Interactive GSS simulator window."""

    def __init__(
        self,
        K: int,
        q: int,
        s: int,
        P: np.ndarray | None = None,
        model=None,
        parent=None,
    ):
        super().__init__(parent)
        self._K = K
        self._q = q
        self._s = s
        # Transition matrix (fixed, not editable)
        self._P = P if P is not None else np.full((K, K), 1.0 / K)

        self._last_data: tuple | None = None   # (ns, rs, xs, ys, seed_used)
        self._worker: _SimWorker | None = None
        self._wait_dlg: _WaitDialog | None = None

        self.setWindowTitle(
            f"FofGss — Simulateur GSS  (K={K}, q={q}, s={s})"
        )

        # ── central widget ────────────────────────────────────────────
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(12)

        # ── left panel ───────────────────────────────────────────────
        left = QWidget()
        left.setFixedWidth(420)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)

        self._param_panel = ParamPanel(K, q, s)
        self._param_panel.validity_changed.connect(self._on_validity_changed)
        left_layout.addWidget(self._param_panel, stretch=1)

        # N field
        n_row = QHBoxLayout()
        n_row.addWidget(QLabel("N (steps) :"))
        self._n_spin = QSpinBox()
        self._n_spin.setRange(1, 1_000_000)
        self._n_spin.setValue(1000)
        self._n_spin.setSingleStep(100)
        n_row.addWidget(self._n_spin)
        left_layout.addLayout(n_row)

        # Seed field
        seed_row = QHBoxLayout()
        seed_row.addWidget(QLabel("Graine (seed) :"))
        self._seed_edit = QLineEdit()
        self._seed_edit.setPlaceholderText("vide = aléatoire")
        self._seed_edit.setMaximumWidth(120)
        seed_row.addWidget(self._seed_edit)
        left_layout.addLayout(seed_row)

        # Buttons
        self._btn_simulate = QPushButton("Simuler")
        self._btn_simulate.setFixedHeight(36)
        self._btn_simulate.clicked.connect(self._on_simulate)
        left_layout.addWidget(self._btn_simulate)

        self._btn_save = QPushButton("Enregistrer CSV")
        self._btn_save.setFixedHeight(36)
        self._btn_save.setEnabled(False)
        self._btn_save.clicked.connect(self._on_save)
        left_layout.addWidget(self._btn_save)

        self._status_bar = QLabel("")
        self._status_bar.setWordWrap(True)
        self._status_bar.setStyleSheet("color: #cc0000; font-size: 10px;")
        left_layout.addWidget(self._status_bar)

        main_layout.addWidget(left)

        # ── right panel ──────────────────────────────────────────────
        self._plot_panel = PlotPanel(q, s)
        self._plot_panel.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        main_layout.addWidget(self._plot_panel, stretch=1)

        # If a model was passed, load its F and Σ_W
        if model is not None:
            self._load_model(model)

        self._refresh_simulate_button()

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_validity_changed(self, _: bool) -> None:
        self._refresh_simulate_button()

    def _on_simulate(self) -> None:
        params = self._build_gss_params()
        if params is None:
            return

        N = self._n_spin.value()
        seed = self._parse_seed()

        self._btn_simulate.setEnabled(False)
        self._btn_save.setEnabled(False)
        self._status_bar.setText("")

        self._wait_dlg = _WaitDialog(self)
        self._wait_dlg.show()

        self._worker = _SimWorker(params, N, seed, parent=self)
        self._worker.finished.connect(self._on_sim_finished)
        self._worker.error.connect(self._on_sim_error)
        self._worker.start()

    def _on_sim_finished(
        self,
        ns: list[int],
        rs: list[int],
        xs: np.ndarray,
        ys: np.ndarray,
    ) -> None:
        if self._wait_dlg:
            self._wait_dlg.accept()
            self._wait_dlg = None

        seed = self._parse_seed()
        self._last_data = (ns, rs, xs, ys, seed)
        self._plot_panel.update_plots(ns, rs, xs, ys, self._K)

        self._btn_save.setEnabled(True)
        self._refresh_simulate_button()

    def _on_sim_error(self, msg: str) -> None:
        if self._wait_dlg:
            self._wait_dlg.reject()
            self._wait_dlg = None

        self._status_bar.setText(f"Erreur : {msg}")
        self._refresh_simulate_button()

    def _on_save(self) -> None:
        if self._last_data is None:
            return
        ns, rs, xs, ys, seed_used = self._last_data

        output_dir = pathlib.Path("data/simulated")
        output_dir.mkdir(parents=True, exist_ok=True)

        N = len(ns)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        seed_str = str(seed_used) if seed_used is not None else "rnd"
        filename = f"simulated_gui_N{N}_seed{seed_str}_{ts}.csv"
        filepath = output_dir / filename

        q, s = self._q, self._s
        header = ["n", "r"] + [f"x_{i}" for i in range(q)] + [f"y_{i}" for i in range(s)]

        import csv
        with filepath.open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(header)
            for n, r, x_row, y_row in zip(ns, rs, xs, ys):
                writer.writerow([n, r] + list(x_row) + list(y_row))

        QMessageBox.information(
            self,
            "CSV enregistré",
            f"Fichier enregistré :\n{filepath.resolve()}",
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _refresh_simulate_button(self) -> None:
        valid = self._param_panel.is_valid()
        self._btn_simulate.setEnabled(valid)
        if valid:
            self._btn_simulate.setStyleSheet("")
            self._status_bar.setText("")
        else:
            self._btn_simulate.setStyleSheet(
                "background-color: #ffcccc; color: #990000;"
            )
            self._status_bar.setText(
                "Paramètre(s) invalide(s) — corriger avant de simuler."
            )

    def _parse_seed(self) -> int | None:
        text = self._seed_edit.text().strip()
        if not text:
            return None
        try:
            return int(text)
        except ValueError:
            return None

    def _build_gss_params(self) -> GSSParams | None:
        """Collect GUI values and build a GSSParams object."""
        K, q, s = self._K, self._q, self._s

        F_list = self._param_panel.get_F_list()
        Sigma_W_list = self._param_panel.get_Sigma_W_list()
        if F_list is None or Sigma_W_list is None:
            self._status_bar.setText("Paramètre(s) invalide(s).")
            return None

        # Decompose Sigma_W(k) into blocks Sigma_U, Delta, Sigma_V
        Sigma_U_list = [sw[:q, :q] for sw in Sigma_W_list]
        Delta_list   = [sw[:q, q:] for sw in Sigma_W_list]
        Sigma_V_list = [sw[q:, q:] for sw in Sigma_W_list]

        # Decompose F(k) into blocks A, B, C, D
        A_list = [f[:q, :q] for f in F_list]
        B_list = [f[:q, q:] for f in F_list]
        C_list = [f[q:, :q] for f in F_list]
        D_list = [f[q:, q:] for f in F_list]

        try:
            f_matrix = FMatrix(K, q, s, A_list, B_list, C_list, D_list)
            noise_cov = GSSNoiseCovariance(K, q, s, Sigma_U_list, Delta_list, Sigma_V_list)

            mu_z0_list    = [np.zeros((q + s, 1)) for _ in range(K)]
            Sigma_z0_list = [np.eye(q + s) for _ in range(K)]

            params = GSSParams(
                K=K, q=q, s=s,
                P=self._P,
                f_matrix=f_matrix,
                noise_cov=noise_cov,
                pi0=None,               # stationary
                mu_z0_list=mu_z0_list,
                Sigma_z0_list=Sigma_z0_list,
            )
        except Exception as exc:  # noqa: BLE001
            self._status_bar.setText(f"Erreur de paramètres : {exc}")
            return None

        return params

    def _load_model(self, model) -> None:
        """Pre-fill tables from a BaseGSSModel instance."""
        from prg.classes.NoiseCovariance import GSSNoiseCovariance as _NC
        from prg.classes.FMatrix import FMatrix as _FM

        p = model.get_params()
        K, q, s = p["K"], p["q"], p["s"]

        # Build block noise cov to get full Σ_W
        nc = _NC(K, q, s, p["Sigma_U_list"], p["Delta_list"], p["Sigma_V_list"])
        fm = _FM(K, q, s, p["A_list"], p["B_list"], p["C_list"], p["D_list"])

        for k in range(K):
            self._param_panel.set_state_params(k, fm.F(k), nc.Sigma_W(k))

        if p.get("P") is not None:
            self._P = np.asarray(p["P"])
