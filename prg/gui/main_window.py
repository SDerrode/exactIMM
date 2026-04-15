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
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QGridLayout,
    QLabel, QSpinBox, QLineEdit, QPushButton,
    QDialog, QMessageBox, QSizePolicy, QSplitter,
    QFileDialog,
)

from prg.classes.FMatrix import FMatrix
from prg.classes.GSSParams import GSSParams
from prg.classes.GSSSimulator import GSSSimulator
from prg.classes.NoiseCovariance import GSSNoiseCovariance
from prg.gui.matrix_widget import StochasticMatrixWidget
from prg.gui.param_panel import ParamPanel
from prg.gui.plot_panel import PlotPanel


# ---------------------------------------------------------------------------
# Background workers
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


class _FilterWorker(QThread):
    """Run GSSFilter step-by-step in a background thread."""

    # E_xs (N,q), Var_xs (N,q), pis (N,K)
    finished = pyqtSignal(object, object, object)
    error = pyqtSignal(str)

    def __init__(
        self,
        params: GSSParams,
        ys: np.ndarray,   # (N, s)
        parent=None,
    ):
        super().__init__(parent)
        self._params = params
        self._ys = ys

    def run(self) -> None:
        try:
            from prg.filter.gss_filter import GSSFilter
            filt = GSSFilter(self._params)
            E_xs_list:   list[np.ndarray] = []
            Var_xs_list: list[np.ndarray] = []
            pis_list:    list[np.ndarray] = []
            for y_row in self._ys:
                res = filt.step(y_row.reshape(-1, 1))
                E_xs_list.append(res.E_x.ravel())
                Var_xs_list.append(res.Var_x.diagonal())
                pis_list.append(res.pi)
            self.finished.emit(
                np.array(E_xs_list),
                np.array(Var_xs_list),
                np.array(pis_list),
            )
        except Exception as exc:  # noqa: BLE001
            self.error.emit(str(exc))


# ---------------------------------------------------------------------------
# Wait dialog (modal, no close button)
# ---------------------------------------------------------------------------

class _WaitDialog(QDialog):
    def __init__(self, message: str = "Please wait…", parent=None):
        super().__init__(parent)
        self.setWindowTitle(message)
        self.setModal(True)
        self.setWindowFlag(Qt.WindowType.WindowCloseButtonHint, False)
        layout = QVBoxLayout(self)
        lbl = QLabel(message)
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
        self._filter_worker: _FilterWorker | None = None
        self._wait_dlg: _WaitDialog | None = None
        self._current_params: GSSParams | None = None

        self.setWindowTitle(
            f"FofGss — GSS Simulator  (K={K}, q={q}, s={s})"
        )

        # ── central widget ────────────────────────────────────────────
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        # ── left panel ───────────────────────────────────────────────
        left = QWidget()
        left.setMinimumWidth(340)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)

        self._param_panel = ParamPanel(K, q, s)
        self._param_panel.validity_changed.connect(self._on_validity_changed)
        left_layout.addWidget(self._param_panel, stretch=1)

        # Transition matrix P
        p_section = QHBoxLayout()
        p_label = QLabel("Transition matrix:")
        p_label.setStyleSheet("font-size: 11px;")
        p_section.addWidget(p_label)
        self._p_widget = StochasticMatrixWidget(K, title=f"P  ({K}×{K})")
        self._p_widget.set_matrix(self._P)
        self._p_widget.validity_changed.connect(self._on_validity_changed)
        p_section.addWidget(self._p_widget)
        p_section.addStretch()
        left_layout.addLayout(p_section)

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
        seed_row.addWidget(QLabel("Seed:"))
        self._seed_edit = QLineEdit()
        self._seed_edit.setPlaceholderText("empty = random")
        self._seed_edit.setMaximumWidth(120)
        seed_row.addWidget(self._seed_edit)
        left_layout.addLayout(seed_row)

        # Buttons — grille 2×2
        #   [Simuler]          [Filtrer]
        #   [Enregistrer CSV]  [Exporter modèle…]
        btn_grid = QGridLayout()
        btn_grid.setSpacing(6)

        self._btn_simulate = QPushButton("Simulate")
        self._btn_simulate.setFixedHeight(36)
        self._btn_simulate.clicked.connect(self._on_simulate)

        self._btn_filter = QPushButton("Filter")
        self._btn_filter.setFixedHeight(36)
        self._btn_filter.setEnabled(False)
        self._btn_filter.clicked.connect(self._on_filter)

        self._btn_save = QPushButton("Save CSV")
        self._btn_save.setFixedHeight(36)
        self._btn_save.setEnabled(False)
        self._btn_save.clicked.connect(self._on_save)

        self._btn_export = QPushButton("Export model…")
        self._btn_export.setFixedHeight(36)
        self._btn_export.setEnabled(False)
        self._btn_export.clicked.connect(self._on_export_model)

        btn_grid.addWidget(self._btn_simulate, 0, 0)
        btn_grid.addWidget(self._btn_filter,   0, 1)
        btn_grid.addWidget(self._btn_save,     1, 0)
        btn_grid.addWidget(self._btn_export,   1, 1)

        left_layout.addLayout(btn_grid)

        self._eqm_label = QLabel("")
        self._eqm_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._eqm_label.setStyleSheet("font-size: 10px; padding: 2px;")
        left_layout.addWidget(self._eqm_label)

        self._status_bar = QLabel("")
        self._status_bar.setWordWrap(True)
        self._status_bar.setStyleSheet("color: #cc0000; font-size: 10px;")
        left_layout.addWidget(self._status_bar)

        splitter.addWidget(left)

        # ── right panel ──────────────────────────────────────────────
        self._plot_panel = PlotPanel(q, s)
        self._plot_panel.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        splitter.addWidget(self._plot_panel)

        # Left panel takes ~420px initially, right panel takes the rest
        splitter.setSizes([420, 800])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

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

        self._current_params = params
        self._btn_simulate.setEnabled(False)
        self._btn_save.setEnabled(False)
        self._btn_filter.setEnabled(False)
        self._eqm_label.setText("")
        self._plot_panel.clear_filter_overlay()
        self._status_bar.setText("")

        self._wait_dlg = _WaitDialog(parent=self)
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
        self._btn_filter.setEnabled(True)
        self._refresh_simulate_button()

    def _on_sim_error(self, msg: str) -> None:
        if self._wait_dlg:
            self._wait_dlg.reject()
            self._wait_dlg = None

        self._status_bar.setText(f"Error: {msg}")
        self._refresh_simulate_button()

    def _on_filter(self) -> None:
        if self._last_data is None or self._current_params is None:
            return

        _, _, _, ys, _ = self._last_data

        self._btn_filter.setEnabled(False)
        self._eqm_label.setText("")
        self._status_bar.setText("")

        self._wait_dlg = _WaitDialog("Filtering…", parent=self)
        self._wait_dlg.show()

        self._filter_worker = _FilterWorker(
            self._current_params, ys, parent=self
        )
        self._filter_worker.finished.connect(self._on_filter_finished)
        self._filter_worker.error.connect(self._on_filter_error)
        self._filter_worker.start()

    def _on_filter_finished(
        self,
        E_xs: np.ndarray,    # (N, q)
        Var_xs: np.ndarray,  # (N, q)
        pis: np.ndarray,     # (N, K)
    ) -> None:
        if self._wait_dlg:
            self._wait_dlg.accept()
            self._wait_dlg = None

        ns, rs, xs, ys, _ = self._last_data

        # Overlay filtered estimates on the plot
        self._plot_panel.add_filter_overlay(ns, E_xs, Var_xs)

        # Compute and display EQMM (mean squared error averaged over time and components)
        sq_err = np.mean((xs - E_xs) ** 2, axis=1)   # (N,)
        eqm = float(sq_err.mean())
        self._eqm_label.setText(f"MSE: {eqm:.5f}")

        self._btn_filter.setEnabled(True)

    def _on_filter_error(self, msg: str) -> None:
        if self._wait_dlg:
            self._wait_dlg.reject()
            self._wait_dlg = None

        self._status_bar.setText(f"Filter error: {msg}")
        self._btn_filter.setEnabled(True)

    def _on_export_model(self) -> None:
        """Open a save-file dialog and write a ready-to-use Python model file."""
        default_name = (
            f"model_gss_K{self._K}_q{self._q}_s{self._s}_custom.py"
        )
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Python model",
            str(pathlib.Path("prg/models") / default_name),
            "Python files (*.py);;All files (*)",
        )
        if not path:
            return   # user cancelled

        code = self._generate_model_code(pathlib.Path(path).stem)
        try:
            pathlib.Path(path).write_text(code, encoding="utf-8")
            QMessageBox.information(
                self,
                "Model exported",
                f"File saved:\n{pathlib.Path(path).resolve()}",
            )
        except OSError as exc:
            QMessageBox.critical(self, "Write error", str(exc))

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
            "CSV saved",
            f"File saved:\n{filepath.resolve()}",
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _refresh_simulate_button(self) -> None:
        valid = self._param_panel.is_valid() and self._p_widget.is_valid()
        self._btn_simulate.setEnabled(valid)
        self._btn_export.setEnabled(valid)
        if valid:
            self._btn_simulate.setStyleSheet("")
            self._status_bar.setText("")
        else:
            self._btn_simulate.setStyleSheet(
                "border: 2px solid #cc0000; color: #cc0000;"
            )
            self._status_bar.setText(
                "Invalid parameter(s) — fix before simulating."
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
        mu_z0_list = self._param_panel.get_mu_z0_list()
        b_list = self._param_panel.get_b_list()
        P = self._p_widget.get_matrix()
        if F_list is None or Sigma_W_list is None or mu_z0_list is None or b_list is None or P is None:
            self._status_bar.setText("Invalid parameter(s).")
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

            Sigma_z0_list = [np.eye(q + s) for _ in range(K)]

            params = GSSParams(
                K=K, q=q, s=s,
                P=P,
                f_matrix=f_matrix,
                noise_cov=noise_cov,
                pi0=None,               # stationary
                mu_z0_list=mu_z0_list,  # from GUI
                Sigma_z0_list=Sigma_z0_list,
                b_list=b_list,          # from GUI
            )
        except Exception as exc:  # noqa: BLE001
            self._status_bar.setText(f"Parameter error: {exc}")
            return None

        return params

    def _generate_model_code(self, file_stem: str) -> str:
        """
        Generate a complete Python model file from the current GUI parameters.

        Parameters
        ----------
        file_stem : str
            Stem of the target file (e.g. ``'model_gss_K2_q1_s1_custom'``).
            Used to derive the class name.
        """
        K, q, s = self._K, self._q, self._s

        # Class name: CamelCase from file stem
        class_name = "".join(w.capitalize() for w in file_stem.split("_"))

        # Collect parameters from GUI
        F_list        = self._param_panel.get_F_list()
        Sigma_W_list  = self._param_panel.get_Sigma_W_list()
        mu_z0_list    = self._param_panel.get_mu_z0_list()
        b_list        = self._param_panel.get_b_list()
        P             = self._p_widget.get_matrix()

        # Decompose blocks
        A_list       = [f[:q,  :q]  for f in F_list]
        B_list       = [f[:q,  q:]  for f in F_list]
        C_list       = [f[q:,  :q]  for f in F_list]
        D_list       = [f[q:,  q:]  for f in F_list]
        Sigma_U_list = [sw[:q, :q]  for sw in Sigma_W_list]
        Delta_list   = [sw[:q, q:]  for sw in Sigma_W_list]
        Sigma_V_list = [sw[q:, q:]  for sw in Sigma_W_list]

        def _fmt_arr(arr: np.ndarray) -> str:
            """Format a 2-D numpy array as a compact np.array(...) literal.

            All rows after the first are aligned under the opening ``[``.
            e.g.  np.array([[0.8, 0.1],
                             [0.0, 0.7]])
            """
            prefix = "np.array(["
            align  = " " * len(prefix)         # align continuation rows
            rows   = []
            for r in range(arr.shape[0]):
                vals = ", ".join(f"{v:.8g}" for v in arr[r])
                rows.append(f"[{vals}]")
            inner = (",\n" + align).join(rows)
            return f"{prefix}{inner}])"

        def _fmt_list(arrays, field_indent: int = 4) -> str:
            """Format a list of arrays with each item on its own line.

            field_indent : spaces before each item (matches class body indent).
            """
            pad = " " * (field_indent + 1)   # one extra to align past '['
            items = [_fmt_arr(a) for a in arrays]
            if len(items) == 1:
                return f"[{items[0]}]"
            joined = (",\n" + pad).join(items)
            return f"[{joined}]"

        lines: list[str] = []
        lines += [
            "#!/usr/bin/env python3",
            "# -*- coding: utf-8 -*-",
            f'"""',
            f"prg/models/{file_stem}.py",
            f"{'=' * (len('prg/models/') + len(file_stem) + 3)}",
            f"GSS model: K={K} states, q={q} (hidden), s={s} (observed).",
            "",
            "Generated by FofGss GUI.",
            f'"""',
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
            f'    """GSS model exported from the GUI (K={K}, q={q}, s={s})."""',
            "",
            f"    K: int = {K}",
            f"    q: int = {q}",
            f"    s: int = {s}",
            "",
            "    # --- Markov chain ---",
            f"    P: np.ndarray = {_fmt_arr(P)}",
            "",
            "    # --- Dynamics: F(k) = [[A_k, B_k], [C_k, D_k]] ---",
            f"    A_list: list[np.ndarray] = {_fmt_list(A_list)}",
            f"    B_list: list[np.ndarray] = {_fmt_list(B_list)}",
            f"    C_list: list[np.ndarray] = {_fmt_list(C_list)}",
            f"    D_list: list[np.ndarray] = {_fmt_list(D_list)}",
            "",
            "    # --- Noise covariances: Sigma_W(k) = [[Sigma_U, Delta], [Delta^T, Sigma_V]] ---",
            f"    Sigma_U_list: list[np.ndarray] = {_fmt_list(Sigma_U_list)}",
            f"    Delta_list:   list[np.ndarray] = {_fmt_list(Delta_list)}",
            f"    Sigma_V_list: list[np.ndarray] = {_fmt_list(Sigma_V_list)}",
            "",
            "    # --- Initial conditions ---",
            "    pi0: np.ndarray | None = None   # None → stationary distribution",
            "",
            f"    mu_z0_list: list[np.ndarray] = {_fmt_list(mu_z0_list)}",
            f"    Sigma_z0_list: list[np.ndarray] = {_fmt_list([np.eye(q + s)] * K)}",
            f"    b_list: list[np.ndarray] = {_fmt_list(b_list)}",
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

    def _load_model(self, model) -> None:
        """Pre-fill tables from a BaseGSSModel instance."""
        from prg.classes.NoiseCovariance import GSSNoiseCovariance as _NC
        from prg.classes.FMatrix import FMatrix as _FM

        p = model.get_params()
        K, q, s = p["K"], p["q"], p["s"]

        # Build block noise cov to get full Σ_W
        nc = _NC(K, q, s, p["Sigma_U_list"], p["Delta_list"], p["Sigma_V_list"])
        fm = _FM(K, q, s, p["A_list"], p["B_list"], p["C_list"], p["D_list"])

        mu_list = p.get("mu_z0_list")
        b_list  = p.get("b_list")
        for k in range(K):
            mu = np.asarray(mu_list[k]) if mu_list is not None else None
            b  = np.asarray(b_list[k])  if b_list  is not None else None
            self._param_panel.set_state_params(k, fm.F(k), nc.Sigma_W(k), mu, b)

        if p.get("P") is not None:
            self._P = np.asarray(p["P"])
            self._p_widget.set_matrix(self._P)
