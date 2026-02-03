"""
GAM calibration using the Rust gnomon CLI.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .base import PGSMethod


class GnomonGAMMethod(PGSMethod):
    """
    GAM calibration via the Rust gnomon CLI.

    This mirrors the mgcv-based GAM but uses gnomon's native spline/REML pipeline.
    """

    def __init__(
        self,
        n_pcs: int = 2,
        pgs_knots: int = 10,
        pgs_degree: int = 3,
        pc_knots: int = 10,
        pc_degree: int = 3,
        penalty_order: int = 2,
        no_calibration: bool = True,
        gnomon_exe: Optional[str] = None,
        keep_work_dir: bool = False,
    ):
        super().__init__(name=f"GAM-Gnomon")
        self.n_pcs = n_pcs
        self.pgs_knots = pgs_knots
        self.pgs_degree = pgs_degree
        self.pc_knots = pc_knots
        self.pc_degree = pc_degree
        self.penalty_order = penalty_order
        self.no_calibration = no_calibration
        self.keep_work_dir = keep_work_dir

        exe = (gnomon_exe or os.environ.get("GNOMON_EXE") or "").strip()
        if exe:
            exe_path = Path(exe)
            if not exe_path.exists():
                raise FileNotFoundError(f"GNOMON_EXE not found: {exe}")
            self.gnomon_exe = str(exe_path)
        else:
            self.gnomon_exe = shutil.which("gnomon")
        if not self.gnomon_exe:
            raise FileNotFoundError(
                "gnomon executable not found. Set GNOMON_EXE or add gnomon to PATH."
            )

        self._work_dir: Optional[Path] = None
        self._model_path: Optional[Path] = None

    def _write_training_tsv(self, path: Path, P: np.ndarray, PC: np.ndarray, y: np.ndarray) -> None:
        n_pcs_actual = min(self.n_pcs, 5, PC.shape[1])
        data = {
            "phenotype": y.astype(float),
            "score": P.astype(float),
            "sex": np.zeros_like(P, dtype=float),
        }
        for i in range(n_pcs_actual):
            data[f"PC{i+1}"] = PC[:, i].astype(float)
        pd.DataFrame(data).to_csv(path, sep="\t", index=False)

    def _write_prediction_tsv(self, path: Path, P: np.ndarray, PC: np.ndarray) -> None:
        n_pcs_actual = min(self.n_pcs, 5, PC.shape[1])
        data = {
            "sample_id": np.arange(1, len(P) + 1),
            "score": P.astype(float),
            "sex": np.zeros_like(P, dtype=float),
        }
        for i in range(n_pcs_actual):
            data[f"PC{i+1}"] = PC[:, i].astype(float)
        pd.DataFrame(data).to_csv(path, sep="\t", index=False)

    @staticmethod
    def _estimate_num_coeffs(
        n_pcs: int,
        pgs_knots: int,
        pgs_degree: int,
        pc_knots: int,
        pc_degree: int,
    ) -> int:
        pgs_basis_coeffs = pgs_knots + pgs_degree + 1
        pgs_main_coeffs = pgs_basis_coeffs - 1
        sex_main_coeffs = 1
        pc_basis_coeffs = pc_knots + pc_degree + 1
        pc_main_coeffs = n_pcs * (pc_basis_coeffs - 1)
        interaction_coeffs = n_pcs * (pgs_basis_coeffs - 1) * (pc_basis_coeffs - 1)
        sex_pgs_interaction_coeffs = pgs_main_coeffs if sex_main_coeffs > 0 else 0
        return (
            1
            + sex_main_coeffs
            + pgs_main_coeffs
            + sex_pgs_interaction_coeffs
            + pc_main_coeffs
            + interaction_coeffs
        )

    def fit(self, P: np.ndarray, PC: np.ndarray, y: np.ndarray) -> "GnomonGAMMethod":
        work_dir = Path(tempfile.mkdtemp(prefix="gnomon_gam_"))
        self._work_dir = work_dir
        train_path = work_dir / "train.tsv"
        self._write_training_tsv(train_path, P, PC, y)

        n_samples = len(y)
        n_pcs_actual = min(self.n_pcs, 5, PC.shape[1])
        pgs_knots = self.pgs_knots
        pc_knots = self.pc_knots
        coeffs = self._estimate_num_coeffs(
            n_pcs_actual, pgs_knots, self.pgs_degree, pc_knots, self.pc_degree
        )
        if coeffs > n_samples:
            while coeffs > n_samples and (pgs_knots > 1 or pc_knots > 1):
                if pc_knots > 1:
                    pc_knots -= 1
                elif pgs_knots > 1:
                    pgs_knots -= 1
                coeffs = self._estimate_num_coeffs(
                    n_pcs_actual, pgs_knots, self.pgs_degree, pc_knots, self.pc_degree
                )
            print(
                "GnomonGAMMethod: reducing knots to avoid over-parameterized model "
                f"(coeffs={coeffs}, samples={n_samples}). "
                f"pgs_knots={pgs_knots}, pc_knots={pc_knots}"
            )

        cmd = [
            self.gnomon_exe,
            "train",
            str(train_path),
            "--num-pcs",
            str(n_pcs_actual),
            "--pgs-knots",
            str(pgs_knots),
            "--pgs-degree",
            str(self.pgs_degree),
            "--pc-knots",
            str(pc_knots),
            "--pc-degree",
            str(self.pc_degree),
            "--penalty-order",
            str(self.penalty_order),
        ]
        if self.no_calibration:
            cmd.append("--no-calibration")

        result = subprocess.run(
            cmd, cwd=work_dir, capture_output=True, text=True
        )
        if result.returncode != 0:
            raise RuntimeError(
                "gnomon train failed:\n"
                f"cmd={' '.join(cmd)}\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )

        model_path = work_dir / "model.toml"
        if not model_path.exists():
            raise RuntimeError("gnomon train did not produce model.toml")

        self._model_path = model_path
        self.is_fitted = True
        return self

    def predict_proba(self, P: np.ndarray, PC: np.ndarray) -> np.ndarray:
        if not self.is_fitted or self._work_dir is None or self._model_path is None:
            raise RuntimeError("Model must be fitted before prediction")

        pred_path = self._work_dir / "predict.tsv"
        self._write_prediction_tsv(pred_path, P, PC)

        cmd = [
            self.gnomon_exe,
            "infer",
            str(pred_path),
            "--model",
            str(self._model_path),
        ]
        if self.no_calibration:
            cmd.append("--no-calibration")

        result = subprocess.run(
            cmd, cwd=self._work_dir, capture_output=True, text=True
        )
        if result.returncode != 0:
            raise RuntimeError(
                "gnomon infer failed:\n"
                f"cmd={' '.join(cmd)}\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )

        out_path = self._work_dir / "predictions.tsv"
        if not out_path.exists():
            raise RuntimeError("gnomon infer did not produce predictions.tsv")

        df = pd.read_csv(out_path, sep="\t")
        if "prediction" not in df.columns:
            raise RuntimeError("predictions.tsv missing 'prediction' column")

        return df["prediction"].to_numpy()

    def __del__(self) -> None:
        if self.keep_work_dir:
            return
        if self._work_dir and self._work_dir.exists():
            shutil.rmtree(self._work_dir, ignore_errors=True)
