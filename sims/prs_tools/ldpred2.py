"""
Wrapper for LDpred2 (via R script).
"""
import subprocess
import pandas as pd
import os
import shutil

class LDpred2:
    def __init__(self, r_script_path="sims/prs_tools/run_ldpred2.R"):
        self.r_script_path = r_script_path

    def _safe_write_text(self, path: str, text: str) -> None:
        try:
            with open(path, "w", encoding="utf-8", errors="replace") as f:
                f.write(text)
        except Exception:
            pass

    def _rscript_diagnostics(self) -> str:
        rscript_exe = shutil.which("Rscript") or "Rscript"
        version = "<unavailable>"
        try:
            v = subprocess.run([rscript_exe, "--version"], capture_output=True, text=True)
            version_out = (v.stdout or "") + ("\n" + v.stderr if v.stderr else "")
            version = version_out.strip() or f"<exit={v.returncode}>"
        except Exception as e:
            version = f"<error: {type(e).__name__}: {e}>"
        return f"rscript_exe={rscript_exe} rscript_version={version} cwd={os.getcwd()}"
        
    def fit(self, bfile_train, pheno_file, bfile_val, out_prefix):
        """
        Run LDpred2 via R wrapper.
        """
        cmd = [
            "Rscript", self.r_script_path,
            bfile_train,
            pheno_file,
            bfile_val,
            out_prefix
        ]
        
        print(f"Running LDpred2: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            self._safe_write_text(f"{out_prefix}.ldpred2.cmd", " ".join(cmd) + "\n")
            self._safe_write_text(f"{out_prefix}.ldpred2.stdout", result.stdout or "")
            self._safe_write_text(f"{out_prefix}.ldpred2.stderr", result.stderr or "")
            raise RuntimeError(
                f"LDpred2 failed ({self._rscript_diagnostics()}):\n{result.stderr}\n\nSTDOUT:\n{result.stdout}"
            )
            
        return f"{out_prefix}.scores"

    def predict(self, bfile_test, score_file, out_prefix):
        """
        Score using PLINK2.
        """
        cmd = [
            "plink2",
            "--bfile", bfile_test,
            "--score", score_file, "1", "2", "3",
            "--out", out_prefix
        ]
        
        print(f"Running Scoring: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"PLINK scoring failed:\n{result.stderr}")
            
        score_path = f"{out_prefix}.sscore"
        results = pd.read_csv(score_path, sep='\t')
        id_col = '#IID' if '#IID' in results.columns else 'IID'
        if id_col not in results.columns or 'SCORE1_AVG' not in results.columns:
            raise RuntimeError(
                "PLINK2 .sscore missing expected columns. "
                f"Path={score_path} Columns={list(results.columns)}"
            )
        return results[[id_col, 'SCORE1_AVG']].rename(columns={id_col: 'IID', 'SCORE1_AVG': 'PRS'})
