"""
Wrapper for GCTB BayesR.
"""
import subprocess
import os
import shutil
from pathlib import Path
import pandas as pd
import numpy as np

class BayesR:
    def __init__(self, gctb_path="gctb"):
        self.gctb_path = gctb_path

    def _gctb_diagnostics(self) -> str:
        exe_path = shutil.which(self.gctb_path) or self.gctb_path
        version = "<unavailable>"
        try:
            v = subprocess.run([exe_path, "--version"], capture_output=True, text=True)
            version_out = (v.stdout or "") + ("\n" + v.stderr if v.stderr else "")
            version = version_out.strip() or f"<exit={v.returncode}>"
        except Exception as e:
            version = f"<error: {type(e).__name__}: {e}>"
        return f"gctb_exe={exe_path} gctb_version={version}"

    def _safe_write_text(self, path: str, text: str) -> None:
        try:
            with open(path, "w", encoding="utf-8", errors="replace") as f:
                f.write(text)
        except Exception:
            pass
        
    def fit(self, bfile_train, pheno_file, out_prefix):
        """
        Run GCTB BayesR on training data.
        
        Args:
            bfile_train: Path prefix to training PLINK files
            pheno_file: Path to phenotype file (FID IID Pheno)
            out_prefix: Output prefix for GCTB results
        """
        cmd = [
            self.gctb_path,
            "--bfile", bfile_train,
            "--pheno", pheno_file,
            "--bayes", "R",
            "--chain-length", "10000",
            "--burn-in", "2000",
            "--out", out_prefix
        ]
        
        print(f"Running BayesR: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            self._safe_write_text(f"{out_prefix}.gctb.stdout", result.stdout or "")
            self._safe_write_text(f"{out_prefix}.gctb.stderr", result.stderr or "")
            raise RuntimeError(f"GCTB BayesR failed:\n{result.stderr}\n\nSTDOUT:\n{result.stdout}")
            
        out_file = f"{out_prefix}.snpRes"
        if not os.path.exists(out_file):
             raise RuntimeError(f"GCTB output file not found: {out_file}")

        # Check if file has content and expected header
        try:
            with open(out_file, 'r') as f:
                header = f.readline()
                if not header.strip():
                     raise RuntimeError(f"GCTB output file {out_file} is empty.")
                cols = header.strip().split()
                cols_l = {c.lower() for c in cols}
                # GCTB versions/flags differ; accept a few common effect-size column names.
                # GCTB BayesR typically uses A1Effect for the posterior mean effect of allele A1.
                effect_candidates = ["a1effect", "a1_effect", "effect", "beta", "b", "mean", "bhat"]
                if not any(c in cols_l for c in effect_candidates):
                     try:
                         size = os.path.getsize(out_file)
                     except Exception:
                         size = None
                     preview_lines = []
                     try:
                         preview_lines.append(header.rstrip("\n"))
                         preview_lines.append(f.readline().rstrip("\n"))
                     except Exception:
                         pass
                     raise RuntimeError(
                         f"GCTB output file {out_file} missing an effect column. "
                         f"Tried={effect_candidates}. "
                         f"DetectedColumns={cols}. "
                         f"FileSizeBytes={size}. "
                         f"HeaderPreview={preview_lines}. "
                         f"{self._gctb_diagnostics()}"
                     )
        except Exception as e:
             self._safe_write_text(f"{out_prefix}.gctb.stdout", result.stdout or "")
             self._safe_write_text(f"{out_prefix}.gctb.stderr", result.stderr or "")
             try:
                 with open(out_file, "r", encoding="utf-8", errors="replace") as f:
                     self._safe_write_text(f"{out_prefix}.snpRes.head", "".join([next(f, "") for _ in range(25)]))
             except Exception:
                 pass
             raise RuntimeError(f"Validation of GCTB output failed: {e}")
             
        print("BayesR training complete.")
        return out_file

    def predict(self, bfile_test, effect_file, out_prefix):
        """
        Score test data using PLINK2 and BayesR effects.
        """
        # BayesR .snpRes format: Id, Name, Chrom, Position, A1, A2, Effect...
        # PLINK2 --score needs: ID, Allele, Effect
        
        # Read effect file
        df = pd.read_csv(effect_file, sep=r'\s+')
        # Columns: Id Name Chrom Position A1 A2 PPIP PIP_0.001 ... Effect

        cols_lower = {c.lower(): c for c in df.columns}
        name_col = cols_lower.get('name') or cols_lower.get('snp') or cols_lower.get('id')
        a1_col = cols_lower.get('a1') or cols_lower.get('allele1')
        effect_col = (
            cols_lower.get('a1effect')
            or cols_lower.get('a1_effect')
            or cols_lower.get('effect')
            or cols_lower.get('beta')
            or cols_lower.get('b')
            or cols_lower.get('mean')
            or cols_lower.get('bhat')
        )

        if name_col is None or a1_col is None or effect_col is None:
            head_preview = df.head(10).to_string(index=False)
            raise RuntimeError(
                "Could not identify required columns in BayesR .snpRes. "
                f"Needed SNP/Name + A1 + Effect. Found columns={list(df.columns)}. "
                f"Head:\n{head_preview}"
            )

        print(
            "BayesR scoring columns: "
            f"snp_id={name_col} allele={a1_col} effect={effect_col}"
        )
        
        score_file = f"{out_prefix}.score"
        df[[name_col, a1_col, effect_col]].to_csv(score_file, sep='\t', index=False, header=False)
        
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
            
        # Read output scores (.sscore)
        # FID IID NMISS_ALLELE_CT NAMED_ALLELE_DOSAGE_SUM SCORE1_AVG
        score_path = f"{out_prefix}.sscore"
        results = pd.read_csv(score_path, sep='\t')
        
        # Return FID, IID, SCORE
        return results[['#IID', 'SCORE1_AVG']].rename(columns={'#IID': 'IID', 'SCORE1_AVG': 'PRS'})
