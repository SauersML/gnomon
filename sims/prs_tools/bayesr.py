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
        self.max_snps = 300_000
        self.thin_seed = 42

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

    def _count_bim_snps(self, bfile_prefix: str) -> int:
        bim_path = f"{bfile_prefix}.bim"
        try:
            with open(bim_path, "r", encoding="utf-8", errors="replace") as f:
                return sum(1 for _ in f)
        except Exception as e:
            raise RuntimeError(f"Failed to count SNPs in {bim_path}: {e}")

    def _resolve_plink2(self) -> str:
        plink_exe = shutil.which("plink2") or "/usr/local/bin/plink2"
        if not os.path.exists(plink_exe) and shutil.which("plink2") is None:
            raise FileNotFoundError(
                f"plink2 executable not found on PATH and fallback missing at {plink_exe}"
            )
        return plink_exe

    def _maybe_thin_bfile(self, bfile_prefix: str, out_prefix: str) -> str:
        n_snps = self._count_bim_snps(bfile_prefix)
        if n_snps <= self.max_snps:
            print(f"BayesR SNPs={n_snps} (<= {self.max_snps}); no thinning needed.")
            return bfile_prefix

        thin_prefix = f"{out_prefix}_thin"
        if Path(f"{thin_prefix}.bed").exists():
            print(f"BayesR SNPs={n_snps}; using existing thinned bfile={thin_prefix}")
            return thin_prefix

        print(
            "BayesR SNPs="
            f"{n_snps} (> {self.max_snps}); thinning to {self.max_snps} for GCTB memory safety."
        )
        plink_exe = self._resolve_plink2()
        cmd = [
            plink_exe,
            "--bfile", bfile_prefix,
            "--thin-count", str(self.max_snps),
            "--seed", str(self.thin_seed),
            "--make-bed",
            "--out", thin_prefix,
        ]
        print(f"Running PLINK2 thinning: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            self._safe_write_text(f"{thin_prefix}.plink2.stdout", result.stdout or "")
            self._safe_write_text(f"{thin_prefix}.plink2.stderr", result.stderr or "")
            raise RuntimeError(
                "PLINK2 thinning failed:\n"
                f"returncode={result.returncode}\n"
                f"STDERR:\n{result.stderr}\n\n"
                f"STDOUT:\n{result.stdout}\n"
            )
        return thin_prefix
        
    def fit(self, bfile_train, pheno_file, out_prefix, covar_file):
        """
        Run GCTB BayesR on training data.

        Args:
            bfile_train: Path prefix to training PLINK files
            pheno_file: Path to phenotype file (FID IID Pheno)
            out_prefix: Output prefix for GCTB results
            covar_file: REQUIRED path to covariate file (FID IID PC1 PC2 ...)
        """
        if covar_file is None:
            raise ValueError("BayesR requires covar_file (cannot be None). Pass path to .covar file with PCs.")
        if not os.path.exists(covar_file):
            raise FileNotFoundError(f"BayesR covariate file does not exist: {covar_file}")

        bfile_train_eff = self._maybe_thin_bfile(bfile_train, out_prefix)

        cmd = [
            self.gctb_path,
            "--bfile", bfile_train_eff,
            "--pheno", pheno_file,
            "--bayes", "R",
            "--covar", covar_file,
            "--chain-length", "10000",
            "--burn-in", "2000",
            "--thread", "4",
            "--out", out_prefix
        ]

        print(f"BayesR using covariates from: {covar_file}")
        
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

    def predict(self, bfile_test, effect_file, out_prefix, freq_file=None):
        """
        Score test data using PLINK2 and BayesR effects.
        """
        # BayesR .snpRes format: Id, Name, Chrom, Position, A1, A2, Effect...
        # PLINK2 --score needs: ID, Allele, Effect
        
        # Read effect file
        print(f"BayesR loading effects from: {effect_file}")
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

        try:
            total_rows = int(df.shape[0])
        except Exception:
            total_rows = None
        if total_rows is not None:
            missing_name = int(df[name_col].isna().sum()) if name_col in df.columns else None
            missing_a1 = int(df[a1_col].isna().sum()) if a1_col in df.columns else None
            missing_eff = int(df[effect_col].isna().sum()) if effect_col in df.columns else None
            print(
                "BayesR effects diagnostics: "
                f"rows={total_rows} missing_name={missing_name} missing_a1={missing_a1} missing_effect={missing_eff}"
            )
        
        score_file = f"{out_prefix}.score"

        score_df = df[[name_col, a1_col, effect_col]].copy()
        before_rows = int(score_df.shape[0])
        score_df = score_df.dropna(subset=[name_col, a1_col, effect_col])
        score_df[name_col] = score_df[name_col].astype(str).str.strip()
        score_df[a1_col] = score_df[a1_col].astype(str).str.strip()
        score_df[effect_col] = pd.to_numeric(score_df[effect_col], errors="coerce")
        score_df = score_df.dropna(subset=[effect_col])
        score_df = score_df[(score_df[name_col] != "") & (score_df[a1_col] != "")]
        after_rows = int(score_df.shape[0])
        print(f"BayesR score rows: before_filter={before_rows} after_filter={after_rows}")

        if score_df.empty:
            raise RuntimeError(
                "BayesR produced no valid scoring rows after filtering missing/invalid values. "
                f"Columns used: snp_id={name_col} allele={a1_col} effect={effect_col}"
            )

        score_df.to_csv(score_file, sep='\t', index=False, header=False)

        try:
            print("BayesR score preview (first 5 rows):")
            print(score_df.head(5).to_string(index=False))
        except Exception as e:
            print(f"BayesR score preview unavailable: {type(e).__name__}: {e}")

        try:
            size_bytes = os.path.getsize(score_file)
            print(f"BayesR wrote score file: {score_file} size_bytes={size_bytes}")
        except Exception as e:
            print(f"BayesR could not stat score file {score_file}: {type(e).__name__}: {e}")

        try:
            with open(score_file, "r", encoding="utf-8", errors="replace") as f:
                first_line = f.readline().rstrip("\n")
                second_line = f.readline().rstrip("\n")
                third_line = f.readline().rstrip("\n")
            print(f"BayesR score line1 repr={first_line!r}")
            if second_line:
                print(f"BayesR score line2 repr={second_line!r}")
            if third_line:
                print(f"BayesR score line3 repr={third_line!r}")
            try:
                line1_cat_a = "".join(ch if ch not in ["\t", "\n", "\r"] else {"\t": "^I", "\n": "^J", "\r": "^M"}[ch] for ch in first_line)
                print(f"BayesR score line1 cat-A={line1_cat_a}")
            except Exception:
                pass
            first_fields = first_line.split()
            print(f"BayesR score line1 fields_count={len(first_fields)} fields={first_fields}")
            if len(first_fields) < 3:
                raise RuntimeError(
                    f"Generated score file {score_file} has fewer than 3 whitespace-delimited fields on line 1: "
                    f"fields={first_fields} raw_line={first_line!r}"
                )
        except Exception as e:
            raise RuntimeError(f"Failed validating generated score file {score_file}: {e}")
        
        cmd = [
            "plink2",
            "--bfile", bfile_test,
            "--score", score_file, "1", "2", "3",
            "--out", out_prefix
        ]
        if freq_file is not None:
            cmd.extend(["--read-freq", str(freq_file)])
        
        print(f"Running Scoring: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            try:
                self._safe_write_text(f"{out_prefix}.plink2.stdout", result.stdout or "")
                self._safe_write_text(f"{out_prefix}.plink2.stderr", result.stderr or "")
            except Exception:
                pass

            try:
                with open(score_file, "r", encoding="utf-8", errors="replace") as f:
                    preview = [next(f, "") for _ in range(10)]
                print("PLINK scoring failure: score file first 10 lines (repr):")
                for i, line in enumerate(preview, start=1):
                    print(f"  L{i}={line.rstrip(chr(10))!r}")
                print("PLINK scoring failure: score file whitespace field counts (first 10 lines):")
                for i, line in enumerate(preview, start=1):
                    if not line:
                        continue
                    fields = line.split()
                    print(f"  L{i} NF={len(fields)} fields={fields}")
            except Exception as e:
                print(f"PLINK scoring failure: could not preview score file {score_file}: {type(e).__name__}: {e}")

            raise RuntimeError(
                "PLINK scoring failed:\n"
                f"returncode={result.returncode}\n"
                f"STDERR:\n{result.stderr}\n\n"
                f"STDOUT:\n{result.stdout}\n\n"
                f"ScoreFile={score_file}\n"
                f"PLINK2StdoutFile={out_prefix}.plink2.stdout\n"
                f"PLINK2StderrFile={out_prefix}.plink2.stderr"
            )
            
        # Read output scores (.sscore)
        score_path = f"{out_prefix}.sscore"
        results = pd.read_csv(score_path, sep='\t')
        
        # Return FID, IID, SCORE
        id_col = '#IID' if '#IID' in results.columns else 'IID'
        if id_col not in results.columns or 'SCORE1_AVG' not in results.columns:
            raise RuntimeError(
                "PLINK2 .sscore missing expected columns. "
                f"Path={score_path} Columns={list(results.columns)}"
            )
        return results[[id_col, 'SCORE1_AVG']].rename(columns={id_col: 'IID', 'SCORE1_AVG': 'PRS'})
