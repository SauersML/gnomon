"""
Wrapper for GCTB BayesR.
"""
import subprocess
import os
from pathlib import Path
import pandas as pd
import numpy as np

class BayesR:
    def __init__(self, gctb_path="gctb"):
        self.gctb_path = gctb_path
        
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
                if 'Effect' not in header:
                     # GCTB 2.03+ usually has Id Name Chrom Position A1 A2 ... Effect
                     raise RuntimeError(f"GCTB output file {out_file} missing 'Effect' column. Header: {header}")
        except Exception as e:
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
        df = pd.read_csv(effect_file, delim_whitespace=True)
        # Columns: Id Name Chrom Position A1 A2 PPIP PIP_0.001 ... Effect
        
        score_file = f"{out_prefix}.score"
        df[['Name', 'A1', 'Effect']].to_csv(score_file, sep='\t', index=False, header=False)
        
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
