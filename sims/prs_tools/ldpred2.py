"""
Wrapper for LDpred2 (via R script).
"""
import subprocess
import pandas as pd
import os

class LDpred2:
    def __init__(self, r_script_path="sims/prs_tools/run_ldpred2.R"):
        self.r_script_path = r_script_path
        
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
            raise RuntimeError(f"LDpred2 failed:\n{result.stderr}\n\nSTDOUT:\n{result.stdout}")
            
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
        return results[['#IID', 'SCORE1_AVG']].rename(columns={'#IID': 'IID', 'SCORE1_AVG': 'PRS'})
