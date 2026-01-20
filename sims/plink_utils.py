"""
Helper functions for managing PLINK and external tools integration.
"""
import subprocess
import os
import shutil
from pathlib import Path

def run_plink_conversion(vcf_path: str, out_prefix: str) -> None:
    """
    Convert VCF to PLINK binary format (.bed/.bim/.fam).
    Uses plink2.
    """
    cmd = [
        "plink2",
        "--vcf", vcf_path,
        "--make-bed",
        "--out", out_prefix,
        "--silent"
    ]
    
    # Check if plink2 is in PATH, otherwise try likely locations
    if shutil.which("plink2") is None:
        # Fallback for GitHub Actions or standard Linux installs
        if Path("/usr/local/bin/plink2").exists():
            cmd[0] = "/usr/local/bin/plink2"
        else:
             print("WARNING: plink2 not found in PATH or /usr/local/bin/plink2")

    print(f"Running PLINK conversion: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"PLINK conversion failed:\n{result.stderr}")
        
    print(f"Created PLINK files: {out_prefix}.bed/bim/fam")

def write_phenotype_file(df, out_path: str) -> None:
    """
    Write phenotype file for GCTB/PLINK.
    Format: FID IID pheno
    """
    # Create FID/IID/Pheno dataframe
    # Assuming individual_id is IID, and we use family_id=individual_id (or 0)
    pheno_df = df[['individual_id', 'individual_id', 'y']].copy()
    pheno_df.columns = ['FID', 'IID', 'pheno']
    
    # GCTB often expects no header, space separated
    pheno_df.to_csv(out_path, sep=' ', index=False, header=False)
    print(f"Written phenotype file: {out_path}")
