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
        "plink2", # Default
        "--vcf", vcf_path,
        "--make-bed",
        "--out", out_prefix,
        "--silent"
    ]
    
    # Debug detection
    plink_in_path = shutil.which("plink2")
    print(f"[DEBUG] shutil.which('plink2') -> {plink_in_path}")
    
    # If not in regular PATH, check common locations
    if plink_in_path:
        cmd[0] = plink_in_path # Use full path found
    elif Path("/usr/local/bin/plink2").exists():
        print(f"[DEBUG] Found at /usr/local/bin/plink2")
        cmd[0] = "/usr/local/bin/plink2"
    elif Path("./plink2").exists():
        print(f"[DEBUG] Found at ./plink2")
        cmd[0] = str(Path("./plink2").resolve())
    else:
        print("[WARNING] plink2 not found in PATH or standard locations. Keeping 'plink2' and hoping for the best.")

    print(f"Running PLINK conversion: '{' '.join(cmd)}'")
    
    # Use shell=False (default), explicitly passing executable if needed? 
    # Usually list args is fine.
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError as e:
         # One last try with shell if strictly needed, but unlikely for plink
         raise RuntimeError(f"Could not execute plink2 command: {e}. PATH={os.environ.get('PATH')}")
    
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
