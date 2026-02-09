"""
Helper functions for managing PLINK and external tools integration.
"""
import subprocess
import os
import shutil
import tempfile
from pathlib import Path
import re

def run_plink_conversion(vcf_path: str, out_prefix: str, cm_map_path: str = None) -> None:
    """
    Convert VCF to PLINK binary format (.bed/.bim/.fam).
    Uses plink2.
    
    If cm_map_path is provided (format: BP cM per line), updates the .bim file
    with correct genetic positions.
    """
    # Resolve PLINK executable: use PATH or fallback to CI location
    plink_exe = shutil.which("plink2") or "/usr/local/bin/plink2"
    
    def _run_plink(vcf_input: str) -> subprocess.CompletedProcess:
        cmd = [
            plink_exe,
            "--vcf", vcf_input,
            "--max-alleles", "2",
            "--allow-extra-chr",
            "--rm-dup", "exclude-all",
            "--make-bed",
            "--out", out_prefix,
            "--silent"
        ]
        print(f"Running PLINK conversion: {' '.join(cmd)}")
        return subprocess.run(cmd, capture_output=True, text=True)

    # Fast path: convert directly to avoid duplicating huge VCF files on disk.
    result = _run_plink(vcf_path)

    # Fallback: normalize CHROM labels if direct conversion fails.
    if result.returncode != 0:
        print("Direct PLINK conversion failed; retrying with normalized CHROM labels...")

        temp_dir = tempfile.mkdtemp(prefix="plink_vcf_")
        vcf_numeric = str(Path(temp_dir) / f"{Path(out_prefix).name}_numeric.vcf")
        chr_prefix_re = re.compile(r"^(chr)([0-9]+|[XYM]|MT)\b", flags=re.IGNORECASE)
        try:
            with open(vcf_path, "r", encoding="utf-8") as fin, open(vcf_numeric, "w", encoding="utf-8") as fout:
                for line in fin:
                    if line.startswith("#"):
                        fout.write(line)
                        continue
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) < 2:
                        fout.write(line)
                        continue
                    chrom = parts[0]
                    m = chr_prefix_re.match(chrom)
                    if m:
                        chrom = chrom[len(m.group(1)) :]
                    parts[0] = chrom
                    fout.write("\t".join(parts) + "\n")
            result = _run_plink(vcf_numeric)
        except Exception as e:
            raise RuntimeError(f"VCF preprocessing fallback failed: {e}")
        finally:
            if os.path.exists(vcf_numeric):
                os.remove(vcf_numeric)
            if os.path.isdir(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)

    if result.returncode != 0:
        raise RuntimeError(f"PLINK conversion failed:\n{result.stderr}")
        
    print(f"Created PLINK files: {out_prefix}.bed/bim/fam")
    
    # Inject Genetic Map if provided
    if cm_map_path and os.path.exists(cm_map_path):
        print(f"Injecting genetic map from {cm_map_path} into {out_prefix}.bim ...")
        
        # Load map: POS(int) -> cM(float)
        pos_to_cm = {}
        with open(cm_map_path, "r") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    pos_to_cm[int(parts[0])] = parts[1] # Keep as string to preserve formatting if needed
        
        bim_path = f"{out_prefix}.bim"
        bim_tmp = f"{out_prefix}.bim.tmp"
        
        updated_count = 0
        try:
            with open(bim_path, "r") as fin, open(bim_tmp, "w") as fout:
                for line in fin:
                    # BIM format: CHR SNP CM BP A1 A2
                    cols = line.strip().split()
                    if len(cols) < 6:
                        fout.write(line)
                        continue
                        
                    bp = int(cols[3])
                    if bp in pos_to_cm:
                        cols[2] = pos_to_cm[bp]
                        updated_count += 1
                    
                    fout.write("\t".join(cols) + "\n")
            
            shutil.move(bim_tmp, bim_path)
            print(f"Updated {updated_count} variants with genetic positions.")

        except Exception as e:
            if os.path.exists(bim_tmp):
                os.remove(bim_tmp)
            raise RuntimeError(
                f"REQUIRED: Failed to update .bim file with genetic map: {e}. "
                f"Genetic positions are critical for LD-aware methods."
            )

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
