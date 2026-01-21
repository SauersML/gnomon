"""
Wrapper for PRS-CSx.
"""
import subprocess
import os
from pathlib import Path
import pandas as pd
import numpy as np

class PRScsx:
    def __init__(self, prscsx_path=None):
        if prscsx_path:
            self.prscsx_path = prscsx_path
        else:
            self.prscsx_path = os.environ.get("PRSCSX_PATH", "PRScsx")
            
    def _run_gwas(self, bfile, pheno_file, out_prefix):
        """
        Run GWAS using PLINK2 to get summary stats.
        """
        # Verify phenotype has variation
        import pandas as pd
        pheno_df = pd.read_csv(pheno_file, sep=r'\s+', header=None, names=['FID', 'IID', 'PHENO'])
        n_cases = (pheno_df['PHENO'] == 1).sum()
        n_controls = (pheno_df['PHENO'] == 0).sum()
        
        if n_cases == 0 or n_controls == 0:
            raise RuntimeError(f"Phenotype has no variation: {n_cases} cases, {n_controls} controls")
        
        print(f"GWAS phenotype: {n_cases} cases, {n_controls} controls")
        
        cmd = [
            "plink2",
            "--bfile", bfile,
            "--pheno", pheno_file,
            "--1",  # Phenotype is case/control coded as 0/1
            "--glm", "allow-no-covars", "hide-covar",
            "--out", out_prefix
        ]
        
        print(f"Running GWAS: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"PLINK GWAS failed:\n{result.stderr}")
            
        # Check for generated file
        # Binary trait PLINK2 output usually ends in .glm.logistic.hybrid
        out_files = list(Path(".").glob(f"{out_prefix}*.glm.logistic.hybrid"))
        if not out_files:
            # Try linear if it fell back or different extension
            out_files = list(Path(".").glob(f"{out_prefix}*.glm.*"))
            
        if not out_files:
             raise RuntimeError(f"No GWAS output found for {out_prefix}")
             
        gwas_file = out_files[0]
        
        # Read and format for PRS-CSx
        try:
            df = pd.read_csv(gwas_file, sep='\t')
        except Exception:
            # PLINK sometimes produces messy files if errors occurred but return code 0
            raise RuntimeError(f"Could not read GWAS output: {gwas_file}")
            
        # PLINK2 cols: #CHROM POS ID REF ALT A1 TEST OBS_CT OR SE T_STAT P
        # PRS-CSx needs: SNP A1 A2 BETA P
        
        # Ensure we have required columns
        required = ['ID', 'A1', 'P']
        if not all(col in df.columns for col in required):
             # Try mapping standard PLINK 1.9 names if PLINK 2 failed?
             # But we used plink2.
             raise RuntimeError(f"GWAS output missing columns. Found: {df.columns}")

        # BETA/OR handling
        if 'OR' in df.columns:
            beta = np.log(df['OR'])
        elif 'BETA' in df.columns:
            beta = df['BETA']
        else:
            raise RuntimeError("No BETA or OR column in GWAS output")
            
        # A2 (Reference allele)
        # PLINK2 has REF and ALT.
        # A1 is usually the ALT allele (effect allele) in PLINK2 unless specified otherwise.
        # Check column 'A1'.
        # If A1 is ALT, then A2 is REF.
        # We'll assume A1 is the effect allele provided by PLINK.
        # We need the OTHER allele for A2.
        # In PLINK2, REF/ALT are strictly defined. A1 is the test allele.
        # If A1 == ALT, A2 = REF. If A1 == REF, A2 = ALT.
        
        a2 = np.where(df['A1'] == df['ALT'], df['REF'], df['ALT'])
        
        out_df = pd.DataFrame({
            'SNP': df['ID'],
            'A1': df['A1'],
            'A2': a2,
            'BETA': beta,
            'P': df['P']
        })
        
        sst_path = f"{out_prefix}.sumstats"
        out_df.to_csv(sst_path, sep='\t', index=False)
        return sst_path

    def fit(self, bfile_train, pheno_file, out_prefix, ref_dir, pop="EUR", chrom="22"):
        """
        Run PRS-CSx.
        """
        # 1. Run GWAS to get summary stats
        sst_file = self._run_gwas(bfile_train, pheno_file, out_prefix)

        def _head(path: str, n: int = 5) -> str:
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    lines = []
                    for _ in range(n):
                        line = f.readline()
                        if not line:
                            break
                        lines.append(line.rstrip("\n"))
                return "\n".join(lines)
            except Exception as e:
                return f"<error reading {path}: {e}>"

        def _bim_chrom_counts(bim_path: str) -> str:
            try:
                chrom_counts = {}
                with open(bim_path, "r", encoding="utf-8", errors="replace") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        chrom = line.split()[0]
                        chrom_counts[chrom] = chrom_counts.get(chrom, 0) + 1
                items = sorted(chrom_counts.items(), key=lambda kv: (-kv[1], kv[0]))
                return "\n".join([f"{c}\t{n}" for c, n in items[:20]])
            except Exception as e:
                return f"<error counting chromosomes in {bim_path}: {e}>"

        bim_path = f"{bfile_train}.bim"
        fam_path = f"{bfile_train}.fam"

        diag_lines = []
        diag_lines.append(f"PRS-CSx diagnostics:")
        diag_lines.append(f"  bfile_train={bfile_train}")
        diag_lines.append(f"  bim_path={bim_path} exists={Path(bim_path).exists()}")
        diag_lines.append(f"  fam_path={fam_path} exists={Path(fam_path).exists()}")
        diag_lines.append(f"  sst_file={sst_file} exists={Path(sst_file).exists()}")
        diag_lines.append(f"  ref_dir={ref_dir} exists={Path(ref_dir).exists()}")
        try:
            diag_lines.append("  ref_dir listing (top 50):")
            diag_lines.extend([f"    {p}" for p in sorted([x.name for x in Path(ref_dir).glob('*')])[:50]])
        except Exception as e:
            diag_lines.append(f"  ref_dir listing error: {e}")

        diag_lines.append("  bim head:")
        diag_lines.append(_head(bim_path, n=5))
        diag_lines.append("  bim chromosome counts (top 20):")
        diag_lines.append(_bim_chrom_counts(bim_path))
        diag_lines.append("  fam head:")
        diag_lines.append(_head(fam_path, n=5))
        diag_lines.append("  sumstats head:")
        diag_lines.append(_head(sst_file, n=5))

        diag = "\n".join(diag_lines)
        print(diag)
        
        # 2. Run PRS-CSx script
        # python PRScsx.py --ref_dir=... --bim_prefix=... --sst_file=... --n_gwas=... --pop=... --out_name=...
        
        # Need sample size N. Get from .fam line count?
        with open(f"{bfile_train}.fam") as f:
            n_gwas = sum(1 for _ in f)
            
        script = os.path.join(self.prscsx_path, "PRScsx.py")
        
        cmd = [
            "python", script,
            f"--ref_dir={ref_dir}",
            f"--bim_prefix={bfile_train}",
            f"--sst_file={sst_file}",
            f"--n_gwas={n_gwas}",
            f"--pop={pop}",
            f"--chrom={chrom}",
            f"--out_dir=.",
            f"--out_name={out_prefix}"
        ]
        
        print(f"Running PRS-CSx: {' '.join(cmd)}")
        # We assume ref_dir exists. If not, the tool or python will raise an error.
        # Strict fail-fast: do not check and skip.
            
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(
                f"PRS-CSx failed:\n{result.stderr}\n\nSTDOUT:\n{result.stdout}\n\n{diag}\n"
            )
            
        # Output is likely {out_prefix}_pst_eff_a1_b0.5_phiauto_chr22.txt
        # We need to find the output file.
        # PRS-CSx output naming is complex.
        # We'll look for the posterior effect file.
        
        return f"{out_prefix}_pst_eff_a1_b0.5_phiauto_chr{chrom}.txt"

    def predict(self, bfile_test, effect_file, out_prefix):
        """
        Score using PLINK2.
        """
        # PRS-CSx output columns: SNP A1 A2 BETA
        # PLINK2 --score needs: ID, Allele, Effect
        
        # Read effect file
        # It's usually tab separated, no header? Or has header?
        # Standard PRS-CS output has no header: SNP A1 A2 BETA
        # Let's check first line.
        
        try:
            df = pd.read_csv(effect_file, sep='\t', header=None)
            if isinstance(df.iloc[0, 3], str): # check if header exists
                 df = pd.read_csv(effect_file, sep='\t')
        except Exception:
             # If empty or fail
             raise RuntimeError(f"Could not read PRS-CSx output: {effect_file}")

        # Assuming NO header based on typical output, cols 0, 1, 3 are SNP, A1, BETA
        # PLINK2 --score expects ID A1 BETA
        
        score_file = f"{out_prefix}.csx_score"
        # Write ID A1 BETA
        df[[0, 1, 3]].to_csv(score_file, sep='\t', index=False, header=False)
        
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
