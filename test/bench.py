import os
import random
import shutil
import subprocess
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import requests

# ==============================================================================
#                              BENCHMARK CONFIGURATION
# ==============================================================================

# --- Workspace and Paths ---
# All generated files and results will be stored here.
WORKDIR = Path("./benchmark_workdir")
# The script will expect the binaries to be in these locations.
GNOMON_BINARY = Path("./target/release/gnomon").resolve()
PLINK2_BINARY = WORKDIR / "plink2"
PLINK2_URL = "https://s3.amazonaws.com/plink2-assets/alpha6/plink2_linux_avx2_20250609.zip"

# --- Benchmark Workloads ---
# The script will run a benchmark for every combination of these dimensions.
DIMENSIONS = [
    {"n_variants": 100_000, "n_individuals": 5_000, "n_scores": 10},
    {"n_variants": 500_000, "n_individuals": 10_000, "n_scores": 50},
    {"n_variants": 1_000_000, "n_individuals": 1_000, "n_scores": 99}, # I/O heavy
    {"n_variants": 500_000, "n_individuals": 5_000, "n_scores": 1}, # Single score
]

# --- Data Realism & Variety Parameters ---
# For each workload, the script will randomly pick from these options to create
# diverse and realistic datasets.

# Allele Frequency (AF) distributions: (distribution_name, param1, param2)
AF_DISTRIBUTIONS = [
    ('beta', 0.2, 0.2),      # U-shaped: mix of rare and common (simulates WGS)
    ('uniform', 0.05, 0.5),  # Mostly common (simulates genotyping arrays)
]

# Effect size (weight) distributions for the score weights.
EFFECT_DISTRIBUTIONS = [
    ('normal', 0, 0.001),   # Simulates a dense polygenic score
    ('laplace', 0, 0.05),   # Simulates a score with sparse, larger "GWAS hits"
]

# The probability that a new score column will be correlated with a previous one.
SCORE_CORRELATION_PROBABILITY = 0.3

# ==============================================================================
#                               HELPER FUNCTIONS
# ==============================================================================

def print_header(title: str, char: str = "="):
    """Prints a formatted header to the console."""
    width = 80
    print("\n" + char * width, flush=True)
    print(f"{char*4} {title} {char*(width - len(title) - 6)}", flush=True)
    print(char * width, flush=True)

def setup_environment():
    """Prepares the CI environment by creating the workspace and downloading plink2."""
    print_header("Benchmark Setup", "-")
    WORKDIR.mkdir(exist_ok=True)
    
    if not GNOMON_BINARY.exists():
        print(f"❌ ERROR: Gnomon binary not found at '{GNOMON_BINARY}'. Please build it first with 'cargo build --release'.")
        return False

    if not PLINK2_BINARY.exists():
        print(f"  > PLINK2 not found. Downloading from {PLINK2_URL}...")
        zip_path = WORKDIR / "plink.zip"
        try:
            with requests.get(PLINK2_URL, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(zip_path, 'wb') as f:
                    shutil.copyfileobj(r.raw, f)
            with zipfile.ZipFile(zip_path, 'r') as z:
                # Find the binary inside the zip and extract it to the correct path
                for member in z.infolist():
                    if member.filename.endswith('plink2') and not member.is_dir():
                        with z.open(member) as source, open(PLINK2_BINARY, 'wb') as target:
                            shutil.copyfileobj(source, target)
                        break
            PLINK2_BINARY.chmod(0o755) # Make it executable
            zip_path.unlink()
            print(f"  > Successfully downloaded and configured PLINK2 at '{PLINK2_BINARY}'.")
        except Exception as e:
            print(f"❌ ERROR: Failed to download or extract PLINK2: {e}")
            return False
    else:
        print(f"  > Found existing PLINK2 binary at '{PLINK2_BINARY}'.")
    
    return True

# ==============================================================================
#                         REALISTIC DATA GENERATOR
# ==============================================================================

class RealisticDataGenerator:
    """Generates realistic, large-scale genomic datasets for benchmarking."""
    def __init__(self, n_variants: int, n_individuals: int, workdir: Path):
        self.n_variants = n_variants
        self.n_individuals = n_individuals
        self.workdir = workdir
        self.run_prefix = workdir / f"data_{n_variants}v_{n_individuals}i"
        self.variants_df = None
        self.score_file_path = None
        
    def _generate_bim(self):
        """Generates the .bim file with realistic allele frequencies."""
        print(f"    > Generating .bim file with {self.n_variants} variants...")
        positions = np.random.choice(np.arange(1, 50_000_000), self.n_variants, replace=False)
        positions.sort()
        
        alleles = np.array(['A', 'C', 'G', 'T'])
        ref_alleles = np.random.choice(alleles, self.n_variants)
        alt_alleles = np.array([np.random.choice(np.setdiff1d(alleles, [r])) for r in ref_alleles])

        dist_name, p1, p2 = random.choice(AF_DISTRIBUTIONS)
        if dist_name == 'beta':
            af = np.random.beta(p1, p2, self.n_variants)
        else: # uniform
            af = np.random.uniform(p1, p2, self.n_variants)

        self.variants_df = pd.DataFrame({
            'chr': '1',
            'id': [f"1:{pos}" for pos in positions],
            'cm': 0,
            'pos': positions,
            'a1': ref_alleles,
            'a2': alt_alleles,
            'af': af
        })
        self.variants_df.to_csv(self.run_prefix.with_suffix(".bim"), sep='\t', header=False, index=False)

    def _generate_fam(self):
        """Generates a simple .fam file."""
        print(f"    > Generating .fam file for {self.n_individuals} individuals...")
        with open(self.run_prefix.with_suffix(".fam"), 'w') as f:
            for i in range(self.n_individuals):
                f.write(f"sample_{i} sample_{i} 0 0 0 -9\n")

    def _generate_bed(self):
        """Generates a packed binary .bed file based on HWE."""
        print("    > Generating .bed file...")
        p = self.variants_df['af'].values
        hwe_probs = np.vstack([ (1-p)**2, 2*p*(1-p), p**2 ]).T # P(hom_ref), P(het), P(hom_alt)
        
        rand_draws = np.random.rand(self.n_variants, self.n_individuals)
        cum_probs = hwe_probs.cumsum(axis=1)
        
        # 0 for hom_ref, 1 for het, 2 for hom_alt
        genotypes = (rand_draws > cum_probs[:, [0]]) + (rand_draws > cum_probs[:, [1]])
        
        # Introduce a tiny amount of missingness (1%)
        missing_mask = np.random.rand(*genotypes.shape) < 0.01
        genotypes[missing_mask] = -1 # Use -1 as a placeholder for missing
        
        # Map to PLINK's 2-bit codes: 00 (hom_alt), 01 (missing), 10 (het), 11 (hom_ref)
        # Note: PLINK's A1/A2 are often minor/major, but here we treat A1=REF, A2=ALT
        # Genotype codes are counts of the A2 (ALT) allele.
        # So, 0 -> hom_ref -> 11; 1 -> het -> 10; 2 -> hom_alt -> 00; -1 -> missing -> 01
        code_map = {0: 0b11, 1: 0b10, 2: 0b00, -1: 0b01}

        with open(self.run_prefix.with_suffix(".bed"), 'wb') as f:
            f.write(bytes([0x6c, 0x1b, 0x01])) # PLINK magic number
            for i in range(self.n_variants):
                byte = 0
                for j in range(0, self.n_individuals, 4):
                    chunk = genotypes[i, j:j+4]
                    byte = 0
                    # Pack 4 genotypes into a single byte
                    for k, geno in enumerate(chunk):
                        byte |= (code_map[int(geno)] << (k * 2))
                    f.write(byte.to_bytes(1, 'little'))
                # Handle padding for individuals not divisible by 4
                if self.n_individuals % 4 != 0:
                    remaining_genos = genotypes[i, (self.n_individuals // 4) * 4:]
                    byte = 0
                    for k, geno in enumerate(remaining_genos):
                         byte |= (code_map[int(geno)] << (k * 2))
                    f.write(byte.to_bytes(1, 'little'))

    def _generate_score_file(self, n_scores):
        """Generates a complex, realistic, gnomon-native score file."""
        print(f"    > Generating .score file with {n_scores} scores...")
        score_data = {"snp_id": self.variants_df['id']}
        score_cols_data = []

        for i in range(n_scores):
            # Decide if this score will be correlated with a previous one
            is_correlated = (i > 0) and (random.random() < SCORE_CORRELATION_PROBABILITY)
            
            current_weights = np.zeros(self.n_variants)
            
            if is_correlated:
                # Pick a random previous score to use as a base
                base_score_col = random.choice(score_cols_data)
                current_weights = base_score_col.copy()
                # Perturb 15% of the non-zero weights
                non_zero_indices = np.where(current_weights != 0)[0]
                n_to_perturb = int(0.15 * len(non_zero_indices))
                if n_to_perturb > 0:
                    perturb_indices = np.random.choice(non_zero_indices, n_to_perturb, replace=False)
                    dist_name, p1, p2 = random.choice(EFFECT_DISTRIBUTIONS)
                    new_weights = np.random.normal(p1, p2, n_to_perturb) if dist_name == 'normal' else np.random.laplace(p1, p2, n_to_perturb)
                    current_weights[perturb_indices] = new_weights
            else: # Independent score
                dist_name, p1, p2 = random.choice(EFFECT_DISTRIBUTIONS)
                sparsity = random.uniform(0.01, 0.8)
                n_non_zero = int(sparsity * self.n_variants)
                
                if n_non_zero > 0:
                    indices = np.random.choice(self.n_variants, n_non_zero, replace=False)
                    weights = np.random.normal(p1, p2, n_non_zero) if dist_name == 'normal' else np.random.laplace(p1, p2, n_non_zero)
                    current_weights[indices] = weights
            
            score_data[f"score_{i+1}"] = current_weights
            score_cols_data.append(current_weights)

        score_df = pd.DataFrame(score_data)
        score_df['effect_allele'] = self.variants_df['a2']
        score_df['other_allele'] = self.variants_df['a1']
        
        # Reorder columns to gnomon-native format: snp_id, effect_allele, other_allele, scores...
        score_name_cols = [f"score_{i+1}" for i in range(n_scores)]
        final_cols = ['snp_id', 'effect_allele', 'other_allele'] + score_name_cols
        score_df = score_df[final_cols]
        
        self.score_file_path = self.run_prefix.with_suffix(".score")
        score_df.to_csv(self.score_file_path, sep='\t', index=False, float_format='%.6g')

    def generate_all_files(self, n_scores):
        """Generate all necessary files for a benchmark run."""
        self._generate_bim()
        self._generate_fam()
        self._generate_bed()
        self._generate_score_file(n_scores)
        return self.run_prefix

    def cleanup(self):
        """Remove the large generated files."""
        print("    > Cleaning up generated data files...")
        for ext in [".bed", ".bim", ".fam", ".score"]:
            try:
                self.run_prefix.with_suffix(ext).unlink()
            except FileNotFoundError:
                pass

# ==============================================================================
#                       EXECUTION & MONITORING ENGINE
# ==============================================================================

def run_and_monitor_process(tool_name: str, command: list, log_prefix: str, cwd: Path) -> dict:
    """Executes a command using its default parallelism and monitors its performance."""
    print(f"  > Running {tool_name} with default max parallelism...")
    command_str_list = [str(c) for c in command]
    print(f"    Command: {' '.join(command_str_list)}")

    start_time = time.monotonic()
    log_file_path = WORKDIR / f"{log_prefix}.log"
    
    try:
        with open(log_file_path, 'w') as log_file:
            process = subprocess.Popen(
                command_str_list,
                cwd=cwd,
                stdout=log_file,
                stderr=subprocess.STDOUT
            )
        
        p = psutil.Process(process.pid)
        peak_rss_mb = 0
        while process.poll() is None:
            try:
                mem_info = p.memory_info()
                peak_rss_mb = max(peak_rss_mb, mem_info.rss / 1024 / 1024)
            except psutil.NoSuchProcess:
                break
            time.sleep(0.02)
        
        wall_time = time.monotonic() - start_time
        returncode = process.returncode
        
        if returncode != 0:
            print(f"  > ❌ {tool_name} FAILED with exit code {returncode}. Log: {log_file_path}")
        else:
            print(f"  > ✅ {tool_name} finished in {wall_time:.2f}s. Peak Memory: {peak_rss_mb:.2f} MB")

    except (FileNotFoundError, psutil.NoSuchProcess) as e:
        print(f"  > ❌ ERROR launching {tool_name}: {e}")
        wall_time, peak_rss_mb, returncode = -1, -1, -1
        
    return {
        "tool": tool_name,
        "time_sec": wall_time,
        "peak_mem_mb": peak_rss_mb,
        "success": returncode == 0,
        "returncode": returncode,
    }

# ==============================================================================
#                             MAIN ORCHESTRATION
# ==============================================================================

def validate_results(gnomon_output_path: Path, plink2_output_path: Path) -> bool:
    """Compares the output files from gnomon and plink2 for concordance."""
    try:
        g_df = pd.read_csv(gnomon_output_path, sep='\t')
        p_df = pd.read_csv(plink2_output_path, sep=r'\s+')
        
        if len(g_df) != len(p_df):
            print(f"  > ❌ VALIDATION FAILED: Output files have different numbers of rows ({len(g_df)} vs {len(p_df)}).")
            return False

        # Find the score columns
        g_score_cols = [c for c in g_df.columns if c.endswith('_AVG')]
        p_score_cols = [c for c in p_df.columns if c.endswith('_AVG')]

        if len(g_score_cols) != len(p_score_cols):
             print(f"  > ❌ VALIDATION FAILED: Different number of score columns found.")
             return False

        # Merge and compare
        # PLINK2 has #FID, gnomon has #IID
        g_df.rename(columns={'#IID': 'IID'}, inplace=True)
        p_df.rename(columns={'#FID': 'FID'}, inplace=True)
        
        merged_df = pd.merge(g_df, p_df, on="IID", suffixes=('_g', '_p'))
        
        for g_col, p_col in zip(sorted(g_score_cols), sorted(p_score_cols)):
            # Gnomon reports per-variant average, plink2 reports sum/2. We need to normalize plink2's output.
            # plink2 SCORE_AVG = (SUM(dosage * weight)) / (2 * num_variants_used)
            # gnomon SCORE_AVG = (SUM(dosage * weight)) / (num_variants_used)
            # So, gnomon's average should be double plink2's average.
            is_close = np.allclose(merged_df[g_col], merged_df[p_col] * 2, rtol=1e-5, atol=1e-8)
            if not is_close:
                print(f"  > ❌ VALIDATION FAILED: Scores in columns '{g_col}' and '{p_col}' do not match after normalization.")
                diff = np.abs(merged_df[g_col] - merged_df[p_col] * 2).max()
                print(f"    Max difference: {diff}")
                return False
        
        print("  > ✅ Validation Successful: Output scores are numerically concordant.")
        return True
    except Exception as e:
        print(f"  > ❌ VALIDATION FAILED with an exception: {e}")
        return False

def report_results(results: list):
    """Prints and saves the final benchmark summary."""
    print_header("Benchmark Summary Report")
    if not results:
        print("No results to report.")
        return

    df = pd.DataFrame(results)
    
    # Create a pivot table for easy comparison
    try:
        report_df = df.pivot_table(
            index=['n_variants', 'n_individuals', 'n_scores'],
            columns='tool',
            values=['time_sec', 'peak_mem_mb']
        )
        report_df.columns = [f"{val}_{tool}" for val, tool in report_df.columns]
        
        if 'time_sec_gnomon' in report_df and 'time_sec_plink2' in report_df:
            report_df['time_ratio_g/p'] = report_df['time_sec_gnomon'] / report_df['time_sec_plink2']
        if 'peak_mem_mb_gnomon' in report_df and 'peak_mem_mb_plink2' in report_df:
            report_df['mem_ratio_g/p'] = report_df['peak_mem_mb_gnomon'] / report_df['peak_mem_mb_plink2']

        print(report_df.to_markdown(floatfmt=".3f"))
    except Exception as e:
        print("Could not generate pivot table report. Printing raw results.")
        print(df.to_markdown(index=False, floatfmt=".3f"))

    summary_path = WORKDIR / "benchmark_summary.csv"
    df.to_csv(summary_path, index=False)
    print(f"\nFull results saved to '{summary_path}'")

def main():
    """Main orchestration function."""
    if not setup_environment():
        exit(1)

    all_results = []
    failed_runs = 0
    run_id = 0

    for workload_params in DIMENSIONS:
        run_id += 1
        print_header(f"Benchmark Run {run_id}/{len(DIMENSIONS)}: {workload_params}")

        generator = RealisticDataGenerator(
            n_variants=workload_params["n_variants"],
            n_individuals=workload_params["n_individuals"],
            workdir=WORKDIR
        )
        data_prefix = generator.generate_all_files(n_scores=workload_params["n_scores"])

        # Run Gnomon
        gnomon_cmd = [str(GNOMON_BINARY), "--score", str(generator.score_file_path), str(data_prefix)]
        gnomon_log_prefix = f"run{run_id}_gnomon"
        gnomon_res = run_and_monitor_process("gnomon", gnomon_cmd, gnomon_log_prefix, WORKDIR)
        gnomon_res.update(workload_params)
        all_results.append(gnomon_res)

        # Run PLINK2
        plink_out_prefix = WORKDIR / f"plink2_run{run_id}"
        # Calculate the score column range for plink2
        first_score_col = 4 # snp_id, effect_allele, other_allele are cols 1, 2, 3
        last_score_col = 3 + n_scores
        score_col_range_str = f"{first_score_col}-{last_score_col}"
        
        # Build the corrected PLINK2 command
        plink2_cmd = [
            str(PLINK2_BINARY),
            "--bfile", str(data_prefix),
            "--score", str(score_file), "1", "2", "header", # No column number here
            "--score-col-nums", score_col_range_str, # Use the dynamic range
            "no-mean-imputation",
            "--out", str(plink_out_prefix)
        ]
        
        # Execute the command
        plink2_res = run_and_monitor_process("plink2", plink2_cmd, plink2_log_prefix, WORKDIR)
        plink2_log_prefix = f"run{run_id}_plink2"
        plink2_res.update(workload_params)
        all_results.append(plink2_res)
        
        # Validate results if both runs were successful
        if gnomon_res["success"] and plink2_res["success"]:
            gnomon_output_path = data_prefix.with_suffix(".sscore")
            plink2_output_path = plink_out_prefix.with_suffix(".sscore")
            if not validate_results(gnomon_output_path, plink2_output_path):
                failed_runs += 1
        else:
            failed_runs += 1
        
        generator.cleanup()

    report_results(all_results)
    
    if failed_runs > 0:
        print(f"\n❌ Benchmark finished with {failed_runs} failed or invalid run(s).")
        exit(1)
    else:
        print("\n🎉 All benchmarks completed and passed validation successfully.")
        exit(0)

if __name__ == "__main__":
    main()
