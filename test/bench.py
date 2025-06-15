import os
import random
import re
import shutil
import subprocess
import sys
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
WORKDIR = Path("./benchmark_workdir")
GNOMON_BINARY_REL = Path("./target/release/gnomon")
PLINK2_BINARY_REL = WORKDIR / "plink2"
PLINK2_URL = "https://s3.amazonaws.com/plink2-assets/alpha6/plink2_linux_avx2_20250609.zip"

# --- Benchmark Workloads ---
DIMENSIONS = [
    {"n_variants": 100_000, "n_individuals": 5_000, "n_scores": 10},
    {"n_variants": 500_000, "n_individuals": 10_000, "n_scores": 50},
    {"n_variants": 1_000_000, "n_individuals": 1_000, "n_scores": 99},
    {"n_variants": 500_000, "n_individuals": 5_000, "n_scores": 1},
]

# --- Data Realism & Variety Parameters ---
AF_DISTRIBUTIONS = [('beta', 0.2, 0.2), ('uniform', 0.05, 0.5)]
EFFECT_DISTRIBUTIONS = [('normal', 0, 0.001), ('laplace', 0, 0.05)]
SCORE_CORRELATION_PROBABILITY = 0.3

# --- Performance Tuning ---
GENERATION_CHUNK_SIZE = 50_000

# ==============================================================================
#                               HELPER FUNCTIONS
# ==============================================================================

def print_header(title: str, char: str = "="):
    """Prints a formatted header to the console."""
    width = 80
    print("\n" + char * width, flush=True)
    print(f"{char*4} {title} {char*(width - len(title) - 6)}", flush=True)
    print(char * width, flush=True)

def setup_environment(gnomon_path, plink_path):
    print_header("Benchmark Setup", "-")
    WORKDIR.mkdir(exist_ok=True)
    if not gnomon_path.exists():
        print(f"❌ ERROR: Gnomon binary not found at '{gnomon_path}'. Please build with 'cargo build --release'.")
        return False
    if not plink_path.exists():
        print(f"  > PLINK2 not found. Downloading from {PLINK2_URL}...")
        zip_path = WORKDIR / "plink.zip"
        try:
            with requests.get(PLINK2_URL, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(zip_path, 'wb') as f: shutil.copyfileobj(r.raw, f)
            with zipfile.ZipFile(zip_path, 'r') as z:
                for member in z.infolist():
                    if member.filename.endswith('plink2') and not member.is_dir():
                        with z.open(member) as source, open(plink_path, 'wb') as target:
                            shutil.copyfileobj(source, target)
                        break
            plink_path.chmod(0o755)
            zip_path.unlink()
            print(f"  > Successfully downloaded and configured PLINK2 at '{plink_path}'.")
        except Exception as e:
            print(f"❌ ERROR: Failed to download or extract PLINK2: {e}")
            return False
    else:
        print(f"  > Found existing PLINK2 binary at '{plink_path}'.")
    return True

# ==============================================================================
#                         REALISTIC DATA GENERATOR
# ==============================================================================

class RealisticDataGenerator:
    def __init__(self, n_variants: int, n_individuals: int, workdir: Path):
        self.n_variants, self.n_individuals, self.workdir = n_variants, n_individuals, workdir
        self.run_prefix = workdir / f"data_{n_variants}v_{n_individuals}i"
        self.variants_df, self.score_file_path = None, None
        
    def _generate_bim(self):
        print(f"    > Generating .bim file for {self.n_variants} variants (vectorized)...")
        positions = np.sort(np.random.choice(np.arange(1, 50_000_000), self.n_variants, replace=False))
        base_alleles = np.array(['A', 'C', 'G', 'T'])
        ref_indices = np.random.randint(0, 4, self.n_variants)
        offsets = np.random.randint(1, 4, self.n_variants)
        alt_indices = (ref_indices + offsets) % 4
        dist_name, p1, p2 = random.choice(AF_DISTRIBUTIONS)
        af = np.random.beta(p1, p2, self.n_variants) if dist_name == 'beta' else np.random.uniform(p1, p2, self.n_variants)
        self.variants_df = pd.DataFrame({'chr': '1', 'id': [f"1:{pos}" for pos in positions], 'cm': 0, 'pos': positions,
                                       'a1': base_alleles[ref_indices], 'a2': base_alleles[alt_indices], 'af': af})
        self.variants_df[['chr', 'id', 'cm', 'pos', 'a1', 'a2']].to_csv(self.run_prefix.with_suffix(".bim"), sep='\t', header=False, index=False)

    def _generate_fam(self):
        print(f"    > Generating .fam file for {self.n_individuals} individuals...")
        with open(self.run_prefix.with_suffix(".fam"), 'w') as f:
            for i in range(self.n_individuals):
                f.write(f"sample_{i} sample_{i} 0 0 0 -9\n")

    def _generate_bed(self):
        print(f"    > Generating .bed file (chunked, {GENERATION_CHUNK_SIZE} variants/chunk)...")
        # 0=A1/A1 -> 00; 1=A1/A2 -> 10; 2=A2/A2 -> 11; missing -> 01
        code_map = {0: 0b00, 1: 0b10, 2: 0b11, -1: 0b01}
        mapping_array = np.array([code_map[key] for key in sorted(code_map)], dtype=np.uint8)
        
        with open(self.run_prefix.with_suffix(".bed"), 'wb') as f:
            f.write(bytes([0x6c, 0x1b, 0x01]))
            for start_idx in range(0, self.n_variants, GENERATION_CHUNK_SIZE):
                end_idx = min(start_idx + GENERATION_CHUNK_SIZE, self.n_variants)
                chunk_size = end_idx - start_idx
                p = self.variants_df['af'].values[start_idx:end_idx, np.newaxis]
                hwe_probs = np.hstack([(1-p)**2, 2*p*(1-p), p**2])
                rand_draws = np.random.rand(chunk_size, self.n_individuals)
                genotypes = (rand_draws > hwe_probs.cumsum(axis=1)[:, [0]]) + (rand_draws > hwe_probs.cumsum(axis=1)[:, [1]])
                genotypes[(np.random.rand(*genotypes.shape) < 0.01)] = -1
                codes = mapping_array[genotypes.astype(int)]
                padded_n_individuals = (self.n_individuals + 3) // 4 * 4
                padded_codes = np.full((chunk_size, padded_n_individuals), code_map[-1], dtype=np.uint8)
                padded_codes[:, :self.n_individuals] = codes
                packed_bytes = (padded_codes[:, 0::4]) | (padded_codes[:, 1::4] << 2) | \
                               (padded_codes[:, 2::4] << 4) | (padded_codes[:, 3::4] << 6)
                f.write(packed_bytes.tobytes())
            
    def _generate_score_file(self, n_scores):
        print(f"    > Generating .score file with {n_scores} scores...")
        score_data, score_cols_data = {"snp_id": self.variants_df['id']}, []
        for i in range(n_scores):
            is_correlated = (i > 0) and (random.random() < SCORE_CORRELATION_PROBABILITY)
            current_weights = np.zeros(self.n_variants)
            if is_correlated:
                base_col = random.choice(score_cols_data)
                current_weights = base_col.copy()
                non_zero_indices = np.where(current_weights != 0)[0]
                n_to_perturb = int(0.15 * len(non_zero_indices))
                if n_to_perturb > 0:
                    perturb_indices = np.random.choice(non_zero_indices, n_to_perturb, replace=False)
                    dist_name, p1, p2 = random.choice(EFFECT_DISTRIBUTIONS)
                    new_weights = np.random.normal(p1, p2, n_to_perturb) if dist_name == 'normal' else np.random.laplace(p1, p2, n_to_perturb)
                    current_weights[perturb_indices] = new_weights
            else:
                dist_name, p1, p2 = random.choice(EFFECT_DISTRIBUTIONS)
                sparsity = random.uniform(0.01, 0.8)
                indices = np.random.choice(self.n_variants, int(sparsity * self.n_variants), replace=False)
                if len(indices) > 0:
                    weights = np.random.normal(p1, p2, len(indices)) if dist_name == 'normal' else np.random.laplace(p1, p2, len(indices))
                    current_weights[indices] = weights
            score_data[f"score_{i+1}"] = current_weights
            score_cols_data.append(current_weights)

        score_df = pd.DataFrame(score_data)
        score_df['effect_allele'] = self.variants_df['a2']
        score_df['other_allele'] = self.variants_df['a1']
        score_name_cols = [f"score_{i+1}" for i in range(n_scores)]
        final_cols = ['snp_id', 'effect_allele', 'other_allele'] + score_name_cols
        self.score_file_path = self.run_prefix.with_suffix(".score")
        score_df[final_cols].to_csv(self.score_file_path, sep='\t', index=False, float_format='%.6g')

    def generate_all_files(self, n_scores):
        self._generate_bim(); self._generate_fam(); self._generate_bed(); self._generate_score_file(n_scores)
        return self.run_prefix

    def cleanup(self):
        print("    > Cleaning up generated data files...")
        for ext in [".bed", ".bim", ".fam", ".score"]:
            try: self.run_prefix.with_suffix(ext).unlink()
            except FileNotFoundError: pass

# ==============================================================================
#                       EXECUTION & MONITORING ENGINE
# ==============================================================================

def run_and_monitor_process(tool_name: str, command: list, cwd: Path) -> dict:
    print(f"  > Running {tool_name} with default max parallelism...")
    command_str_list = [str(c) for c in command]
    print(f"    Command: {' '.join(command_str_list)}")
    start_time = time.monotonic()
    
    try:
        process = subprocess.Popen(command_str_list, cwd=cwd, stdout=sys.stdout, stderr=sys.stderr, text=True, encoding='utf-8')
        p = psutil.Process(process.pid)
        peak_rss_mb = 0
        while process.poll() is None:
            try:
                peak_rss_mb = max(peak_rss_mb, p.memory_info().rss / 1024 / 1024)
            except psutil.NoSuchProcess: break
            time.sleep(0.02)
        wall_time = time.monotonic() - start_time
        returncode = process.returncode
        if returncode != 0:
            print(f"  > ❌ {tool_name} FAILED with exit code {returncode}.")
        else:
            print(f"  > ✅ {tool_name} finished in {wall_time:.2f}s. Peak Memory: {peak_rss_mb:.2f} MB")
    except (FileNotFoundError, psutil.NoSuchProcess) as e:
        print(f"  > ❌ ERROR launching {tool_name}: {e}")
        wall_time, peak_rss_mb, returncode = -1, -1, -1
    return {"tool": tool_name, "time_sec": wall_time, "peak_mem_mb": peak_rss_mb, "success": returncode == 0}

# ==============================================================================
#                             VALIDATION & REPORTING
# ==============================================================================

def validate_results(gnomon_output_path: Path, plink2_output_path: Path) -> bool:
    try:
        g_df = pd.read_csv(gnomon_output_path, sep='\t').rename(columns={'#IID': 'IID'})
        p_df = pd.read_csv(plink2_output_path, sep=r'\s+').rename(columns={'#FID': 'FID'})
        if len(g_df) != len(p_df):
            print(f"  > ❌ VALIDATION FAILED: Row count mismatch ({len(g_df)} vs {len(p_df)})."); return False
        g_score_cols = {int(re.search(r'score_(\d+)_AVG', c).group(1)): c for c in g_df.columns if c.endswith('_AVG')}
        p_score_cols = {int(re.search(r'SCORE(\d+)_AVG', c).group(1)): c for c in p_df.columns if c.endswith('_AVG')}
        if len(g_score_cols) != len(p_score_cols):
             print(f"  > ❌ VALIDATION FAILED: Score column count mismatch."); return False
        merged_df = pd.merge(g_df, p_df, on="IID")
        for i in sorted(g_score_cols.keys()):
            g_col, p_col = g_score_cols[i], p_score_cols[i]
            # PLINK2's --score average is over total allele observations (2*variants),
            # while gnomon's average is over variants. We multiply by 2 to reconcile.
            is_close = np.allclose(merged_df[g_col], merged_df[p_col] * 2, rtol=1e-4, atol=1e-7, equal_nan=True)
            if not is_close:
                diff = np.nanmax(np.abs(merged_df[g_col] - merged_df[p_col] * 2))
                print(f"  > ❌ VALIDATION FAILED: Scores in '{g_col}' and '{p_col}' do not match. Max diff: {diff}"); return False
        print("  > ✅ Validation Successful: Output scores are numerically concordant.")
        return True
    except Exception as e:
        print(f"  > ❌ VALIDATION FAILED with an exception: {e}"); return False

def report_results(results: list):
    print_header("Benchmark Summary Report")
    if not results: print("No results to report."); return
    df = pd.DataFrame(results)
    try:
        report_df = df.pivot_table(index=['n_variants', 'n_individuals', 'n_scores'], columns='tool', values=['time_sec', 'peak_mem_mb'])
        report_df.columns = [f"{val}_{tool}" for val, tool in report_df.columns]
        print("---\n- Ratios (< 1.0) indicate gnomon is faster or uses less memory.\n- Time is wall-clock seconds. Memory is peak RSS in Megabytes.\n---")
        if 'time_sec_gnomon' in report_df and 'time_sec_plink2' in report_df: report_df['time_ratio_g/p'] = report_df['time_sec_gnomon'] / report_df['time_sec_plink2']
        if 'peak_mem_mb_gnomon' in report_df and 'peak_mem_mb_plink2' in report_df: report_df['mem_ratio_g/p'] = report_df['peak_mem_mb_gnomon'] / report_df['peak_mem_mb_plink2']
        print(report_df.to_markdown(floatfmt=".3f"))
    except Exception as e:
        print("Could not generate pivot table. Printing raw results."); print(df.to_markdown(index=False, floatfmt=".3f"))
    summary_path = WORKDIR / "benchmark_summary.csv"
    df.to_csv(summary_path, index=False)
    print(f"\nFull results saved to '{summary_path}'")

def main():
    np.random.seed(42); random.seed(42)
    gnomon_abs_path, plink2_abs_path = GNOMON_BINARY_REL.resolve(), PLINK2_BINARY_REL.resolve()
    if not setup_environment(gnomon_abs_path, plink2_abs_path): exit(1)

    all_results, failed_runs, run_id = [], 0, 0
    for workload_params in DIMENSIONS:
        run_id += 1
        print_header(f"Benchmark Run {run_id}/{len(DIMENSIONS)}: {workload_params}")
        generator = RealisticDataGenerator(n_variants=workload_params["n_variants"], n_individuals=workload_params["n_individuals"], workdir=WORKDIR)
        data_prefix = generator.generate_all_files(n_scores=workload_params["n_scores"])

        gnomon_cmd = [str(gnomon_abs_path), "--score", generator.score_file_path.name, data_prefix.name]
        gnomon_res = run_and_monitor_process("gnomon", gnomon_cmd, WORKDIR)
        gnomon_res.update(workload_params); all_results.append(gnomon_res)
        
        n_scores = workload_params["n_scores"]
        score_col_range, plink_out_prefix = f"4-{3 + n_scores}", WORKDIR / f"plink2_run{run_id}"
        
        # The PLINK2 command requires careful ordering of modifiers.
        # --score takes several modifiers, such as 'header' and 'no-mean-imputation'.
        # --score-col-nums is a separate flag that must come after all --score modifiers.
        plink2_cmd = [str(plink2_abs_path), "--bfile", data_prefix.name, "--score", generator.score_file_path.name,
                      "1", "2", "header", "no-mean-imputation", "--score-col-nums", score_col_range, "--out", plink_out_prefix.name]
        plink2_res = run_and_monitor_process("plink2", plink2_cmd, WORKDIR)
        plink2_res.update(workload_params); all_results.append(plink2_res)
        
        if gnomon_res["success"] and plink2_res["success"]:
            gnomon_original_out = data_prefix.with_suffix(".sscore")
            gnomon_unique_out = WORKDIR / f"run{run_id}_gnomon.sscore"
            try:
                shutil.move(gnomon_original_out, gnomon_unique_out)
                if not validate_results(gnomon_unique_out, plink_out_prefix.with_suffix(".sscore")):
                    failed_runs += 1
            except FileNotFoundError:
                print(f"  > ❌ VALIDATION SKIPPED: Gnomon output file not found at {gnomon_original_out}"); failed_runs += 1
        else:
            failed_runs += 1
        
        generator.cleanup()

    report_results(all_results)
    if failed_runs > 0:
        print(f"\n❌ Benchmark finished with {failed_runs} failed or invalid run(s)."); exit(1)
    else:
        print("\n🎉 All benchmarks completed and passed validation successfully."); exit(0)

if __name__ == "__main__":
    main()
