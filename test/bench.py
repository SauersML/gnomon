import os
import random
import re
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Dict, List, Any

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

# --- The Crucible: Realistic Benchmark Scenarios ---
# This new configuration drives a generator that creates messy, realistic data
# to test the architectures under real-world conditions.
REALISTIC_DIMENSIONS = [
    {
        "test_name": "Large_GWAS_High_Overlap",
        "n_individuals": 50_000,
        "genome_variants": 20_000_000,
        "target_variants": 3_000_000,
        "score_files": [
            {
                "name": "large_gwas",
                "n_scores": 10,
                "gwas_source_variants": 2_500_000,
                "overlap_pct": 0.95,
                "flip_pct": 0.15,
                "missing_weight_pct": 0.01,
            }
        ],
    },
    {
        "test_name": "Multi_Panel_Modest_Overlap",
        "n_individuals": 10_000,
        "genome_variants": 20_000_000,
        "target_variants": 1_000_000,
        "score_files": [
            {"name": "panel_A", "n_scores": 5, "gwas_source_variants": 500, "overlap_pct": 0.80, "flip_pct": 0.10, "missing_weight_pct": 0.02},
            {"name": "panel_B", "n_scores": 8, "gwas_source_variants": 1200, "overlap_pct": 0.85, "flip_pct": 0.20, "missing_weight_pct": 0.05},
            {"name": "panel_C", "n_scores": 2, "gwas_source_variants": 800, "overlap_pct": 0.90, "flip_pct": 0.05, "missing_weight_pct": 0.0},
        ],
    },
    {
        "test_name": "Stress_Test_Low_Overlap",
        "n_individuals": 5_000,
        "genome_variants": 20_000_000,
        "target_variants": 500_000,
        "score_files": [
            {
                "name": "discovery_gwas",
                "n_scores": 20,
                "gwas_source_variants": 4_000_000,
                "overlap_pct": 0.10,
                "flip_pct": 0.10,
                "missing_weight_pct": 0.01,
            }
        ],
    },
    {
        "test_name": "Dense_Score_Sparse_Geno",
        "n_individuals": 15_000,
        "genome_variants": 10_000_000,
        "target_variants": 500_000,
        "score_files": [
            {
                "name": "dense_score",
                "n_scores": 64,
                "gwas_source_variants": 400_000,
                "overlap_pct": 0.98,
                "flip_pct": 0.10,
                "missing_weight_pct": 0.0,
                "score_sparsity": 1.0, # 1.0 = fully dense
            }
        ],
    },
    {
        "test_name": "Ultra_Scale_Test",
        "n_individuals": 150_000,
        "genome_variants": 30_000_000,
        "target_variants": 4_000_000,
        "score_files": [
            {
                "name": "ultra_gwas",
                "n_scores": 80,
                "gwas_source_variants": 3_000_000,
                "overlap_pct": 0.95,
                "flip_pct": 0.15,
                "missing_weight_pct": 0.01,
            }
        ],
    },
]

# --- Data Realism & Variety Parameters ---
EFFECT_DISTRIBUTIONS = [('normal', 0, 0.001), ('laplace', 0, 0.05)]

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
    """Generates messy, realistic, decoupled genomic data for benchmarking."""
    def __init__(self, workload_params: Dict[str, Any], workdir: Path):
        self.params = workload_params
        self.workdir = workdir
        self.run_prefix = workdir / f"data_{workload_params['test_name']}"
        self.target_variants_df: pd.DataFrame = None
        self.gwas_universe_df: pd.DataFrame = None

    def _generate_variant_universes(self):
        print("    > Generating variant universes...")
        n_genome = self.params['genome_variants']
        positions = np.sort(np.random.choice(np.arange(1, n_genome * 10), n_genome, replace=False))
        base_alleles = np.array(['A', 'C', 'G', 'T'])
        ref_indices = np.random.randint(0, 4, n_genome)
        alt_indices = (ref_indices + np.random.randint(1, 4, n_genome)) % 4
        genome_df = pd.DataFrame({
            'chr': 1, 'pos': positions, 'id': [f"1:{p}" for p in positions],
            'a1': base_alleles[ref_indices], 'a2': base_alleles[alt_indices]
        }).set_index('id', drop=False)

        self.target_variants_df = genome_df.sample(n=self.params['target_variants'], random_state=42).sort_values('pos').reset_index(drop=True)
        self.gwas_universe_df = genome_df.copy() # The GWAS universe is the whole genome pool

    def _generate_bim_and_fam(self):
        print(f"    > Generating .bim for {len(self.target_variants_df)} variants and .fam for {self.params['n_individuals']} individuals...")
        bim_df = self.target_variants_df.copy()
        bim_df['cm'] = 0
        bim_df[['chr', 'id', 'cm', 'pos', 'a1', 'a2']].to_csv(self.run_prefix.with_suffix(".bim"), sep='\t', header=False, index=False)
        with open(self.run_prefix.with_suffix(".fam"), 'w') as f:
            for i in range(self.params['n_individuals']):
                f.write(f"sample_{i}\tsample_{i}\t0\t0\t0\t-9\n")

    def _generate_bed(self):
        print(f"    > Generating .bed file (memory-efficient, row-by-row)...")
        n_variants, n_individuals = len(self.target_variants_df), self.params['n_individuals']
        dist_name, p1, p2 = random.choice(AF_DISTRIBUTIONS)
        af = np.random.beta(p1, p2, n_variants) if dist_name == 'beta' else np.random.uniform(p1, p2, n_variants)
        
        code_map = {0: 0b00, 1: 0b10, 2: 0b11, -1: 0b01}
        mapping_array = np.array([code_map[key] for key in sorted(code_map)], dtype=np.uint8)
        
        with open(self.run_prefix.with_suffix(".bed"), 'wb') as f:
            f.write(bytes([0x6c, 0x1b, 0x01]))
            padded_n_individuals = (n_individuals + 3) // 4 * 4
            bytes_per_row = padded_n_individuals // 4

            for i in range(n_variants):
                p = af[i]
                hwe_probs = np.array([(1-p)**2, 2*p*(1-p), p**2])
                # Generate genotypes for one variant across all individuals
                rand_draws = np.random.rand(n_individuals)
                genotypes = (rand_draws > hwe_probs[0]) + (rand_draws > hwe_probs[0] + hwe_probs[1])
                genotypes[(np.random.rand(n_individuals) < 0.01)] = -1
                
                codes = mapping_array[genotypes.astype(int)]
                padded_codes = np.full(padded_n_individuals, code_map[-1], dtype=np.uint8)
                padded_codes[:n_individuals] = codes
                
                packed_bytes = np.zeros(bytes_per_row, dtype=np.uint8)
                packed_bytes = (padded_codes[0::4]) | (padded_codes[1::4] << 2) | \
                               (padded_codes[2::4] << 4) | (padded_codes[3::4] << 6)
                f.write(packed_bytes.tobytes())

    def _generate_score_files(self) -> List[Path]:
        print("    > Generating score files with realistic overlap and ambiguity...")
        generated_paths = []
        target_variant_ids = set(self.target_variants_df['id'])
        
        for sf_config in self.params['score_files']:
            n_overlap = int(sf_config['gwas_source_variants'] * sf_config['overlap_pct'])
            n_non_overlap = sf_config['gwas_source_variants'] - n_overlap
            
            overlapping_ids = np.random.choice(list(target_variant_ids), n_overlap, replace=False)
            
            gwas_non_target_ids = self.gwas_universe_df[~self.gwas_universe_df.index.isin(target_variant_ids)].index
            non_overlapping_ids = np.random.choice(gwas_non_target_ids, n_non_overlap, replace=False)
            
            score_variant_ids = np.concatenate([overlapping_ids, non_overlapping_ids])
            score_df_source = self.gwas_universe_df.loc[score_variant_ids].copy()
            score_df_source = score_df_source.sort_values(['chr', 'pos']).reset_index(drop=True)
            
            score_df_source['effect_allele'] = score_df_source['a2']
            score_df_source['other_allele'] = score_df_source['a1']
            
            overlap_mask = score_df_source['id'].isin(overlapping_ids)
            flip_indices = score_df_source[overlap_mask].sample(frac=sf_config['flip_pct'], random_state=42).index
            
            orig_eff = score_df_source.loc[flip_indices, 'effect_allele'].copy()
            score_df_source.loc[flip_indices, 'effect_allele'] = score_df_source.loc[flip_indices, 'other_allele']
            score_df_source.loc[flip_indices, 'other_allele'] = orig_eff
            
            final_score_df = score_df_source[['id', 'effect_allele', 'other_allele']].rename(columns={'id': 'snp_id'})
            n_variants_in_score = len(final_score_df)
            for i in range(sf_config['n_scores']):
                dist_name, p1, p2 = random.choice(EFFECT_DISTRIBUTIONS)
                sparsity = sf_config.get('score_sparsity', random.uniform(0.01, 0.8))
                
                weights = np.zeros(n_variants_in_score)
                n_weighted = int(sparsity * n_variants_in_score)
                indices = np.random.choice(n_variants_in_score, n_weighted, replace=False)
                
                if n_weighted > 0:
                    eff_weights = np.random.normal(p1, p2, n_weighted) if dist_name == 'normal' else np.random.laplace(p1, p2, n_weighted)
                    weights[indices] = eff_weights
                
                if sf_config['missing_weight_pct'] > 0 and n_weighted > 0:
                    n_missing = int(sf_config['missing_weight_pct'] * n_weighted)
                    missing_indices = np.random.choice(indices, n_missing, replace=False)
                    weights[missing_indices] = np.nan

                final_score_df[f"score_{sf_config['name']}_{i+1}"] = weights

            score_file_path = self.run_prefix.with_suffix(f".{sf_config['name']}.score")
            final_score_df.to_csv(score_file_path, sep='\t', index=False, float_format='%.6g', na_rep='')
            generated_paths.append(score_file_path)
            
        return generated_paths

    def generate_all_files(self) -> (Path, List[Path]):
        self._generate_variant_universes()
        self._generate_bim_and_fam()
        self._generate_bed()
        score_file_paths = self._generate_score_files()
        return self.run_prefix, score_file_paths

    def cleanup(self):
        print("    > Cleaning up generated data files...")
        extensions = [".bed", ".bim", ".fam"] + [f".{sf['name']}.score" for sf in self.params['score_files']]
        for ext in extensions:
            try: self.run_prefix.with_suffix(ext).unlink(missing_ok=True)
            except IsADirectoryError: pass

# ==============================================================================
#                       EXECUTION & MONITORING ENGINE
# ==============================================================================

def run_and_monitor_process(tool_name: str, command: List[str], cwd: Path) -> Dict[str, Any]:
    print(f"  > Running {tool_name} with default max parallelism...")
    print(f"    Command: {' '.join(command)}")
    start_time = time.monotonic()
    
    try:
        process = subprocess.Popen(command, cwd=cwd, stdout=sys.stdout, stderr=sys.stderr, text=True, encoding='utf-8')
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

def simplified_validation(gnomon_success: bool, plink2_success: bool):
    """A simplified validation check focusing on successful execution."""
    if gnomon_success and plink2_success:
        print("  > ✅ Simplified Validation: Both tools completed successfully.")
        return True
    else:
        print("  > ❌ VALIDATION FAILED: One or both tools failed to complete.")
        return False
    
def report_results(results: list):
    print_header("Benchmark Summary Report")
    if not results: print("No results to report."); return
    df = pd.DataFrame(results)
    
    index_cols = ['test_name']
    for col in index_cols:
        if col not in df.columns:
            df[col] = 'N/A'
            
    try:
        report_df = df.pivot_table(index=index_cols, columns='tool', values=['time_sec', 'peak_mem_mb'])
        report_df.columns = [f"{val}_{tool}" for val, tool in report_df.columns]
        print("---\n- Ratios (< 1.0) indicate gnomon is faster or uses less memory.\n- Time is wall-clock seconds. Memory is peak RSS in Megabytes.\n---")
        if 'time_sec_gnomon' in report_df and 'time_sec_plink2' in report_df: report_df['time_ratio_g/p'] = report_df['time_sec_gnomon'] / report_df['time_sec_plink2']
        if 'peak_mem_mb_gnomon' in report_df and 'peak_mem_mb_plink2' in report_df: report_df['mem_ratio_g/p'] = report_df['peak_mem_mb_gnomon'] / report_df['peak_mem_mb_plink2']
        print(report_df.to_markdown(floatfmt=".3f"))
    except Exception as e:
        print(f"Could not generate pivot table. Printing raw results.\n{e}"); print(df.to_markdown(index=False, floatfmt=".3f"))
    
    summary_path = WORKDIR / "benchmark_summary.csv"
    df.to_csv(summary_path, index=False)
    print(f"\nFull results saved to '{summary_path}'")

def main():
    np.random.seed(42); random.seed(42)
    gnomon_abs_path, plink2_abs_path = GNOMON_BINARY_REL.resolve(), PLINK2_BINARY_REL.resolve()
    if not setup_environment(gnomon_abs_path, plink2_abs_path): sys.exit(1)

    all_results, failed_runs = [], 0
    for run_id, workload_params in enumerate(REALISTIC_DIMENSIONS, 1):
        params = workload_params.copy()
        print_header(f"Benchmark Run {run_id}/{len(REALISTIC_DIMENSIONS)}: {params['test_name']}")
        
        generator = RealisticDataGenerator(workload_params=params, workdir=WORKDIR)
        data_prefix, score_files = generator.generate_all_files()

        # --- Gnomon Execution ---
        gnomon_cmd = [str(gnomon_abs_path)]
        for sf in score_files:
            gnomon_cmd.extend(["--score", sf.name])
        gnomon_cmd.append(data_prefix.name)
        gnomon_res = run_and_monitor_process("gnomon", gnomon_cmd, WORKDIR)
        gnomon_res.update(params); all_results.append(gnomon_res)
        
        # --- PLINK2 Execution ---
        plink_out_prefix = WORKDIR / f"plink2_run{run_id}"
        plink2_cmd = [str(plink2_abs_path), "--bfile", data_prefix.name, "--out", plink_out_prefix.name, "--threads", str(os.cpu_count())]
        for sf in score_files:
            with open(sf, 'r') as f:
                header = f.readline().strip()
                n_file_scores = len(header.split('\t')) - 3
            score_col_range = "4" if n_file_scores == 1 else f"4-{3 + n_file_scores}"
            # PLINK2 requires a full --score block for each file
            plink2_cmd.extend(["--score", sf.name, "1", "2", "header", "no-mean-imputation", "list-variants", "--score-col-nums", score_col_range])
            
        plink2_res = run_and_monitor_process("plink2", plink2_cmd, WORKDIR)
        plink2_res.update(params); all_results.append(plink2_res)
        
        # --- Validation ---
        if not simplified_validation(gnomon_res["success"], plink2_res["success"]):
            failed_runs += 1
        
        generator.cleanup()

    report_results(all_results)
    if failed_runs > 0:
        print(f"\n❌ Benchmark finished with {failed_runs} failed or invalid run(s)."); sys.exit(1)
    else:
        print("\n🎉 All benchmarks completed and passed validation successfully."); sys.exit(0)

if __name__ == "__main__":
    main()
