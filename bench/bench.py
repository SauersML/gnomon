import subprocess
import requests
import zipfile
import random
import time
import sys
import threading
from pathlib import Path

import pandas as pd
import psutil

# --- Configuration ---
random.seed(2025)
DATA_URL_BASE = "https://github.com/SauersML/genomic_pca/blob/main/data/"
FILES_TO_DOWNLOAD = {
    "chr22_subset50.bed.zip": "chr22_subset50.bed",
    "chr22_subset50.bim.zip": "chr22_subset50.bim",
    "chr22_subset50.fam.zip": "chr22_subset50.fam",
}
DATA_DIR = Path("./bench_data")
PLINK_PREFIX = DATA_DIR / "chr22_subset50"
SCORE_FILE = DATA_DIR / "scores.tsv"
GNOMON_BINARY = Path("../target/release/gnomon")
NUM_RUNS = 3 # Number of benchmark repetitions for stable results

# --- Helper Functions (Duplicated for standalone use) ---

def setup_environment():
    """Create data directory, download, and unzip files."""
    print("--- 셋업: Setting up benchmark environment ---")
    DATA_DIR.mkdir(exist_ok=True)
    for zip_name, final_name in FILES_TO_DOWNLOAD.items():
        zip_path = DATA_DIR / zip_name
        final_path = DATA_DIR / final_name
        if not final_path.exists():
            print(f"Downloading {zip_name}...")
            url = f"{DATA_URL_BASE}{zip_name}?raw=true"
            # Using requests here for simplicity
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(zip_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Unzipping {zip_name}...")
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(DATA_DIR)
            zip_path.unlink()
    print("Environment setup complete.")

def generate_score_file():
    """Creates a synthetic score file from the .bim file."""
    print("--- 셋업: Generating synthetic score file ---")
    if SCORE_FILE.exists():
        return
    bim_df = pd.read_csv(PLINK_PREFIX.with_suffix(".bim"), sep='\t', header=None,
                         names=['chr', 'snp_id', 'cm', 'pos', 'a1', 'a2'])
    score_data = []
    for _, row in bim_df.iterrows():
        effect_allele = random.choice([row['a1'], row['a2']])
        score1 = random.uniform(-1.0, 1.0)
        score2 = random.uniform(-0.5, 0.5)
        score_data.append([row['snp_id'], effect_allele, score1, score2])
    score_df = pd.DataFrame(score_data, columns=['snp_id', 'effect_allele', 'PRS1', 'PRS2'])
    score_df.to_csv(SCORE_FILE, sep='\t', index=False)
    print(f"Generated score file with {len(score_df)} variants.")


def monitor_memory(p, results):
    """Polls process memory usage in a separate thread."""
    peak_mem = 0
    results['peak_mem_mb'] = 0
    while p.is_running() and p.status() != psutil.STATUS_ZOMBIE:
        try:
            mem_info = p.memory_info()
            # RSS: Resident Set Size, non-swapped physical memory
            peak_mem = max(peak_mem, mem_info.rss)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            break
        time.sleep(0.01) # Poll every 10ms
    results['peak_mem_mb'] = peak_mem / (1024 * 1024)


def run_benchmark(tool_name, command_list):
    """
    Runs a command, measuring wall-clock time and peak memory usage.
    Returns a dictionary of results.
    """
    print(f"--- 벤치마킹: {tool_name} ---")
    
    start_time = time.perf_counter()
    
    # Popen allows us to get the process handle for monitoring
    process = subprocess.Popen(command_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    monitor_results = {}
    mem_thread = threading.Thread(target=monitor_memory, args=(process, monitor_results))
    mem_thread.start()
    
    # Wait for the process to complete
    process.wait()
    mem_thread.join()
    
    end_time = time.perf_counter()
    
    return {
        'tool': tool_name,
        'time_sec': end_time - start_time,
        'peak_mem_mb': monitor_results.get('peak_mem_mb', 0)
    }

# --- Main Execution Logic ---
if __name__ == "__main__":
    setup_environment()
    generate_score_file()

    # Define commands for benchmark
    cmd_gnomon = [str(GNOMON_BINARY), "--input-path", str(DATA_DIR), "--score", str(SCORE_FILE)]
    cmd_plink2 = ["plink2", "--bfile", str(PLINK_PREFIX), "--score", str(SCORE_FILE), "header", "no-mean-imputation",
                  "--score-col-nums", "3-4", "--out", str(DATA_DIR / "plink2_run")]
    
    all_results = []
    for i in range(NUM_RUNS):
        print(f"\n--- Starting Benchmark Run {i+1}/{NUM_RUNS} ---")
        all_results.append(run_benchmark("gnomon", cmd_gnomon))
        all_results.append(run_benchmark("plink2", cmd_plink2))
    
    # --- Analyze and Report ---
    print("\n\n--- BENCHMARK SUMMARY ---")
    results_df = pd.DataFrame(all_results)
    summary = results_df.groupby('tool').agg(
        mean_time=('time_sec', 'mean'),
        std_time=('time_sec', 'std'),
        mean_mem=('peak_mem_mb', 'mean'),
        std_mem=('peak_mem_mb', 'std')
    ).reset_index()

    print(summary.to_string(index=False, float_format="%.3f"))

    # Calculate and print performance factors
    gnomon_stats = summary[summary['tool'] == 'gnomon'].iloc[0]
    plink2_stats = summary[summary['tool'] == 'plink2'].iloc[0]
    
    time_factor = plink2_stats['mean_time'] / gnomon_stats['mean_time']
    mem_factor = plink2_stats['mean_mem'] / gnomon_stats['mean_mem']

    print("\n--- PERFORMANCE FACTORS (PLINK2 vs Gnomon) ---")
    print(f"Time:       gnomon is {time_factor:.2f}x faster")
    print(f"Memory:     gnomon uses {mem_factor:.2f}x less memory (or {1/mem_factor:.2f}x memory of plink2)")
