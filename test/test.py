import subprocess
import requests
import zipfile
import gzip
import shutil
import time
import sys
import threading
import os
from pathlib import Path

import pandas as pd
import numpy as np
import psutil

# ========================================================================================
#                             CONFIGURATION
# ========================================================================================

# --- Artifacts & Paths ---
CI_WORKDIR = Path("./ci_workdir")
GNOMON_BINARY = Path("./target/release/gnomon")
PLINK1_BINARY = CI_WORKDIR / "plink"
PLINK2_BINARY = CI_WORKDIR / "plink2"
PLINK_PREFIX = CI_WORKDIR / "chr22_subset50"

# --- Data Sources ---
# Tools
PLINK1_URL = "https://s3.amazonaws.com/plink1-assets/plink_linux_x86_64_20231211.zip"
PLINK2_URL = "https://s3.amazonaws.com/plink2-assets/alpha6/plink2_linux_avx2_20250609.zip"

# Genotype Data
GENOTYPE_URL_BASE = "https://github.com/SauersML/genomic_pca/blob/main/data/"
GENOTYPE_FILES = {
    "chr22_subset50.bed.zip": "chr22_subset50.bed",
    "chr22_subset50.bim.zip": "chr22_subset50.bim",
    "chr22_subset50.fam.zip": "chr22_subset50.fam",
}

# Real-world PGS Catalog Score Files to test against
PGS_SCORES = {
    "PGS004696": "https://ftp.ebi.ac.uk/pub/databases/spot/pgs/scores/PGS004696/ScoringFiles/Harmonized/PGS004696_hmPOS_GRCh38.txt.gz",
    "PGS003725": "https://ftp.ebi.ac.uk/pub/databases/spot/pgs/scores/PGS003725/ScoringFiles/Harmonized/PGS003725_hmPOS_GRCh38.txt.gz",
    "PGS001780": "https://ftp.ebi.ac.uk/pub/databases/spot/pgs/scores/PGS001780/ScoringFiles/Harmonized/PGS001780_hmPOS_GRCh38.txt.gz",
}

# --- Validation Thresholds ---
CORRELATION_THRESHOLD = 0.9999
NUMERICAL_TOLERANCE = 1e-5


# ========================================================================================
#                             HELPER FUNCTIONS
# ========================================================================================

def print_header(title: str):
    """Prints a formatted header."""
    print("\n" + "=" * 80)
    print(f"===== {title.upper()} =====")
    print("=" * 80)

def download_and_extract(url: str, dest_dir: Path):
    """Downloads a file and extracts it if it is a .zip or .gz archive."""
    base_url = url.split('?')[0]
    filename = Path(base_url.split("/")[-1])
    download_path = dest_dir / filename
    
    print(f"Downloading {filename}...")
    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        with open(download_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    except requests.exceptions.RequestException as e:
        print(f"❌ FAILED to download {url}: {e}")
        sys.exit(1)

    if str(filename).endswith(".zip"):
        print(f"Unzipping {filename}...")
        with zipfile.ZipFile(download_path, 'r') as zf:
            zf.extractall(dest_dir)
        download_path.unlink()
    elif str(filename).endswith(".gz"):
        print(f"Decompressing {filename}...")
        unzipped_path = dest_dir / filename.stem
        with gzip.open(download_path, 'rb') as f_in:
            with open(unzipped_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        download_path.unlink()

def setup_environment():
    """Prepares the CI workspace by downloading and setting up all required artifacts."""
    print_header("Environment Setup")
    CI_WORKDIR.mkdir(exist_ok=True)
    
    # Download and set up tools
    download_and_extract(PLINK1_URL, CI_WORKDIR)
    download_and_extract(PLINK2_URL, CI_WORKDIR)
    PLINK1_BINARY.chmod(0o755)
    PLINK2_BINARY.chmod(0o755)
    
    # Download genotype data
    for zip_name in GENOTYPE_FILES:
        url = f"{GENOTYPE_URL_BASE}{zip_name}?raw=true"
        download_and_extract(url, CI_WORKDIR)
        
    # Download score files
    for url in PGS_SCORES.values():
        download_and_extract(url, CI_WORKDIR)

    print("\n--- Environment setup complete ---\n")

def monitor_memory(p: subprocess.Popen, results: dict):
    """Polls process memory usage in a separate thread."""
    peak_mem = 0
    results['peak_mem_mb'] = 0
    try:
        proc = psutil.Process(p.pid)
        while p.poll() is None:
            try:
                mem_info = proc.memory_info()
                peak_mem = max(peak_mem, mem_info.rss)
            except psutil.NoSuchProcess:
                break
            time.sleep(0.01)
    except psutil.NoSuchProcess:
        pass # Process may have finished before the thread started
    results['peak_mem_mb'] = peak_mem / (1024 * 1024)

def inspect_file_head(file_path: Path, num_lines: int = 50):
    """Prints the first few lines of a file for debugging."""
    print("-" * 30)
    print(f"Inspecting file: {file_path.name}")
    print(f"Full path: {file_path.resolve()}")
    if not file_path.exists():
        print(">>> FILE DOES NOT EXIST <<<")
        print("-" * 30)
        return
        
    print("-" * 30)
    try:
        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                if i >= num_lines:
                    break
                print(f"{i+1: >3}: {line.strip()}")
        print("-" * 30)
    except Exception as e:
        print(f"Could not read file: {e}")
        print("-" * 30)


def run_and_measure(command: list, tool_name: str, files_to_inspect_on_error: list = None) -> dict:
    print(f"\n--- Running: {tool_name} ---")
    start_time = time.perf_counter()
    
    # We now stream stderr directly to the console (sys.stderr) so that
    # log messages from gnomon are visible in real-time.
    process = subprocess.Popen(
        command, 
        stdout=subprocess.PIPE, 
        stderr=sys.stderr,  # Stream stderr directly
        text=True, 
        encoding='utf-8'
    )
    
    monitor_results = {}
    mem_thread = threading.Thread(target=monitor_memory, args=(process, monitor_results))
    mem_thread.start()
    
    stdout, _ = process.communicate()
    mem_thread.join()
    
    end_time = time.perf_counter()
    
    if process.returncode != 0:
        print_header(f"ERROR running {tool_name}")
        print(f"Exit Code: {process.returncode}")
        
        print("\n--- COMMAND ---")
        print(' '.join(map(str, command)))
        
        print("\n--- STDOUT (Captured) ---")
        print(stdout if stdout else "[EMPTY]")
        
        # Stderr is already printed, so no need to print it again here.
        
        if files_to_inspect_on_error:
            print_header("INSPECTING RELEVANT FILES FOR DEBUGGING")
            for file_path in files_to_inspect_on_error:
                inspect_file_head(Path(file_path))
        
        print_header("TEST SUITE HALTED DUE TO ERROR")
        sys.exit(1)
        
    print(f"--- Finished {tool_name} in {end_time - start_time:.2f}s ---")
    return {
        'tool': tool_name,
        'time_sec': end_time - start_time,
        'peak_mem_mb': monitor_results.get('peak_mem_mb', 0)
    }

def find_reformatted_file(original_pgs_path: Path) -> Path:
    """Constructs the expected path of a gnomon-reformatted score file."""
    expected_name = f"{original_pgs_path.stem}.gnomon_format.tsv"
    expected_path = original_pgs_path.parent / expected_name
    if not expected_path.exists():
        print(f"❌ Could not find the expected reformatted file at: {expected_path}")
        # This is a critical failure of the gnomon run, so we should exit.
        sys.exit(1)
    print(f"Found reformatted score file: {expected_path.name}")
    return expected_path

def validate_outputs(gnomon_path: Path, plink2_path: Path, plink1_path: Path, pgs_id: str) -> bool:
    """Compares the output files from all tools for identity and correlation."""
    print_header(f"Validating outputs for {pgs_id}")
    try:
        gnomon_df = pd.read_csv(gnomon_path, sep='\t').rename(columns={"#IID": "IID"}).set_index("IID")
        plink2_df = pd.read_csv(plink2_path, sep='\t').rename(columns={"#IID": "IID"}).set_index("IID")
        plink1_df = pd.read_csv(plink1_path, delim_whitespace=True).set_index("IID")
        
        print("Successfully loaded output files from all tools.")

        score_name_gnomon = gnomon_df.columns[0]
        score_name_plink2 = f"{score_name_gnomon}_SUM"
        
        merged = gnomon_df.join(plink2_df[[score_name_plink2]]).join(plink1_df[['SCORE']])
        
        is_close_p2 = np.isclose(merged[score_name_gnomon], merged[score_name_plink2], atol=NUMERICAL_TOLERANCE)
        mismatches_p2 = merged[~is_close_p2]
        if not mismatches_p2.empty:
            print(f"❌ Numerical mismatch between Gnomon and PLINK2 for {pgs_id}")
            print("Sample of mismatched scores:")
            print(mismatches_p2.head())
            return False
        print(f"✅ Numerical Identity: Gnomon == PLINK2")
        
        is_close_p1 = np.isclose(merged[score_name_gnomon], merged['SCORE'], atol=NUMERICAL_TOLERANCE)
        mismatches_p1 = merged[~is_close_p1]
        if not mismatches_p1.empty:
            print(f"❌ Numerical mismatch between Gnomon and PLINK1 for {pgs_id}")
            print("Sample of mismatched scores:")
            print(mismatches_p1.head())
            return False
        print(f"✅ Numerical Identity: Gnomon == PLINK1")
        
        correlation = merged.corr().iloc[0, 1]
        if not (correlation > CORRELATION_THRESHOLD):
            print(f"❌ Correlation below threshold ({correlation:.5f}) for {pgs_id}")
            return False
        print(f"✅ Correlation > {CORRELATION_THRESHOLD} (actual: {correlation:.6f})")
        
        return True

    except FileNotFoundError as e:
        print(f"❌ VALIDATION FAILED for {pgs_id}: Output file not found: {e}")
        return False
    except Exception as e:
        print(f"❌ VALIDATION FAILED for {pgs_id} with an unexpected error: {e}")
        return False

# ========================================================================================
#                             MAIN EXECUTION
# ========================================================================================

if __name__ == "__main__":
    print_header("Running Gnomon CI Test & Benchmark Suite")
    setup_environment()
    
    all_perf_results = []
    any_test_failed = False
    
    for pgs_id, pgs_url in PGS_SCORES.items():
        print_header(f"TESTING SCORE: {pgs_id}")
        original_score_file = CI_WORKDIR / Path(pgs_url.split("/")[-1]).stem
        reformatted_file_path = original_score_file.parent / f"{original_score_file.stem}.gnomon_format.tsv"

        # --- Run Gnomon (triggers auto-reformatting) ---
        cmd_gnomon = [str(GNOMON_BINARY), "--score", str(original_score_file), str(CI_WORKDIR)]
        # Define the key files to inspect if gnomon fails
        gnomon_debug_files = [
            PLINK_PREFIX.with_suffix(".bim"),
            original_score_file,
            reformatted_file_path  # This might not exist if failure is early
        ]
        all_perf_results.append(run_and_measure(cmd_gnomon, f"gnomon_{pgs_id}", gnomon_debug_files))
        
        # --- Find reformatted file for PLINK ---
        reformatted_file = find_reformatted_file(original_score_file)
        
        # --- Run PLINK2 ---
        plink2_out_prefix = CI_WORKDIR / f"plink2_{pgs_id}"
        cmd_plink2 = [str(PLINK2_BINARY), "--bfile", str(PLINK_PREFIX), 
                      "--score", str(reformatted_file), "header", "no-mean-imputation",
                      "--score-col-nums", "3", "--out", str(plink2_out_prefix),
                      "--variant-id", "@:#"]
        all_perf_results.append(run_and_measure(cmd_plink2, f"plink2_{pgs_id}"))

        # --- Run PLINK1 ---
        plink1_out_prefix = CI_WORKDIR / f"plink1_{pgs_id}"
        cmd_plink1 = [str(PLINK1_BINARY), "--bfile", str(PLINK_PREFIX), 
                      "--score", str(reformatted_file), "1", "2", "3", "header", "sum", "no-mean-imputation",
                      "--out", str(plink1_out_prefix)]
        all_perf_results.append(run_and_measure(cmd_plink1, f"plink1_{pgs_id}"))
        
        # --- Validate Outputs ---
        gnomon_out = original_score_file.parent / f"{original_score_file.name}.sscore"
        plink2_out = plink2_out_prefix.with_suffix(".sscore")
        plink1_out = plink1_out_prefix.with_suffix(".profile")
        
        if not validate_outputs(gnomon_out, plink2_out, plink1_out, pgs_id):
            any_test_failed = True

    # --- Analyze and Report Performance ---
    print_header("PERFORMANCE SUMMARY")
    results_df = pd.DataFrame(all_perf_results)
    results_df['pgs_id'] = results_df['tool'].apply(lambda x: x.split('_')[1])
    results_df['tool_base'] = results_df['tool'].apply(lambda x: x.split('_')[0])
    
    summary = results_df.groupby('tool_base').agg(
        mean_time_sec=('time_sec', 'mean'),
        std_time_sec=('time_sec', 'std'),
        mean_mem_mb=('peak_mem_mb', 'mean'),
        std_mem_mb=('peak_mem_mb', 'std')
    ).reset_index()

    print(summary.to_markdown(index=False, floatfmt=".3f"))
    
    if 'gnomon' in summary['tool_base'].values and 'plink2' in summary['tool_base'].values:
        gnomon_stats = summary[summary['tool_base'] == 'gnomon'].iloc[0]
        plink2_stats = summary[summary['tool_base'] == 'plink2'].iloc[0]
        
        time_factor = plink2_stats['mean_time_sec'] / gnomon_stats['mean_time_sec']
        mem_factor = gnomon_stats['mean_mem_mb'] / plink2_stats['mean_mem_mb']
        
        print("\n--- PERFORMANCE FACTORS (Gnomon vs PLINK2) ---")
        print(f"Time:       Gnomon is {time_factor:.2f}x faster on average.")
        print(f"Memory:     Gnomon uses {mem_factor:.2f}x the memory of PLINK2 on average.")
    
    # --- Final Exit ---
    if any_test_failed:
        print_header("CI CHECK FAILED")
        print("❌ One or more correctness tests did not pass.")
        sys.exit(1)
    else:
        print_header("CI CHECK PASSED")
        print("🎉 All correctness and performance tests completed successfully.")
        sys.exit(0)
