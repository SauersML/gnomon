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

def download_and_extract(url: str, dest_dir: Path):
    """Downloads a file and extracts it if it is a .zip or .gz archive."""
    filename = Path(url.split("/")[-1])
    download_path = dest_dir / filename
    
    print(f"Downloading {filename}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(download_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

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

    print("--- Environment setup complete ---\n")

def monitor_memory(p: subprocess.Popen, results: dict):
    """Polls process memory usage in a separate thread."""
    peak_mem = 0
    results['peak_mem_mb'] = 0
    proc = psutil.Process(p.pid)
    while p.poll() is None:
        try:
            mem_info = proc.memory_info()
            peak_mem = max(peak_mem, mem_info.rss)
        except psutil.NoSuchProcess:
            break
        time.sleep(0.01) # Poll every 10ms
    results['peak_mem_mb'] = peak_mem / (1024 * 1024)

def run_and_measure(command: list, tool_name: str) -> dict:
    """Runs a command, measuring its wall-clock time and peak memory usage."""
    print(f"--- {tool_name} ---")
    start_time = time.perf_counter()
    
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    monitor_results = {}
    mem_thread = threading.Thread(target=monitor_memory, args=(process, monitor_results))
    mem_thread.start()
    
    stdout, stderr = process.communicate()
    mem_thread.join()
    
    end_time = time.perf_counter()
    
    if process.returncode != 0:
        print(f"❌ ERROR running {tool_name}. Exit code: {process.returncode}")
        print("--- STDOUT ---")
        print(stdout)
        print("--- STDERR ---")
        print(stderr)
        sys.exit(1)
        
    return {
        'tool': tool_name,
        'time_sec': end_time - start_time,
        'peak_mem_mb': monitor_results.get('peak_mem_mb', 0)
    }

def find_reformatted_file(original_pgs_path: Path) -> Path:
    """Constructs the expected path of a gnomon-reformatted score file."""
    expected_name = f"{original_pgs_path.stem}.gnomon_format.tsv"
    expected_path = original_pgs_path.parent / expected_name
    assert expected_path.exists(), f"Reformatted file not found at {expected_path}"
    return expected_path

def validate_outputs(gnomon_path: Path, plink2_path: Path, plink1_path: Path, pgs_id: str) -> bool:
    """Compares the output files from all tools for identity and correlation."""
    print(f"--- 검증: Validating outputs for {pgs_id} ---")
    try:
        gnomon_df = pd.read_csv(gnomon_path, sep='\t').rename(columns={"#IID": "IID"}).set_index("IID")
        plink2_df = pd.read_csv(plink2_path, sep='\t').rename(columns={"#IID": "IID"}).set_index("IID")
        plink1_df = pd.read_csv(plink1_path, delim_whitespace=True).set_index("IID")

        # Get score column names
        score_name_gnomon = gnomon_df.columns[0]
        score_name_plink2 = f"{score_name_gnomon}_SUM"
        
        # Merge for comparison
        merged = gnomon_df.join(plink2_df[[score_name_plink2]]).join(plink1_df[['SCORE']])
        
        # 1. Numerical Identity Check
        is_close_p2 = np.isclose(merged[score_name_gnomon], merged[score_name_plink2], atol=NUMERICAL_TOLERANCE)
        is_close_p1 = np.isclose(merged[score_name_gnomon], merged['SCORE'], atol=NUMERICAL_TOLERANCE)
        
        assert is_close_p2.all(), f"Numerical mismatch between Gnomon and PLINK2 for {pgs_id}"
        print(f"✅ Numerical Identity: Gnomon == PLINK2")
        
        assert is_close_p1.all(), f"Numerical mismatch between Gnomon and PLINK1 for {pgs_id}"
        print(f"✅ Numerical Identity: Gnomon == PLINK1")
        
        # 2. Correlation Check (as a redundant sanity check)
        correlation = merged.corr().iloc[0, 1]
        assert correlation > CORRELATION_THRESHOLD, f"Correlation below threshold ({correlation:.5f}) for {pgs_id}"
        print(f"✅ Correlation > {CORRELATION_THRESHOLD} (actual: {correlation:.6f})")
        
        return True

    except Exception as e:
        print(f"❌ VALIDATION FAILED for {pgs_id}: {e}")
        return False


# ========================================================================================
#                             MAIN EXECUTION
# ========================================================================================

if __name__ == "__main__":
    print("===== Running Gnomon CI Test & Benchmark Suite =====")
    setup_environment()
    
    all_perf_results = []
    any_test_failed = False
    
    for pgs_id, pgs_url in PGS_SCORES.items():
        print(f"\n===== TESTING SCORE: {pgs_id} =====")
        original_score_file = CI_WORKDIR / Path(pgs_url.split("/")[-1]).stem

        # --- Run Gnomon (triggers auto-reformatting) ---
        cmd_gnomon = [str(GNOMON_BINARY), "--input-path", str(CI_WORKDIR), "--score", str(original_score_file)]
        all_perf_results.append(run_and_measure(cmd_gnomon, f"gnomon_{pgs_id}"))
        
        # --- Find reformatted file for PLINK ---
        reformatted_file = find_reformatted_file(original_score_file)
        
        # --- Run PLINK2 ---
        cmd_plink2 = [str(PLINK2_BINARY), "--bfile", str(PLINK_PREFIX), 
                      "--score", str(reformatted_file), "header", "no-mean-imputation",
                      "--score-col-nums", "3", "--out", str(CI_WORKDIR / f"plink2_{pgs_id}")]
        all_perf_results.append(run_and_measure(cmd_plink2, f"plink2_{pgs_id}"))

        # --- Run PLINK1 ---
        cmd_plink1 = [str(PLINK1_BINARY), "--bfile", str(PLINK_PREFIX), 
                      "--score", str(reformatted_file), "1", "2", "3", "header", "sum", "no-mean-imputation",
                      "--out", str(CI_WORKDIR / f"plink1_{pgs_id}")]
        all_perf_results.append(run_and_measure(cmd_plink1, f"plink1_{pgs_id}"))
        
        # --- Validate Outputs ---
        gnomon_out = original_score_file.parent / f"{original_score_file.name}.sscore"
        plink2_out = CI_WORKDIR / f"plink2_{pgs_id}.sscore"
        plink1_out = CI_WORKDIR / f"plink1_{pgs_id}.profile"
        
        if not validate_outputs(gnomon_out, plink2_out, plink1_out, pgs_id):
            any_test_failed = True

    # --- Analyze and Report Performance ---
    print("\n\n===== PERFORMANCE SUMMARY =====")
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
    
    gnomon_stats = summary[summary['tool_base'] == 'gnomon'].iloc[0]
    plink2_stats = summary[summary['tool_base'] == 'plink2'].iloc[0]
    
    time_factor = plink2_stats['mean_time_sec'] / gnomon_stats['mean_time_sec']
    mem_factor = gnomon_stats['mean_mem_mb'] / plink2_stats['mean_mem_mb']
    
    print("\n--- PERFORMANCE FACTORS (Gnomon vs PLINK2) ---")
    print(f"Time:       Gnomon is {time_factor:.2f}x faster on average.")
    print(f"Memory:     Gnomon uses {mem_factor:.2f}x the memory of PLINK2 on average.")
    
    # --- Final Exit ---
    if any_test_failed:
        print("\n\n❌ CI CHECK FAILED: One or more correctness tests did not pass.")
        sys.exit(1)
    else:
        print("\n\n🎉 CI CHECK PASSED: All correctness and performance tests completed successfully.")
        sys.exit(0)
