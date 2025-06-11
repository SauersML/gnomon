import subprocess
import requests
import zipfile
import gzip
import shutil
import time
import sys
import threading
import os
import re
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

# --- Data Paths ---
# Gnomon & PLINK2 use the modern, chr:pos standard
GNOMON_NATIVE_PREFIX = CI_WORKDIR / "gnomon_native_data"
# PLINK1 requires a universally safe chr_pos format
P1_COMPAT_PREFIX = CI_WORKDIR / "p1_compatible_data"
# Source data path
ORIGINAL_PLINK_PREFIX = CI_WORKDIR / "chr22_subset50_original"

# --- Data Sources ---
PLINK1_URL = "https://s3.amazonaws.com/plink1-assets/plink_linux_x86_64_20231211.zip"
PLINK2_URL = "https://s3.amazonaws.com/plink2-assets/alpha6/plink2_linux_avx2_20250609.zip"

GENOTYPE_URL_BASE = "https://github.com/SauersML/genomic_pca/blob/main/data/"
GENOTYPE_FILES = {
    "chr22_subset50.bed.zip": "chr22_subset50.bed",
    "chr22_subset50.bim.zip": "chr22_subset50.bim",
    "chr22_subset50.fam.zip": "chr22_subset50.fam",
}

PGS_SCORES = {
    "PGS004696": "https://ftp.ebi.ac.uk/pub/databases/spot/pgs/scores/PGS004696/ScoringFiles/Harmonized/PGS004696_hmPOS_GRCh38.txt.gz",
    "PGS003725": "https://ftp.ebi.ac.uk/pub/databases/spot/pgs/scores/PGS003725/ScoringFiles/Harmonized/PGS003725_hmPOS_GRCh38.txt.gz",
    "PGS001780": "https://ftp.ebi.ac.uk/pub/databases/spot/pgs/scores/PGS001780/ScoringFiles/Harmonized/PGS001780_hmPOS_GRCh38.txt.gz",
}

# --- Validation Thresholds ---
NUMERICAL_TOLERANCE = 1e-4
CORR_THRESHOLD = 0.9999

# ========================================================================================
#                             HELPER FUNCTIONS
# ========================================================================================

def print_header(title: str, char: str = "="):
    """Prints a distinct, formatted header."""
    width = 80
    print("\n" + char * width)
    print(f"{char*4} {title.upper()} {' ' * (width - len(title) - 10)} {char*4}")
    print(char * width)

def download_and_extract(url: str, dest_dir: Path):
    """Downloads a file and extracts it if it is a .zip or .gz archive."""
    base_url = url.split('?')[0]
    filename = Path(base_url.split("/")[-1])
    download_path = dest_dir / filename
    
    print(f"Downloading {filename}...")
    try:
        response = requests.get(url, stream=True, timeout=120)
        response.raise_for_status()
        with open(download_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    except requests.exceptions.RequestException as e:
        print(f"❌ FAILED to download {url}: {e}")
        sys.exit(1)

    if str(filename).endswith(".zip"):
        with zipfile.ZipFile(download_path, 'r') as zf:
            zf.extractall(dest_dir)
        download_path.unlink()
    elif str(filename).endswith(".gz"):
        unzipped_path = dest_dir / filename.stem
        with gzip.open(download_path, 'rb') as f_in:
            with open(unzipped_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        download_path.unlink()

def inspect_file_head(file_path: Path, num_lines: int = 10, title: str = ""):
    """Prints the first few lines of a file for quick inspection."""
    header_title = title if title else f"Inspecting file: {file_path.name}"
    print_header(header_title, char='-')
    print(f"Full path: {file_path.resolve()}")
    if not file_path.exists():
        print(">>> FILE DOES NOT EXIST <<<")
    else:
        try:
            with open(file_path, 'r', errors='ignore') as f:
                for i, line in enumerate(f):
                    if i >= num_lines: break
                    print(f"{i+1: >3}: {line.strip()}")
        except Exception as e:
            print(f"Could not read file: {e}")
    print("-" * 80)

def monitor_memory(p: subprocess.Popen, results: dict):
    """Monitors the peak RSS memory of a process in a separate thread."""
    peak_mem = 0
    results['peak_mem_mb'] = 0
    try:
        proc = psutil.Process(p.pid)
        while p.poll() is None:
            try: peak_mem = max(peak_mem, proc.memory_info().rss)
            except psutil.NoSuchProcess: break
            time.sleep(0.01)
    except psutil.NoSuchProcess: pass
    results['peak_mem_mb'] = peak_mem / (1024 * 1024)

def run_and_measure(command: list, tool_name: str, log_path: Path) -> dict:
    """Runs a command, measures its time and memory, and parses its log."""
    print_header(f"RUNNING: {tool_name}", char='-')
    print("Command:")
    print(f"  {' '.join(map(str, command))}\n")
    
    start_time = time.perf_counter()
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8')
    
    monitor_results = {}
    mem_thread = threading.Thread(target=monitor_memory, args=(process, monitor_results))
    mem_thread.start()
    stdout, stderr = process.communicate()
    mem_thread.join()
    
    duration = time.perf_counter() - start_time
    
    results = {
        'tool': tool_name, 'time_sec': duration,
        'peak_mem_mb': monitor_results.get('peak_mem_mb', 0),
        'returncode': process.returncode, 'success': process.returncode == 0
    }

    if not results['success']:
        print(f"❌ {tool_name} FAILED (Exit Code: {process.returncode})")
        print_header("STDOUT", char='.')
        print(stdout.strip() or "[EMPTY]")
        print_header("STDERR", char='.')
        print(stderr.strip() or "[EMPTY]")
    else:
        print(f"✅ {tool_name} finished in {duration:.2f}s.")
        
    return results

def setup_environment():
    """Prepares the CI workspace, downloading and pre-processing all data."""
    print_header("ENVIRONMENT SETUP")
    CI_WORKDIR.mkdir(exist_ok=True)
    
    # Download and set up tools
    for url in [PLINK1_URL, PLINK2_URL]:
        download_and_extract(url, CI_WORKDIR)
    PLINK1_BINARY.chmod(0o755)
    PLINK2_BINARY.chmod(0o755)
    
    # Download and prepare genotype data
    for zip_name, final_name in GENOTYPE_FILES.items():
        download_and_extract(f"{GENOTYPE_URL_BASE}{zip_name}?raw=true", CI_WORKDIR)
        (CI_WORKDIR / final_name).rename(ORIGINAL_PLINK_PREFIX.with_suffix(f".{final_name.split('.')[-1]}"))
        
    # Download score files
    for url in PGS_SCORES.values():
        download_and_extract(url, CI_WORKDIR)

    print_header("DATA PRE-PROCESSING: A ROBUST, TOOL-DRIVEN WORKFLOW")
    try:
        # Create Gnomon/PLINK2-native data (chr:pos IDs)
        print("Step 1a: Creating Gnomon-native data (chr:pos format)...")
        cmd_dedup_gnomon = [
            str(PLINK2_BINARY), "--bfile", str(ORIGINAL_PLINK_PREFIX), 
            "--set-all-var-ids", "@:#", "--rm-dup", "force-first",
            "--make-bed", "--out", str(GNOMON_NATIVE_PREFIX)
        ]
        subprocess.run(cmd_dedup_gnomon, check=True, capture_output=True, text=True)
        print("✅ Gnomon-native data created successfully.")
        inspect_file_head(GNOMON_NATIVE_PREFIX.with_suffix(".bim"), title="Verifying Gnomon-Native .bim File")

        # Create PLINK1-compatible data (chr_pos IDs)
        print("\nStep 1b: Creating PLINK1-compatible data (chr_pos format)...")
        cmd_dedup_p1 = [
            str(PLINK2_BINARY), "--bfile", str(ORIGINAL_PLINK_PREFIX),
            "--set-all-var-ids", "chr@_#", "--rm-dup", "force-first",
            "--make-bed", "--out", str(P1_COMPAT_PREFIX)
        ]
        subprocess.run(cmd_dedup_p1, check=True, capture_output=True, text=True)
        print("✅ PLINK1-compatible data created successfully.")
        inspect_file_head(P1_COMPAT_PREFIX.with_suffix(".bim"), title="Verifying PLINK1-Compatible .bim File")

    except subprocess.CalledProcessError as e:
        print(f"❌ FAILED to pre-process genotype data: {e.stderr}")
        sys.exit(1)

def find_score_column(df: pd.DataFrame, tool_name: str) -> str:
    """Intelligently finds the score column in a results DataFrame."""
    # PLINK1: The column is 'SCORESUM' with the 'sum' modifier, 'SCORE' otherwise.
    if "plink1" in tool_name:
        if 'SCORESUM' in df.columns: return 'SCORESUM'
        if 'SCORE' in df.columns: return 'SCORE'
        raise KeyError("Could not find 'SCORESUM' or 'SCORE' in PLINK1 output.")
    
    # PLINK2: The column is usually 'NAMED_ALLELE_DOSAGE_SUM'. Fall back to others.
    if "plink2" in tool_name:
        priority_list = ['NAMED_ALLELE_DOSAGE_SUM', 'SCORE1_SUM']
        for col in priority_list:
            if col in df.columns: return col
        # Generic fallback
        sum_cols = [c for c in df.columns if 'SUM' in c]
        if sum_cols: return sum_cols[0]
        raise KeyError("Could not find a score sum column in PLINK2 output.")
        
    # Gnomon: The score column is named after the PGS ID itself.
    if "gnomon" in tool_name:
        # It's the only column that isn't the ID column.
        score_cols = [c for c in df.columns if c not in ['#IID', 'IID']]
        if len(score_cols) == 1: return score_cols[0]
        raise KeyError("Could not uniquely identify score column in Gnomon output.")
        
    raise ValueError(f"Unknown tool: {tool_name}")

# ========================================================================================
#                             MAIN EXECUTION & VALIDATION
# ========================================================================================

def main():
    """Main execution function for the CI suite."""
    print_header("GNOMON CI TEST & BENCHMARK SUITE")
    setup_environment()
    
    all_perf_results = []
    failed_tests = []
    
    for pgs_id, pgs_url in PGS_SCORES.items():
        print_header(f"TESTING SCORE: {pgs_id}")
        original_score_file = CI_WORKDIR / Path(pgs_url.split("/")[-1]).stem
        
        # --- 1. Run Gnomon ---
        cmd_gnomon = [str(GNOMON_BINARY), "--score", str(original_score_file), str(GNOMON_NATIVE_PREFIX)]
        gnomon_result = run_and_measure(cmd_gnomon, f"gnomon_{pgs_id}", CI_WORKDIR / "gnomon.log")
        all_perf_results.append(gnomon_result)
        
        if not gnomon_result['success']:
            failed_tests.append(f"{pgs_id} (gnomon_execution_failed)"); continue

        gnomon_reformatted_file = original_score_file.with_suffix(".gnomon_format.tsv")
        gnomon_final_score_file = original_score_file.with_suffix(".txt.sscore")
        
        if not gnomon_reformatted_file.exists() or not gnomon_final_score_file.exists():
            print(f"❌ CRITICAL: Gnomon did not produce its expected output files for {pgs_id}.")
            failed_tests.append(f"{pgs_id} (gnomon_output_missing)"); continue
        
        # --- 2. Run PLINK2 for apples-to-apples comparison ---
        plink2_out_prefix = CI_WORKDIR / f"plink2_{pgs_id}"
        cmd_plink2 = [
            str(PLINK2_BINARY), "--bfile", str(GNOMON_NATIVE_PREFIX),
            "--score", str(gnomon_reformatted_file), "1", "2", "3", "header", "no-mean-imputation",
            "--out", str(plink2_out_prefix)
        ]
        plink2_result = run_and_measure(cmd_plink2, f"plink2_{pgs_id}", plink2_out_prefix.with_suffix(".log"))
        all_perf_results.append(plink2_result)

        # --- 3. Run PLINK1 on its compatible dataset ---
        df_p1 = pd.read_csv(gnomon_reformatted_file, sep='\t')
        df_p1['snp_id'] = 'chr' + df_p1['snp_id'].str.replace(':', '_', regex=False)
        p1_compat_score_file = CI_WORKDIR / f"{pgs_id}_p1_compat.tsv"
        df_p1.to_csv(p1_compat_score_file, sep='\t', index=False)
        
        plink1_out_prefix = CI_WORKDIR / f"plink1_{pgs_id}"
        # Use 'sum' to get a simple sum of scores, which is directly comparable to gnomon.
        cmd_plink1 = [
            str(PLINK1_BINARY), "--bfile", str(P1_COMPAT_PREFIX),
            "--score", str(p1_compat_score_file), "1", "2", "3", "header", "sum",
            "--out", str(plink1_out_prefix)
        ]
        plink1_result = run_and_measure(cmd_plink1, f"plink1_{pgs_id}", plink1_out_prefix.with_suffix(".log"))
        all_perf_results.append(plink1_result)
        
        # --- 4. Validate Outputs ---
        if not (gnomon_result['success'] and plink1_result['success'] and plink2_result['success']):
            print(f"\nSkipping validation for {pgs_id} due to tool execution failure(s).")
            if not plink1_result['success']: failed_tests.append(f"{pgs_id} (plink1_execution_failed)")
            if not plink2_result['success']: failed_tests.append(f"{pgs_id} (plink2_execution_failed)")
            continue

        print_header(f"VALIDATING OUTPUTS for {pgs_id}", char='~')
        try:
            gnomon_df = pd.read_csv(gnomon_final_score_file, sep='\t').rename(columns={"#IID": "IID"}).set_index("IID")
            plink2_df = pd.read_csv(plink2_out_prefix.with_suffix(".sscore"), sep='\t').rename(columns={"#IID": "IID"}).set_index("IID")
            plink1_df = pd.read_csv(plink1_out_prefix.with_suffix(".profile"), sep=r'\s+', engine='python').set_index("IID")

            merged = pd.DataFrame({
                'gnomon': gnomon_df[find_score_column(gnomon_df, 'gnomon')],
                'plink1': plink1_df[find_score_column(plink1_df, 'plink1')],
                'plink2': plink2_df[find_score_column(plink2_df, 'plink2')]
            })
            
            print("\nComparing scores from Gnomon, PLINK1, and PLINK2:")
            print(merged.head(15).to_markdown(floatfmt=".6g"))
            
            print("\n--- Score Correlations ---")
            corr_matrix = merged.corr()
            print(corr_matrix.to_markdown(floatfmt=".8f"))
            
            # Check for Gnomon bug where all scores are zero
            if (merged['gnomon'] == 0).all():
                failed_tests.append(f"{pgs_id} (gnomon_all_zeros_bug)")
                print(f"❌ CRITICAL FAILURE: Gnomon produced all-zero scores for {pgs_id}.")

            # Check for numerical identity
            if not np.isclose(merged['gnomon'], merged['plink2'], atol=NUMERICAL_TOLERANCE).all():
                failed_tests.append(f"{pgs_id} (Gnomon_vs_PLINK2_mismatch)")
            else:
                print(f"\n✅ NUMERICAL IDENTITY: Gnomon == PLINK2 (Tolerance: {NUMERICAL_TOLERANCE})")

            if not np.isclose(merged['gnomon'], merged['plink1'], atol=NUMERICAL_TOLERANCE).all():
                failed_tests.append(f"{pgs_id} (Gnomon_vs_PLINK1_mismatch)")
            else:
                print(f"✅ NUMERICAL IDENTITY: Gnomon == PLINK1 (Tolerance: {NUMERICAL_TOLERANCE})")
            
            # Check for high correlation as a sanity check
            if corr_matrix.loc['gnomon', 'plink1'] < CORR_THRESHOLD:
                failed_tests.append(f"{pgs_id} (gnomon_vs_plink1_low_correlation)")
                print(f"📉 WARNING: Gnomon vs PLINK1 correlation is below threshold ({CORR_THRESHOLD})")

        except Exception as e:
            failed_tests.append(f"{pgs_id} (validation_script_error: {e})")
            print(f"❌ An error occurred during the validation step: {e}")

    # --- Final Summary ---
    print_header("PERFORMANCE SUMMARY")
    if all_perf_results:
        results_df = pd.DataFrame(all_perf_results)
        results_df['tool_base'] = results_df['tool'].apply(lambda x: x.split('_')[0])
        summary = results_df[results_df['success']].groupby('tool_base').agg(
            mean_time_sec=('time_sec', 'mean'),
            std_time_sec=('time_sec', 'std'),
            mean_mem_mb=('peak_mem_mb', 'mean'),
            std_mem_mb=('peak_mem_mb', 'std')
        ).reset_index()
        if not summary.empty:
            print(summary.to_markdown(index=False, floatfmt=".3f"))
        else:
            print("No tools completed successfully. Cannot generate performance summary.")
    
    # --- Final CI Check ---
    if failed_tests:
        print_header("CI CHECK FAILED")
        print("❌ One or more tests did not pass. Failed components:")
        for test in sorted(list(set(failed_tests))):
            print(f"  - {test}")
        sys.exit(1)
    else:
        print_header("CI CHECK PASSED")
        print("🎉 All correctness and performance tests completed successfully.")
        sys.exit(0)

if __name__ == "__main__":
    main()
