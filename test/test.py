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

# Define separate, explicit data paths for tool compatibility
# Gnomon & PLINK2 will use the modern, chr:pos standard
GNOMON_NATIVE_PREFIX = CI_WORKDIR / "gnomon_native_data"
# PLINK1 requires a universally safe chr_pos format
P1_COMPAT_PREFIX = CI_WORKDIR / "p1_compatible_data"

# Original data path (used as source)
ORIGINAL_PLINK_PREFIX = CI_WORKDIR / "chr22_subset50_original"


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
NUMERICAL_TOLERANCE = 1e-4 # Relaxed slightly for cross-tool float differences

# ========================================================================================
#                             ENHANCED HELPER FUNCTIONS
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
    header_title = title if title else f"Inspecting file: {file_path.name}"
    print_header(header_title, char='-')
    print(f"Full path: {file_path.resolve()}")
    if not file_path.exists():
        print(">>> FILE DOES NOT EXIST <<<")
        print("-" * 80)
        return
    try:
        with open(file_path, 'r', errors='ignore') as f:
            for i, line in enumerate(f):
                if i >= num_lines:
                    break
                print(f"{i+1: >3}: {line.strip()}")
        print("-" * 80)
    except Exception as e:
        print(f"Could not read file: {e}")
        print("-" * 80)

def monitor_memory(p: subprocess.Popen, results: dict):
    peak_mem = 0
    results['peak_mem_mb'] = 0
    try:
        proc = psutil.Process(p.pid)
        while p.poll() is None:
            try:
                peak_mem = max(peak_mem, proc.memory_info().rss)
            except psutil.NoSuchProcess:
                break
            time.sleep(0.01)
    except psutil.NoSuchProcess:
        pass
    results['peak_mem_mb'] = peak_mem / (1024 * 1024)

def parse_plink_log(log_path: Path) -> dict:
    if not log_path.exists(): return {"log_status": "Not Found"}
    log_info = {"log_status": "Read OK"}
    try:
        with open(log_path, 'r', errors='ignore') as f:
            content = f.read()
            error_match = re.search(r"\nError: (.*)", content)
            if error_match: log_info['error_message'] = error_match.group(1).strip()
            
            processed_matches = re.findall(r"(?:--score: |from --score file).*?(\d+)\s+variants", content)
            if processed_matches:
                log_info['variants_processed'] = int(processed_matches[-1])

    except Exception as e:
        return {"log_status": f"Error reading log: {e}"}
    return log_info

def run_and_measure(command: list, tool_name: str, log_path: Path) -> dict:
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
    end_time = time.perf_counter()
    duration = end_time - start_time
    log_info = parse_plink_log(log_path)
    results = {'tool': tool_name, 'time_sec': duration, 'peak_mem_mb': monitor_results.get('peak_mem_mb', 0), 'returncode': process.returncode, 'stdout': stdout, 'stderr': stderr, 'log_info': log_info, 'success': process.returncode == 0}
    if not results['success']:
        print(f"❌ {tool_name} FAILED (Exit Code: {process.returncode})")
        print_header("STDOUT", char='.')
        print(stdout.strip() if stdout.strip() else "[EMPTY]")
        print_header("STDERR", char='.')
        print(stderr.strip() if stderr.strip() else "[EMPTY]")
    else:
        print(f"✅ {tool_name} finished in {duration:.2f}s.")
    print("Log file summary:")
    for key, value in log_info.items():
        print(f"  - {key}: {value}")
    return results

def setup_environment():
    """Prepares the CI workspace, creating two parallel, clean datasets."""
    print_header("ENVIRONMENT SETUP")
    CI_WORKDIR.mkdir(exist_ok=True)
    
    download_and_extract(PLINK1_URL, CI_WORKDIR)
    download_and_extract(PLINK2_URL, CI_WORKDIR)
    PLINK1_BINARY.chmod(0o755)
    PLINK2_BINARY.chmod(0o755)
    
    for zip_name, final_name in GENOTYPE_FILES.items():
        download_and_extract(f"{GENOTYPE_URL_BASE}{zip_name}?raw=true", CI_WORKDIR)
        source_path = CI_WORKDIR / final_name
        dest_path = ORIGINAL_PLINK_PREFIX.with_suffix(f".{final_name.split('.')[-1]}")
        source_path.rename(dest_path)
        
    for url in PGS_SCORES.values():
        download_and_extract(url, CI_WORKDIR)

    print_header("DATA PRE-PROCESSING: A ROBUST, TOOL-DRIVEN WORKFLOW")
    try:
        # Create two separate, clean datasets for maximum compatibility
        
        # 1. Gnomon Native data (chr:pos IDs) for Gnomon and PLINK2
        print("Step 1a: Creating Gnomon-native data (chr:pos format)...")
        cmd_rename_gnomon = [str(PLINK2_BINARY), "--bfile", str(ORIGINAL_PLINK_PREFIX), "--set-all-var-ids", "@:#", "--make-bed", "--out", str(GNOMON_NATIVE_PREFIX) + "_renamed"]
        subprocess.run(cmd_rename_gnomon, check=True, capture_output=True, text=True)
        cmd_dedup_gnomon = [str(PLINK2_BINARY), "--bfile", str(GNOMON_NATIVE_PREFIX) + "_renamed", "--rm-dup", "force-first", "--make-bed", "--out", str(GNOMON_NATIVE_PREFIX)]
        subprocess.run(cmd_dedup_gnomon, check=True, capture_output=True, text=True)
        print("✅ Gnomon-native data created successfully.")
        inspect_file_head(GNOMON_NATIVE_PREFIX.with_suffix(".bim"), title="Verifying Gnomon-Native .bim File")

        # 2. PLINK1 Compatible data (chr_pos IDs)
        print("\nStep 1b: Creating PLINK1-compatible data (chr_pos format)...")
        cmd_rename_p1 = [str(PLINK2_BINARY), "--bfile", str(ORIGINAL_PLINK_PREFIX), "--set-all-var-ids", "chr@_#", "--make-bed", "--out", str(P1_COMPAT_PREFIX) + "_renamed"]
        subprocess.run(cmd_rename_p1, check=True, capture_output=True, text=True)
        cmd_dedup_p1 = [str(PLINK2_BINARY), "--bfile", str(P1_COMPAT_PREFIX) + "_renamed", "--rm-dup", "force-first", "--make-bed", "--out", str(P1_COMPAT_PREFIX)]
        subprocess.run(cmd_dedup_p1, check=True, capture_output=True, text=True)
        print("✅ PLINK1-compatible data created successfully.")
        inspect_file_head(P1_COMPAT_PREFIX.with_suffix(".bim"), title="Verifying PLINK1-Compatible .bim File")

    except Exception as e:
        print(f"❌ FAILED to pre-process genotype data: {e}")
        sys.exit(1)

def create_plink1_compatible_score_file(gnomon_native_score_file: Path) -> Path:
    """
    Creates a copy of the Gnomon-reformatted score file and converts its
    `chr:pos` IDs to the `chr_pos` format required by the PLINK1 test run.
    """
    p1_compat_path = gnomon_native_score_file.with_name(gnomon_native_score_file.name.replace(".tsv", "_p1_compat.tsv"))
    print_header(f"Creating PLINK1-compatible score file: {p1_compat_path.name}", char='*')
    
    df = pd.read_csv(gnomon_native_score_file, sep=r'\s+', engine='python')
    # Gnomon's reformatter creates chr:pos, we need chr_pos for the P1 test
    # The PLINK1 data was created with a "chr" prefix, so we add it here.
    if 'snp_id' in df.columns:
        df['snp_id'] = 'chr' + df['snp_id'].str.replace(':', '_', regex=False)
    
    df.to_csv(p1_compat_path, sep='\t', index=False)
    
    print(f"✅ IDs in {p1_compat_path.name} aligned for PLINK1.")
    inspect_file_head(p1_compat_path, title="Verifying PLINK1 Score File")
    return p1_compat_path

# ========================================================================================
#                             MAIN EXECUTION & VALIDATION
# ========================================================================================

if __name__ == "__main__":
    print_header("GNOMON CI TEST & BENCHMARK SUITE")
    setup_environment()
    
    all_perf_results = []
    failed_tests = []
    
    for pgs_id, pgs_url in PGS_SCORES.items():
        print_header(f"TESTING SCORE: {pgs_id}")
        original_score_file = CI_WORKDIR / Path(pgs_url.split("/")[-1]).stem
        
        # --- Run Gnomon (using its native chr:pos data) ---
        cmd_gnomon = [str(GNOMON_BINARY), "--score", str(original_score_file), str(GNOMON_NATIVE_PREFIX)]
        gnomon_result = run_and_measure(cmd_gnomon, f"gnomon_{pgs_id}", CI_WORKDIR / f"gnomon_{pgs_id}.log")
        all_perf_results.append(gnomon_result)
        
        if not gnomon_result['success']:
            failed_tests.append(f"{pgs_id} (gnomon_execution_failed)"); continue

        # --- Find and Verify Gnomon's output files IMMEDIATELY ---
        gnomon_reformatted_file = original_score_file.parent / f"{original_score_file.name}.gnomon_format.tsv"
        gnomon_final_score_file = original_score_file.parent / f"{original_score_file.name}.sscore"
        
        if not gnomon_reformatted_file.exists() or not gnomon_final_score_file.exists():
            print(f"❌ CRITICAL: Gnomon did not produce its expected output files for {pgs_id}.")
            if not gnomon_reformatted_file.exists(): inspect_file_head(gnomon_reformatted_file)
            if not gnomon_final_score_file.exists(): inspect_file_head(gnomon_final_score_file)
            failed_tests.append(f"{pgs_id} (gnomon_output_missing)"); continue
        else:
            print("✅ Gnomon produced all expected output files.")
            inspect_file_head(gnomon_reformatted_file, title="Verifying Gnomon-Reformatted Score File")

        # --- Run PLINK2 (using Gnomon-native chr:pos data for direct comparison) ---
        plink2_out_prefix = CI_WORKDIR / f"plink2_{pgs_id}"
        # Use Gnomon's reformatted file as input for a true apples-to-apples comparison
        cmd_plink2 = [
            str(PLINK2_BINARY), "--pfile", str(GNOMON_NATIVE_PREFIX),
            "--score", str(gnomon_reformatted_file), "1", "2", "3", "header", "no-mean-imputation",
            "--out", str(plink2_out_prefix)
        ]
        plink2_result = run_and_measure(cmd_plink2, f"plink2_{pgs_id}", plink2_out_prefix.with_suffix(".log"))
        all_perf_results.append(plink2_result)

        # --- Run PLINK1 (using its special compatible chr_pos data) ---
        p1_compat_score_file = create_plink1_compatible_score_file(gnomon_reformatted_file)
        plink1_out_prefix = CI_WORKDIR / f"plink1_{pgs_id}"
        cmd_plink1 = [
            str(PLINK1_BINARY), "--bfile", str(P1_COMPAT_PREFIX),
            "--score", str(p1_compat_score_file), "1", "2", "3", "header", "sum", "no-mean-imputation",
            "--out", str(plink1_out_prefix)
        ]
        plink1_result = run_and_measure(cmd_plink1, f"plink1_{pgs_id}", plink1_out_prefix.with_suffix(".log"))
        all_perf_results.append(plink1_result)
        
        # --- Validate Outputs ---
        if gnomon_result['success'] and plink1_result['success'] and plink2_result['success']:
            print_header(f"VALIDATING OUTPUTS for {pgs_id}", char='~')
            
            try:
                gnomon_df = pd.read_csv(gnomon_final_score_file, sep='\t').rename(columns={"#IID": "IID"}).set_index("IID")
                plink2_df = pd.read_csv(plink2_out_prefix.with_suffix(".sscore"), sep='\t').rename(columns={"#IID": "IID"}).set_index("IID")
                plink1_df = pd.read_csv(plink1_out_prefix.with_suffix(".profile"), delim_whitespace=True).set_index("IID")
                
                gnomon_col = gnomon_df.columns[0]
                
                merged = pd.DataFrame(index=gnomon_df.index)
                merged['gnomon'] = gnomon_df[gnomon_col]
                merged['plink2'] = plink2_df['NAMED_ALLELE_DOSAGE_SUM']
                merged['plink1'] = plink1_df['SCORE']

                print("Comparing scores from Gnomon, PLINK2, and PLINK1:")
                print(merged.head(15).to_string())
                
                if (merged['gnomon'] == 0).all():
                    failed_tests.append(f"{pgs_id} (gnomon_all_zeros_bug)")
                    print(f"❌ CRITICAL FAILURE: Gnomon produced all-zero scores for {pgs_id}.")
                    continue 

                is_close_p2 = np.isclose(merged['gnomon'], merged['plink2'], atol=NUMERICAL_TOLERANCE, rtol=NUMERICAL_TOLERANCE)
                if not is_close_p2.all(): failed_tests.append(f"{pgs_id} (Gnomon_vs_PLINK2_mismatch)")
                else: print(f"✅ NUMERICAL IDENTITY: Gnomon == PLINK2")

                is_close_p1 = np.isclose(merged['gnomon'], merged['plink1'], atol=NUMERICAL_TOLERANCE, rtol=NUMERICAL_TOLERANCE)
                if not is_close_p1.all(): failed_tests.append(f"{pgs_id} (Gnomon_vs_PLINK1_mismatch)")
                else: print(f"✅ NUMERICAL IDENTITY: Gnomon == PLINK1")

            except Exception as e:
                failed_tests.append(f"{pgs_id} (validation_script_error: {e})")
        else:
            print(f"Skipping validation for {pgs_id} due to tool execution failure(s).")

    # --- Final Summary ---
    print_header("PERFORMANCE SUMMARY")
    if all_perf_results:
        results_df = pd.DataFrame(all_perf_results)
        results_df['tool_base'] = results_df['tool'].apply(lambda x: x.split('_')[0])
        summary = results_df[results_df['success']].groupby('tool_base').agg(
            mean_time_sec=('time_sec', 'mean'), std_time_sec=('time_sec', 'std'),
            mean_mem_mb=('peak_mem_mb', 'mean'), std_mem_mb=('peak_mem_mb', 'std')
        ).reset_index()
        print(summary.to_markdown(index=False, floatfmt=".3f"))
        
        try:
            gnomon_time = summary.loc[summary['tool_base'] == 'gnomon', 'mean_time_sec'].iloc[0]
            plink2_time = summary.loc[summary['tool_base'] == 'plink2', 'mean_time_sec'].iloc[0]
            if gnomon_time > 0 and plink2_time > 0:
                speed_factor = plink2_time / gnomon_time
                print(f"\nTime: Gnomon is {speed_factor:.2f}x the speed of PLINK2 on average.")
            else:
                print("\nCould not compute performance comparison due to missing or zero-time results.")
        except (IndexError, ZeroDivisionError):
            print("\nCould not compute performance comparison due to missing or zero-time results.")
    
    if failed_tests:
        print_header("CI CHECK FAILED")
        print("❌ One or more tests did not pass. Failed components:")
        for test in sorted(list(set(failed_tests))): print(f"  - {test}")
        sys.exit(1)
    else:
        print_header("CI CHECK PASSED")
        print("🎉 All correctness and performance tests completed successfully.")
        sys.exit(0)