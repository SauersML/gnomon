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
# We now use a more explicit naming scheme for clarity
ORIGINAL_PLINK_PREFIX = CI_WORKDIR / "chr22_subset50_original"
RENAMED_PLINK_PREFIX = CI_WORKDIR / "chr22_subset50_renamed"
CLEANED_PLINK_PREFIX = CI_WORKDIR / "chr22_subset50_cleaned"


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
            processed_match = re.search(r"--score: (\d+) variants processed", content)
            if processed_match: log_info['variants_processed'] = int(processed_match.group(1))
            processed_match_p1 = re.search(r"(\d+) variants loaded from --score file", content)
            if processed_match_p1: log_info['variants_processed'] = int(processed_match_p1.group(1))
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
    """Prepares the CI workspace by downloading and setting up all required artifacts."""
    print_header("ENVIRONMENT SETUP")
    CI_WORKDIR.mkdir(exist_ok=True)
    
    download_and_extract(PLINK1_URL, CI_WORKDIR)
    download_and_extract(PLINK2_URL, CI_WORKDIR)
    PLINK1_BINARY.chmod(0o755)
    PLINK2_BINARY.chmod(0o755)
    
    temp_prefix = CI_WORKDIR / "chr22_subset50"
    for zip_name, final_name in GENOTYPE_FILES.items():
        download_and_extract(f"{GENOTYPE_URL_BASE}{zip_name}?raw=true", CI_WORKDIR)
        (temp_prefix.parent / final_name).rename(ORIGINAL_PLINK_PREFIX.with_suffix(f".{final_name.split('.')[-1]}"))
        
    for url in PGS_SCORES.values():
        download_and_extract(url, CI_WORKDIR)

    print_header("DATA PRE-PROCESSING: A ROBUST, TOOL-DRIVEN WORKFLOW")
    try:
        # Step 1: RENAME variants to chr:pos format using PLINK. This creates a new, synchronized fileset.
        print("Step 1: Renaming variants to 'chr:pos' format using PLINK...")
        cmd_rename = [
            str(PLINK1_BINARY), # Using PLINK1 for --set-missing-var-ids
            "--bfile", str(ORIGINAL_PLINK_PREFIX),
            # The template '@:#' means 'chromosome:position'
            "--set-missing-var-ids", "@:#",
            "--make-bed",
            "--out", str(RENAMED_PLINK_PREFIX)
        ]
        # We need to run this on the original data, which has '.' as missing IDs
        # To make it robust, we create a temporary bim file with all IDs set to '.'
        original_bim = pd.read_csv(ORIGINAL_PLINK_PREFIX.with_suffix('.bim'), sep='\\s+', header=None)
        original_bim[1] = '.'
        temp_bim_path = CI_WORKDIR / "temp_for_rename.bim"
        original_bim.to_csv(temp_bim_path, sep='\t', header=False, index=False)
        
        cmd_rename_p2 = [
             str(PLINK2_BINARY),
             "--bed", str(ORIGINAL_PLINK_PREFIX.with_suffix('.bed')),
             "--bim", str(temp_bim_path), # Use the temp bim with '.' for IDs
             "--fam", str(ORIGINAL_PLINK_PREFIX.with_suffix('.fam')),
             "--set-all-var-ids", "@:#",
             "--make-bed",
             "--out", str(RENAMED_PLINK_PREFIX)
        ]
        result_rename = subprocess.run(cmd_rename_p2, check=True, capture_output=True, text=True)
        print("✅ Renaming successful.")
        inspect_file_head(RENAMED_PLINK_PREFIX.with_suffix(".bim"), title="Verifying Renamed .bim File")

        # Step 2: DE-DUPLICATE the renamed fileset using PLINK's dedicated command.
        print("\nStep 2: De-duplicating the renamed fileset...")
        cmd_dedup = [
            str(PLINK2_BINARY),
            "--bfile", str(RENAMED_PLINK_PREFIX),
            # This mode finds variants with the same ID and keeps only the first one.
            "--rm-dup", "force-first",
            "--make-bed",
            "--out", str(CLEANED_PLINK_PREFIX)
        ]
        result_dedup = subprocess.run(cmd_dedup, check=True, capture_output=True, text=True)
        print("✅ De-duplication successful. The data is now clean and consistent.")
        inspect_file_head(CLEANED_PLINK_PREFIX.with_suffix(".bim"), title="Verifying Final Cleaned .bim File")

    except Exception as e:
        print(f"❌ FAILED to pre-process genotype data: {e}")
        if 'result_rename' in locals() and result_rename.returncode != 0:
            print("--- RENAME STEP FAILED ---")
            print(result_rename.stderr)
        if 'result_dedup' in locals() and result_dedup.returncode != 0:
            print("--- DE-DUPLICATION STEP FAILED ---")
            print(result_dedup.stderr)
        sys.exit(1)

def find_reformatted_file(original_pgs_path: Path) -> Path:
    expected_name = f"{original_pgs_path.stem}.gnomon_format.tsv"
    expected_path = original_pgs_path.parent / expected_name
    if not expected_path.exists():
        print(f"❌ CRITICAL: Could not find the expected reformatted file at: {expected_path}")
        return None
    print(f"Found Gnomon's reformatted score file: {expected_path.name}")
    inspect_file_head(expected_path, title="Verifying reformatted score file")
    return expected_path

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
        
        cmd_gnomon = [str(GNOMON_BINARY), "--score", str(original_score_file), str(CLEANED_PLINK_PREFIX)]
        gnomon_result = run_and_measure(cmd_gnomon, f"gnomon_{pgs_id}", CI_WORKDIR / f"gnomon_{pgs_id}.log")
        all_perf_results.append(gnomon_result)
        
        if not gnomon_result['success']:
            failed_tests.append(f"{pgs_id} (gnomon)"); continue

        reformatted_file = find_reformatted_file(original_score_file)
        if reformatted_file is None:
            failed_tests.append(f"{pgs_id} (gnomon_reformat)"); continue

        pfile_prefix = CLEANED_PLINK_PREFIX.with_suffix("")
        if not pfile_prefix.with_suffix(".pgen").exists():
             print_header("One-time PLINK2 setup: creating .pfile from CLEANED data", char='*')
             cmd_make_pgen = [str(PLINK2_BINARY), "--bfile", str(CLEANED_PLINK_PREFIX), "--make-pgen", "--out", str(pfile_prefix)]
             subprocess.run(cmd_make_pgen, check=True, capture_output=True, text=True)
        
        plink2_out_prefix = CI_WORKDIR / f"plink2_{pgs_id}"
        cmd_plink2 = [str(PLINK2_BINARY), "--pfile", str(pfile_prefix), "--score", str(reformatted_file), "1", "2", "3", "header", "--out", str(plink2_out_prefix)]
        plink2_result = run_and_measure(cmd_plink2, f"plink2_{pgs_id}", plink2_out_prefix.with_suffix(".log"))
        all_perf_results.append(plink2_result)

        plink1_out_prefix = CI_WORKDIR / f"plink1_{pgs_id}"
        cmd_plink1 = [str(PLINK1_BINARY), "--bfile", str(CLEANED_PLINK_PREFIX), "--score", str(reformatted_file), "1", "2", "3", "header", "sum", "--out", str(plink1_out_prefix)]
        plink1_result = run_and_measure(cmd_plink1, f"plink1_{pgs_id}", plink1_out_prefix.with_suffix(".log"))
        all_perf_results.append(plink1_result)
        
        gnomon_out_file = CLEANED_PLINK_PREFIX.with_suffix(".sscore")
        
        if gnomon_result['success'] and plink1_result['success'] and plink2_result['success']:
            print_header(f"VALIDATING OUTPUTS for {pgs_id}", char='~')
            plink2_out_file = plink2_out_prefix.with_suffix(".sscore")
            plink1_out_file = plink1_out_prefix.with_suffix(".profile")
            
            try:
                gnomon_df = pd.read_csv(gnomon_out_file, sep='\t').rename(columns={"#IID": "IID"}).set_index("IID")
                plink2_df = pd.read_csv(plink2_out_file, sep='\t').rename(columns={"#IID": "IID"}).set_index("IID")
                plink1_df = pd.read_csv(plink1_out_file, delim_whitespace=True).set_index("IID")
                
                score_name_gnomon = gnomon_df.columns[0]
                score_name_plink2 = f"{score_name_gnomon}_SUM"

                merged = gnomon_df.join(plink2_df[[score_name_plink2]]).join(plink1_df[['SCORE']])
                merged.columns = ['gnomon', 'plink2', 'plink1']
                
                print(merged.head(15).to_string())
                
                is_close_p2 = np.isclose(merged['gnomon'], merged['plink2'], atol=NUMERICAL_TOLERANCE, rtol=NUMERICAL_TOLERANCE)
                if not is_close_p2.all(): failed_tests.append(f"{pgs_id} (Gnomon!=PLINK2)")
                else: print(f"✅ NUMERICAL IDENTITY: Gnomon == PLINK2")

                is_close_p1 = np.isclose(merged['gnomon'], merged['plink1'], atol=NUMERICAL_TOLERANCE, rtol=NUMERICAL_TOLERANCE)
                if not is_close_p1.all(): failed_tests.append(f"{pgs_id} (Gnomon!=PLINK1)")
                else: print(f"✅ NUMERICAL IDENTITY: Gnomon == PLINK1")
            except Exception as e:
                failed_tests.append(f"{pgs_id} (validation_error: {e})")
        else:
            print(f"Skipping validation for {pgs_id} due to tool failure(s).")
            if not plink1_result['success']: failed_tests.append(f"{pgs_id} (plink1)")
            if not plink2_result['success']: failed_tests.append(f"{pgs_id} (plink2)")

    print_header("PERFORMANCE SUMMARY")
    if all_perf_results:
        results_df = pd.DataFrame(all_perf_results)
        results_df['pgs_id'] = results_df['tool'].apply(lambda x: x.split('_')[1])
        results_df['tool_base'] = results_df['tool'].apply(lambda x: x.split('_')[0])
        summary = results_df[results_df['success']].groupby('tool_base').agg(
            mean_time_sec=('time_sec', 'mean'), std_time_sec=('time_sec', 'std'),
            mean_mem_mb=('peak_mem_mb', 'mean'), std_mem_mb=('peak_mem_mb', 'std')
        ).reset_index()
        print(summary.to_markdown(index=False, floatfmt=".3f"))
        gnomon_stats = summary[summary['tool_base'] == 'gnomon']
        plink2_stats = summary[summary['tool_base'] == 'plink2']
        if not gnomon_stats.empty and not plink2_stats.empty:
            time_factor = plink2_stats['mean_time_sec'].iloc[0] / gnomon_stats['mean_time_sec'].iloc[0]
            print(f"\nTime: Gnomon is {time_factor:.2f}x faster than PLINK2 on average.")
    
    if failed_tests:
        print_header("CI CHECK FAILED")
        print("❌ One or more tests did not pass. Failed components:")
        for test in sorted(list(set(failed_tests))): print(f"  - {test}")
        sys.exit(1)
    else:
        print_header("CI CHECK PASSED")
        print("🎉 All correctness and performance tests completed successfully.")
        sys.exit(0)
