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

CI_WORKDIR = Path("./ci_workdir")
GNOMON_BINARY = Path("./target/release/gnomon")
PLINK1_BINARY = CI_WORKDIR / "plink"
PLINK2_BINARY = CI_WORKDIR / "plink2"

GNOMON_NATIVE_PREFIX = CI_WORKDIR / "gnomon_native_data"
P1_COMPAT_PREFIX = CI_WORKDIR / "p1_compatible_data"
ORIGINAL_PLINK_PREFIX = CI_WORKDIR / "chr22_subset50_original"

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

def run_and_measure(command: list, tool_name: str) -> dict:
    """Runs a command, measures its time and memory, and captures results."""
    print_header(f"RUNNING: {tool_name}", char='-')
    print("Command:")
    print(f"  {' '.join(map(str, command))}\n")
    
    start_time = time.perf_counter()
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8')
    
    # Memory monitoring logic (simplified for brevity)
    peak_mem = 0
    try:
        proc = psutil.Process(process.pid)
        while process.poll() is None:
            try: peak_mem = max(peak_mem, proc.memory_info().rss)
            except psutil.NoSuchProcess: break
            time.sleep(0.01)
    except psutil.NoSuchProcess: pass
    
    stdout, stderr = process.communicate()
    duration = time.perf_counter() - start_time
    
    results = {
        'tool': tool_name, 'time_sec': duration,
        'peak_mem_mb': peak_mem / (1024 * 1024),
        'returncode': process.returncode, 'success': process.returncode == 0
    }

    if not results['success']:
        print(f"❌ {tool_name} FAILED (Exit Code: {process.returncode})")
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
        
    for url in PGS_SCORES.values():
        download_and_extract(url, CI_WORKDIR)

    print_header("DATA PRE-PROCESSING")
    try:
        # Create Gnomon/PLINK2-native data (chr:pos IDs)
        print("Step 1a: Creating Gnomon-native data (chr:pos format)...")
        cmd_dedup_gnomon = [
            str(PLINK2_BINARY), "--bfile", str(ORIGINAL_PLINK_PREFIX), 
            "--set-all-var-ids", "@:#", "--rm-dup", "force-first",
            "--make-bed", "--out", str(GNOMON_NATIVE_PREFIX)
        ]
        subprocess.run(cmd_dedup_gnomon, check=True, capture_output=True, text=True)

        # Create PLINK1-compatible data (chr_pos IDs)
        print("Step 1b: Creating PLINK1-compatible data (chr_pos format)...")
        cmd_dedup_p1 = [
            str(PLINK2_BINARY), "--bfile", str(ORIGINAL_PLINK_PREFIX),
            "--set-all-var-ids", "chr@_#", "--rm-dup", "force-first",
            "--make-bed", "--out", str(P1_COMPAT_PREFIX)
        ]
        subprocess.run(cmd_dedup_p1, check=True, capture_output=True, text=True)
        print("✅ Data pre-processing successful.")

    except subprocess.CalledProcessError as e:
        print(f"❌ FAILED to pre-process genotype data: {e.stderr}")
        sys.exit(1)

def find_score_column(df: pd.DataFrame, priority_list: list) -> str:
    """Intelligently finds the score column in a results DataFrame from a priority list."""
    for col in priority_list:
        if col in df.columns:
            return col
    # Fallback for dynamic PLINK2 columns
    for col in df.columns:
        if col.startswith('SCORE') and col.endswith('_AVG'):
            return col
    raise KeyError(f"Could not find any of the candidate score columns {priority_list} in the DataFrame.")


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
        gnomon_result = run_and_measure(cmd_gnomon, f"gnomon_{pgs_id}")
        all_perf_results.append(gnomon_result)
        
        if not gnomon_result['success']:
            failed_tests.append(f"{pgs_id} (gnomon_execution_failed)"); continue

        gnomon_reformatted_file = original_score_file.with_suffix(".gnomon_format.tsv")
        gnomon_final_score_file = original_score_file.with_suffix(".txt.sscore")
        
        if not gnomon_reformatted_file.exists() or not gnomon_final_score_file.exists():
            failed_tests.append(f"{pgs_id} (gnomon_output_missing)"); continue
        
        # --- 2. Run PLINK2 ---
        plink2_out_prefix = CI_WORKDIR / f"plink2_{pgs_id}"
        cmd_plink2 = [
            str(PLINK2_BINARY), "--bfile", str(GNOMON_NATIVE_PREFIX),
            "--score", str(gnomon_reformatted_file), "1", "2", "3", "header", "no-mean-imputation",
            "--out", str(plink2_out_prefix)
        ]
        plink2_result = run_and_measure(cmd_plink2, f"plink2_{pgs_id}")
        all_perf_results.append(plink2_result)

        # --- 3. Run PLINK1 ---
        df_p1 = pd.read_csv(gnomon_reformatted_file, sep='\t')
        df_p1['snp_id'] = 'chr' + df_p1['snp_id'].str.replace(':', '_', regex=False)
        p1_compat_score_file = CI_WORKDIR / f"{pgs_id}_p1_compat.tsv"
        df_p1.to_csv(p1_compat_score_file, sep='\t', index=False)
        
        plink1_out_prefix = CI_WORKDIR / f"plink1_{pgs_id}"
        cmd_plink1 = [
            str(PLINK1_BINARY), "--bfile", str(P1_COMPAT_PREFIX),
            "--score", str(p1_compat_score_file), "1", "2", "3", "header", "sum",
            "--out", str(plink1_out_prefix)
        ]
        plink1_result = run_and_measure(cmd_plink1, f"plink1_{pgs_id}")
        all_perf_results.append(plink1_result)
        
        # --- 4. Validate Outputs (Apples-to-Apples Comparison) ---
        if not (gnomon_result['success'] and plink1_result['success'] and plink2_result['success']):
            if not plink1_result['success']: failed_tests.append(f"{pgs_id} (plink1_execution_failed)")
            if not plink2_result['success']: failed_tests.append(f"{pgs_id} (plink2_execution_failed)")
            continue

        print_header(f"VALIDATING OUTPUTS for {pgs_id}", char='~')
        try:
            # Load all result files
            gnomon_df = pd.read_csv(gnomon_final_score_file, sep='\t').rename(columns={"#IID": "IID"}).set_index("IID")
            plink2_df = pd.read_csv(plink2_out_prefix.with_suffix(".sscore"), sep='\t').rename(columns={"#IID": "IID"}).set_index("IID")
            plink1_df = pd.read_csv(plink1_out_prefix.with_suffix(".profile"), sep=r'\s+', engine='python').set_index("IID")

            # --- IMPLEMENTATION OF THE PLAN ---
            # 1. Get the pure weighted sum from PLINK1. This is our ground truth.
            plink1_sum_col = find_score_column(plink1_df, ['SCORESUM', 'SCORE'])
            
            # 2. Derive the pure weighted sum from PLINK2's average score.
            plink2_avg_col = find_score_column(plink2_df, ['SCORE1_AVG'])
            # ALLELE_CT is 2 * num_variants. Divide by 2 to get num_variants.
            num_variants = plink2_df['ALLELE_CT'] / 2
            plink2_df['SCORE_SUM'] = plink2_df[plink2_avg_col] * num_variants

            # 3. Get the gnomon score.
            gnomon_score_col = find_score_column(gnomon_df, [pgs_id])
            
            # 4. Create a clean, merged DataFrame for comparison of SUMS.
            merged = pd.DataFrame({
                'gnomon': gnomon_df[gnomon_score_col],
                'plink1_sum': plink1_df[plink1_sum_col],
                'plink2_sum': plink2_df['SCORE_SUM']
            })
            
            print("\nComparing calculated weighted SUMS:")
            print(merged.head(15).to_markdown(floatfmt=".6g"))
            
            print("\n--- Score Correlations ---")
            corr_matrix = merged.corr()
            print(corr_matrix.to_markdown(floatfmt=".8f"))
            
            # Validation checks
            if (merged['gnomon'] == 0).all():
                failed_tests.append(f"{pgs_id} (gnomon_all_zeros_bug)")
                print(f"\n❌ CRITICAL FAILURE: Gnomon produced all-zero scores for {pgs_id}.")

            # Now, compare gnomon to the true sums from plink1 and plink2
            if not np.isclose(merged['gnomon'], merged['plink1_sum'], atol=NUMERICAL_TOLERANCE).all():
                failed_tests.append(f"{pgs_id} (Gnomon_vs_PLINK1_mismatch)")
            else:
                print(f"\n✅ NUMERICAL IDENTITY: Gnomon == PLINK1 (SUM)")

            if not np.isclose(merged['gnomon'], merged['plink2_sum'], atol=NUMERICAL_TOLERANCE).all():
                failed_tests.append(f"{pgs_id} (Gnomon_vs_PLINK2_mismatch)")
            else:
                print(f"✅ NUMERICAL IDENTITY: Gnomon == PLINK2 (SUM)")
            
            if corr_matrix.loc['plink1_sum', 'plink2_sum'] < CORR_THRESHOLD:
                failed_tests.append(f"{pgs_id} (plink1_vs_plink2_low_correlation)")
                print(f"📉 WARNING: PLINK1 vs PLINK2 sum correlation is unexpectedly low!")


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
            mean_mem_mb=('peak_mem_mb', 'mean')
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
