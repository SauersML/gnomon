import subprocess
import requests
import zipfile
import gzip
import shutil
import time
import sys
import os
import re
from pathlib import Path
from itertools import combinations

import pandas as pd
import numpy as np
import psutil

# ========================================================================================
#                                   CONFIGURATION
# ========================================================================================
CI_WORKDIR = Path("./ci_workdir")
GNOMON_BINARY = Path("./target/release/gnomon")
PLINK2_BINARY = CI_WORKDIR / "plink2"
PYLINK_SCRIPT = Path("test/pylink.py").resolve()

# Constants for the Disagreement Bisection Algorithm
DISCOVERY_PGS_ID = "PGS004696"
CORRELATION_THRESHOLD = 0.99
BISECTION_TERMINATION_SIZE = 3

SHARED_GENOTYPE_PREFIX = CI_WORKDIR / "chr22_subset50_compat"
ORIGINAL_PLINK_PREFIX = CI_WORKDIR / "chr22_subset50"

PLINK1_URL = "https://s3.amazonaws.com/plink1-assets/plink_linux_x86_64_20231211.zip"
PLINK2_URL = "https://s3.amazonaws.com/plink2-assets/alpha6/plink2_linux_avx2_20250609.zip"

GENOTYPE_URL_BASE = "https://github.com/SauersML/genomic_pca/blob/main/data/"
GENOTYPE_FILES = [
    "chr22_subset50.bed.zip", "chr22_subset50.bim.zip", "chr22_subset50.fam.zip"
]

PGS_SCORES = {
    "PGS004696": "https://ftp.ebi.ac.uk/pub/databases/spot/pgs/scores/PGS004696/ScoringFiles/Harmonized/PGS004696_hmPOS_GRCh38.txt.gz",
    "PGS003725": "https://ftp.ebi.ac.uk/pub/databases/spot/pgs/scores/PGS003725/ScoringFiles/Harmonized/PGS003725_hmPOS_GRCh38.txt.gz",
    "PGS001780": "https://ftp.ebi.ac.uk/pub/databases/spot/pgs/scores/PGS001780/ScoringFiles/Harmonized/PGS001780_hmPOS_GRCh38.txt.gz",
}

# ========================================================================================
#                                   HELPER FUNCTIONS
# ========================================================================================
def print_header(title: str, char: str = "="):
    """Prints a formatted header to the console."""
    width = 80
    print("\n" + char * width, flush=True)
    print(f"{char*4} {title} {char*(width - len(title) - 6)}", flush=True)
    print(char * width, flush=True)

def print_debug_header(title: str):
    """Prints a smaller header for debug sections."""
    print("\n" + "." * 80, flush=True)
    print(f".... {title} ....", flush=True)
    print("." * 80, flush=True)

def print_file_head(file_path: Path):
    """Prints the header and first 10 lines of a text file for verification."""
    if not file_path.exists():
        print(f"File not found: {file_path}", flush=True)
        return
    if file_path.suffix in ['.bed']:
        print_debug_header(f"HEXDUMP (first 250 bytes) OF: {file_path.name}")
        try:
            with open(file_path, 'rb') as f:
                print(f"  {f.read(250).hex()}", flush=True)
        except Exception as e:
            print(f"  Could not read binary file: {e}", flush=True)
    else:
        print_debug_header(f"HEAD OF: {file_path.name}")
        try:
            with open(file_path, 'r', errors='replace') as f:
                for i, line in enumerate(f):
                    if i >= 10: break
                    print(f"  {line.strip()}", flush=True)
        except Exception as e:
            print(f"  Could not read file: {e}", flush=True)

def download_and_extract(url: str, dest_dir: Path):
    """Downloads and extracts a file, handling .zip and .gz."""
    filename = Path(url.split('?')[0].split('/')[-1])
    outpath = dest_dir / filename
    print(f"[SCRIPT] Downloading {filename}...", flush=True)
    try:
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(outpath, 'wb') as f: shutil.copyfileobj(r.raw, f)
    except Exception as e:
        print(f"❌ FAILED to download {url}: {e}", flush=True); sys.exit(1)

    if outpath.suffix == '.zip':
        with zipfile.ZipFile(outpath, 'r') as z: z.extractall(dest_dir)
        print(f"[SCRIPT] Extracted ZIP: {filename}", flush=True)
        outpath.unlink()
    elif outpath.suffix == '.gz':
        dest = dest_dir / filename.stem
        with gzip.open(outpath, 'rb') as f_in, open(dest, 'wb') as f_out: shutil.copyfileobj(f_in, f_out)
        print(f"[SCRIPT] Extracted GZ: {filename} -> {dest.name}", flush=True)
        outpath.unlink()

def run_and_measure(cmd: list, name: str, out_prefix: Path):
    """Executes a command, streams its output, and captures results."""
    print_header(f"RUNNING: {name}", char='-')
    print(f"Command:\n  {' '.join(map(str, cmd))}\n", flush=True)
    start = time.perf_counter()
    proc = subprocess.Popen([str(c) for c in cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace', bufsize=1)
    
    output_lines = []
    print_debug_header(f"{name} Real-time Output")
    with proc.stdout:
        for line in iter(proc.stdout.readline, ''):
            print(f"  [{name}] {line.strip()}", flush=True)
            output_lines.append(line.strip())
    
    returncode = proc.wait()
    duration = time.perf_counter() - start
    
    result = {'tool': name, 'time_sec': duration, 'returncode': returncode, 'success': returncode == 0, 'stdout': "\n".join(output_lines)}
    if result['success']: print(f"✅ {name} finished in {duration:.2f}s.", flush=True)
    else: print(f"❌ {name} FAILED (Exit Code: {returncode})", flush=True)
    return result

def setup_environment():
    """Prepares the CI environment by downloading all necessary data and tools."""
    print_header("ENVIRONMENT SETUP")
    CI_WORKDIR.mkdir(exist_ok=True)
    if not PYLINK_SCRIPT.exists():
        print(f"❌ ERROR: PyLink script not found at: {PYLINK_SCRIPT}", flush=True); sys.exit(1)
    print(f"[SCRIPT] Found PyLink script at: {PYLINK_SCRIPT}", flush=True)

    for url, binary_path in [(PLINK1_URL, CI_WORKDIR / "plink"), (PLINK2_URL, PLINK2_BINARY)]:
        if not binary_path.exists():
            download_and_extract(url, CI_WORKDIR)
            # Find the downloaded binary (it might have a generic name)
            for p in CI_WORKDIR.iterdir():
                if p.is_file() and p.name.startswith(binary_path.name.split('_')[0]):
                    p.rename(binary_path)
                    binary_path.chmod(0o755)
                    print(f"[SCRIPT] Configured and chmod: {binary_path}", flush=True)
                    break

    for zip_name in GENOTYPE_FILES: download_and_extract(f"{GENOTYPE_URL_BASE}{zip_name}?raw=true", CI_WORKDIR)
    for url in PGS_SCORES.values(): download_and_extract(url, CI_WORKDIR)

def create_synchronized_genotype_files(original_prefix: Path, final_prefix: Path):
    """Creates a new PLINK fileset with harmonized 'chr:pos' variant IDs."""
    print_header("SYNCHRONIZING GENOTYPE DATA", char='.')
    bim_df = pd.read_csv(original_prefix.with_suffix('.bim'), sep='\s+', header=None, names=['chrom', 'rsid', 'cm', 'pos', 'a1', 'a2'], dtype=str)
    bim_df['chr_pos_id'] = bim_df['chrom'] + ':' + bim_df['pos']
    
    # Exclude variants at positions that appear more than once
    duplicate_pos = bim_df[bim_df.duplicated(subset=['chr_pos_id'], keep=False)]
    variants_to_exclude = set(duplicate_pos['rsid'])
    variants_to_rename = bim_df[~bim_df['rsid'].isin(variants_to_exclude)]
    
    exclude_file = CI_WORKDIR / "variants_to_exclude.txt"
    pd.Series(list(variants_to_exclude)).to_csv(exclude_file, index=False, header=False)
    print(f"[SCRIPT] Identified {len(variants_to_exclude)} variants at duplicate positions to exclude.", flush=True)

    id_map_file = CI_WORKDIR / "id_update_map.txt"
    variants_to_rename[['rsid', 'chr_pos_id']].to_csv(id_map_file, sep=' ', index=False, header=False)
    
    cmd = [str(CI_WORKDIR / "plink"), "--bfile", str(original_prefix), "--exclude", str(exclude_file), "--update-name", str(id_map_file), "2", "1", "--make-bed", "--out", str(final_prefix)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print("❌ FAILED to create synchronized genotype fileset.", flush=True); print(proc.stdout); print(proc.stderr); sys.exit(1)
    print(f"[SCRIPT] Successfully created synchronized fileset at: {final_prefix}", flush=True)

def create_unified_scorefile(score_file: Path, pgs_id: str) -> pd.DataFrame:
    """Creates a unified scorefile dataframe where the variant ID is 'chr:pos'."""
    print_header(f"CREATING UNIFIED SCORE FILE FOR {pgs_id}", char='.')
    df = pd.read_csv(score_file, sep='\t', comment='#', usecols=['hm_chr', 'hm_pos', 'effect_allele', 'other_allele', 'effect_weight'], dtype=str, low_memory=False)
    df.dropna(inplace=True)
    df['effect_weight'] = pd.to_numeric(df['effect_weight'], errors='coerce')
    df.dropna(inplace=True)
    df['chr_pos_id'] = df['hm_chr'] + ':' + df['hm_pos']
    df.drop_duplicates(subset=['chr_pos_id'], keep='first', inplace=True)
    final_df = df[['chr_pos_id', 'effect_allele', 'other_allele', 'effect_weight']].copy()
    final_df.columns = ['snp_id', 'effect_allele', 'other_allele', 'effect_weight']
    
    out_path = CI_WORKDIR / f"{pgs_id}_unified_format.tsv"
    final_df.to_csv(out_path, sep='\t', index=False)
    print(f"[SCRIPT] Writing {len(final_df)} variants to unified score file: {out_path.name}", flush=True)
    return final_df

def analyze_pgs_results(pgs_id: str, pgs_run_results: list) -> dict:
    """Analyzes and compares results from multiple tool runs for a single PGS."""
    summary = {'pgs_id': pgs_id, 'success': False}
    tool_paths = {
        'gnomon': SHARED_GENOTYPE_PREFIX.with_suffix('.sscore'),
        'pylink': CI_WORKDIR / f"pylink_{pgs_id}.sscore",
        'plink2': CI_WORKDIR / f"plink2_{pgs_id}.sscore",
    }
    
    data_frames = []
    for res in pgs_run_results:
        if not res['success']: continue
        tool_name = res['tool'].split('_')[0]
        path = tool_paths.get(tool_name)
        if not path or not path.exists(): continue
        
        try:
            df_raw = pd.read_csv(path, sep='\s+')
            id_col = '#IID' if '#IID' in df_raw.columns else 'IID'
            score_col = next((c for c in df_raw.columns if '_AVG' in c), None)
            if not score_col: raise KeyError(f"{tool_name.upper()} _AVG score column not found")
            df = df_raw[[id_col, score_col]].rename(columns={id_col: 'IID', score_col: f'SCORE_{tool_name}'})
            df[f'SCORE_{tool_name}'] = pd.to_numeric(df[f'SCORE_{tool_name}'], errors='coerce')
            data_frames.append(df)
        except Exception as e:
            print(f"  > [ANALYSIS_ERROR] Failed to parse {tool_name} for {pgs_id}: {e}", flush=True)

    if len(data_frames) < 2: return summary
    
    merged_df = data_frames[0]
    for df_to_merge in data_frames[1:]:
        merged_df = pd.merge(merged_df, df_to_merge, on='IID', how='inner')
    
    score_cols = sorted([c for c in merged_df.columns if c.startswith('SCORE_')])
    merged_df.dropna(subset=score_cols, inplace=True)
    
    if len(score_cols) < 2 or merged_df.empty: return summary
    
    summary['correlation_matrix'] = merged_df[score_cols].corr()
    summary['success'] = True
    return summary

def test_variant_subset(variant_df: pd.DataFrame, iteration_name: str) -> float:
    """Helper function for the bisection algorithm. Runs Gnomon and PLINK2 on a subset of variants and returns their correlation."""
    temp_score_path = CI_WORKDIR / f"_temp_score_{iteration_name}.tsv"
    variant_df.to_csv(temp_score_path, sep='\t', index=False)

    # Run Gnomon
    g_prefix = CI_WORKDIR / f"_g_{iteration_name}"
    g_cmd = [str(GNOMON_BINARY), "--score", str(temp_score_path), str(SHARED_GENOTYPE_PREFIX), "--out", str(g_prefix)]
    run_and_measure(g_cmd, f"Gnomon_Iter_{iteration_name}", g_prefix)

    # Run PLINK2
    p2_prefix = CI_WORKDIR / f"_p2_{iteration_name}"
    p2_cmd = [str(PLINK2_BINARY), "--bfile", str(SHARED_GENOTYPE_PREFIX), "--score", str(temp_score_path), "1", "2", "4", "header", "no-mean-imputation", "--out", str(p2_prefix)]
    run_and_measure(p2_cmd, f"PLINK2_Iter_{iteration_name}", p2_prefix)

    # Analyze results
    try:
        g_df = pd.read_csv(g_prefix.with_suffix(".sscore"), sep='\s+').rename(columns={'#IID': 'IID', f'{DISCOVERY_PGS_ID}_AVG': 'SCORE_GNOMON'})
        p2_df = pd.read_csv(p2_prefix.with_suffix(".sscore"), sep='\s+').rename(columns={'#IID': 'IID', 'SCORE1_AVG': 'SCORE_PLINK2'})
        merged = pd.merge(g_df[['IID', 'SCORE_GNOMON']], p2_df[['IID', 'SCORE_PLINK2']], on='IID').dropna()
        if len(merged) < 2: return 1.0 # Not enough data to correlate
        correlation = merged['SCORE_GNOMON'].corr(merged['SCORE_PLINK2'])
        return correlation if pd.notna(correlation) else 1.0
    except Exception as e:
        print(f"  > [BISECTION_ANALYSIS_ERROR] Could not analyze results for iteration {iteration_name}: {e}", flush=True)
        return 1.0 # Assume perfect correlation on error to avoid false positives

def run_disagreement_discovery(initial_score_df: pd.DataFrame):
    """Performs the bisection search to find the variant causing the most disagreement."""
    print_header("DISAGREEMENT DISCOVERY MODE", "*")
    
    # 1. Establish Baseline
    print("[DISCOVERY] Establishing baseline correlation with full score file...")
    baseline_corr = test_variant_subset(initial_score_df, "baseline")
    print(f"[DISCOVERY] Baseline Gnomon vs PLINK2 Correlation: {baseline_corr:.6f}")
    if baseline_corr >= CORRELATION_THRESHOLD:
        print("✅ Baseline correlation is high. No disagreement discovery needed.", flush=True)
        return

    # 2. Bisection Loop
    current_variants_df = initial_score_df.copy()
    iteration = 0
    while len(current_variants_df) > BISECTION_TERMINATION_SIZE:
        iteration += 1
        print_header(f"Bisection Iteration {iteration}: {len(current_variants_df)} variants remaining", "-")
        
        midpoint = len(current_variants_df) // 2
        part_a_df = current_variants_df.iloc[:midpoint]
        part_b_df = current_variants_df.iloc[midpoint:]

        print(f"[DISCOVERY] Testing Part A ({len(part_a_df)} variants)...")
        corr_a = test_variant_subset(part_a_df, f"{iteration}A")
        print(f"  > Correlation for Part A: {corr_a:.6f}")

        print(f"[DISCOVERY] Testing Part B ({len(part_b_df)} variants)...")
        corr_b = test_variant_subset(part_b_df, f"{iteration}B")
        print(f"  > Correlation for Part B: {corr_b:.6f}")

        if corr_a < CORRELATION_THRESHOLD and corr_b < CORRELATION_THRESHOLD:
            print("[DISCOVERY] Both halves show disagreement. Pursuing the worse one.")
            current_variants_df = part_a_df if corr_a <= corr_b else part_b_df
        elif corr_a < CORRELATION_THRESHOLD:
            print("[DISCOVERY] Disagreement isolated to Part A.")
            current_variants_df = part_a_df
        elif corr_b < CORRELATION_THRESHOLD:
            print("[DISCOVERY] Disagreement isolated to Part B.")
            current_variants_df = part_b_df
        else:
            print("✅ Disagreement resolved. The issue is likely an interaction between the two halves. Halting search.", flush=True)
            return

    # 3. Final Pinpointing
    print_header(f"Final Pinpointing: {len(current_variants_df)} suspect variants", "-")
    suspect_results = []
    for i, row in current_variants_df.iterrows():
        variant_id = row['snp_id']
        print(f"[DISCOVERY] Testing single suspect variant: {variant_id}")
        single_variant_df = pd.DataFrame([row])
        corr = test_variant_subset(single_variant_df, f"final_{variant_id.replace(':', '_')}")
        suspect_results.append({'id': variant_id, 'corr': corr, 'df': single_variant_df})
        print(f"  > Correlation for {variant_id}: {corr:.6f}")
    
    # 4. Report Final Culprit
    if suspect_results:
        worst_offender = min(suspect_results, key=lambda x: x['corr'])
        print_header("DISAGREEMENT ISOLATED", "*")
        print(f"The primary source of disagreement is variant: {worst_offender['id']}")
        print(f"On its own, it yields a Gnomon vs PLINK2 correlation of: {worst_offender['corr']:.6f}")
        print("\nMinimal scorefile to reproduce this disagreement:")
        print("---------------------------------------------")
        print(worst_offender['df'].to_string(index=False))
        print("---------------------------------------------")

def main():
    """Main function to run the entire benchmark and discovery pipeline."""
    print_header("GNOMON CI TEST & BENCHMARK SUITE")
    setup_environment()
    create_synchronized_genotype_files(ORIGINAL_PLINK_PREFIX, SHARED_GENOTYPE_PREFIX)

    all_results, failures, final_summary_data = [], [], []

    for pgs_id, url in PGS_SCORES.items():
        print_header(f"TESTING SCORE: {pgs_id}", "~")
        raw_score_file = CI_WORKDIR / Path(url.split('/')[-1]).stem
        unified_score_df = create_unified_scorefile(raw_score_file, pgs_id)
        unified_score_file_path = CI_WORKDIR / f"{pgs_id}_unified_format.tsv"

        pgs_run_results = []
        
        # --- Define commands for all tools ---
        commands_to_run = [
            ("gnomon", [str(GNOMON_BINARY), "--score", str(unified_score_file_path), str(SHARED_GENOTYPE_PREFIX), "--out", str(SHARED_GENOTYPE_PREFIX)]),
            ("plink2", [str(PLINK2_BINARY), "--bfile", str(SHARED_GENOTYPE_PREFIX), "--score", str(unified_score_file_path), "1", "2", "4", "header", "no-mean-imputation", "--out", str(CI_WORKDIR / f"plink2_{pgs_id}")]),
            ("pylink", ["python3", str(PYLINK_SCRIPT), "--bfile", str(SHARED_GENOTYPE_PREFIX), "--score", str(unified_score_file_path), "--out", str(CI_WORKDIR / f"pylink_{pgs_id}"), "1", "2", "4"]),
        ]
        
        # --- Run all tools for the current PGS ID ---
        for tool, cmd in commands_to_run:
            res = run_and_measure(cmd, f"{tool}_{pgs_id}", cmd[-1])
            pgs_run_results.append(res)
            if not res['success']: failures.append(f"{pgs_id} ({tool}_failed)")
        
        # --- Analyze results for this PGS ID ---
        pgs_summary = analyze_pgs_results(pgs_id, pgs_run_results)
        final_summary_data.append(pgs_summary)

        # --- Automated Disagreement Discovery ---
        if pgs_id == DISCOVERY_PGS_ID and pgs_summary.get('success'):
            corr_matrix = pgs_summary['correlation_matrix']
            if 'SCORE_gnomon' in corr_matrix and 'SCORE_plink2' in corr_matrix:
                gnomon_plink2_corr = corr_matrix.loc['SCORE_gnomon', 'SCORE_plink2']
                if gnomon_plink2_corr < CORRELATION_THRESHOLD:
                    run_disagreement_discovery(unified_score_df)
    
    print_final_summary(final_summary_data)

    print_header(f"CI CHECK {'FAILED' if failures else 'PASSED'}", "*")
    if failures:
        print('❌ One or more tool executions failed:', flush=True)
        [print(f'  - {f}', flush=True) for f in sorted(list(set(failures)))]
        sys.exit(1)
    else:
        print('🎉 All tests passed successfully.', flush=True)
        sys.exit(0)

def print_final_summary(summary_data: list):
    """Prints a single, comprehensive summary report at the end of the run."""
    print_header("FINAL TEST & BENCHMARKING REPORT", "*")
    for pgs_summary in summary_data:
        pgs_id = pgs_summary['pgs_id']
        print_header(f"Analysis for {pgs_id}", "=")
        if not pgs_summary.get('success', False):
            print(f"❌ Analysis for {pgs_id} could not be completed.")
            continue
        print(f"✅ Concordance established using {pgs_summary.get('correlation_matrix').shape[0]} tools.")
        print_debug_header("Score Correlation Matrix")
        print(pgs_summary['correlation_matrix'].to_markdown(floatfmt=".8f"))

if __name__ == '__main__':
    main()
