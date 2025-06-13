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
#                                     CONFIGURATION
# ========================================================================================
CI_WORKDIR = Path("./ci_workdir")
GNOMON_BINARY = Path("./target/release/gnomon")
PLINK1_BINARY = CI_WORKDIR / "plink"
PLINK2_BINARY = CI_WORKDIR / "plink2"

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
#                                     HELPER FUNCTIONS
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

def download_and_extract(url: str, dest_dir: Path):
    """Downloads and extracts a file, handling .zip and .gz, and cleans up archives."""
    base = url.split('?')[0]
    filename = Path(base.split('/')[-1])
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
        print(f"[SCRIPT] Extracted ZIP: {filename}", flush=True); outpath.unlink()
    elif outpath.suffix == '.gz':
        dest = dest_dir / filename.stem
        with gzip.open(outpath, 'rb') as f_in, open(dest, 'wb') as f_out: shutil.copyfileobj(f_in, f_out)
        print(f"[SCRIPT] Extracted GZ: {filename} -> {dest.name}", flush=True); outpath.unlink()

def run_and_measure(cmd: list, name: str):
    """Executes a command, streaming its output in real-time, and measures performance."""
    print_header(f"RUNNING: {name}", char='-')
    print(f"Command:\n  {' '.join(map(str, cmd))}\n", flush=True)
    start = time.perf_counter()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace', bufsize=1)
    
    output_lines = []
    print_debug_header(f"{name} Real-time Output")
    with proc.stdout:
        for line in iter(proc.stdout.readline, ''):
            clean_line = line.strip()
            print(f"  [{name}] {clean_line}", flush=True)
            output_lines.append(clean_line)
    
    returncode = proc.wait()
    duration = time.perf_counter() - start
    result = {'tool': name, 'time_sec': duration, 'returncode': returncode, 'success': returncode == 0, 'stdout': "\n".join(output_lines)}
    if result['success']: print(f"✅ {name} finished in {duration:.2f}s.", flush=True)
    else: print(f"❌ {name} FAILED (Exit Code: {returncode})", flush=True)
    return result

def setup_environment():
    """Downloads, extracts, renames, and sets permissions for all required binaries."""
    print_header("ENVIRONMENT SETUP")
    CI_WORKDIR.mkdir(exist_ok=True)
    
    for url, binary_path in [(PLINK1_URL, PLINK1_BINARY), (PLINK2_URL, PLINK2_BINARY)]:
        if not binary_path.exists():
            download_and_extract(url, CI_WORKDIR)
            for p in CI_WORKDIR.iterdir():
                if p.is_file() and p.name.startswith(binary_path.name.split('_')[0]):
                    if p != binary_path: p.rename(binary_path)
                    binary_path.chmod(0o755)
                    print(f"[SCRIPT] Configured and chmod: {binary_path}", flush=True)
                    break
    
    for zip_name in GENOTYPE_FILES: download_and_extract(f"{GENOTYPE_URL_BASE}{zip_name}?raw=true", CI_WORKDIR)
    for url in PGS_SCORES.values(): download_and_extract(url, CI_WORKDIR)

def create_synchronized_genotype_files(original_prefix: Path, final_prefix: Path):
    """Creates a new, de-duplicated PLINK fileset where variant IDs are 'chr:pos'."""
    print_header("SYNCHRONIZING GENOTYPE DATA", char='.')
    original_bim = original_prefix.with_suffix('.bim')
    bim_df = pd.read_csv(original_bim, sep='\s+', header=None, names=['chrom', 'rsid', 'cm', 'pos', 'a1', 'a2'], dtype=str)
    
    bim_df['chr_pos_id'] = bim_df['chrom'] + ':' + bim_df['pos']
    
    duplicate_pos = bim_df[bim_df.duplicated(subset=['chr_pos_id'], keep=False)]
    variants_to_exclude = set(duplicate_pos['rsid'])
    variants_to_rename = bim_df[~bim_df['rsid'].isin(variants_to_exclude)]
    
    exclude_file = CI_WORKDIR / "variants_to_exclude.txt"
    pd.Series(list(variants_to_exclude)).to_csv(exclude_file, index=False, header=False)
    print(f"[SCRIPT] Identified {len(variants_to_exclude)} variants at duplicate positions to exclude.", flush=True)

    id_map_file = CI_WORKDIR / "id_update_map.txt"
    variants_to_rename[['rsid', 'chr_pos_id']].to_csv(id_map_file, sep=' ', index=False, header=False)
    print(f"[SCRIPT] Created ID mapping file for {len(variants_to_rename)} variants.", flush=True)

    cmd = [str(PLINK1_BINARY), "--bfile", str(original_prefix), "--exclude", str(exclude_file), "--update-name", str(id_map_file), "2", "1", "--make-bed", "--out", str(final_prefix)]
    
    print("[SCRIPT] Creating synchronized fileset using PLINK...", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print("❌ FAILED to create synchronized genotype fileset.", flush=True); print(proc.stdout); print(proc.stderr); sys.exit(1)
    
    print(f"[SCRIPT] Successfully created synchronized fileset at: {final_prefix}", flush=True)

def create_unified_scorefile(score_file: Path, pgs_id: str) -> Path:
    """Creates a unified scorefile where the variant ID is 'chr:pos'."""
    print_header(f"CREATING UNIFIED SCORE FILE FOR {pgs_id}", char='.')
    df = pd.read_csv(score_file, sep='\t', comment='#', usecols=['hm_chr', 'hm_pos', 'effect_allele', 'other_allele', 'effect_weight'], dtype=str, low_memory=False)
    df.dropna(inplace=True)
    df['effect_weight'] = pd.to_numeric(df['effect_weight'], errors='coerce')
    df.dropna(inplace=True)
    df['chr_pos_id'] = df['hm_chr'] + ':' + df['hm_pos']
    df.drop_duplicates(subset=['chr_pos_id'], keep='first', inplace=True)
    final_df = df[['chr_pos_id', 'effect_allele', 'other_allele', 'effect_weight']].copy()
    final_df.columns = ['snp_id', 'effect_allele', 'other_allele', pgs_id]
    out_path = CI_WORKDIR / f"{pgs_id}_unified_format.tsv"
    print(f"[SCRIPT] Writing {len(final_df)} variants to unified score file: {out_path.name}", flush=True)
    final_df.to_csv(out_path, sep='\t', index=False)
    return out_path

def analyze_pgs_results(pgs_id: str, pgs_run_results: list) -> dict:
    """Analyzes results for a single PGS, focusing on AVERAGE scores."""
    summary = {'pgs_id': pgs_id, 'success': False}
    tool_paths = {
        'gnomon': SHARED_GENOTYPE_PREFIX.with_suffix('.sscore'),
        'plink1': CI_WORKDIR / f"plink1_{pgs_id}.profile",
        'plink2': CI_WORKDIR / f"plink2_{pgs_id}.sscore",
    }
    
    data_frames = []
    num_variants = 0
    for res in pgs_run_results:
        if not res['success']: continue
        tool_name = res['tool'].split('_')[0]
        path = tool_paths.get(tool_name)
        if not path or not path.exists(): continue
        
        try:
            df_raw = pd.read_csv(path, sep='\s+')
            id_col = '#IID' if '#IID' in df_raw.columns else 'IID'
            
            if tool_name == 'gnomon':
                score_col = next((c for c in df_raw.columns if '_AVG' in c), None)
                if not score_col: raise KeyError("Gnomon _AVG score column not found")
                df = df_raw[[id_col, score_col]].rename(columns={id_col: 'IID', score_col: f'SCORE_{tool_name}'})
                match = re.search(r'(\d+)\s+overlapping\s+variants', res['stdout'])
                if match and num_variants == 0: num_variants = int(match.group(1))

            elif tool_name == 'plink1':
                # Without 'sum', the 'SCORE' column is the average score.
                df = df_raw[['IID', 'SCORE']].rename(columns={'SCORE': f'SCORE_{tool_name}'})
                match = re.search(r'(\d+)\s+valid\s+predictors', res['stdout'])
                if match and num_variants == 0: num_variants = int(match.group(1))

            elif tool_name == 'plink2':
                # With 'no-mean-imputation', plink2 provides an _AVG column.
                score_col = next((c for c in df_raw.columns if '_AVG' in c), None)
                if not score_col: raise KeyError("PLINK2 _AVG score column not found")
                df = df_raw[[id_col, score_col]].rename(columns={id_col: 'IID', score_col: f'SCORE_{tool_name}'})
                match = re.search(r'(\d+)\s+variants\s+processed', res['stdout'])
                if match and num_variants == 0: num_variants = int(match.group(1))
            
            if 'df' in locals() and df is not None: data_frames.append(df)
        except Exception as e:
            print(f"  > [ANALYSIS_ERROR] Failed to parse {tool_name} for {pgs_id}: {e}", flush=True)

    if len(data_frames) < 2: return summary
    
    merged_df = data_frames[0]
    for df_to_merge in data_frames[1:]:
        merged_df = pd.merge(merged_df, df_to_merge, on='IID', how='inner')
    
    score_cols = sorted([c for c in merged_df.columns if c.startswith('SCORE_')])
    if len(score_cols) < 2: return summary

    summary['n_variants_used'] = num_variants
    summary['sample_scores'] = merged_df[['IID'] + score_cols].head(5)
    summary['correlation_matrix'] = merged_df[score_cols].corr()

    mad_results, mrd_results = {}, {}
    for col1, col2 in combinations(score_cols, 2):
        key = f"{col1.replace('SCORE_', '')}_vs_{col2.replace('SCORE_', '')}"
        mad_results[key] = (merged_df[col1] - merged_df[col2]).abs().mean()
        denominator = (merged_df[col1].abs() + merged_df[col2].abs()) / 2.0
        valid_rows = denominator > 1e-12 # Use a small epsilon for floating point safety
        mrd = ((merged_df.loc[valid_rows, col1] - merged_df.loc[valid_rows, col2]).abs() / denominator[valid_rows]).mean()
        mrd_results[key] = mrd if not np.isnan(mrd) else 0.0

    summary['mad'] = mad_results
    summary['mrd'] = mrd_results
    summary['success'] = True
    return summary

def print_final_summary(summary_data: list, performance_results: list):
    """Prints a single, comprehensive summary report at the end of the run."""
    print_header("FINAL TEST & BENCHMARKING REPORT", "*")

    for pgs_summary in summary_data:
        pgs_id = pgs_summary['pgs_id']
        print_header(f"Analysis for {pgs_id}", "=")
        
        if not pgs_summary.get('success', False):
            print(f"❌ Analysis for {pgs_id} could not be completed due to failed runs or parsing errors.")
            continue

        print(f"✅ Concordance established using {pgs_summary.get('n_variants_used', 'N/A'):,} variants.")

        print_debug_header("Sample of Computed AVERAGE Scores")
        print(pgs_summary['sample_scores'].to_markdown(index=False, floatfmt=".8f"))
        
        print_debug_header("Score Correlation Matrix")
        print(pgs_summary['correlation_matrix'].to_markdown(floatfmt=".8f"))
        
        print_debug_header("Mean Absolute Difference (MAD)")
        mad_df = pd.DataFrame.from_dict(pgs_summary['mad'], orient='index', columns=['Difference'])
        print(mad_df.to_markdown(floatfmt=".8f"))
        
        print_debug_header("Mean Relative Difference (MRD %)")
        mrd_df = pd.DataFrame.from_dict(pgs_summary['mrd'], orient='index', columns=['Difference (%)'])
        mrd_df['Difference (%)'] *= 100
        print(mrd_df.to_markdown(floatfmt=".6f"))

    print_header("Performance Summary", "=")
    if performance_results:
        df = pd.DataFrame([r for r in performance_results if r['success']])
        if not df.empty:
            df['tool_base'] = df['tool'].str.split('_').str[0]
            summary = df.groupby('tool_base').agg(
                mean_time_sec=('time_sec','mean'),
                min_time_sec=('time_sec','min'),
                max_time_sec=('time_sec','max'),
            ).reset_index()
            print(summary.to_markdown(index=False, floatfmt='.3f'))
        else:
            print("No successful tool runs to summarize performance.")

def main():
    print_header("GNOMON CI TEST & BENCHMARK SUITE")
    setup_environment()
    create_synchronized_genotype_files(ORIGINAL_PLINK_PREFIX, SHARED_GENOTYPE_PREFIX)

    all_results, failures, final_summary_data = [], [], []
    for pgs_id, url in PGS_SCORES.items():
        print_header(f"TESTING SCORE: {pgs_id}", "~")
        raw_score_file = CI_WORKDIR / Path(url.split('/')[-1]).stem
        pgs_run_results = []
        unified_score_file = create_unified_scorefile(raw_score_file, pgs_id)
        
        gnomon_cmd = [str(GNOMON_BINARY), "--score", str(unified_score_file), str(SHARED_GENOTYPE_PREFIX)]
        res_g = run_and_measure(gnomon_cmd, f"gnomon_{pgs_id}")
        all_results.append(res_g); pgs_run_results.append(res_g)
        if not res_g['success']: failures.append(f"{pgs_id} (gnomon_failed)")

        # NOTE: Both plink commands now use 'no-mean-imputation' and do NOT use 'sum'
        # This makes them calculate the average score, matching gnomon's primary output.
        out2_prefix = CI_WORKDIR / f"plink2_{pgs_id}"
        plink2_cmd = [str(PLINK2_BINARY), "--bfile", str(SHARED_GENOTYPE_PREFIX), "--score", str(unified_score_file), "1", "2", "4", "header", "no-mean-imputation", "--out", str(out2_prefix)]
        res_p2 = run_and_measure(plink2_cmd, f"plink2_{pgs_id}")
        all_results.append(res_p2); pgs_run_results.append(res_p2)
        if not res_p2['success']: failures.append(f"{pgs_id} (plink2_failed)")
        
        out1_prefix = CI_WORKDIR / f"plink1_{pgs_id}"
        plink1_cmd = [str(PLINK1_BINARY), "--bfile", str(SHARED_GENOTYPE_PREFIX), "--score", str(unified_score_file), "1", "2", "4", "header", "no-mean-imputation", "--out", str(out1_prefix)]
        res_p1 = run_and_measure(plink1_cmd, f"plink1_{pgs_id}")
        all_results.append(res_p1); pgs_run_results.append(res_p1)
        if not res_p1['success']: failures.append(f"{pgs_id} (plink1_failed)")

        pgs_summary = analyze_pgs_results(pgs_id, pgs_run_results)
        final_summary_data.append(pgs_summary)

    print_final_summary(final_summary_data, all_results)

    print_header(f"CI CHECK {'FAILED' if failures else 'PASSED'}", "*")
    if failures:
        print('❌ One or more tool executions failed:', flush=True)
        [print(f'  - {f}', flush=True) for f in sorted(list(set(failures)))]
        sys.exit(1)
    else:
        print('🎉 All tests passed successfully.', flush=True)
        sys.exit(0)

if __name__ == '__main__':
    main()
