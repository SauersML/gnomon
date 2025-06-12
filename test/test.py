import subprocess
import requests
import zipfile
import gzip
import shutil
import time
import sys
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
ORIGINAL_PLINK_PREFIX = CI_WORKDIR / "chr22_subset50"

PLINK1_URL = "https://s3.amazonaws.com/plink1-assets/plink_linux_x86_64_20231211.zip"
PLINK2_URL = "https://s3.amazonaws.com/plink2-assets/alpha6/plink2_linux_avx2_20250609.zip"

GENOTYPE_URL_BASE = "https://github.com/SauersML/genomic_pca/blob/main/data/"
GENOTYPE_FILES = [
    "chr22_subset50.bed.zip",
    "chr22_subset50.bim.zip",
    "chr22_subset50.fam.zip",
]

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
    width = 80
    print("\n" + char * width)
    print(f"{char*4} {title.upper()} {' ' * (width - len(title) - 10)} {char*4}")
    print(char * width)

def download_and_extract(url: str, dest_dir: Path):
    base = url.split('?')[0]
    filename = Path(base).name
    outpath = dest_dir / filename
    print(f"Downloading {filename}...")
    try:
        r = requests.get(url, stream=True, timeout=120)
        r.raise_for_status()
        with open(outpath, 'wb') as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)
    except Exception as e:
        print(f"❌ FAILED to download {url}: {e}")
        sys.exit(1)

    if filename.endswith(".zip"):
        with zipfile.ZipFile(outpath, 'r') as z:
            z.extractall(dest_dir)
        print(f"Extracted ZIP: {filename}")
        outpath.unlink()
    elif filename.endswith(".gz"):
        dest = dest_dir / outpath.stem
        with gzip.open(outpath, 'rb') as f_in, open(dest, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
        print(f"Extracted GZ: {filename} -> {dest.name}")
        outpath.unlink()

def run_and_measure(cmd: list, name: str, score_file: Path = None):
    print_header(f"RUNNING: {name}", char='-')
    print("Command:")
    print(f"  {' '.join(map(str, cmd))}\n")
    start = time.perf_counter()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    peak = 0
    try:
        p = psutil.Process(proc.pid)
        while proc.poll() is None:
            try:
                peak = max(peak, p.memory_info().rss)
            except psutil.NoSuchProcess:
                break
            time.sleep(0.01)
    except psutil.NoSuchProcess:
        pass
    out, err = proc.communicate()
    dur = time.perf_counter() - start
    result = {
        'tool': name,
        'time_sec': dur,
        'peak_mem_mb': peak / (1024*1024),
        'returncode': proc.returncode,
        'success': proc.returncode == 0
    }
    if result['success']:
        print(f"✅ {name} finished in {dur:.2f}s.")
    else:
        print(f"❌ {name} FAILED (Exit Code: {proc.returncode})")
        print_header("STDERR", char='.')
        print(err.strip() or "[EMPTY]")
    return result

def setup_environment():
    print_header("ENVIRONMENT SETUP")
    CI_WORKDIR.mkdir(exist_ok=True)
    for url, binary in [(PLINK1_URL, PLINK1_BINARY), (PLINK2_URL, PLINK2_BINARY)]:
        download_and_extract(url, CI_WORKDIR)
        for p in CI_WORKDIR.iterdir():
            if p.is_file() and p.name.startswith(binary.name.split('_')[0]):
                p.rename(binary)
                binary.chmod(0o755)
                print(f"Renamed and chmod: {binary}")
                break
    for zip_name in GENOTYPE_FILES:
        download_and_extract(f"{GENOTYPE_URL_BASE}{zip_name}?raw=true", CI_WORKDIR)
    print_header("WORKDIR CONTENTS AFTER DOWNLOADS", char='.')
    for p in sorted(CI_WORKDIR.iterdir()):
        print(f" - {p.name}")
    for url in PGS_SCORES.values():
        download_and_extract(url, CI_WORKDIR)

def make_gnomon_format(score_file: Path) -> Path:
    df = pd.read_csv(score_file, sep='\t', comment='#')
    # The last column is the weight column; keep only the first three plus it.
    weight_col = df.columns[-1]
    fmt_file = score_file.with_suffix('.gnomon_format.tsv')
    out = df[['snp_id', 'effect_allele', 'other_allele', weight_col]].copy()
    out.rename(columns={weight_col: 'weight'}, inplace=True)
    out.to_csv(fmt_file, sep='\t', index=False)
    print_header(f"GNOMON-FORMAT for {score_file.name}", char='.')
    print(out.head(5).to_markdown(index=False))
    print(f"Total variants formatted: {len(out)}")
    return fmt_file

# ========================================================================================
#                             MAIN EXECUTION & VALIDATION
# ========================================================================================
def main():
    print_header("GNOMON CI TEST & BENCHMARK SUITE")
    setup_environment()
    all_results, failures = [], []

    for pgs_id, url in PGS_SCORES.items():
        print_header(f"TESTING SCORE: {pgs_id}")
        score_file = CI_WORKDIR / Path(url.split('/')[-1]).stem
        print(f"Score file path: {score_file}")

        # 1) Gnomon
        res_g = run_and_measure(
            [str(GNOMON_BINARY), "--score", str(score_file), str(ORIGINAL_PLINK_PREFIX)],
            f"gnomon_{pgs_id}"
        )
        all_results.append(res_g)
        if not res_g['success']:
            failures.append(f"{pgs_id} (gnomon_execution_failed)")
            continue

        # prepare for PLINK2
        fmt_file = make_gnomon_format(score_file)

        # 2) PLINK2
        out2 = CI_WORKDIR / f"plink2_{pgs_id}"
        res_p2 = run_and_measure([
            str(PLINK2_BINARY),
            "--bfile", str(ORIGINAL_PLINK_PREFIX),
            "--score", str(fmt_file), "1", "2", "3", "header", "no-mean-imputation",
            "--out", str(out2)
        ], f"plink2_{pgs_id}")
        all_results.append(res_p2)

        # 3) PLINK1
        df1 = pd.read_csv(fmt_file, sep='\t')
        df1['snp_id'] = 'chr' + df1['snp_id'].str.replace(':', '_', regex=False)
        compat = CI_WORKDIR / f"{pgs_id}_p1_compat.tsv"
        df1.to_csv(compat, sep='\t', index=False)
        print(f"Wrote PLINK1-compatible: {compat}")
        out1 = CI_WORKDIR / f"plink1_{pgs_id}"
        res_p1 = run_and_measure([
            str(PLINK1_BINARY),
            "--bfile", str(ORIGINAL_PLINK_PREFIX),
            "--score", str(compat), "1", "2", "3", "header", "sum",
            "--out", str(out1)
        ], f"plink1_{pgs_id}")
        all_results.append(res_p1)

        # 4) Validate
        if not (res_p2['success'] and res_p1['success']):
            if not res_p1['success']:
                failures.append(f"{pgs_id} (plink1_execution_failed)")
            if not res_p2['success']:
                failures.append(f"{pgs_id} (plink2_execution_failed)")
            continue

        print_header(f"VALIDATING OUTPUTS for {pgs_id}", char='~')
        try:
            gdf = pd.read_csv(str(ORIGINAL_PLINK_PREFIX.with_suffix(".sscore")), sep='\t')\
                    .rename(columns={'#IID':'IID'}).set_index('IID')
            p2df = pd.read_csv(f'{out2}.sscore', sep='\t')\
                    .rename(columns={'#IID':'IID'}).set_index('IID')
            p1df = pd.read_csv(f'{out1}.profile', delim_whitespace=True).set_index('IID')
            print(f"Loaded: gnomon={len(gdf)}, plink2={len(p2df)}, plink1={len(p1df)}")

            col1 = [c for c in p1df.columns if c.upper().startswith("SCORE")][0]
            col2 = [c for c in p2df.columns if c.upper().startswith("SCORE1")][0]
            nv = p2df['ALLELE_CT'] / 2
            p2df['SCORE_SUM'] = p2df[col2] * nv
            colg = pgs_id

            merged = pd.DataFrame({
                'gnomon': gdf[colg],
                'plink1_sum': p1df[col1],
                'plink2_sum': p2df['SCORE_SUM']
            })
            print("Merged head:")
            print(merged.head().to_markdown())
            print("Correlations:")
            corr = merged.corr()
            print(corr.to_markdown(floatfmt='.8f'))

            if (merged['gnomon'] == 0).all():
                failures.append(f"{pgs_id} (gnomon_all_zeros_bug)")
            if not np.allclose(merged['gnomon'], merged['plink1_sum'], atol=NUMERICAL_TOLERANCE):
                failures.append(f"{pgs_id} (Gnomon_vs_PLINK1_mismatch)")
            if not np.allclose(merged['gnomon'], merged['plink2_sum'], atol=NUMERICAL_TOLERANCE):
                failures.append(f"{pgs_id} (Gnomon_vs_PLINK2_mismatch)")
            if corr.loc['plink1_sum','plink2_sum'] < CORR_THRESHOLD:
                failures.append(f"{pgs_id} (plink1_vs_plink2_low_correlation)")

        except Exception as e:
            failures.append(f"{pgs_id} (validation_script_error: {e})")
            print(f"❌ Error during validation: {e}")

    # Summary
    print_header("PERFORMANCE SUMMARY")
    if all_results:
        df_res = pd.DataFrame(all_results)
        df_res['tool_base'] = df_res['tool'].str.split('_').str[0]
        summary = df_res[df_res['success']].groupby('tool_base').agg(
            mean_time_sec=('time_sec','mean'),
            mean_mem_mb=('peak_mem_mb','mean')
        ).reset_index()
        print(summary.to_markdown(index=False, floatfmt='.3f'))

    print_header(f"CI CHECK {'PASSED' if not failures else 'FAILED'}")
    if failures:
        print("❌ Tests failed:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("🎉 All tests passed.")
    sys.exit(0)

if __name__ == '__main__':
    main()
