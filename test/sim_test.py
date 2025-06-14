import numpy as np
import pandas as pd
import sys
import os
import subprocess
import requests
import zipfile
import shutil
import time
import math
from pathlib import Path

# --- Configuration Parameters ---
N_VARIANTS = 50_000
N_INDIVIDUALS = 50
CHR = '22'
CHR_LENGTH = 40_000_000
ALT_EFFECT_PROB = 0.7
MISSING_RATE = 0.10

# --- CI/Validation Configuration ---
WORKDIR = Path("./sim_workdir")
OUTPUT_PREFIX = WORKDIR / "simulated_data"
# New validation thresholds
CORR_THRESHOLD = 0.99999
MAD_THRESHOLD = 0.00001


def generate_variants_and_weights():
    """
    FIRST: Derives normally and randomly distributed weights over 50K positions.
    """
    print(f"Step 1: Simulating {N_VARIANTS} variants on chr{CHR}...")

    positions = np.random.choice(np.arange(1, CHR_LENGTH + 1), N_VARIANTS, replace=False)
    positions.sort()

    alleles = np.array(['A', 'C', 'G', 'T'])
    ref_idx = np.random.randint(0, 4, N_VARIANTS)
    alt_offset = np.random.randint(1, 4, N_VARIANTS)
    alt_idx = (ref_idx + alt_offset) % 4
    
    ref_alleles = alleles[ref_idx]
    alt_alleles = alleles[alt_idx]

    is_alt_effect = np.random.rand(N_VARIANTS) < ALT_EFFECT_PROB
    effect_alleles = np.where(is_alt_effect, alt_alleles, ref_alleles)
    effect_weights = np.random.normal(loc=0.5, scale=0.5, size=N_VARIANTS)

    variants_df = pd.DataFrame({
        'chr': CHR,
        'pos': positions,
        'ref': ref_alleles,
        'alt': alt_alleles,
        'effect_allele': effect_alleles,
        'effect_weight': effect_weights
    })

    print("...Variant and weight simulation complete.")
    return variants_df

def generate_genotypes(variants_df):
    """
    THEN: Creates genotypes for 50 individuals based on HWE.
    """
    print(f"Step 2: Simulating genotypes for {N_INDIVIDUALS} individuals...")
    n_vars = len(variants_df)
    p = np.random.uniform(0.01, 0.99, size=n_vars)
    q = 1 - p
    hwe_probs = np.vstack([p**2, 2*p*q, q**2]).T
    rand_draws = np.random.rand(n_vars, N_INDIVIDUALS)
    cum_probs = hwe_probs.cumsum(axis=1)
    genotypes = (rand_draws > cum_probs[:, 0, np.newaxis]) + \
                (rand_draws > cum_probs[:, 1, np.newaxis])

    print("...Genotype simulation complete.")
    return genotypes.astype(int)


def introduce_missingness(genotypes):
    """
    Inserts random missingness into the genotype matrix.
    """
    print(f"Step 4: Introducing {MISSING_RATE*100:.1f}% missingness...")
    n_variants, n_individuals = genotypes.shape
    missing_mask = np.random.rand(n_variants, n_individuals) < MISSING_RATE
    genotypes_with_missing = genotypes.astype(float)
    genotypes_with_missing[missing_mask] = -1
    
    print("...Missingness introduced.")
    return genotypes_with_missing.astype(int)

def calculate_ground_truth_prs(genotypes, variants_df):
    """
    Calculates the 'ground truth' polygenic score for each individual using a
    vectorized, high-precision method.
    """
    print("Step 5: Calculating ground truth polygenic scores (high-precision)...")
    
    # --- 1. Vectorized Data Preparation ---
    effect_weights = variants_df['effect_weight'].values
    is_alt_effect = (variants_df['effect_allele'] == variants_df['alt']).values
    
    valid_mask = (genotypes != -1)
    
    dosages = genotypes.astype(float)
    
    # **FIXED LINE**: Select all rows where the ref allele is the effect allele
    # and update all individuals for those rows.
    rows_to_flip = ~is_alt_effect
    dosages[rows_to_flip, :] = 2 - dosages[rows_to_flip, :]
    
    # --- 2. High-Precision Score Calculation ---
    # Reshape weights for broadcasting and calculate score components
    # Set component to 0 for missing genotypes so it doesn't affect the sum
    score_components = np.where(valid_mask, dosages * effect_weights[:, np.newaxis], 0)

    # Use math.fsum for a high-precision sum for each individual (column-wise)
    # This avoids the floating-point error accumulation of np.sum()
    score_sums = np.array([math.fsum(col) for col in score_components.T])

    # --- 3. Final Averaging ---
    # Count the number of valid (non-missing) variants for each individual
    variant_counts = valid_mask.sum(axis=0)

    # Calculate the average score, handling the case of zero valid variants
    score_avg = np.divide(score_sums, variant_counts, 
                          out=np.zeros_like(score_sums, dtype=float), 
                          where=(variant_counts != 0))

    # --- 4. Format Output ---
    results_df = pd.DataFrame({
        'FID': [f"sample_{i+1}" for i in range(N_INDIVIDUALS)],
        'IID': [f"sample_{i+1}" for i in range(N_INDIVIDUALS)],
        'PRS_AVG': score_avg
    })
    
    print("...PRS calculation complete.")
    return results_df

def write_output_files(prs_results, variants_df, genotypes_with_missing, prefix: Path):
    """
    Writes all specified output files to the workspace directory.
    """
    print(f"Step 6: Writing all output files to prefix '{prefix}'...")

    # a. Ground Truth .sscore file
    sscore_filename = prefix.with_suffix(".truth.sscore")
    prs_results_ordered = prs_results[['FID', 'IID', 'PRS_AVG']]
    with open(sscore_filename, 'w') as f:
        f.write('#' + '\t'.join(prs_results_ordered.columns) + '\n')
        prs_results_ordered.to_csv(f, sep='\t', header=False, index=False, na_rep='NA')
    print(f"...Ground truth PRS results written to {sscore_filename}")

    # b. Gnomon-native Scorefile
    gnomon_scorefile = prefix.with_suffix(".gnomon.score")
    gnomon_df = variants_df.copy()
    gnomon_df['snp_id'] = gnomon_df['chr'].astype(str) + ':' + gnomon_df['pos'].astype(str)
    gnomon_df['other_allele'] = np.where(gnomon_df['effect_allele'] == gnomon_df['ref'], gnomon_df['alt'], gnomon_df['ref'])
    gnomon_df.rename(columns={'effect_weight': 'simulated_score'}, inplace=True)
    gnomon_df[['snp_id', 'effect_allele', 'other_allele', 'simulated_score']].to_csv(gnomon_scorefile, sep='\t', index=False)
    print(f"...Gnomon-native scorefile written to {gnomon_scorefile}")

    # c. PLINK fileset (.bed, .bim, .fam)
    with open(prefix.with_suffix(".fam"), 'w') as f:
        for i in range(N_INDIVIDUALS):
            f.write(f"sample_{i+1} sample_{i+1} 0 0 0 -9\n")
    
    bim_df = pd.DataFrame({
        'chr': variants_df['chr'],
        'id': variants_df['chr'].astype(str) + ':' + variants_df['pos'].astype(str),
        'cm': 0, 'pos': variants_df['pos'],
        'a1': variants_df['ref'], 'a2': variants_df['alt']
    })
    bim_df.to_csv(prefix.with_suffix(".bim"), sep='\t', header=False, index=False)
    
    code_map = {0: 0b00, -1: 0b01, 1: 0b10, 2: 0b11}
    with open(prefix.with_suffix(".bed"), 'wb') as f:
        f.write(bytes([0x6c, 0x1b, 0x01]))
        for i in range(genotypes_with_missing.shape[0]):
            for j in range(0, N_INDIVIDUALS, 4):
                byte = 0
                for k, geno in enumerate(genotypes_with_missing[i, j:j+4]):
                    byte |= (code_map[geno] << (k * 2))
                f.write(byte.to_bytes(1, 'little'))
    
    print(f"...PLINK files written: {prefix}.bed/.bim/.fam")

def run_simple_dosage_test(workdir: Path, gnomon_path: Path, plink_path: Path, pylink_path: Path, run_cmd_func):
    """
    Runs a complex, programmatically generated test case to check edge cases.
    Returns True if successful, False otherwise.
    """
    prefix = workdir / "simple_test"
    print("\n" + "="*80)
    print("= Running Comprehensive Simple Test Case")
    print("="*80)
    print(f"Test files will be prefixed: {prefix}")

    # --- Test Data Generation (Inner Functions) ---
    def _generate_test_data():
        individuals = ['id_hom_ref', 'id_het', 'id_hom_alt', 'id_new_person', 'id_special_case', 'id_multiallelic']
        bim_data = [
            {'chr': '1', 'id': '1:1000', 'cm': 0, 'pos': 1000, 'a1': 'A', 'a2': 'G'},
            {'chr': '1', 'id': '1:2000', 'cm': 0, 'pos': 2000, 'a1': 'C', 'a2': 'T'},
            {'chr': '1', 'id': '1:3000', 'cm': 0, 'pos': 3000, 'a1': 'A', 'a2': 'T'}
        ]
        for i in range(50):
            pos = 10000 + i
            bim_data.append({'chr': '1', 'id': f'1:{pos}', 'cm': 0, 'pos': pos, 'a1': 'A', 'a2': 'T'})
        bim_data.extend([
            {'chr': '1', 'id': '1:50000:A:C', 'cm': 0, 'pos': 50000, 'a1': 'A', 'a2': 'C'},
            {'chr': '1', 'id': '1:50000:A:G', 'cm': 0, 'pos': 50000, 'a1': 'A', 'a2': 'G'},
            {'chr': '1', 'id': '1:60000:T:C', 'cm': 0, 'pos': 60000, 'a1': 'T', 'a2': 'C'}
        ])
        bim_df = pd.DataFrame(bim_data)
        
        score_data = [
            {'snp_id': '1:1000', 'effect_allele': 'G', 'other_allele': 'A', 'simple_score': 0.5},
            {'snp_id': '1:2000', 'effect_allele': 'T', 'other_allele': 'C', 'simple_score': -0.2},
            {'snp_id': '1:3000', 'effect_allele': 'A', 'other_allele': 'T', 'simple_score': -0.7}
        ]
        for i in range(50):
            pos = 10000 + i
            score_data.append({'snp_id': f'1:{pos}', 'effect_allele': 'A', 'other_allele': 'T', 'simple_score': 0.1})
        score_data.extend([
            {'snp_id': '1:50000', 'effect_allele': 'A', 'other_allele': 'T', 'simple_score': 10.0},
            {'snp_id': '1:60000', 'effect_allele': 'C', 'other_allele': 'T', 'simple_score': 1.0}
        ])
        score_df = pd.DataFrame(score_data)

        genotypes_df = pd.DataFrame(-1, index=bim_df['id'], columns=individuals)
        genotypes_df.loc['1:1000', 'id_hom_ref'] = 0
        genotypes_df.loc['1:2000', 'id_hom_ref'] = 0
        genotypes_df.loc['1:1000', 'id_het'] = 1
        genotypes_df.loc['1:1000', 'id_hom_alt'] = 2
        genotypes_df.loc['1:2000', 'id_hom_alt'] = 2
        genotypes_df.loc['1:3000', 'id_new_person'] = 1
        special_case_ids = [f'1:{10000+i}' for i in range(50)]
        genotypes_df.loc[special_case_ids[0:10], 'id_special_case'] = 0
        genotypes_df.loc[special_case_ids[10:40], 'id_special_case'] = 1
        genotypes_df.loc[special_case_ids[40:50], 'id_special_case'] = 2
        genotypes_df.loc['1:50000:A:C', 'id_multiallelic'] = 1
        genotypes_df.loc['1:50000:A:G', 'id_multiallelic'] = 1
        genotypes_df.loc['1:60000:T:C', 'id_multiallelic'] = 1
        return bim_df, score_df, individuals, genotypes_df

    def _write_plink_files(prefix, bim_df, individuals, genotypes_df):
        with open(prefix.with_suffix(".fam"), 'w') as f:
            for iid in individuals: f.write(f"{iid} {iid} 0 0 0 -9\n")
        bim_df[['chr', 'id', 'cm', 'pos', 'a1', 'a2']].to_csv(prefix.with_suffix(".bim"), sep='\t', header=False, index=False)
        code_map = {0: 0b00, 1: 0b10, 2: 0b11, -1: 0b01}
        n_individuals = len(individuals)
        with open(prefix.with_suffix(".bed"), 'wb') as f:
            f.write(bytes([0x6c, 0x1b, 0x01]))
            for _, geno_series in genotypes_df.iterrows():
                for j in range(0, n_individuals, 4):
                    byte = 0
                    chunk = geno_series.iloc[j:j+4]
                    for k, geno in enumerate(chunk):
                        byte |= (code_map[int(geno)] << (k * 2))
                    f.write(byte.to_bytes(1, 'little'))
        print(f"Programmatically wrote {prefix}.bed/.bim/.fam")

    def _write_score_file(filename, score_df):
        score_df.to_csv(filename, sep='\t', index=False)
        print(f"Programmatically wrote {filename}")

    # --- Main Logic ---
    bim_df, score_df, individuals, genotypes_df = _generate_test_data()
    truth_df = pd.DataFrame({'IID': individuals, 'SCORE_TRUTH': [0.0, 0.5, 0.3, -0.7, 0.1, 1.0]})
    _write_plink_files(prefix, bim_df, individuals, genotypes_df)
    _write_score_file(prefix.with_suffix(".score"), score_df)

    # Run tools, checking for command success
    if not run_cmd_func([gnomon_path, "--score", prefix.with_suffix(".score").name, prefix.name], "Simple Gnomon Test", cwd=workdir): return False
    if not run_cmd_func([f"./{plink_path.name}", "--bfile", prefix.name, "--score", prefix.with_suffix(".score").name, "1", "2", "4", "header", "no-mean-imputation", "--out", "simple_plink_results"], "Simple PLINK2 Test", cwd=workdir): return False
    if not run_cmd_func(["python3", pylink_path.as_posix(), "--bfile", prefix.name, "--score", prefix.with_suffix(".score").name, "--out", "simple_pylink_results", "1", "2", "4"], "Simple PyLink Test", cwd=workdir): return False

    print("\n--- Analyzing Simple Test Results ---")
    gnomon_results = pd.read_csv(workdir / "simple_test.sscore", sep='\t').rename(columns={'#IID': 'IID', 'simple_score_AVG': 'SCORE_GNOMON'})[['IID', 'SCORE_GNOMON']]
    plink_results_raw = pd.read_csv(workdir / "simple_plink_results.sscore", sep=r'\s+').rename(columns={'#IID': 'IID'})
    plink_results = plink_results_raw.assign(SCORE_PLINK2=plink_results_raw['SCORE1_AVG'] * 2.0)[['IID', 'SCORE_PLINK2']]
    pylink_results_raw = pd.read_csv(workdir / "simple_pylink_results.sscore", sep='\t').rename(columns={'#IID': 'IID'})
    pylink_results = pylink_results_raw.assign(SCORE_PYLINK=pylink_results_raw['SCORE1_AVG'] * 2.0)[['IID', 'SCORE_PYLINK']]
    merged_df = pd.merge(truth_df, gnomon_results, on='IID', how='left').merge(plink_results, on='IID', how='left').merge(pylink_results, on='IID', how='left')

    print("\n--- Comparison of Scores ---")
    print(merged_df.to_markdown(index=False, floatfmt=".6f"))

    multiallelic_row = merged_df[merged_df['IID'] == 'id_multiallelic']
    other_rows = merged_df[merged_df['IID'] != 'id_multiallelic']
    
    is_gnomon_ok = np.allclose(other_rows['SCORE_TRUTH'], other_rows['SCORE_GNOMON']) and \
                   np.allclose(multiallelic_row['SCORE_TRUTH'], multiallelic_row['SCORE_GNOMON'])
    is_plink_ok = np.allclose(other_rows['SCORE_TRUTH'], other_rows['SCORE_PLINK2']) and \
                  pd.isna(multiallelic_row['SCORE_PLINK2'].iloc[0])
    is_pylink_ok = np.allclose(other_rows['SCORE_TRUTH'], other_rows['SCORE_PYLINK']) and \
                   pd.isna(multiallelic_row['SCORE_PYLINK'].iloc[0])
    
    all_ok = is_gnomon_ok and is_plink_ok and is_pylink_ok

    if all_ok:
        print("\n✅ Simple Dosage Test Successful: All tools behaved as expected.")
    else:
        print("\n❌ Simple Dosage Test FAILED:")
        if not is_gnomon_ok: print("  - Gnomon scores do not match ground truth.")
        if not is_plink_ok: print("  - PLINK2 scores/behavior do not match ground truth.")
        if not is_pylink_ok: print("  - PyLink scores/behavior do not match ground truth.")
    return all_ok


def run_and_validate_tools(runtimes):
    """
    Downloads tools, runs them, validates results, and manages runtime collection.
    Returns True if all validations pass, False otherwise.
    """
    PLINK2_URL = "https://s3.amazonaws.com/plink2-assets/alpha6/plink2_linux_avx2_20250609.zip"
    PLINK2_BINARY_PATH = WORKDIR / "plink2"
    GNOMON_BINARY_PATH = Path("./target/release/gnomon").resolve()
    PYLINK_SCRIPT_PATH = Path("test/pylink.py").resolve()
    
    overall_success = True

    def _print_header(title: str, char: str = "-"):
        width = 70
        print(f"\n{char*4} {title} {'-'*(width - len(title) - 5)}")

    def run_command(cmd: list, step_name: str, cwd: Path):
        _print_header(f"Executing: {step_name}")
        cmd_str = [str(c) for c in cmd]
        print(f"  > Command: {' '.join(cmd_str)}")
        print(f"  > CWD: {cwd}")
        
        proc_env = os.environ.copy()
        if "gnomon" in str(cmd_str[0]):
            proc_env["RUST_BACKTRACE"] = "1"

        start_time = time.monotonic()
        try:
            # Use run with capture_output=True and timeout
            result = subprocess.run(
                cmd_str, capture_output=True, text=True, encoding='utf-8',
                errors='replace', cwd=cwd, env=proc_env, check=False, timeout=600
            )
            print("--- OUTPUT ---")
            print(result.stdout)
            if result.stderr:
                print("--- STDERR ---")
                print(result.stderr)

            end_time = time.monotonic()
            duration = end_time - start_time

            if result.returncode == 0:
                print(f"\n  > Success. (Completed in {duration:.4f}s)")
                if "Large-Scale" in step_name:
                    method_name = step_name.split(" ")[-1]
                    runtimes.append({"Method": method_name, "Runtime (s)": duration})
                return True
            else:
                print(f"\n  > ❌ ERROR: {step_name} failed with exit code {result.returncode}.")
                return False
        except FileNotFoundError:
            print(f"  > ❌ ERROR: Command '{cmd[0]}' not found.")
            return False
        except subprocess.TimeoutExpired:
            print(f"  > ❌ ERROR: {step_name} timed out.")
            return False

    def setup_tools():
        _print_header("Step A: Setting up tools")
        if not GNOMON_BINARY_PATH.exists():
            print(f"  > ❌ ERROR: Gnomon binary not found at '{GNOMON_BINARY_PATH}'.")
            return False
        print(f"  > Found 'gnomon' executable at: {GNOMON_BINARY_PATH}")
        if PLINK2_BINARY_PATH.exists():
            print("  > 'plink2' executable already exists. Skipping download.")
            return True
        print(f"  > Downloading PLINK2 to '{PLINK2_BINARY_PATH}'...")
        zip_path = WORKDIR / "plink.zip"
        try:
            with requests.get(PLINK2_URL, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(zip_path, 'wb') as f: shutil.copyfileobj(r.raw, f)
            with zipfile.ZipFile(zip_path, 'r') as z:
                for member in z.infolist():
                    if member.filename.endswith('plink2') and not member.is_dir():
                        with z.open(member) as source, open(PLINK2_BINARY_PATH, 'wb') as target:
                            shutil.copyfileobj(source, target)
                        break
            zip_path.unlink()
            PLINK2_BINARY_PATH.chmod(0o755)
            print("  > PLINK2 downloaded and extracted successfully.")
            return True
        except Exception as e:
            print(f"  > ❌ FAILED to download or extract PLINK2: {e}")
            return False
    
    def analyze_large_scale_results():
        _print_header("Step D: Analyzing and Comparing Large-Scale Simulation Results")
        try:
            truth_df = pd.read_csv(OUTPUT_PREFIX.with_suffix(".truth.sscore"), sep='\t').rename(columns={'#FID': 'FID', 'PRS_AVG': 'SCORE_TRUTH'})[['IID', 'SCORE_TRUTH']]
            gnomon_df = pd.read_csv(OUTPUT_PREFIX.with_suffix(".sscore"), sep='\t').rename(columns={'#IID': 'IID', 'simulated_score_AVG': 'SCORE_GNOMON'})[['IID', 'SCORE_GNOMON']]
            plink2_df_raw = pd.read_csv(WORKDIR / "plink2_results.sscore", sep=r'\s+').rename(columns={'#IID': 'IID'})
            plink2_df = plink2_df_raw.assign(SCORE_PLINK2=plink2_df_raw['SCORE1_AVG'] * 2.0)[['IID', 'SCORE_PLINK2']]
            pylink_df_raw = pd.read_csv(WORKDIR / "pylink_results.sscore", sep='\t').rename(columns={'#IID': 'IID'})
            pylink_df = pylink_df_raw.assign(SCORE_PYLINK=pylink_df_raw['SCORE1_AVG'] * 2.0)[['IID', 'SCORE_PYLINK']]
        except (FileNotFoundError, KeyError) as e:
            print(f"  > ❌ ERROR: Failed to load or parse a result file. Error: {e}. Aborting comparison.")
            return False

        merged_df = pd.merge(truth_df, gnomon_df, on='IID').merge(plink2_df, on='IID').merge(pylink_df, on='IID')
        score_cols = ['SCORE_TRUTH', 'SCORE_GNOMON', 'SCORE_PLINK2', 'SCORE_PYLINK']
        merged_df[score_cols] = merged_df[score_cols].astype(float)

        print("\n--- Score Correlation Matrix (Large-Scale) ---")
        corr_matrix = merged_df[score_cols].corr()
        print(corr_matrix.to_markdown(floatfmt=".8f"))

        print("\n--- Mean Absolute Difference (MAD) vs. Ground Truth (Large-Scale) ---")
        mad_g = (merged_df['SCORE_TRUTH'] - merged_df['SCORE_GNOMON']).abs().mean()
        mad_p = (merged_df['SCORE_TRUTH'] - merged_df['SCORE_PLINK2']).abs().mean()
        mad_py = (merged_df['SCORE_TRUTH'] - merged_df['SCORE_PYLINK']).abs().mean()
        print(f"Gnomon vs. Truth: {mad_g:.10f}")
        print(f"PLINK2 vs. Truth: {mad_p:.10f}")
        print(f"PyLink vs. Truth: {mad_py:.10f}")

        # --- New Validation Logic ---
        corr_g = corr_matrix.loc['SCORE_TRUTH', 'SCORE_GNOMON']
        corr_p = corr_matrix.loc['SCORE_TRUTH', 'SCORE_PLINK2']
        corr_py = corr_matrix.loc['SCORE_TRUTH', 'SCORE_PYLINK']

        gnomon_ok = corr_g > CORR_THRESHOLD and mad_g < MAD_THRESHOLD
        plink_ok = corr_p > CORR_THRESHOLD and mad_p < MAD_THRESHOLD
        pylink_ok = corr_py > CORR_THRESHOLD and mad_py < MAD_THRESHOLD

        if gnomon_ok and plink_ok and pylink_ok:
            print(f"\n✅ Large-Scale Validation SUCCESS: All tools passed criteria (Corr > {CORR_THRESHOLD}, MAD < {MAD_THRESHOLD}).")
            return True
        else:
            print(f"\n❌ Large-Scale Validation FAILED: One or more tools did not meet criteria.")
            if not gnomon_ok: print(f"  - Gnomon: Corr={corr_g:.8f}, MAD={mad_g:.10f}")
            if not plink_ok: print(f"  - PLINK2: Corr={corr_p:.8f}, MAD={mad_p:.10f}")
            if not pylink_ok: print(f"  - PyLink: Corr={corr_py:.8f}, MAD={mad_py:.10f}")
            return False

    # --- Main execution flow for this function ---
    if not setup_tools(): return False
    
    if not run_simple_dosage_test(WORKDIR, GNOMON_BINARY_PATH, PLINK2_BINARY_PATH, PYLINK_SCRIPT_PATH, run_command):
        overall_success = False

    print("\n" + "="*80)
    print("= Running Large-Scale Simulation and Validation")
    print("="*80)

    # Sequentially run tools and check for command failures
    if not run_command([GNOMON_BINARY_PATH, "--score", OUTPUT_PREFIX.with_suffix(".gnomon.score").name, OUTPUT_PREFIX.name], "Large-Scale Gnomon", WORKDIR): overall_success = False
    if not run_command([f"./{PLINK2_BINARY_PATH.name}", "--bfile", OUTPUT_PREFIX.name, "--score", OUTPUT_PREFIX.with_suffix(".gnomon.score").name, "1", "2", "4", "header", "no-mean-imputation", "--out", "plink2_results"], "Large-Scale PLINK2", WORKDIR): overall_success = False
    if not run_command(["python3", PYLINK_SCRIPT_PATH.as_posix(), "--bfile", OUTPUT_PREFIX.name, "--score", OUTPUT_PREFIX.with_suffix(".gnomon.score").name, "--out", "pylink_results", "1", "2", "4"], "Large-Scale PyLink", WORKDIR): overall_success = False
    
    # If any command failed, we can't analyze results. Return the current failure status.
    if not overall_success:
        print("\nSkipping large-scale analysis due to command failure.")
        return False
        
    # Run analysis and update overall success status
    if not analyze_large_scale_results():
        overall_success = False

    return overall_success

def print_runtime_summary(runtimes):
    """Prints the mean runtime summary table."""
    print("\n" + "="*50)
    print("= Mean Runtime for Large-Scale Methods")
    print("="*50)
    if runtimes:
        runtime_df = pd.DataFrame(runtimes)
        mean_runtimes = runtime_df.groupby('Method')['Runtime (s)'].mean().reset_index()
        print(mean_runtimes.to_markdown(index=False, floatfmt=".4f"))
    else:
        print("No large-scale runtimes were recorded (likely due to an early error).")

def cleanup():
    """Removes the generated workspace directory."""
    print("\n" + "-"*80)
    print(f"Cleaning up workspace directory: {WORKDIR}")
    if WORKDIR.exists():
        shutil.rmtree(WORKDIR)
        print("...Workspace cleaned up.")

def main():
    """Main function to run the entire simulation and validation pipeline."""
    np.random.seed(42)
    exit_code = 0
    runtimes = []

    # Ensure the isolated workspace exists and is empty
    if WORKDIR.exists():
        shutil.rmtree(WORKDIR)
    WORKDIR.mkdir()

    try:
        print("--- Starting Full Simulation and File Writing Pipeline ---")
        print(f"All generated files will be placed in: {WORKDIR}")

        variants_df = generate_variants_and_weights()
        genotypes_pristine = generate_genotypes(variants_df)
        genotypes_with_missing = introduce_missingness(genotypes_pristine)
        prs_results_df = calculate_ground_truth_prs(genotypes_with_missing, variants_df)
        write_output_files(prs_results_df, variants_df, genotypes_with_missing, OUTPUT_PREFIX)
        print("\n--- Simulation and File Writing Finished Successfully ---")

        # Run the validation, which now returns a success/fail boolean
        if not run_and_validate_tools(runtimes):
            exit_code = 1 # Mark for exit, but don't exit yet

    except Exception as e:
        print(f"\n--- An unexpected error occurred: {e} ---")
        import traceback
        traceback.print_exc()
        exit_code = 1
        
    finally:
        # This block is GUARANTEED to run, regardless of success, failure, or exceptions.
        print_runtime_summary(runtimes)
        if 'keep' not in sys.argv:
            cleanup()
        else:
            print("\nSkipping cleanup because 'keep' argument was provided.")
    
    if exit_code == 0:
        print("\n--- Full Simulation and Validation Pipeline Finished Successfully ---")
    else:
        # Use a generic error message as the specific reason for failure
        # has already been printed.
        print("\nError: Process completed with exit code 1.")
        
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
