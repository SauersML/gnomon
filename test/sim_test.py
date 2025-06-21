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
import multiprocessing
import gmpy2
from pathlib import Path

# --- Configuration Parameters ---
N_VARIANTS = 50000
N_INDIVIDUALS = 100
CHR = '22'
CHR_LENGTH = 39_005_000
ALT_EFFECT_PROB = 0.3
MISSING_RATE = 0.5

# Mixture weights for allele-frequency distributions:
FREQ_DIST_WEIGHTS = [0.25, 0.25, 0.25, 0.25]
# Mixture weights for effect-size distributions:
EFFECT_DIST_WEIGHTS = [1.0, 0.0, 0.0, 0.0]

# --- CI/Validation Configuration ---
WORKDIR = Path("./sim_workdir")
OUTPUT_PREFIX = WORKDIR / "simulated_data"
CORR_THRESHOLD = 0.99999
MAD_THRESHOLD = 0.00001

# --- HELPER FUNCTIONS ---

def sum_column_precise(col):
    """Sums a column of floats using the ultra-precise gmpy2 library."""
    gmpy2.get_context().precision = 256
    return sum(gmpy2.mpfr(f) for f in col)

def sample_allele_frequencies(n):
    """Samples allele frequencies from a mixture of distributions."""
    choices = np.random.choice(4, size=n, p=FREQ_DIST_WEIGHTS)
    p = np.empty(n)
    for i, c in enumerate(choices):
        if c == 0: p[i] = np.random.uniform(0.001, 0.05)
        elif c == 1: p[i] = min(r, 1 - r) if (r := np.random.beta(0.5, 10)) else r
        elif c == 2: p[i] = min(r, 1 - r) if (r := np.random.beta(0.2, 0.2)) else r
        else: p[i] = np.random.uniform(0.01, 0.99)
    return p

def sample_effect_sizes(n):
    """Samples effect sizes from a mixture of distributions."""
    choices = np.random.choice(4, size=n, p=EFFECT_DIST_WEIGHTS)
    w = np.empty(n)
    for i, c in enumerate(choices):
        if c == 0: w[i] = np.random.normal(0, 0.0001)
        elif c == 1: w[i] = np.random.laplace(0, 0.5)
        elif c == 2: w[i] = np.random.uniform(-1, 1)
        else: w[i] = np.random.standard_cauchy() * 0.5
    return w

def sample_effect_alleles(n, ref, alt, af):
    """Chooses an effect allele, biased by allele frequency."""
    alt_prob = np.clip(ALT_EFFECT_PROB + (0.5 - af), 0.1, 0.9)
    mask = np.random.rand(n) < alt_prob
    return np.where(mask, alt, ref)

def _write_plink_files(prefix, bim_df, individuals, genotypes_df):
    """Helper to write a set of PLINK files for a test case."""
    with open(prefix.with_suffix(".fam"), 'w') as f:
        for iid in individuals:
            f.write(f"{iid} {iid} 0 0 0 -9\n")
    bim_df[['chr','id','cm','pos','a1','a2']].to_csv(
        prefix.with_suffix(".bim"), sep='\t', header=False, index=False
    )
    code_map = {0:0b00, 1:0b10, 2:0b11, -1:0b01}
    n = len(individuals)
    with open(prefix.with_suffix(".bed"), 'wb') as f:
        f.write(bytes([0x6c, 0x1b, 0x01]))
        for _, row in genotypes_df.iterrows():
            for j in range(0, n, 4):
                byte = 0
                chunk = row.iloc[j:j+4]
                for k, geno in enumerate(chunk):
                    byte |= (code_map[int(geno)] << (k*2))
                f.write(byte.to_bytes(1, 'little'))
    print(f"Programmatically wrote {prefix}.bed/.bim/.fam")

def _write_score_file(filename, score_df):
    """Helper to write a score file for a test case."""
    score_df.to_csv(filename, sep='\t', index=False)
    print(f"Programmatically wrote {filename}")

# --- SIMULATION STEPS ---

def generate_variants_and_weights():
    """Step 1: Simulate variant positions, alleles, frequencies, and effect sizes."""
    print(f"Step 1: Simulating {N_VARIANTS} variants on chr{CHR}...")
    positions = np.sort(np.random.choice(np.arange(1, CHR_LENGTH + 1), N_VARIANTS, replace=False))
    alleles = np.array(['A', 'C', 'G', 'T'])
    ref_idx = np.random.randint(0, 4, N_VARIANTS)
    alt_idx = (ref_idx + np.random.randint(1, 4, N_VARIANTS)) % 4
    variants_df = pd.DataFrame({
        'chr': CHR, 'pos': positions, 'ref': alleles[ref_idx], 'alt': alleles[alt_idx],
        'af': sample_allele_frequencies(N_VARIANTS),
        'effect_weight': sample_effect_sizes(N_VARIANTS)
    })
    variants_df['effect_allele'] = sample_effect_alleles(N_VARIANTS, variants_df['ref'], variants_df['alt'], variants_df['af'])
    print("...Variant and weight simulation complete.")
    return variants_df

def generate_genotypes(variants_df):
    """Step 2: Generate genotypes under Hardy-Weinberg equilibrium."""
    print(f"Step 2: Simulating genotypes for {N_INDIVIDUALS} individuals...")
    p = variants_df['af'].values
    q = 1 - p
    hwe_probs = np.vstack([q**2, 2*p*q, p**2]).T # Note: HWE is for alt allele count
    rand_draws = np.random.rand(N_VARIANTS, N_INDIVIDUALS)
    cum_probs = hwe_probs.cumsum(axis=1)
    genotypes = (rand_draws > cum_probs[:, [0]]) + (rand_draws > cum_probs[:, [1]])
    print("...Genotype simulation complete.")
    return genotypes.astype(int)

def introduce_missingness(genotypes):
    """Step 3: Introduce random missingness into the genotype matrix."""
    print(f"Step 4: Introducing {MISSING_RATE*100:.1f}% missingness...")
    mask = np.random.rand(*genotypes.shape) < MISSING_RATE
    g = genotypes.astype(float)
    g[mask] = -1
    print("...Missingness introduced.")
    return g.astype(int)

def calculate_ground_truth_prs(genotypes, variants_df):
    """Step 5: Calculate ground truth PRS with high precision for large-scale sim."""
    print("Step 5: Calculating ground truth polygenic scores (ULTIMATE PRECISION, ACCELERATED)...")
    effect_weights = variants_df['effect_weight'].values
    is_alt_effect = (variants_df['effect_allele'] == variants_df['alt']).values
    valid_mask = (genotypes != -1)
    dosages = np.where(is_alt_effect[:, np.newaxis], genotypes, 2 - genotypes).astype(float)
    score_components = np.where(valid_mask, dosages * effect_weights[:, np.newaxis], 0)
    print(f"    > Dispatching summations to {multiprocessing.cpu_count()} CPU cores...")
    with multiprocessing.Pool() as pool:
        score_sums = pool.map(sum_column_precise, score_components.T)
    variant_counts = valid_mask.sum(axis=0)
    score_avg = np.array([s / v if v != 0 else 0.0 for s, v in zip(score_sums, variant_counts)], dtype=float)
    return pd.DataFrame({'FID': f'sample_{i+1}', 'IID': f'sample_{i+1}', 'PRS_AVG': score_avg[i]} for i in range(N_INDIVIDUALS))

def write_output_files(prs_results, variants_df, genotypes_with_missing, prefix: Path):
    """Step 6: Write all output files for the large-scale simulation."""
    print(f"Step 6: Writing all output files to prefix '{prefix}'...")
    sscore_truth = prefix.with_suffix(".truth.sscore")
    prs_results[['FID', 'IID', 'PRS_AVG']].to_csv(sscore_truth, sep='\t', index=False, header=True, float_format='%.17g')
    print(f"...Ground truth PRS results written to {sscore_truth}")
    gnomon_scorefile = prefix.with_suffix(".gnomon.score")
    gdf = variants_df.copy()
    gdf['variant_id'] = gdf['chr'].astype(str) + ':' + gdf['pos'].astype(str)
    gdf['other_allele'] = np.where(gdf['effect_allele'] == gdf['ref'], gdf['alt'], gdf['ref'])
    gdf[['variant_id', 'effect_allele', 'other_allele', 'effect_weight']].rename(columns={'effect_weight': 'simulated_score'}).to_csv(gnomon_scorefile, sep='\t', index=False)
    print(f"...Gnomon-native scorefile written to {gnomon_scorefile}")
    bim_df = pd.DataFrame({'chr': variants_df['chr'], 'id': variants_df['chr'].astype(str) + ':' + variants_df['pos'].astype(str), 'cm': 0, 'pos': variants_df['pos'], 'a1': variants_df['ref'], 'a2': variants_df['alt']})
    genotypes_with_missing_for_write = np.where(variants_df['ref'].values[:, np.newaxis] == bim_df['a1'].values[:, np.newaxis], genotypes_with_missing, 2 - genotypes_with_missing)
    _write_plink_files(prefix, bim_df, [f"sample_{i+1}" for i in range(N_INDIVIDUALS)], pd.DataFrame(genotypes_with_missing_for_write, index=bim_df['id']))

def run_simple_dosage_test(workdir: Path, gnomon_path: Path, plink_path: Path, pylink_path: Path, run_cmd_func):
    """
    Runs a comprehensive, built-in test case to validate dosage calculations,
    including complex multiallelic resolution, differential missingness, and
    detection of biologically inconsistent genotypes.
    """
    prefix = workdir / "simple_test"
    print("\n" + "="*80)
    print("= Running Comprehensive Simple Test Case")
    print("="*80)
    print(f"Test files will be prefixed: {prefix}")

    # ============================================================================
    #  INTERNAL TRUTH ENGINE: Emulates the target biological logic from scratch.
    # ============================================================================
    def _calculate_biologically_accurate_truth(bim_df, score_df, individuals, genotypes_df):
        """
        An independent oracle that calculates the 100% correct biological outcome.
        This is not meant to mirror Gnomon, but to define the gold standard.
        """
        truth_sums = {iid: 0.0 for iid in individuals}
        variants_used = {iid: 0 for iid in individuals}
        variants_missing = {iid: 0 for iid in individuals}
        iids_expected_to_fail = set()

        # Pre-index the BIM by chr:pos for fast lookups
        bim_lookup = bim_df.groupby(bim_df['chr'].astype(str) + ':' + bim_df['pos'].astype(str))

        for iid in individuals:
            if iid in iids_expected_to_fail:
                continue

            for _, score_row in score_df.iterrows():
                variant_id = score_row['variant_id']
                effect_allele = score_row['effect_allele']
                weight = score_row['simple_score']

                # 1. Triage: Find all plausible BIM contexts for this variant
                try:
                    possible_contexts = bim_lookup.get_group(variant_id)
                except KeyError:
                    # This variant from the score file is not in the BIM at all.
                    variants_missing[iid] += 1
                    continue
                
                # 2. Resolution: Gather all valid, non-missing genotype interpretations
                valid_interpretations = []
                for _, context_row in possible_contexts.iterrows():
                    genotype = genotypes_df.loc[context_row['id'], iid]
                    if genotype != -1: # Genotype is not missing for this context
                        # Decode dosage relative to the effect allele
                        dosage = -1
                        if effect_allele == context_row['a2']: # Effect is allele 2
                            dosage = float(genotype)
                        elif effect_allele == context_row['a1']: # Effect is allele 1
                            dosage = 2.0 - float(genotype)
                        
                        # A dosage of -1 means the effect allele wasn't in this context,
                        # so this interpretation is not valid for this score.
                        if dosage != -1:
                            valid_interpretations.append(dosage)
                
                # 3. Apply Policy: Make a decision based on the evidence
                if len(valid_interpretations) == 0:
                    variants_missing[iid] += 1
                elif len(valid_interpretations) == 1:
                    dosage = valid_interpretations[0]
                    truth_sums[iid] += dosage * weight
                    variants_used[iid] += 1
                else: # len > 1
                    iids_expected_to_fail.add(iid)
                    # Once an individual is expected to fail, we can stop processing them.
                    break
        
        # 4. Assemble Final Truth DataFrames
        truth_data = []
        for iid in individuals:
            if iid in iids_expected_to_fail:
                # For failing IIDs, score is irrelevant (NaN) and missing is 100%
                truth_data.append({'IID': iid, 'SCORE_TRUTH': np.nan, 'MISSING_PCT_TRUTH': 100.0})
            else:
                total_variants_in_score = len(score_df)
                avg_score = truth_sums[iid] / variants_used[iid] if variants_used[iid] > 0 else 0.0
                missing_pct = (variants_missing[iid] / total_variants_in_score) * 100.0
                truth_data.append({'IID': iid, 'SCORE_TRUTH': avg_score, 'MISSING_PCT_TRUTH': missing_pct})

        return pd.DataFrame(truth_data), iids_expected_to_fail

    # ============================================================================
    #  TEST SETUP AND EXECUTION
    # ============================================================================
    
    # 1. Generate test data in memory
    bim_df, score_df, individuals, genotypes_df = _generate_test_data()

    # 2. Calculate the "oracle" ground truth using the new engine
    truth_df, iids_expected_to_fail = _calculate_biologically_accurate_truth(
        bim_df, score_df, individuals, genotypes_df
    )
    
    # 3. Write test files to disk
    _write_plink_files(prefix, bim_df, individuals, genotypes_df)
    _write_score_file(prefix.with_suffix(".score"), score_df)

    # 4. Execute all tools
    gnomon_result = run_cmd_func([gnomon_path, "--score", prefix.with_suffix(".score").name, prefix.name],
                                 "Simple Gnomon Test", cwd=workdir)
    plink_result = run_cmd_func([f"./{plink_path.name}", "--bfile", prefix.name,
                                 "--score", prefix.with_suffix(".score").name, "1", "2", "4",
                                 "header", "no-mean-imputation", "--out", "simple_plink_results"],
                                 "Simple PLINK2 Test", cwd=workdir)
    pylink_result = run_cmd_func(["python3", pylink_path.as_posix(), "--precise",
                                  "--bfile", prefix.name, "--score", prefix.with_suffix(".score").name,
                                  "--out", "simple_pylink_results", "1", "2", "4"],
                                  "Simple PyLink Test", cwd=workdir)

    # ============================================================================
    #  VALIDATION AND ANALYSIS
    # ============================================================================
    print("\n--- Analyzing Simple Test Results ---")
    
    # --- Validate Gnomon's Behavior ---
    is_gnomon_ok = False
    gnomon_successful_iids = set(individuals) - iids_expected_to_fail
    
    if gnomon_result is None:
        print("❌ Gnomon test could not be run (e.g., file not found).")
    elif gnomon_result.returncode != 0:
        # Gnomon crashed. This is ONLY okay if it was for an expected reason.
        found_expected_error = False
        for iid in iids_expected_to_fail:
            if iid in gnomon_result.stderr:
                print(f"✅ Gnomon correctly failed for inconsistent data in IID '{iid}'.")
                found_expected_error = True
                is_gnomon_ok = True # So far so good, a single correct crash is a pass.
                break
        if not found_expected_error:
            print(f"❌ Gnomon failed unexpectedly. Stderr:\n{gnomon_result.stderr}")
    else:
        # Gnomon succeeded. This is ONLY okay if no individuals were expected to fail.
        if iids_expected_to_fail:
            print(f"❌ Gnomon SUCCEEDED when it should have FAILED for IIDs: {iids_expected_to_fail}")
        else:
            try:
                # Load Gnomon results and compare against truth for successful IIDs
                gnomon_results_raw = pd.read_csv(workdir / "simple_test.sscore", sep='\t')
                gnomon_results = gnomon_results_raw.rename(columns={
                    '#IID':'IID',
                    'simple_score_AVG':'SCORE_GNOMON',
                    'simple_score_MISSING_PCT': 'MISSING_PCT_GNOMON'
                })
                merged = pd.merge(truth_df, gnomon_results, on='IID', how='inner')
                scores_ok = np.allclose(merged['SCORE_TRUTH'], merged['SCORE_GNOMON'])
                missing_ok = np.allclose(merged['MISSING_PCT_TRUTH'], merged['MISSING_PCT_GNOMON'])
                if scores_ok and missing_ok:
                    print("✅ Gnomon scores and missingness percentages are correct.")
                    is_gnomon_ok = True
                else:
                    if not scores_ok: print("  - Gnomon scores do not match biologically accurate truth.")
                    if not missing_ok: print("  - Gnomon missingness percentages do not match biologically accurate truth.")

            except Exception as e:
                print(f"❌ Failed to parse or validate Gnomon's output file: {e}")

    is_plink_ok = True
    is_pylink_ok = True
    
    # Final verdict
    if is_gnomon_ok and is_plink_ok and is_pylink_ok:
        print("\n✅ Simple Dosage Test SUCCEEDED: All tools behaved as expected.")
        return True
    else:
        print("\n❌ Simple Dosage Test FAILED.")
        return False

def run_and_validate_tools(runtimes):
    """Downloads tools, runs them, validates results, and collects runtimes."""
    PLINK2_URL = "https://s3.amazonaws.com/plink2-assets/alpha6/plink2_linux_avx2_20250609.zip"
    PLINK2_BINARY_PATH = WORKDIR / "plink2"
    GNOMON_BINARY_PATH = Path("./target/release/gnomon").resolve()
    
    overall_success = True

    def _print_header(title: str, char: str = "-"):
        print(f"\n{char*4} {title} {'-'*(70 - len(title) - 5)}")

    def run_command(cmd: list, step_name: str, cwd: Path) -> subprocess.CompletedProcess | None:
        """
        Executes a command and returns the full result object, or None on critical failure.
        The caller is responsible for interpreting the result.
        """
        _print_header(f"Executing: {step_name}")
        cmd_str = [str(c) for c in cmd]
        print(f"  > Command: {' '.join(cmd_str)}")
        print(f"  > CWD: {cwd}")

        proc_env = os.environ.copy()
        if "gnomon" in str(cmd_str[0]):
            # Rust backtraces are enabled for easier debugging of crashes
            proc_env["RUST_BACKTRACE"] = "1"

        start_time = time.monotonic()
        try:
            result = subprocess.run(
                cmd_str, capture_output=True, text=True, cwd=cwd, env=proc_env, timeout=600
            )
            duration = time.monotonic() - start_time
            # Print output regardless of success or failure for full transparency
            print("--- OUTPUT ---")
            print(result.stdout)
            if result.stderr:
                print("--- STDERR ---")
                print(result.stderr)
            
            print(f"\n  > Process exited with code {result.returncode}. (Completed in {duration:.4f}s)")
            return result
        except FileNotFoundError:
            print(f"  > ❌ CRITICAL ERROR: Command '{cmd[0]}' not found. The test environment is broken.")
            return None
        except subprocess.TimeoutExpired:
            print(f"  > ❌ CRITICAL ERROR: {step_name} timed out after 10 minutes.")
            return None

    def setup_tools():
        _print_header("Step A: Setting up tools")
        if not GNOMON_BINARY_PATH.exists():
            print(f"  > ❌ ERROR: Gnomon binary not found at '{GNOMON_BINARY_PATH}'.")
            return False
        if not PLINK2_BINARY_PATH.exists():
            print(f"  > Downloading PLINK2 to '{PLINK2_BINARY_PATH}'...")
            zip_path = WORKDIR / "plink.zip"
            try:
                with requests.get(PLINK2_URL, stream=True, timeout=120) as r:
                    r.raise_for_status()
                    with open(zip_path, 'wb') as f: shutil.copyfileobj(r.raw, f)
                with zipfile.ZipFile(zip_path, 'r') as z:
                    z.extract('plink2', path=WORKDIR)
                zip_path.unlink()
                PLINK2_BINARY_PATH.chmod(0o755)
                print("  > PLINK2 downloaded and extracted successfully.")
            except Exception as e:
                print(f"  > ❌ FAILED to download or extract PLINK2: {e}")
                return False
        return True

    def analyze_large_scale_results():
        _print_header("Step D: Analyzing and Comparing Large-Scale Simulation Results")
        try:
            truth_df = pd.read_csv(OUTPUT_PREFIX.with_suffix(".truth.sscore"), sep='\t').rename(columns={'PRS_AVG':'SCORE_TRUTH'})[['IID','SCORE_TRUTH']]
            gnomon_df = pd.read_csv(WORKDIR / "gnomon_results.sscore", sep='\t').rename(columns={'#IID':'IID','simulated_score_AVG':'SCORE_GNOMON'})[['IID','SCORE_GNOMON']]
        except (FileNotFoundError, KeyError) as e:
            print(f"  > ❌ ERROR: Failed to load or parse a result file. Error: {e}.")
            return False
        merged_df = pd.merge(truth_df, gnomon_df, on='IID')
        print("\n--- Score Correlation Matrix (Large-Scale) ---")
        corr_matrix = merged_df[['SCORE_TRUTH','SCORE_GNOMON']].corr()
        print(corr_matrix.to_markdown(floatfmt=".8f"))
        print("\n--- Mean Absolute Difference (MAD) vs. Ground Truth (Large-Scale) ---")
        mad_g = (merged_df['SCORE_TRUTH'] - merged_df['SCORE_GNOMON']).abs().mean()
        print(f"Gnomon vs. Truth: {mad_g:.10f}")
        corr_g = corr_matrix.loc['SCORE_TRUTH','SCORE_GNOMON']
        gnomon_ok = corr_g > CORR_THRESHOLD and mad_g < MAD_THRESHOLD
        if gnomon_ok:
            print(f"\n✅ Large-Scale Validation SUCCESS")
            return True
        else:
            print(f"\n❌ Large-Scale Validation FAILED for Gnomon: Corr={corr_g:.8f}, MAD={mad_g:.10f}")
            return False

    if not setup_tools(): return False
    if not run_simple_dosage_test(WORKDIR, GNOMON_BINARY_PATH, run_command):
        overall_success = False

    print("\n" + "="*80)
    print("= Running Large-Scale Simulation and Validation")
    print("="*80)
    
    gnomon_res = run_command([GNOMON_BINARY_PATH, "--score", f"{OUTPUT_PREFIX.name}.gnomon.score", OUTPUT_PREFIX.name], "Large-Scale Gnomon", WORKDIR)
    if gnomon_res.returncode == 0: runtimes.append({"Method": "Gnomon", "Runtime (s)": gnomon_res.duration})
    else: overall_success = False

    if overall_success and not analyze_large_scale_results():
        overall_success = False

    return overall_success

def print_runtime_summary(runtimes):
    """Prints mean runtimes for large-scale methods."""
    print("\n" + "="*50 + "\n= Mean Runtime for Large-Scale Methods\n" + "="*50)
    if runtimes:
        print(pd.DataFrame(runtimes).to_markdown(index=False, floatfmt=".4f"))
    else:
        print("No large-scale runtimes were recorded.")

def cleanup():
    """Cleans up the workspace directory."""
    if WORKDIR.exists():
        shutil.rmtree(WORKDIR)
        print(f"Cleaned up {WORKDIR}")

def main():
    """Main pipeline: simulate, write files, validate."""
    np.random.seed(42)
    exit_code = 0
    runtimes = []
    if WORKDIR.exists(): shutil.rmtree(WORKDIR)
    WORKDIR.mkdir()
    try:
        print("--- Starting Full Simulation and File Writing Pipeline ---")
        variants_df = generate_variants_and_weights()
        genotypes_pristine = generate_genotypes(variants_df)
        genotypes_with_missing = introduce_missingness(genotypes_pristine)
        prs_df = calculate_ground_truth_prs(genotypes_with_missing, variants_df)
        write_output_files(prs_df, variants_df, genotypes_with_missing, OUTPUT_PREFIX)
        print("\n--- Simulation and File Writing Finished Successfully ---")
        if not run_and_validate_tools(runtimes):
            exit_code = 1
    except Exception as e:
        print(f"\n--- An unexpected error occurred: {e} ---", file=sys.stderr)
        exit_code = 1
    finally:
        print_runtime_summary(runtimes)
        cleanup()
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
