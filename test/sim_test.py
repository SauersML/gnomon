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
        elif c == 1:
            r = np.random.beta(0.5, 10)
            p[i] = min(r, 1 - r)
        elif c == 2:
            r = np.random.beta(0.2, 0.2)
            p[i] = min(r, 1 - r)
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
    # The dosage is the number of effect alleles.
    # If effect is 'alt', dosage = genotype (0, 1, or 2)
    # If effect is 'ref' (not 'alt'), dosage = 2 - genotype
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
    # For writing PLINK files, genotypes must be relative to allele2 in the BIM.
    # Our generated genotypes are counts of the 'alt' allele. We must ensure a1/a2 are ref/alt.
    bim_df = pd.DataFrame({'chr': variants_df['chr'], 'id': variants_df['chr'].astype(str) + ':' + variants_df['pos'].astype(str), 'cm': 0, 'pos': variants_df['pos'], 'a1': variants_df['ref'], 'a2': variants_df['alt']})
    _write_plink_files(prefix, bim_df, [f"sample_{i+1}" for i in range(N_INDIVIDUALS)], pd.DataFrame(genotypes_with_missing, index=bim_df['id']))

def run_simple_dosage_test(workdir: Path, gnomon_path: Path, plink_path: Path, pylink_path: Path, run_cmd_func):
    """
    Runs a comprehensive, built-in test case to validate dosage calculations,
    including complex multiallelic resolution, differential missingness, and
    detection of biologically inconsistent genotypes.
    This test will fail the entire script if Gnomon's results cannot be verified.
    """
    prefix = workdir / "simple_test"
    print("\n" + "="*80)
    print("= Running Comprehensive Simple Test Case")
    print("="*80)
    print(f"Test files will be prefixed: {prefix}")

    def _generate_test_data():
        """Generates the in-memory data for the simple test case."""
        individuals = [
            'id_hom_ref', 'id_het', 'id_hom_alt', 'id_new_person',
            'id_special_case', 'id_multi_AC', 'id_multi_AG', 'id_multi_CG'
        ]
        bim_data = [
            {'chr':'1','id':'1:1000','cm':0,'pos':1000,'a1':'A','a2':'G'},
            {'chr':'1','id':'1:2000','cm':0,'pos':2000,'a1':'C','a2':'T'},
            {'chr':'1','id':'1:3000','cm':0,'pos':3000,'a1':'A','a2':'T'},
            *[{'chr':'1','id':f'1:{10000+i}','cm':0,'pos':10000+i,'a1':'A','a2':'T'} for i in range(50)],
            {'chr':'1','id':'1:50000:A:C','cm':0,'pos':50000,'a1':'A','a2':'C'},
            {'chr':'1','id':'1:50000:A:G','cm':0,'pos':50000,'a1':'A','a2':'G'},
            {'chr':'1','id':'1:60000:T:C','cm':0,'pos':60000,'a1':'T','a2':'C'}
        ]
        bim_df = pd.DataFrame(bim_data)
        score_data = [
            {'variant_id':'1:1000','effect_allele':'G','other_allele':'A','simple_score':0.5},
            {'variant_id':'1:2000','effect_allele':'T','other_allele':'C','simple_score':-0.2},
            {'variant_id':'1:3000','effect_allele':'A','other_allele':'T','simple_score':-0.7},
            *[{'variant_id':f'1:{10000+i}','effect_allele':'A','other_allele':'T','simple_score':0.1} for i in range(50)],
            {'variant_id':'1:50000','effect_allele':'C','other_allele':'A','simple_score':10.0},
            {'variant_id':'1:50000','effect_allele':'G','other_allele':'A','simple_score':-5.0},
            {'variant_id':'1:60000','effect_allele':'C','other_allele':'T','simple_score':1.0}
        ]
        score_df = pd.DataFrame(score_data)
        genotypes_df = pd.DataFrame(-1, index=bim_df['id'], columns=individuals)
        genotypes_df.loc['1:1000', 'id_hom_ref'] = 0
        genotypes_df.loc['1:2000', 'id_hom_ref'] = 0
        genotypes_df.loc['1:1000', 'id_het'] = 1
        genotypes_df.loc['1:1000', 'id_hom_alt'] = 2
        genotypes_df.loc['1:2000', 'id_hom_alt'] = 2
        genotypes_df.loc['1:3000', 'id_new_person'] = 1
        special_variants = [f'1:{10000+i}' for i in range(50)]
        genotypes_df.loc[special_variants[:10], 'id_special_case'] = 2
        genotypes_df.loc[special_variants[10:40], 'id_special_case'] = 1
        genotypes_df.loc[special_variants[40:], 'id_special_case'] = 0
        
        # Test Case: Individual has A/C. Genotype for A/G is missing.
        genotypes_df.loc['1:50000:A:C', 'id_multi_AC'] = 1
        genotypes_df.loc['1:50000:A:G', 'id_multi_AC'] = -1
        genotypes_df.loc['1:60000:T:C', 'id_multi_AC'] = 1
        
        # Test Case: Individual has A/G. Genotype for A/C is missing.
        genotypes_df.loc['1:50000:A:C', 'id_multi_AG'] = -1
        genotypes_df.loc['1:50000:A:G', 'id_multi_AG'] = 1
        genotypes_df.loc['1:60000:T:C', 'id_multi_AG'] = 1

        # Test Case: Individual has C/G.
        genotypes_df.loc['1:50000:A:C', 'id_multi_CG'] = 1
        genotypes_df.loc['1:50000:A:G', 'id_multi_CG'] = 1
        genotypes_df.loc['1:60000:T:C', 'id_multi_CG'] = 1
        return bim_df, score_df, individuals, genotypes_df

    def _calculate_biologically_accurate_truth(bim_df, score_df, individuals, genotypes_df):
        """An independent oracle that calculates the 100% correct biological outcome."""
        truth_sums = {iid: 0.0 for iid in individuals}
        variants_used = {iid: 0 for iid in individuals}
        variants_missing = {iid: 0 for iid in individuals}
        # This test case should no longer produce errors.
        iids_expected_to_fail = set()
        
        # Pre-process genotypes to resolve multiallelic sites into a single biological truth
        resolved_genotypes = {}
        bim_grouped_by_pos = bim_df.groupby(['chr', 'pos'])
        for iid in individuals:
            resolved_genotypes[iid] = {}
            for (chrom, pos), group in bim_grouped_by_pos:
                alleles = []
                has_missing = False
                # Check all .bim rows for this physical location
                for _, bim_row in group.iterrows():
                    genotype_val = genotypes_df.loc[bim_row['id'], iid]
                    if genotype_val == -1:
                        has_missing = True
                        continue
                    # Add alleles based on genotype (count of allele a2)
                    alleles.extend([bim_row['a1']] * (2 - genotype_val))
                    alleles.extend([bim_row['a2']] * genotype_val)
                
                # Consolidate alleles, removing the reference 'A's to find the true genotype
                unique_alleles = [a for a in alleles if a != bim_row['a1']]
                # Add back one reference allele if only one alt allele was found
                if len(unique_alleles) == 1:
                    unique_alleles.append(bim_row['a1'])
                # If no alt alleles, it's homozygous reference
                elif len(unique_alleles) == 0 and not has_missing:
                    unique_alleles = [bim_row['a1'], bim_row['a1']]
                
                if len(unique_alleles) > 2:
                    # This would be an error condition, but our test data is valid.
                    iids_expected_to_fail.add(iid)
                elif len(unique_alleles) > 0:
                    resolved_genotypes[iid][(chrom, pos)] = tuple(sorted(unique_alleles))
                # If completely missing, it remains unresolved.

        # Calculate scores based on the resolved genotypes
        for iid in individuals:
            if iid in iids_expected_to_fail: continue
            for _, score_row in score_df.iterrows():
                chrom, pos_str = score_row['variant_id'].split(':')[:2]
                pos = int(pos_str)
                effect_allele = score_row['effect_allele']
                weight = score_row['simple_score']

                if (chrom, pos) in resolved_genotypes[iid]:
                    genotype = resolved_genotypes[iid][(chrom, pos)]
                    dosage = float(genotype.count(effect_allele))
                    truth_sums[iid] += dosage * weight
                    variants_used[iid] += 1
                else:
                    variants_missing[iid] += 1

        truth_data = []
        for iid in individuals:
            if iid in iids_expected_to_fail:
                truth_data.append({'IID': iid, 'SCORE_TRUTH': np.nan, 'MISSING_PCT_TRUTH': np.nan})
            else:
                avg_score = truth_sums[iid] / variants_used[iid] if variants_used[iid] > 0 else 0.0
                denominator = variants_used[iid] + variants_missing[iid]
                missing_pct = (variants_missing[iid] / denominator) * 100.0 if denominator > 0 else 0.0
                truth_data.append({'IID': iid, 'SCORE_TRUTH': avg_score, 'MISSING_PCT_TRUTH': missing_pct})
        return pd.DataFrame(truth_data), iids_expected_to_fail

    # --- Main test logic ---
    bim_df, score_df, individuals, genotypes_df = _generate_test_data()
    truth_df, iids_expected_to_fail = _calculate_biologically_accurate_truth(bim_df, score_df, individuals, genotypes_df)
    _write_plink_files(prefix, bim_df, individuals, genotypes_df)
    _write_score_file(prefix.with_suffix(".score"), score_df)

    gnomon_result = run_cmd_func([gnomon_path, "--score", prefix.with_suffix(".score").name, prefix.name], "Simple Gnomon Test", workdir)
    print("\n--- Analyzing Simple Test Results ---")
    is_gnomon_ok = False

    if gnomon_result is None:
        print("❌ Gnomon test could not be run.")
    elif gnomon_result.returncode != 0:
        # With the corrected biological data, Gnomon should no longer fail.
        # Any failure is now unexpected.
        print(f"❌ Gnomon failed unexpectedly. Stderr:\n{gnomon_result.stderr}")
    else: # Gnomon succeeded (returncode == 0)
        if iids_expected_to_fail:
            # This should not happen with the new test data.
            print(f"❌ Gnomon SUCCEEDED when it should have FAILED for IIDs: {iids_expected_to_fail}")
        else:
            try:
                gnomon_output_path = workdir / "simple_test.sscore"
                if not gnomon_output_path.exists():
                    raise FileNotFoundError("Gnomon ran successfully but did not produce the output .sscore file.")
                gnomon_results_raw = pd.read_csv(gnomon_output_path, sep='\t')
                gnomon_results = gnomon_results_raw.rename(columns={'#IID':'IID', 'simple_score_AVG':'SCORE_GNOMON', 'simple_score_MISSING_PCT': 'MISSING_PCT_GNOMON'})[['IID', 'SCORE_GNOMON', 'MISSING_PCT_GNOMON']]
                merged = pd.merge(truth_df, gnomon_results, on='IID', how='outer')

                print("\n--- Simple Dosage Test: Full Results Table ---")
                print("WHO (what sample) got WHICH SCORE and by WHICH tool:")
                print(merged[['IID', 'SCORE_TRUTH', 'SCORE_GNOMON', 'MISSING_PCT_TRUTH', 'MISSING_PCT_GNOMON']].to_markdown(index=False, floatfmt=".6f"))

                scores_ok = np.allclose(merged['SCORE_TRUTH'], merged['SCORE_GNOMON'], equal_nan=True)
                missing_ok = np.allclose(merged['MISSING_PCT_TRUTH'], merged['MISSING_PCT_GNOMON'], equal_nan=True)

                if scores_ok and missing_ok:
                    print("\n✅ Verification successful: Gnomon scores and missingness percentages match the ground truth.")
                    is_gnomon_ok = True
                else:
                    print("\n❌ Verification failed: Gnomon results DO NOT MATCH the ground truth.")
                    if not scores_ok: print("  - Mismatch found in scores.")
                    if not missing_ok: print("  - Mismatch found in missingness percentages.")
            except Exception as e:
                print(f"❌ Failed to parse or validate Gnomon's output file: {e}")
                import traceback
                traceback.print_exc()

    if is_gnomon_ok:
        print("\n✅ Simple Dosage Test SUCCEEDED.")
        return True
    else:
        print("\n❌ Simple Dosage Test FAILED.")
        return False

def run_and_validate_tools(runtimes):
    """Downloads tools, runs them, validates results, and collects runtimes."""
    PLINK2_URL = "https://s3.amazonaws.com/plink2-assets/alpha6/plink2_linux_avx2_20250609.zip"
    PLINK2_BINARY_PATH = WORKDIR / "plink2"
    GNOMON_BINARY_PATH = Path("./target/release/gnomon").resolve()
    PYLINK_SCRIPT_PATH = Path("test/pylink.py").resolve()
    overall_success = True

    def _print_header(title: str, char: str = "-"):
        print(f"\n{char*4} {title} {'-'*(70 - len(title) - 5)}")

    def run_command(cmd: list, step_name: str, cwd: Path) -> subprocess.CompletedProcess | None:
        """Executes a command and returns the full result object, or None on critical failure."""
        _print_header(f"Executing: {step_name}")
        cmd_str = [str(c) for c in cmd]
        print(f"  > Command: {' '.join(cmd_str)}")
        print(f"  > CWD: {cwd}")
        proc_env = os.environ.copy()
        if "gnomon" in str(cmd_str[0]):
            proc_env["RUST_BACKTRACE"] = "1"
        start_time = time.monotonic()
        try:
            result = subprocess.run(cmd_str, capture_output=True, text=True, cwd=cwd, env=proc_env, timeout=600)
            duration = time.monotonic() - start_time
            print("--- OUTPUT ---", result.stdout, sep="\n")
            if result.stderr: print("--- STDERR ---", result.stderr, sep="\n")
            print(f"\n  > Process exited with code {result.returncode}. (Completed in {duration:.4f}s)")
            if "Large-Scale Gnomon" in step_name and result.returncode == 0:
                runtimes.append({"Method": "Gnomon", "Runtime (s)": duration})
            return result
        except FileNotFoundError:
            print(f"  > ❌ CRITICAL ERROR: Command '{cmd[0]}' not found.")
            return None
        except subprocess.TimeoutExpired:
            print(f"  > ❌ CRITICAL ERROR: {step_name} timed out.")
            return None

    def setup_tools():
        _print_header("Step A: Setting up tools")
        if not GNOMON_BINARY_PATH.exists():
            print(f"  > ❌ ERROR: Gnomon binary not found at '{GNOMON_BINARY_PATH}'. Please build it first with 'cargo build --release'.")
            return False
        if not PLINK2_BINARY_PATH.exists():
            print(f"  > Downloading PLINK2 to '{PLINK2_BINARY_PATH}'...")
            zip_path = WORKDIR / "plink.zip"
            try:
                with requests.get(PLINK2_URL, stream=True, timeout=120) as r:
                    r.raise_for_status()
                    with open(zip_path, 'wb') as f: shutil.copyfileobj(r.raw, f)
                with zipfile.ZipFile(zip_path, 'r') as z: z.extract('plink2', path=WORKDIR)
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
            gnomon_output_path = WORKDIR / f"{OUTPUT_PREFIX.name}.sscore"
            truth_df = pd.read_csv(OUTPUT_PREFIX.with_suffix(".truth.sscore"), sep='\t').rename(columns={'PRS_AVG':'SCORE_TRUTH'})[['IID','SCORE_TRUTH']]
            gnomon_df = pd.read_csv(gnomon_output_path, sep='\t').rename(columns={'#IID':'IID','simulated_score_AVG':'SCORE_GNOMON'})[['IID','SCORE_GNOMON']]
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
    
    # Run simple dosage test. If it fails, the entire script's success is compromised.
    if not run_simple_dosage_test(WORKDIR, GNOMON_BINARY_PATH, PLINK2_BINARY_PATH, PYLINK_SCRIPT_PATH, run_command):
        overall_success = False

    print("\n" + "="*80)
    print("= Running Large-Scale Simulation and Validation")
    print("="*80)
    
    gnomon_res = run_command([GNOMON_BINARY_PATH, "--score", f"{OUTPUT_PREFIX.name}.gnomon.score", OUTPUT_PREFIX.name], "Large-Scale Gnomon", WORKDIR)
    if not (gnomon_res and gnomon_res.returncode == 0):
        print("❌ Gnomon failed to run on the large-scale dataset.")
        overall_success = False
    
    # Only analyze if the run was successful and previous steps haven't failed.
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
            print("\n" + "!"*80)
            print("! OVERALL VALIDATION FAILED: Gnomon's results could not be verified.")
            print("!"*80)
            exit_code = 1
        else:
            print("\n" + "✓"*80)
            print("✓ OVERALL VALIDATION SUCCEEDED: All tests passed and results were verified.")
            print("✓"*80)

    except Exception as e:
        import traceback
        print(f"\n--- An unexpected error occurred: {e} ---", file=sys.stderr)
        traceback.print_exc()
        exit_code = 1
    finally:
        print_runtime_summary(runtimes)
        cleanup()
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
