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
#   FREQ_DIST_WEIGHTS[0] = weight for Uniform(0.001, 0.05)
#   FREQ_DIST_WEIGHTS[1] = weight for Beta(0.5, 10)
#   FREQ_DIST_WEIGHTS[2] = weight for Beta(0.2, 0.2)
#   FREQ_DIST_WEIGHTS[3] = weight for Uniform(0.01, 0.99)
FREQ_DIST_WEIGHTS = [
    0.25,  # Uniform(0.001, 0.05)
    0.25,  # Beta(0.5, 10)
    0.25,  # Beta(0.2, 0.2)
    0.25,  # Uniform(0.01, 0.99)
]

# Mixture weights for effect-size distributions:
#   EFFECT_DIST_WEIGHTS[0] = weight for Normal(0, 0.5)
#   EFFECT_DIST_WEIGHTS[1] = weight for Laplace(0, 0.5)
#   EFFECT_DIST_WEIGHTS[2] = weight for Uniform(-1, 1)
#   EFFECT_DIST_WEIGHTS[3] = weight for Cauchy(0, 0.5)
EFFECT_DIST_WEIGHTS = [
    1.0,  # Normal(0, 0.5)
    0.0,  # Laplace(0, 0.5)
    0.0,  # Uniform(-1, 1)
    0.0,  # Cauchy(0, 0.5)
]

# --- CI/Validation Configuration ---
WORKDIR = Path("./sim_workdir")
OUTPUT_PREFIX = WORKDIR / "simulated_data"
CORR_THRESHOLD = 0.99999
MAD_THRESHOLD = 0.00001

# --- HELPER FUNCTIONS ---

def sum_column_precise(col):
    """
    A worker function for multiprocessing. Sums a column of floats
    using the ultra-precise and fast gmpy2 library.
    """
    gmpy2.get_context().precision = 256
    return sum(gmpy2.mpfr(f) for f in col)

def sample_allele_frequencies(n):
    """
    Sample allele frequencies from a mixture of distributions:
      - Uniform(0.001, 0.05)
      - Beta(0.5, 10)
      - Beta(0.2, 0.2)
      - Uniform(0.01, 0.99)
    """
    choices = np.random.choice(4, size=n, p=FREQ_DIST_WEIGHTS)
    p = np.empty(n)
    for i, c in enumerate(choices):
        if c == 0:
            p[i] = np.random.uniform(0.001, 0.05)
        elif c == 1:
            r = np.random.beta(0.5, 10)
            p[i] = min(r, 1 - r)
        elif c == 2:
            r = np.random.beta(0.2, 0.2)
            p[i] = min(r, 1 - r)
        else:
            p[i] = np.random.uniform(0.01, 0.99)
    return p

def sample_effect_sizes(n):
    """
    Sample effect sizes from a mixture of distributions:
      - Normal(0, 0.5)
      - Laplace(0, 0.5)
      - Uniform(-1, 1)
      - Cauchy(0, 0.5)
    """
    choices = np.random.choice(4, size=n, p=EFFECT_DIST_WEIGHTS)
    w = np.empty(n)
    for i, c in enumerate(choices):
        if c == 0:
            w[i] = np.random.normal(0, 0.0001)
        elif c == 1:
            w[i] = np.random.laplace(0, 0.5)
        elif c == 2:
            w[i] = np.random.uniform(-1, 1)
        else:
            w[i] = np.random.standard_cauchy() * 0.5
    return w

def sample_effect_alleles(n, ref, alt, af):
    """
    Choose effect allele per-variant, biased by allele frequency.
    """
    alt_prob = np.clip(ALT_EFFECT_PROB + (0.5 - af), 0.1, 0.9)
    mask = np.random.rand(n) < alt_prob
    return np.where(mask, alt, ref)

# --- SIMULATION STEPS ---

def generate_variants_and_weights():
    """
    Step 1: Simulate variant positions, alleles, frequencies, and effect sizes.
    """
    print(f"Step 1: Simulating {N_VARIANTS} variants on chr{CHR}...")
    positions = np.random.choice(np.arange(1, CHR_LENGTH + 1), N_VARIANTS, replace=False)
    positions.sort()

    alleles = np.array(['A', 'C', 'G', 'T'])
    ref_idx = np.random.randint(0, 4, N_VARIANTS)
    alt_idx = (ref_idx + np.random.randint(1, 4, N_VARIANTS)) % 4
    ref_alleles = alleles[ref_idx]
    alt_alleles = alleles[alt_idx]

    af = sample_allele_frequencies(N_VARIANTS)
    effect_weights = sample_effect_sizes(N_VARIANTS)
    effect_alleles = sample_effect_alleles(N_VARIANTS, ref_alleles, alt_alleles, af)

    variants_df = pd.DataFrame({
        'chr': CHR,
        'pos': positions,
        'ref': ref_alleles,
        'alt': alt_alleles,
        'effect_allele': effect_alleles,
        'effect_weight': effect_weights,
        'af': af
    })

    print("...Variant and weight simulation complete.")
    return variants_df

def generate_genotypes(variants_df):
    """
    Step 2: Generate genotypes under Hardy-Weinberg equilibrium.
    """
    print(f"Step 2: Simulating genotypes for {N_INDIVIDUALS} individuals...")
    p = variants_df['af'].values
    q = 1 - p
    hwe_probs = np.vstack([p**2, 2*p*q, q**2]).T
    rand_draws = np.random.rand(N_VARIANTS, N_INDIVIDUALS)
    cum_probs = hwe_probs.cumsum(axis=1)
    genotypes = (rand_draws > cum_probs[:, [0]]) + (rand_draws > cum_probs[:, [1]])
    print("...Genotype simulation complete.")
    return genotypes.astype(int)

def introduce_missingness(genotypes):
    """
    Step 3: Introduce random missingness into the genotype matrix.
    """
    print(f"Step 4: Introducing {MISSING_RATE*100:.1f}% missingness...")
    mask = np.random.rand(*genotypes.shape) < MISSING_RATE
    g = genotypes.astype(float)
    g[mask] = -1
    print("...Missingness introduced.")
    return g.astype(int)

def calculate_ground_truth_prs(genotypes, variants_df):
    """
    Step 5: Calculate ground truth PRS with high precision.
    """
    print("Step 5: Calculating ground truth polygenic scores (ULTIMATE PRECISION, ACCELERATED)...")
    effect_weights = variants_df['effect_weight'].values
    is_alt_effect = (variants_df['effect_allele'] == variants_df['alt']).values
    valid_mask = (genotypes != -1)
    dosages = genotypes.astype(float)
    rows_to_flip = ~is_alt_effect
    dosages[rows_to_flip, :] = 2 - dosages[rows_to_flip, :]
    score_components = np.where(valid_mask, dosages * effect_weights[:, np.newaxis], 0)

    print(f"    > Dispatching summations to {multiprocessing.cpu_count()} CPU cores...")
    with multiprocessing.Pool() as pool:
        score_sums = pool.map(sum_column_precise, score_components.T)

    variant_counts = valid_mask.sum(axis=0)
    score_avg_list = [s / v if v != 0 else 0.0 for s, v in zip(score_sums, variant_counts)]
    score_avg = np.array(score_avg_list, dtype=float)

    results_df = pd.DataFrame({
        'FID': [f"sample_{i+1}" for i in range(N_INDIVIDUALS)],
        'IID': [f"sample_{i+1}" for i in range(N_INDIVIDUALS)],
        'PRS_AVG': score_avg
    })

    print("...PRS calculation complete.")
    return results_df

def write_output_files(prs_results, variants_df, genotypes_with_missing, prefix: Path):
    """
    Step 6: Write all output files: truth.sscore, gnomon.score, PLINK .bed/.bim/.fam.
    """
    print(f"Step 6: Writing all output files to prefix '{prefix}'...")

    # a. Ground Truth .truth.sscore
    sscore_truth = prefix.with_suffix(".truth.sscore")
    prs_results[['FID', 'IID', 'PRS_AVG']].to_csv(
        sscore_truth, sep='\t', index=False, header=True, float_format='%.17g'
    )
    print(f"...Ground truth PRS results written to {sscore_truth}")

    # b. Gnomon-native scorefile
    gnomon_scorefile = prefix.with_suffix(".gnomon.score")
    gdf = variants_df.copy()
    gdf['variant_id'] = gdf['chr'].astype(str) + ':' + gdf['pos'].astype(str)
    gdf['other_allele'] = np.where(gdf['effect_allele'] == gdf['ref'], gdf['alt'], gdf['ref'])
    gdf.rename(columns={'effect_weight': 'simulated_score'}, inplace=True)
    gdf[['variant_id', 'effect_allele', 'other_allele', 'simulated_score']].to_csv(
        gnomon_scorefile, sep='\t', index=False
    )
    print(f"...Gnomon-native scorefile written to {gnomon_scorefile}")

    # c. PLINK fileset
    fam_file = prefix.with_suffix(".fam")
    with open(fam_file, 'w') as f:
        for i in range(N_INDIVIDUALS):
            f.write(f"sample_{i+1} sample_{i+1} 0 0 0 -9\n")
    print(f"...PLINK .fam written to {fam_file}")

    bim_file = prefix.with_suffix(".bim")
    pd.DataFrame({
        'chr': variants_df['chr'],
        'id': variants_df['chr'].astype(str) + ':' + variants_df['pos'].astype(str),
        'cm': 0,
        'pos': variants_df['pos'],
        'a1': variants_df['ref'],
        'a2': variants_df['alt']
    }).to_csv(bim_file, sep='\t', header=False, index=False)
    print(f"...PLINK .bim written to {bim_file}")

    bed_file = prefix.with_suffix(".bed")
    code_map = {0:0b00, -1:0b01, 1:0b10, 2:0b11}
    with open(bed_file, 'wb') as f:
        f.write(bytes([0x6c, 0x1b, 0x01]))
        for i in range(genotypes_with_missing.shape[0]):
            for j in range(0, N_INDIVIDUALS, 4):
                byte = 0
                for k, geno in enumerate(genotypes_with_missing[i, j:j+4]):
                    byte |= (code_map[int(geno)] << (k*2))
                f.write(byte.to_bytes(1, 'little'))
    print(f"...PLINK .bed written to {bed_file}")

def run_simple_dosage_test(workdir: Path, gnomon_path: Path, plink_path: Path, pylink_path: Path, run_cmd_func):
    """
    Runs a comprehensive, built-in test case to validate dosage calculations,
    including complex multiallelic resolution, fallback from missing data, and
    detection of biologically inconsistent genotypes.
    """
    prefix = workdir / "simple_test"
    print("\n" + "="*80)
    print("= Running Comprehensive Simple Test Case")
    print("="*80)
    print(f"Test files will be prefixed: {prefix}")

    def _generate_test_data():
        """Generates the in-memory data for the test case."""
        individuals = [
            'id_hom_ref', 'id_het', 'id_hom_alt', 'id_new_person',
            'id_special_case', 'id_multiallelic_fail', 'id_fallback_success'
        ]
        bim_data = [
            {'chr':'1','id':'1:1000','cm':0,'pos':1000,'a1':'A','a2':'G'},
            {'chr':'1','id':'1:2000','cm':0,'pos':2000,'a1':'C','a2':'T'},
            # A block of 50 variants to test missingness calculation
            *[{'chr':'1','id':f'1:{10000+i}','cm':0,'pos':10000+i,'a1':'A','a2':'T'} for i in range(50)],
            # The multiallelic site contexts for chr1:50000
            {'chr':'1','id':'1:50000:A:C','cm':0,'pos':50000,'a1':'A','a2':'C'}, # Context 1
            {'chr':'1','id':'1:50000:A:T','cm':0,'pos':50000,'a1':'A','a2':'T'}, # Context 2
        ]
        bim_df = pd.DataFrame(bim_data)
        score_data = [
            {'variant_id':'1:1000','effect_allele':'G','other_allele':'A','simple_score':0.5},
            {'variant_id':'1:2000','effect_allele':'T','other_allele':'C','simple_score':-0.2},
            *[{'variant_id':f'1:{10000+i}','effect_allele':'A','other_allele':'T','simple_score':0.1} for i in range(50)],
            # Score for the multiallelic site. NOTE: 'other_allele' is irrelevant to Gnomon's new logic.
            {'variant_id':'1:50000','effect_allele':'A','other_allele':'G','simple_score':10.0},
        ]
        score_df = pd.DataFrame(score_data)
        genotypes_df = pd.DataFrame(0, index=bim_df['id'], columns=individuals) # Default to hom-ref
        genotypes_df.loc['1:1000', 'id_het'] = 1
        genotypes_df.loc['1:1000', 'id_hom_alt'] = 2
        genotypes_df.loc['1:2000', 'id_hom_alt'] = 2
        genotypes_df.loc['1:1000', 'id_new_person'] = -1 # Explicitly missing
        special_variants = [f'1:{10000+i}' for i in range(50)]
        genotypes_df.loc[special_variants[10:40], 'id_special_case'] = 1
        genotypes_df.loc[special_variants[40:], 'id_special_case'] = 2
        
        # id_multiallelic_fail: Genotype is valid and different in both contexts. This is a biological
        # contradiction and MUST cause Gnomon to fail for this person.
        genotypes_df.loc['1:50000:A:C', 'id_multiallelic_fail'] = 1 # A/C
        genotypes_df.loc['1:50000:A:T', 'id_multiallelic_fail'] = 2 # T/T
        
        # id_fallback_success: Genotype is MISSING for the A/C context, but valid for A/T.
        # Gnomon MUST use the valid data from the A/T context.
        genotypes_df.loc['1:50000:A:C', 'id_fallback_success'] = -1 # Missing
        genotypes_df.loc['1:50000:A:T', 'id_fallback_success'] = 1  # A/T
        
        return bim_df, score_df, individuals, genotypes_df

    def _calculate_simple_truth(bim_df, score_df, individuals, genotypes_df):
        """A truth engine that emulates the desired biological logic."""
        truth = {iid: {'sum': 0.0, 'used': 0, 'missing': 0} for iid in individuals}
        iids_expected_to_fail = set()

        # Pre-process BIM for faster lookups
        bim_lookup = bim_df.groupby('pos')

        for iid in individuals:
            if iid in iids_expected_to_fail: continue
            for _, score_row in score_df.iterrows():
                pos = int(score_row['variant_id'].split(':')[1])
                effect_allele = score_row['effect_allele']
                weight = score_row['simple_score']

                # 1. Triage: Find all plausible contexts
                possible_contexts = bim_lookup.get_group(pos)

                # 2. Resolution: Gather evidence
                valid_interpretations = []
                for _, context_row in possible_contexts.iterrows():
                    genotype = genotypes_df.loc[context_row['id'], iid]
                    if genotype != -1:
                        # Decode dosage relative to the effect allele for THIS score
                        a1, a2 = context_row['a1'], context_row['a2']
                        if effect_allele == a1:
                            dosage = {0: 2.0, 1: 1.0, 2: 0.0}[genotype]
                            valid_interpretations.append(dosage)
                        elif effect_allele == a2:
                            dosage = {0: 0.0, 1: 1.0, 2: 2.0}[genotype]
                            valid_interpretations.append(dosage)
                        # If effect allele not in context, it's not a valid interpretation for this score

                # 3. Apply Policy
                if len(set(valid_interpretations)) > 1:
                    iids_expected_to_fail.add(iid)
                    break # Stop processing this contradictory individual
                elif len(valid_interpretations) == 1:
                    truth[iid]['sum'] += valid_interpretations[0] * weight
                    truth[iid]['used'] += 1
                else: # 0 valid interpretations
                    truth[iid]['missing'] += 1
        
        # Finalize DataFrames
        results = []
        for iid, data in truth.items():
            if iid in iids_expected_to_fail: continue
            total_variants = data['used'] + data['missing']
            avg_score = data['sum'] / data['used'] if data['used'] > 0 else 0.0
            missing_pct = (data['missing'] / total_variants) * 100.0 if total_variants > 0 else 0.0
            results.append({'IID': iid, 'SCORE_TRUTH': avg_score, 'MISSING_PCT_TRUTH': missing_pct})

        return pd.DataFrame(results), iids_expected_to_fail

    # ---- Main Test Logic ----
    bim_df, score_df, individuals, genotypes_df = _generate_test_data()
    truth_df, iids_expected_to_fail = _calculate_simple_truth(bim_df, score_df, individuals, genotypes_df)
    
    _write_plink_files(prefix, bim_df, individuals, genotypes_df)
    _write_score_file(prefix.with_suffix(".score"), score_df)

    # --- Execute and Validate Gnomon ---
    gnomon_result = run_cmd_func([gnomon_path, "--score", prefix.with_suffix(".score").name, prefix.name],
                                "Simple Gnomon Test", cwd=workdir)
    
    is_gnomon_ok = False
    if gnomon_result.returncode != 0:
        # Gnomon crashed, check if it was for the right reason
        failed_iid = next((iid for iid in iids_expected_to_fail if iid in gnomon_result.stderr), None)
        if failed_iid:
            print(f"✅ Gnomon correctly failed due to data inconsistency for '{failed_iid}'.")
            is_gnomon_ok = True
        else:
            print("❌ Gnomon failed unexpectedly. See stderr for details.")
    else:
        # Gnomon succeeded, check if it should have failed
        if iids_expected_to_fail:
            print(f"❌ Gnomon SUCCEEDED when it should have FAILED for IIDs: {iids_expected_to_fail}")
        else:
            # Gnomon succeeded as expected, now validate the results
            gnomon_df = pd.read_csv(workdir / "simple_test.sscore", sep='\t').rename(columns={
                '#IID':'IID', 'simple_score_AVG':'SCORE_GNOMON', 'simple_score_MISSING_PCT': 'MISSING_PCT_GNOMON'
            })
            merged_df = pd.merge(truth_df, gnomon_df, on='IID', how='left')
            print("\n--- Comparison of Scores (Simple Test) ---")
            print(merged_df.to_markdown(index=False, floatfmt=".6f"))

            scores_ok = np.allclose(merged_df['SCORE_TRUTH'], merged_df['SCORE_GNOMON'])
            missing_ok = np.allclose(merged_df['MISSING_PCT_TRUTH'], merged_df['MISSING_PCT_GNOMON'])

            if scores_ok and missing_ok:
                print("✅ Gnomon scores and missingness percentages are correct.")
                is_gnomon_ok = True
            else:
                if not scores_ok: print("❌ Gnomon scores do not match ground truth.")
                if not missing_ok: print("❌ Gnomon missingness percentages are incorrect.")
    
    if is_gnomon_ok:
        print("\n✅ Simple Dosage Test Successful: Gnomon behaved as expected.")
        return True
    else:
        print("\n❌ Simple Dosage Test FAILED.")
        return False
def run_and_validate_tools(runtimes):
    """
    Downloads tools, runs them, validates results, and collects runtimes.
    """
    PLINK2_URL = "https://s3.amazonaws.com/plink2-assets/alpha6/plink2_linux_avx2_20250609.zip"
    PLINK2_BINARY_PATH = WORKDIR / "plink2"
    GNOMON_BINARY_PATH = Path("./target/release/gnomon").resolve()
    PYLINK_SCRIPT_PATH = Path("test/pylink.py").resolve()

    overall_success = True

    def _print_header(title: str, char: str = "-"):
        width = 70
        print(f"\n{char*4} {title} {'-'*(width - len(title) - 5)}")

def run_command(cmd: list, step_name: str, cwd: Path) -> subprocess.CompletedProcess:
    """
    Executes a command and returns the full CompletedProcess object.

    This function allows the caller to inspect the return code,
    stdout, and stderr, which is essential for validating both successful
    runs and expected failures.
    """
    _print_header(f"Executing: {step_name}")
    cmd_str = [str(c) for c in cmd]
    print(f"  > Command: {' '.join(cmd_str)}")
    print(f"  > CWD: {cwd}")

    proc_env = os.environ.copy()
    if "gnomon" in str(cmd_str[0]):
        proc_env["RUST_BACKTRACE"] = "1"

    start_time = time.monotonic()
    try:
        result = subprocess.run(
            cmd_str, capture_output=True, text=True, cwd=cwd, env=proc_env, timeout=600
        )
        print("--- OUTPUT ---")
        if result.stdout: print(result.stdout)
        if result.stderr:
            print("--- STDERR ---")
            print(result.stderr)

        duration = time.monotonic() - start_time
        if result.returncode == 0:
            print(f"\n  > Success. (Completed in {duration:.4f}s)")
        else:
            print(f"\n  > Process exited with code {result.returncode}. (Completed in {duration:.4f}s)")
        
        return result

    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        print(f"  > ❌ FAILED to execute command: {e}")
        # Return a dummy CompletedProcess object to avoid crashing the test harness
        return subprocess.CompletedProcess(args=cmd_str, returncode=127, stdout="", stderr=str(e))

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
                    with open(zip_path, 'wb') as f:
                        shutil.copyfileobj(r.raw, f)
                with zipfile.ZipFile(zip_path, 'r') as z:
                    for member in z.infolist():
                        if member.filename.endswith('plink2') and not member.is_dir():
                            with z.open(member) as source, open(PLINK2_BINARY_PATH, 'wb') as target:
                                shutil.copyfileobj(source, target)
                            break
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
            truth_df = pd.read_csv(OUTPUT_PREFIX.with_suffix(".truth.sscore"), sep='\t')\
                          .rename(columns={'PRS_AVG':'SCORE_TRUTH'})[['IID','SCORE_TRUTH']]
            gnomon_df = pd.read_csv(OUTPUT_PREFIX.with_suffix(".sscore"), sep='\t')\
                           .rename(columns={'#IID':'IID','simulated_score_AVG':'SCORE_GNOMON'})[['IID','SCORE_GNOMON']]
            plink2_df_raw = pd.read_csv(WORKDIR/"plink2_results.sscore", sep=r'\s+')\
                                .rename(columns={'#IID':'IID'})
            plink2_df = plink2_df_raw.assign(SCORE_PLINK2=plink2_df_raw['SCORE1_AVG']*2.0)[['IID','SCORE_PLINK2']]
            pylink_df_raw = pd.read_csv(WORKDIR/"pylink_results.sscore", sep='\t')\
                                 .rename(columns={'#IID':'IID'})
            pylink_df = pylink_df_raw.assign(SCORE_PYLINK=pylink_df_raw['SCORE1_AVG']*2.0)[['IID','SCORE_PYLINK']]
        except (FileNotFoundError, KeyError) as e:
            print(f"  > ❌ ERROR: Failed to load or parse a result file. Error: {e}.")
            return False

        merged_df = truth_df.merge(gnomon_df, on='IID')\
                            .merge(plink2_df, on='IID')\
                            .merge(pylink_df, on='IID')
        score_cols = ['SCORE_TRUTH','SCORE_GNOMON','SCORE_PLINK2','SCORE_PYLINK']
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

        corr_g = corr_matrix.loc['SCORE_TRUTH','SCORE_GNOMON']
        corr_p = corr_matrix.loc['SCORE_TRUTH','SCORE_PLINK2']
        corr_py = corr_matrix.loc['SCORE_TRUTH','SCORE_PYLINK']

        gnomon_ok = corr_g > CORR_THRESHOLD and mad_g < MAD_THRESHOLD
        plink_ok = corr_p > CORR_THRESHOLD and mad_p < MAD_THRESHOLD
        pylink_ok = corr_py > CORR_THRESHOLD and mad_py < MAD_THRESHOLD

        if gnomon_ok and plink_ok and pylink_ok:
            print(f"\n✅ Large-Scale Validation SUCCESS: All tools passed criteria (Corr > {CORR_THRESHOLD}, MAD < {MAD_THRESHOLD}).")
            return True
        else:
            print(f"\n❌ Large-Scale Validation FAILED:")
            if not gnomon_ok:
                print(f"  - Gnomon: Corr={corr_g:.8f}, MAD={mad_g:.10f}")
            if not plink_ok:
                print(f"  - PLINK2: Corr={corr_p:.8f}, MAD={mad_p:.10f}")
            if not pylink_ok:
                print(f"  - PyLink: Corr={corr_py:.8f}, MAD={mad_py:.10f}")
            return False

    if not setup_tools():
        return False

    if not run_simple_dosage_test(WORKDIR, GNOMON_BINARY_PATH, PLINK2_BINARY_PATH, PYLINK_SCRIPT_PATH, run_command):
        overall_success = False

    print("\n" + "="*80)
    print("= Running Large-Scale Simulation and Validation")
    print("="*80)

    if not run_command(
        [GNOMON_BINARY_PATH, "--score", OUTPUT_PREFIX.with_suffix(".gnomon.score").name, OUTPUT_PREFIX.name],
        "Large-Scale Gnomon",
        WORKDIR
    ):
        overall_success = False

    if not run_command(
        [f"./{PLINK2_BINARY_PATH.name}", "--bfile", OUTPUT_PREFIX.name,
         "--score", OUTPUT_PREFIX.with_suffix(".gnomon.score").name,
         "1", "2", "4", "header", "no-mean-imputation", "--out", "plink2_results"],
        "Large-Scale PLINK2",
        WORKDIR
    ):
        overall_success = False

    if not run_command(
        ["python3", PYLINK_SCRIPT_PATH.as_posix(), "--bfile", OUTPUT_PREFIX.name,
         "--score", OUTPUT_PREFIX.with_suffix(".gnomon.score").name,
         "--out", "pylink_results", "1", "2", "4"],
        "Large-Scale PyLink",
        WORKDIR
    ):
        overall_success = False

    if overall_success:
        if not analyze_large_scale_results():
            overall_success = False

    return overall_success

def print_runtime_summary(runtimes):
    """
    Print mean runtimes for large-scale methods.
    """
    print("\n" + "="*50)
    print("= Mean Runtime for Large-Scale Methods")
    print("="*50)
    if runtimes:
        df = pd.DataFrame(runtimes)
        summary = df.groupby('Method')['Runtime (s)'].mean().reset_index()
        print(summary.to_markdown(index=False, floatfmt=".4f"))
    else:
        print("No large-scale runtimes were recorded.")

def cleanup():
    """
    Clean up the workspace directory.
    """
    if WORKDIR.exists():
        shutil.rmtree(WORKDIR)
        print(f"Cleaned up {WORKDIR}")

def main():
    """
    Main pipeline: simulate, write files, validate.
    """
    np.random.seed(42)
    exit_code = 0
    runtimes = []

    if WORKDIR.exists():
        shutil.rmtree(WORKDIR)
    WORKDIR.mkdir()

    try:
        print("--- Starting Full Simulation and File Writing Pipeline ---")
        variants_df = generate_variants_and_weights()
        gen_pristine = generate_genotypes(variants_df)
        gen_missing = introduce_missingness(gen_pristine)
        prs_df = calculate_ground_truth_prs(gen_missing, variants_df)
        write_output_files(prs_df, variants_df, gen_missing, OUTPUT_PREFIX)
        print("\n--- Simulation and File Writing Finished Successfully ---")

        if not run_and_validate_tools(runtimes):
            exit_code = 1

    except Exception as e:
        print(f"\n--- An unexpected error occurred: {e} ---")
        exit_code = 1

    finally:
        print_runtime_summary(runtimes)
        cleanup()

    sys.exit(exit_code)

if __name__ == "__main__":
    main()
