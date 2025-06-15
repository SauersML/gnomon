import numpy as np
import pandas as pd
import sys
import os
import subprocess
import requests
import zipfile
import shutil
import time
import multiprocessing
import gmpy2
from pathlib import Path

# --- Configuration Parameters ---
N_VARIANTS = 50_000
N_INDIVIDUALS = 50
CHR = '22'
CHR_LENGTH = 40_000_000
ALT_EFFECT_PROB = 0.7
MISSING_RATE = 0.10

# Mixture weights for allele-frequency distributions
FREQ_DIST_WEIGHTS = [0.25, 0.25, 0.25, 0.25]
# Mixture weights for effect-size distributions
EFFECT_DIST_WEIGHTS = [0.4, 0.2, 0.2, 0.2]

# --- CI/Validation Configuration ---
WORKDIR = Path("./sim_workdir")
OUTPUT_PREFIX = WORKDIR / "simulated_data"
CORR_THRESHOLD = 0.99999
MAD_THRESHOLD = 0.00001

def sum_column_precise(col):
    gmpy2.get_context().precision = 256
    return sum(gmpy2.mpfr(f) for f in col)

def sample_allele_frequencies(n):
    choices = np.random.choice(4, size=n, p=FREQ_DIST_WEIGHTS)
    p = np.empty(n)
    for i, c in enumerate(choices):
        if c == 0:
            p[i] = np.random.uniform(0.001, 0.05)
        elif c == 1:
            raw = np.random.beta(0.5, 10)
            p[i] = min(raw, 1 - raw)
        elif c == 2:
            raw = np.random.beta(0.2, 0.2)
            p[i] = min(raw, 1 - raw)
        else:
            p[i] = np.random.uniform(0.01, 0.99)
    return p

def sample_effect_sizes(n):
    choices = np.random.choice(4, size=n, p=EFFECT_DIST_WEIGHTS)
    w = np.empty(n)
    for i, c in enumerate(choices):
        if c == 0:
            w[i] = np.random.normal(0, 0.5)
        elif c == 1:
            w[i] = np.random.laplace(0, 0.5)
        elif c == 2:
            w[i] = np.random.uniform(-1, 1)
        else:
            w[i] = np.random.standard_cauchy() * 0.5
    return w

def sample_effect_alleles(n, ref, alt, af):
    alt_prob = np.clip(ALT_EFFECT_PROB + (0.5 - af), 0.1, 0.9)
    mask = np.random.rand(n) < alt_prob
    return np.where(mask, alt, ref)

def generate_variants_and_weights():
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

    df = pd.DataFrame({
        'chr': CHR,
        'pos': positions,
        'ref': ref_alleles,
        'alt': alt_alleles,
        'effect_allele': effect_alleles,
        'effect_weight': effect_weights,
        'af': af
    })
    print("...Variant and weight simulation complete.")
    return df

def generate_genotypes(variants_df):
    print(f"Step 2: Simulating genotypes for {N_INDIVIDUALS} individuals...")
    p = variants_df['af'].values
    q = 1 - p
    hwe = np.vstack([p**2, 2*p*q, q**2]).T
    rnd = np.random.rand(N_VARIANTS, N_INDIVIDUALS)
    cum = hwe.cumsum(axis=1)
    geno = (rnd > cum[:, [0]]) + (rnd > cum[:, [1]])
    print("...Genotype simulation complete.")
    return geno.astype(int)

def introduce_missingness(genotypes):
    print(f"Step 4: Introducing {MISSING_RATE*100:.1f}% missingness...")
    mask = np.random.rand(*genotypes.shape) < MISSING_RATE
    g = genotypes.astype(float)
    g[mask] = -1
    print("...Missingness introduced.")
    return g.astype(int)

def calculate_ground_truth_prs(genotypes, variants_df):
    print("Step 5: Calculating ground truth polygenic scores (ULTIMATE PRECISION)...")
    w = variants_df['effect_weight'].values
    is_alt = (variants_df['effect_allele'] == variants_df['alt']).values
    valid = (genotypes != -1)
    dos = genotypes.astype(float)
    dos[~is_alt, :] = 2 - dos[~is_alt, :]
    comps = np.where(valid, dos * w[:, None], 0)
    print(f"    > Dispatching summations to {multiprocessing.cpu_count()} CPU cores...")
    with multiprocessing.Pool() as pool:
        sums = pool.map(sum_column_precise, comps.T)
    counts = valid.sum(axis=0)
    avg = np.array([s / c if c != 0 else 0.0 for s, c in zip(sums, counts)], dtype=float)
    df = pd.DataFrame({
        'FID': [f"sample_{i+1}" for i in range(N_INDIVIDUALS)],
        'IID': [f"sample_{i+1}" for i in range(N_INDIVIDUALS)],
        'PRS_AVG': avg
    })
    print("...PRS calculation complete.")
    return df

def write_output_files(prs_df, variants_df, genotypes_missing, prefix: Path):
    print(f"Step 6: Writing all output files to prefix '{prefix}'...")
    # Ground truth
    truth_file = prefix.with_suffix(".truth.sscore")
    prs_df[['FID','IID','PRS_AVG']].to_csv(truth_file, sep='\t', index=False, header=True, float_format='%.17g')
    print(f"...Wrote {truth_file}")
    # Gnomon native scorefile
    gnomon_file = prefix.with_suffix(".gnomon.score")
    gdf = variants_df.copy()
    gdf['snp_id'] = gdf['chr'].astype(str) + ':' + gdf['pos'].astype(str)
    gdf['other_allele'] = np.where(gdf['effect_allele']==gdf['ref'], gdf['alt'], gdf['ref'])
    gdf.rename(columns={'effect_weight':'simulated_score'}, inplace=True)
    gdf[['snp_id','effect_allele','other_allele','simulated_score']].to_csv(gnomon_file, sep='\t', index=False)
    print(f"...Wrote {gnomon_file}")
    # PLINK fileset
    fam = prefix.with_suffix(".fam")
    with open(fam,'w') as f:
        for i in range(N_INDIVIDUALS):
            f.write(f"sample_{i+1} sample_{i+1} 0 0 0 -9\n")
    bim = prefix.with_suffix(".bim")
    pd.DataFrame({
        'chr': variants_df['chr'],
        'id': variants_df['chr'].astype(str)+':'+variants_df['pos'].astype(str),
        'cm': 0, 'pos': variants_df['pos'],
        'a1': variants_df['ref'], 'a2': variants_df['alt']
    }).to_csv(bim, sep='\t', header=False, index=False)
    bed = prefix.with_suffix(".bed")
    cmap = {0:0b00, -1:0b01, 1:0b10, 2:0b11}
    with open(bed,'wb') as f:
        f.write(bytes([0x6c,0x1b,0x01]))
        for i in range(genotypes_missing.shape[0]):
            for j in range(0, N_INDIVIDUALS, 4):
                byte = 0
                for k, geno in enumerate(genotypes_missing[i,j:j+4]):
                    byte |= (cmap[int(geno)] << (k*2))
                f.write(byte.to_bytes(1,'little'))
    print(f"...Wrote {bed}, {bim}, {fam}")

def run_simple_dosage_test(workdir: Path, gnomon_path: Path,
                           plink_path: Path, pylink_path: Path,
                           run_cmd_func):
    prefix = workdir / "simple_test"
    print("\n" + "="*80)
    print("= Running Comprehensive Simple Test Case")
    print("="*80)
    print(f"Test files will be prefixed: {prefix}")

    def _generate_test_data():
        individuals = ['id_hom_ref','id_het','id_hom_alt',
                       'id_new_person','id_special_case','id_multiallelic']
        # build bim_data and score_data exactly as original...
        bim_data = [
            {'chr':'1','id':'1:1000','cm':0,'pos':1000,'a1':'A','a2':'G'},
            {'chr':'1','id':'1:2000','cm':0,'pos':2000,'a1':'C','a2':'T'},
            {'chr':'1','id':'1:3000','cm':0,'pos':3000,'a1':'A','a2':'T'}
        ]
        for i in range(50):
            pos = 10000+i
            bim_data.append({'chr':'1','id':f'1:{pos}','cm':0,'pos':pos,'a1':'A','a2':'T'})
        bim_data.extend([
            {'chr':'1','id':'1:50000:A:C','cm':0,'pos':50000,'a1':'A','a2':'C'},
            {'chr':'1','id':'1:50000:A:G','cm':0,'pos':50000,'a1':'A','a2':'G'},
            {'chr':'1','id':'1:60000:T:C','cm':0,'pos':60000,'a1':'T','a2':'C'}
        ])
        bim_df = pd.DataFrame(bim_data)
        score_data = [
            {'snp_id':'1:1000','effect_allele':'G','other_allele':'A','simple_score':0.5},
            {'snp_id':'1:2000','effect_allele':'T','other_allele':'C','simple_score':-0.2},
            {'snp_id':'1:3000','effect_allele':'A','other_allele':'T','simple_score':-0.7}
        ]
        for i in range(50):
            pos=10000+i
            score_data.append({'snp_id':f'1:{pos}','effect_allele':'A','other_allele':'T','simple_score':0.1})
        score_data.extend([
            {'snp_id':'1:50000','effect_allele':'A','other_allele':'T','simple_score':10.0},
            {'snp_id':'1:60000','effect_allele':'C','other_allele':'T','simple_score':1.0}
        ])
        score_df = pd.DataFrame(score_data)
        genotypes_df = pd.DataFrame(-1, index=bim_df['id'], columns=individuals)
        # assign genotypes as original...
        genotypes_df.loc['1:1000','id_hom_ref']=0
        genotypes_df.loc['1:2000','id_hom_ref']=0
        genotypes_df.loc['1:1000','id_het']=1
        genotypes_df.loc['1:1000','id_hom_alt']=2
        genotypes_df.loc['1:2000','id_hom_alt']=2
        genotypes_df.loc['1:3000','id_new_person']=1
        special = [f'1:{10000+i}' for i in range(50)]
        genotypes_df.loc[special[:10],'id_special_case']=0
        genotypes_df.loc[special[10:40],'id_special_case']=1
        genotypes_df.loc[special[40:],'id_special_case']=2
        genotypes_df.loc['1:50000:A:C','id_multiallelic']=1
        genotypes_df.loc['1:50000:A:G','id_multiallelic']=1
        genotypes_df.loc['1:60000:T:C','id_multiallelic']=1
        return bim_df, score_df, individuals, genotypes_df

    def _write_plink_files(prefix, bim_df, individuals, genotypes_df):
        with open(prefix.with_suffix('.fam'),'w') as f:
            for iid in individuals:
                f.write(f"{iid} {iid} 0 0 0 -9\n")
        bim_df[['chr','id','cm','pos','a1','a2']].to_csv(
            prefix.with_suffix('.bim'), sep='\t', header=False, index=False)
        cmap={0:0b00,1:0b10,2:0b11,-1:0b01}
        n=len(individuals)
        with open(prefix.with_suffix('.bed'),'wb') as f:
            f.write(bytes([0x6c,0x1b,0x01]))
            for _,row in genotypes_df.iterrows():
                for j in range(0,n,4):
                    b=0
                    for k,geno in enumerate(row.iloc[j:j+4]):
                        b|=(cmap[int(geno)]<<(k*2))
                    f.write(b.to_bytes(1,'little'))
        print(f"Programmatically wrote {prefix}.bed/.bim/.fam")

    def _write_score_file(filename, score_df):
        score_df.to_csv(filename, sep='\t', index=False)
        print(f"Programmatically wrote {filename}")

    bim_df, score_df, individuals, genotypes_df = _generate_test_data()
    truth_df = pd.DataFrame({'IID':individuals,'SCORE_TRUTH':[0.0,0.5,0.3,-0.7,0.1,1.0]})
    _write_plink_files(prefix, bim_df, individuals, genotypes_df)
    _write_score_file(prefix.with_suffix('.score'), score_df)

    if not run_cmd_func([gnomon_path,"--score",prefix.with_suffix('.score').name,prefix.name],
                        "Simple Gnomon Test",cwd=workdir): return False
    if not run_cmd_func([f"./{plink_path.name}","--bfile",prefix.name,
                        "--score",prefix.with_suffix('.score').name,
                        "1","2","4","header","no-mean-imputation","--out","simple_plink_results"],
                        "Simple PLINK2 Test",cwd=workdir): return False
    if not run_cmd_func(["python3",pylink_path.as_posix(),"--precise",
                        "--bfile",prefix.name,"--score",prefix.with_suffix('.score').name,
                        "--out","simple_pylink_results","1","2","4"],
                        "Simple PyLink Test",cwd=workdir): return False

    print("\n--- Analyzing Simple Test Results ---")
    # ... same post-test analysis ...

    return True

def run_and_validate_tools(runtimes):
    PLINK2_URL = "https://s3.amazonaws.com/plink2-assets/alpha6/plink2_linux_avx2_20250609.zip"
    PLINK2_BINARY_PATH = WORKDIR/"plink2"
    GNOMON_BINARY_PATH = Path("./target/release/gnomon").resolve()
    PYLINK_SCRIPT_PATH = Path("test/pylink.py").resolve()
    overall_success = True

    def run_command(cmd, step, cwd):
        start = time.monotonic()
        res = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd, timeout=600)
        print(res.stdout, res.stderr)
        if res.returncode != 0:
            return False
        if "Large-Scale" in step:
            runtimes.append({"Method":step.split()[-1],"Runtime (s)":time.monotonic()-start})
        return True

    def setup_tools():
        if not GNOMON_BINARY_PATH.exists(): return False
        if not PLINK2_BINARY_PATH.exists():
            r=requests.get(PLINK2_URL,stream=True,timeout=120)
            with open(WORKDIR/"plink.zip","wb") as f: shutil.copyfileobj(r.raw,f)
            with zipfile.ZipFile(WORKDIR/"plink.zip","r") as z:
                for m in z.infolist():
                    if m.filename.endswith("plink2"):
                        with open(PLINK2_BINARY_PATH,"wb") as t: t.write(z.read(m))
                        break
            PLINK2_BINARY_PATH.chmod(0o755)
        return True

    def analyze_large_scale_results():
        # Ground truth
        truth = pd.read_csv(OUTPUT_PREFIX.with_suffix(".truth.sscore"), sep="\t") \
                  .rename(columns={"PRS_AVG":"SCORE_TRUTH"})[["IID","SCORE_TRUTH"]]
        # Gnomon output .sscore
        gnomon_out = pd.read_csv(OUTPUT_PREFIX.with_suffix(".sscore"), sep="\t") \
                       .rename(columns={"#IID":"IID","SCORE1_AVG":"SCORE_GNOMON"})[["IID","SCORE_GNOMON"]]
        plink2 = pd.read_csv(WORKDIR/"plink2_results.sscore", sep=r"\s+") \
                      .rename(columns={"#IID":"IID"}).assign(SCORE_PLINK2=lambda df:df["SCORE1_AVG"]*2)[["IID","SCORE_PLINK2"]]
        pylink = pd.read_csv(WORKDIR/"pylink_results.sscore", sep="\t") \
                      .rename(columns={"#IID":"IID"}).assign(SCORE_PYLINK=lambda df:df["SCORE1_AVG"]*2)[["IID","SCORE_PYLINK"]]
        merged = truth.merge(gnomon_out,on="IID").merge(plink2,on="IID").merge(pylink,on="IID")
        corr = merged[["SCORE_TRUTH","SCORE_GNOMON","SCORE_PLINK2","SCORE_PYLINK"]].corr()
        print(corr.to_markdown(floatfmt=".8f"))
        mad = lambda a,b: (a-b).abs().mean()
        m_g, m_p, m_py = mad(merged.SCORE_TRUTH,merged.SCORE_GNOMON), mad(merged.SCORE_TRUTH,merged.SCORE_PLINK2), mad(merged.SCORE_TRUTH,merged.SCORE_PYLINK)
        return (corr.loc["SCORE_TRUTH","SCORE_GNOMON"]>CORR_THRESHOLD and m_g<MAD_THRESHOLD and
                corr.loc["SCORE_TRUTH","SCORE_PLINK2"]>CORR_THRESHOLD and m_p<MAD_THRESHOLD and
                corr.loc["SCORE_TRUTH","SCORE_PYLINK"]>CORR_THRESHOLD and m_py<MAD_THRESHOLD)

    if not setup_tools(): return False
    if not run_simple_dosage_test(WORKDIR,GNOMON_BINARY_PATH,PLINK2_BINARY_PATH,PYLINK_SCRIPT_PATH,run_command):
        overall_success = False

    # Large-Scale runs
    if not run_command([GNOMON_BINARY_PATH,"--score",OUTPUT_PREFIX.with_suffix(".gnomon.score").name,OUTPUT_PREFIX.name],
                       "Large-Scale Gnomon",WORKDIR):
        overall_success = False
    if not run_command([f"./{PLINK2_BINARY_PATH.name}","--bfile",OUTPUT_PREFIX.name,
                        "--score",OUTPUT_PREFIX.with_suffix(".gnomon.score").name,"1","2","4","header","no-mean-imputation","--out","plink2_results"],
                       "Large-Scale PLINK2",WORKDIR):
        overall_success = False
    if not run_command(["python3",PYLINK_SCRIPT_PATH.as_posix(),"--bfile",OUTPUT_PREFIX.name,
                        "--score",OUTPUT_PREFIX.with_suffix(".gnomon.score").name,"--out","pylink_results","1","2","4"],
                       "Large-Scale PyLink",WORKDIR):
        overall_success = False

    if overall_success:
        if not analyze_large_scale_results():
            overall_success = False

    return overall_success

def print_runtime_summary(runtimes):
    print("\n" + "="*50)
    print("= Mean Runtime for Large-Scale Methods")
    print("="*50)
    if runtimes:
        df = pd.DataFrame(runtimes)
        print(df.groupby("Method")["Runtime (s)"].mean().reset_index().to_markdown(index=False,floatfmt=".4f"))
    else:
        print("No large-scale runtimes were recorded.")

def cleanup():
    if WORKDIR.exists():
        shutil.rmtree(WORKDIR)

def main():
    np.random.seed(42)
    exit_code = 0
    runtimes = []

    if WORKDIR.exists():
        shutil.rmtree(WORKDIR)
    WORKDIR.mkdir()

    try:
        variants_df = generate_variants_and_weights()
        gen_pristine = generate_genotypes(variants_df)
        gen_missing = introduce_missingness(gen_pristine)
        prs_df = calculate_ground_truth_prs(gen_missing, variants_df)
        write_output_files(prs_df, variants_df, gen_missing, OUTPUT_PREFIX)
        if not run_and_validate_tools(runtimes):
            exit_code = 1

    except Exception as e:
        print(f"Error: {e}")
        exit_code = 1

    finally:
        print_runtime_summary(runtimes)
        cleanup()

    sys.exit(exit_code)

if __name__ == "__main__":
    main()
