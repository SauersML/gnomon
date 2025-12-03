#!/usr/bin/env python3
"""
Automated Ancient DNA PGS Generation Pipeline.

1. Downloads Selection Summary Statistics from Harvard Dataverse.
2. Downloads UCSC liftOver tools.
3. Generates hg19 Additive Polygenic Score files.
4. Lifts data from hg19 to hg38.
5. Generates hg38 Additive Polygenic Score files.
6. Saves the full hg38 dataset.
"""

import gzip
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import urllib.request
import pandas as pd

# --- Constants ---
DATAVERSE_DOI = "doi:10.7910/DVN/7RVV9N"
DATAVERSE_BASE = "https://dataverse.harvard.edu"
TARGET_FILENAME = "Selection_Summary_Statistics_01OCT2025.tsv.gz"
CHAIN_URL = "https://hgdownload.soe.ucsc.edu/goldenPath/hg19/liftOver/hg19ToHg38.over.chain.gz"

def download_with_headers(url, dest_path):
    """Downloads a file using Mozilla/5.0 headers to avoid 403 errors."""
    print(f"Downloading {url}...")
    headers = {"User-Agent": "Mozilla/5.0"}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as response, open(dest_path, 'wb') as out_file:
        shutil.copyfileobj(response, out_file)
    print(f"Saved to {dest_path}")

def get_dataverse_file():
    """Queries Dataverse API with headers and downloads the specific file."""
    print("Querying Dataverse metadata...")
    url = f"{DATAVERSE_BASE}/api/datasets/:persistentId/?persistentId={DATAVERSE_DOI}"
    headers = {"Accept": "application/json", "User-Agent": "Mozilla/5.0"}
    
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as response:
        metadata = json.load(response)
    
    # Extract File ID
    target_file_data = [f for f in metadata['data']['latestVersion']['files'] 
                        if f['dataFile']['filename'] == TARGET_FILENAME][0]
    
    file_id = target_file_data['dataFile']['id']
    download_link = f"{DATAVERSE_BASE}/api/access/datafile/{file_id}"
    
    dest = os.path.join(os.getcwd(), TARGET_FILENAME)
    download_with_headers(download_link, dest)
    return dest

def get_liftover_binary():
    """Downloads the correct liftOver binary for the OS."""
    base_url = "http://hgdownload.soe.ucsc.edu/admin/exe"
    url = f"{base_url}/macOSX.x86_64/liftOver" if sys.platform == "darwin" else f"{base_url}/linux.x86_64/liftOver"
    
    dest = os.path.join(os.getcwd(), "liftOver")
    download_with_headers(url, dest)
    os.chmod(dest, os.stat(dest).st_mode | stat.S_IEXEC)
    return dest

def get_chain_file():
    """Downloads the hg19ToHg38 chain file."""
    dest = os.path.join(os.getcwd(), "hg19ToHg38.over.chain.gz")
    download_with_headers(CHAIN_URL, dest)
    return dest

def generate_pgs_files(df, build_name):
    """Creates Positive and Negative PGS files for the given dataframe."""
    print(f"Generating PGS files for {build_name}...")
    
    # Prepare base dataframe
    base_df = pd.DataFrame()
    base_df['rsID'] = df['RSID'].fillna('.')
    base_df['chr_name'] = df['CHROM']
    base_df['chr_position'] = df['POS']
    base_df['effect_allele'] = df['ALT']
    base_df['other_allele'] = df['REF']
    base_df['weight_type'] = 'selection_coefficient_magnitude'
    base_df['S_raw'] = df['S']

    # 1. Positive Selection (S > 0)
    pos_df = base_df[base_df['S_raw'] > 0].copy()
    pos_df['effect_weight'] = pos_df['S_raw'].abs()
    pos_df = pos_df.drop(columns=['S_raw'])
    
    fname_pos = f"ancient_dna_positive_selection_{build_name}.txt"
    with open(fname_pos, 'w') as f:
        f.write("#format=PGS_2.0\n")
        f.write(f"#genome_build={build_name}\n")
        f.write("#selection_direction=positive\n")
        pos_df.to_csv(f, sep='\t', index=False)
    print(f"Written: {fname_pos}")

    # 2. Negative Selection (S < 0)
    neg_df = base_df[base_df['S_raw'] < 0].copy()
    neg_df['effect_weight'] = neg_df['S_raw'].abs()
    neg_df = neg_df.drop(columns=['S_raw'])
    
    fname_neg = f"ancient_dna_negative_selection_{build_name}.txt"
    with open(fname_neg, 'w') as f:
        f.write("#format=PGS_2.0\n")
        f.write(f"#genome_build={build_name}\n")
        f.write("#selection_direction=negative\n")
        neg_df.to_csv(f, sep='\t', index=False)
    print(f"Written: {fname_neg}")

def run_liftover(df, binary_path, chain_path):
    """Executes UCSC liftOver via subprocess."""
    print("Performing LiftOver (hg19 -> hg38)...")
    
    # Prepare BED input
    bed_input = pd.DataFrame()
    bed_input['chrom'] = 'chr' + df['CHROM'].astype(str).str.lstrip('chr')
    bed_input['start'] = df['POS'] - 1
    bed_input['end'] = df['POS']
    bed_input['id'] = df.index
    
    tmp_in = tempfile.NamedTemporaryFile(delete=False, mode='w')
    tmp_out = tempfile.NamedTemporaryFile(delete=False, mode='w')
    tmp_err = tempfile.NamedTemporaryFile(delete=False, mode='w')
    
    bed_input.to_csv(tmp_in.name, sep='\t', header=False, index=False)
    tmp_in.close()
    
    subprocess.run(
        [binary_path, tmp_in.name, chain_path, tmp_out.name, tmp_err.name],
        check=True
    )
    
    lifted = pd.read_csv(tmp_out.name, sep='\t', header=None, names=['chrom', 'start', 'end', 'id'])
    
    os.remove(tmp_in.name)
    os.remove(tmp_out.name)
    os.remove(tmp_err.name)
    
    merged = df.loc[lifted['id']].copy()
    merged['CHROM'] = lifted['chrom'].str.replace('chr', '').values
    merged['POS'] = lifted['end'].values
    
    return merged

def main():
    # 1. Download Everything
    input_file = get_dataverse_file()
    liftover_bin = get_liftover_binary()
    chain_file = get_chain_file()

    # 2. Load hg19 Data
    print("Loading Data...")
    df = pd.read_csv(input_file, sep='\t', compression='gzip', comment='#')
    
    # 3. Generate hg19 Scores
    generate_pgs_files(df, "hg19")
    
    # 4. LiftOver to hg38
    df_hg38 = run_liftover(df, liftover_bin, chain_file)
    
    # 5. Save Full hg38 Dataset
    hg38_filename = "Selection_Summary_Statistics_hg38.tsv.gz"
    df_hg38.to_csv(hg38_filename, sep='\t', index=False, compression='gzip')
    print(f"Written: {hg38_filename}")
    
    # 6. Generate hg38 Scores
    generate_pgs_files(df_hg38, "hg38")

    print("Pipeline Complete.")

if __name__ == "__main__":
    main()
