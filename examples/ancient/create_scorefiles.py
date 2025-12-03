import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import urllib.request
import pandas as pd
import gzip

# --- Configuration ---
CHUNK_SIZE = 50000  # Rows to process at a time
DATAVERSE_DOI = "doi:10.7910/DVN/7RVV9N"
DATAVERSE_BASE = "https://dataverse.harvard.edu"
TARGET_FILENAME = "Selection_Summary_Statistics_01OCT2025.tsv.gz"
CHAIN_URL = "https://hgdownload.soe.ucsc.edu/goldenPath/hg19/liftOver/hg19ToHg38.over.chain.gz"

# --- Helpers ---

def print_progress(current, total, prefix="Progress"):
    """Simple text-based progress bar."""
    if total:
        percent = 100 * (current / float(total))
        bar = '█' * int(percent // 2) + '-' * (50 - int(percent // 2))
        sys.stdout.write(f"\r{prefix} |{bar}| {percent:.1f}%")
    else:
        sys.stdout.write(f"\r{prefix}: {current / 1024 / 1024:.2f} MB")
    sys.stdout.flush()

def download_file(url, dest_path):
    """Downloads with resume check, headers (403 fix), and progress bar."""
    if os.path.exists(dest_path):
        print(f"File already exists: {dest_path}")
        return

    print(f"Downloading {url}...")
    headers = {"User-Agent": "Mozilla/5.0"}
    req = urllib.request.Request(url, headers=headers)
    
    with urllib.request.urlopen(req) as response:
        total_size = response.getheader('Content-Length')
        if total_size: 
            total_size = int(total_size)
        
        with open(dest_path, 'wb') as out_file:
            downloaded = 0
            while True:
                buffer = response.read(8192)
                if not buffer:
                    break
                downloaded += len(buffer)
                out_file.write(buffer)
                print_progress(downloaded, total_size, prefix="Downloading")
    print() # Newline

def get_dataverse_file():
    """Queries metadata and downloads dataset."""
    dest = os.path.join(os.getcwd(), TARGET_FILENAME)
    if os.path.exists(dest):
        print(f"Dataset already exists: {dest}")
        return dest

    print("Querying Dataverse API...")
    url = f"{DATAVERSE_BASE}/api/datasets/:persistentId/?persistentId={DATAVERSE_DOI}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    
    with urllib.request.urlopen(req) as response:
        metadata = json.load(response)
    
    files = metadata['data']['latestVersion']['files']
    target_data = next(f for f in files if f['dataFile']['filename'] == TARGET_FILENAME)
    file_id = target_data['dataFile']['id']
    
    download_link = f"{DATAVERSE_BASE}/api/access/datafile/{file_id}"
    download_file(download_link, dest)
    return dest

def get_liftover_tools():
    """Gets binary and chain file."""
    # Chain File
    chain_dest = os.path.join(os.getcwd(), "hg19ToHg38.over.chain.gz")
    download_file(CHAIN_URL, chain_dest)

    # Binary
    bin_name = "liftOver"
    bin_dest = os.path.join(os.getcwd(), bin_name)
    
    base = "http://hgdownload.soe.ucsc.edu/admin/exe"
    url = f"{base}/macOSX.x86_64/liftOver" if sys.platform == "darwin" else f"{base}/linux.x86_64/liftOver"
    
    download_file(url, bin_dest)
    os.chmod(bin_dest, os.stat(bin_dest).st_mode | stat.S_IEXEC)
    
    return bin_dest, chain_dest

def write_pgs_chunk(df, f_pos, f_neg, build):
    """Filters a chunk and appends to the open file handles."""
    # Standardize columns
    subset = pd.DataFrame()
    subset['rsID'] = df['RSID'].fillna('.')
    subset['chr_name'] = df['CHROM']
    subset['chr_position'] = df['POS']
    subset['effect_allele'] = df['ALT']
    subset['other_allele'] = df['REF']
    subset['weight_type'] = 'selection_coefficient_magnitude'
    subset['S_raw'] = df['S']

    # Positive Selection
    pos = subset[subset['S_raw'] > 0].copy()
    if not pos.empty:
        pos['effect_weight'] = pos['S_raw'].abs()
        pos.drop(columns=['S_raw'], inplace=True)
        pos.to_csv(f_pos, sep='\t', index=False, header=False, mode='a')

    # Negative Selection
    neg = subset[subset['S_raw'] < 0].copy()
    if not neg.empty:
        neg['effect_weight'] = neg['S_raw'].abs()
        neg.drop(columns=['S_raw'], inplace=True)
        neg.to_csv(f_neg, sep='\t', index=False, header=False, mode='a')

def init_pgs_file(filename, build, direction):
    """Creates a PGS file and writes the header."""
    with open(filename, 'w') as f:
        f.write("#format=PGS_2.0\n")
        f.write(f"#genome_build={build}\n")
        f.write(f"#selection_direction={direction}\n")
        # Write column headers
        cols = ["rsID", "chr_name", "chr_position", "effect_allele", "other_allele", "effect_weight", "weight_type"]
        f.write("\t".join(cols) + "\n")
    return filename

def process_hg19_stream(input_file):
    """Reads input in chunks and writes hg19 PGS files."""
    print("Processing hg19 (Streamed)...")
    
    f_pos_name = "ancient_dna_positive_selection_hg19.txt"
    f_neg_name = "ancient_dna_negative_selection_hg19.txt"
    
    init_pgs_file(f_pos_name, "hg19", "positive")
    init_pgs_file(f_neg_name, "hg19", "negative")
    
    # Open handles for appending
    with open(f_pos_name, 'a') as f_pos, open(f_neg_name, 'a') as f_neg:
        # Read gz in chunks
        iterator = pd.read_csv(input_file, sep='\t', compression='gzip', comment='#', chunksize=CHUNK_SIZE)
        
        count = 0
        for chunk in iterator:
            write_pgs_chunk(chunk, f_pos, f_neg, "hg19")
            count += len(chunk)
            sys.stdout.write(f"\rRows processed: {count:,}")
            sys.stdout.flush()
    print()

def process_hg38_stream(input_file, liftover_bin, chain_file):
    """
    1. Reads hg19 chunk.
    2. Writes temp BED.
    3. Runs liftOver.
    4. Merges result.
    5. Writes hg38 dataset and PGS scores.
    """
    print("Processing hg38 conversion and scoring (Streamed)...")
    
    # Output filenames
    out_dataset = "Selection_Summary_Statistics_hg38.tsv.gz"
    out_pos_pgs = "ancient_dna_positive_selection_hg38.txt"
    out_neg_pgs = "ancient_dna_negative_selection_hg38.txt"
    
    # Initialize PGS files
    init_pgs_file(out_pos_pgs, "hg38", "positive")
    init_pgs_file(out_neg_pgs, "hg38", "negative")
    
    # Check if dataset exists, remove if so to start fresh
    if os.path.exists(out_dataset):
        os.remove(out_dataset)

    # Prepare Iterator
    iterator = pd.read_csv(input_file, sep='\t', compression='gzip', comment='#', chunksize=CHUNK_SIZE)
    
    total_rows = 0
    
    # Open output handles
    # For the dataset, we will write header first, then append mode
    # For PGS, we already initialized, so just append
    
    with open(out_pos_pgs, 'a') as f_pgs_pos, \
         open(out_neg_pgs, 'a') as f_pgs_neg, \
         gzip.open(out_dataset, 'wt') as f_dataset:
        
        dataset_header_written = False
        
        for chunk in iterator:
            # 1. Create Temp BED for this chunk
            # ID column matches chunk index to map back later
            chunk['tmp_idx'] = range(len(chunk))
            
            bed_df = pd.DataFrame()
            bed_df['chrom'] = 'chr' + chunk['CHROM'].astype(str).str.lstrip('chr')
            bed_df['start'] = chunk['POS'] - 1
            bed_df['end'] = chunk['POS']
            bed_df['id'] = chunk['tmp_idx']
            
            with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp_in, \
                 tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp_out, \
                 tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp_unmapped:
                
                bed_df.to_csv(tmp_in.name, sep='\t', header=False, index=False)
                tmp_in.close() # Close to flush
                
                # 2. Run LiftOver
                subprocess.run(
                    [liftover_bin, tmp_in.name, chain_file, tmp_out.name, tmp_unmapped.name],
                    check=False, stderr=subprocess.DEVNULL # Ignore warnings
                )
                
                # 3. Read Lifted
                # If file is empty (no mappings), handle gracefully
                if os.stat(tmp_out.name).st_size > 0:
                    lifted = pd.read_csv(tmp_out.name, sep='\t', header=None, names=['chrom', 'start', 'end', 'id'])
                    
                    # 4. Merge back to chunk data
                    # Inner merge implies we drop unmapped variants
                    merged = chunk.merge(lifted, left_on='tmp_idx', right_on='id', how='inner')
                    
                    if not merged.empty:
                        # Update Coords
                        merged['CHROM'] = merged['chrom'].str.replace('chr', '')
                        merged['POS'] = merged['end']
                        
                        # Cleanup columns
                        keep_cols = ['CHROM', 'POS', 'REF', 'ALT', 'S', 'RSID']
                        # Ensure columns exist (RSID might be missing in some raw files)
                        final_cols = [c for c in keep_cols if c in merged.columns]
                        clean_chunk = merged[final_cols].copy()
                        
                        # 5. Write to hg38 Dataset
                        if not dataset_header_written:
                            clean_chunk.to_csv(f_dataset, sep='\t', index=False)
                            dataset_header_written = True
                        else:
                            clean_chunk.to_csv(f_dataset, sep='\t', index=False, header=False)
                            
                        # 6. Write to PGS files
                        write_pgs_chunk(clean_chunk, f_pgs_pos, f_pgs_neg, "hg38")
                
                # Cleanup Temps
                os.remove(tmp_in.name)
                os.remove(tmp_out.name)
                os.remove(tmp_unmapped.name)

            total_rows += len(chunk)
            sys.stdout.write(f"\rRows Processed/Lifted: {total_rows:,}")
            sys.stdout.flush()
            
    print(f"\nSaved full hg38 dataset to {out_dataset}")

def main():
    # 1. Setup
    input_file = get_dataverse_file()
    liftover_bin, chain_file = get_liftover_tools()
    
    # 2. hg19 Generation
    process_hg19_stream(input_file)
    
    # 3. hg38 Generation (LiftOver + Scoring)
    process_hg38_stream(input_file, liftover_bin, chain_file)
    
    print("Done.")

if __name__ == "__main__":
    main()
