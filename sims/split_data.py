
"""
Split Simulation Data into Training (EUR) and Test (All) sets.
"""
import sys
import os
import shutil
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

def _resolve_sim_prefix(sim_arg: str) -> str:
    sim_arg = str(sim_arg).strip()
    if os.path.exists(f"{sim_arg}.tsv"):
        return sim_arg
    raise FileNotFoundError(f"Simulation TSV not found: {sim_arg}.tsv")

def setup_directories(sim_arg):
    """Create work directories."""
    work_dir = Path(f"{sim_arg}_work")
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir()
    return work_dir

def split_data(sim_arg, work_dir):
    """
    Split PLINK data into Training (EUR) and Test (All).
    """
    # Load info from TSV to identify IDs
    sim_prefix = _resolve_sim_prefix(sim_arg)
    tsv_path = f"{sim_prefix}.tsv"
        
    df = pd.read_csv(tsv_path, sep='\t')

    bfile_orig = sim_prefix  # sim_pops.py produces {sim_name}.bed/bim/fam
    fam_path = f"{bfile_orig}.fam"
    bim_path = f"{bfile_orig}.bim"
    bed_path = f"{bfile_orig}.bed"
    missing_inputs = [p for p in [fam_path, bim_path, bed_path] if not os.path.exists(p)]
    if missing_inputs:
        raise FileNotFoundError(f"Missing PLINK input files for prefix '{bfile_orig}': {missing_inputs}")

    fam = pd.read_csv(
        fam_path,
        sep=r"\s+",
        header=None,
        names=["FID", "IID", "PID", "MID", "SEX", "PHENO"],
        dtype={"FID": str, "IID": str},
    )
    fam_iid_set = set(fam["IID"].astype(str).tolist())
    
    sim_name = sim_prefix
    if sim_name == "confounding":
        # Train on all ancestries, 80/20 split overall.
        train_idx, test_idx = train_test_split(df.index.to_numpy(), test_size=0.2, random_state=42)
    else:
        # Portability: train EUR only; test = 20% EUR + all non-EUR.
        train_mask = df['pop_label'] == 'EUR'
        eur_indices = df[train_mask].index.to_numpy()
        if len(eur_indices) > 0:
            train_idx, test_eur_idx = train_test_split(eur_indices, test_size=0.2, random_state=42)
        else:
            print("Warning: No EUR found? Using random 60% split.")
            train_idx, test_eur_idx = train_test_split(df.index.to_numpy(), test_size=0.4, random_state=42)
        non_eur_idx = df[~df.index.isin(eur_indices)].index.to_numpy()
        test_idx = np.concatenate([test_eur_idx, non_eur_idx])
    
    # Ensure IDs are strings
    df['individual_id'] = df['individual_id'].astype(str)

    df_iid_set = set(df["individual_id"].tolist())
    missing_in_fam = sorted(list(df_iid_set - fam_iid_set))
    if missing_in_fam:
        preview = missing_in_fam[:20]
        raise RuntimeError(
            "Simulation TSV 'individual_id' values do not match PLINK .fam IIDs. "
            f"Missing {len(missing_in_fam)} IDs from {fam_path}. Example: {preview}"
        )

    iid_to_fid = dict(zip(fam["IID"].astype(str), fam["FID"].astype(str)))
    
    train_iid = df.loc[train_idx, "individual_id"].tolist()
    test_iid = df.loc[test_idx, "individual_id"].tolist()
    train_ids = pd.DataFrame({"FID": [iid_to_fid[i] for i in train_iid], "IID": train_iid})
    test_ids = pd.DataFrame({"FID": [iid_to_fid[i] for i in test_iid], "IID": test_iid})
    
    train_keep = work_dir / "train.keep"
    test_keep = work_dir / "test.keep"
    
    train_ids.to_csv(train_keep, sep='\t', index=False, header=False)
    test_ids.to_csv(test_keep, sep='\t', index=False, header=False)

    plink_exe = shutil.which("plink2") or "/usr/local/bin/plink2"
    if not os.path.exists(plink_exe) and shutil.which("plink2") is None:
        raise FileNotFoundError(f"plink2 executable not found on PATH and fallback missing at {plink_exe}")

    def _count_fam(prefix: Path) -> int:
        fp = Path(f"{prefix}.fam")
        if not fp.exists():
            return -1
        with open(fp, "r") as f:
            return sum(1 for _ in f)

    def _run_plink(cmd, label: str):
        log_path = work_dir / f"{label}.plink.log"
        result = subprocess.run(cmd, capture_output=True, text=True)
        with open(log_path, "w") as f:
            f.write(f"CMD: {' '.join(cmd)}\n")
            f.write(f"RETURN_CODE: {result.returncode}\n")
            f.write("STDOUT:\n")
            f.write(result.stdout or "")
            f.write("\nSTDERR:\n")
            f.write(result.stderr or "")
        if result.returncode != 0:
            raise RuntimeError(
                f"PLINK2 failed ({label}) with exit code {result.returncode}. "
                f"See log: {log_path}"
            )
        return result

    n_total_fam = _count_fam(Path(bfile_orig))
    print(f"  PLINK input: {bfile_orig} (samples in .fam={n_total_fam})")
    
    # Training subset
    cmd_train = [
        plink_exe, "--bfile", bfile_orig, "--keep", str(train_keep),
        "--make-bed", "--out", str(work_dir / "train")
    ]
    _run_plink(cmd_train, "split_train")

    # Testing subset
    cmd_test = [
        plink_exe, "--bfile", bfile_orig, "--keep", str(test_keep),
        "--make-bed", "--out", str(work_dir / "test")
    ]
    _run_plink(cmd_test, "split_test")

    n_train_out = _count_fam(work_dir / "train")
    n_test_out = _count_fam(work_dir / "test")
    if n_train_out <= 0 or n_test_out <= 0:
        raise RuntimeError(
            "PLINK2 split produced empty dataset. "
            f"train samples={n_train_out}, test samples={n_test_out}. "
        )
    
    # Write phenotype files (FID IID Pheno)
    pd.DataFrame(
        {
            "FID": [iid_to_fid[i] for i in train_iid],
            "IID": train_iid,
            "y": df.loc[train_idx, "y"].to_numpy(),
        }
    ).to_csv(work_dir / "train.phen", sep=' ', index=False, header=False)

    pd.DataFrame(
        {
            "FID": [iid_to_fid[i] for i in test_iid],
            "IID": test_iid,
            "y": df.loc[test_idx, "y"].to_numpy(),
        }
    ).to_csv(work_dir / "test.phen", sep=' ', index=False, header=False)

    # Write covariate files (FID IID PC1 PC2 ... PC20) for BayesR
    # Detect number of PCs dynamically from column names
    pc_cols = [c for c in df.columns if c.startswith('pc') and c[2:].isdigit()]
    pc_cols_sorted = sorted(pc_cols, key=lambda x: int(x[2:]))  # Sort pc1, pc2, ..., pc20

    if len(pc_cols_sorted) == 0:
        raise RuntimeError(
            f"REQUIRED: No PC columns found in {tsv_path}. "
            "Expected columns like pc1, pc2, ..., pcN. "
            "Simulation must compute PCs before splitting data."
        )

    train_covar_data = {"FID": [iid_to_fid[i] for i in train_iid], "IID": train_iid}
    for pc_col in pc_cols_sorted:
        train_covar_data[pc_col] = df.loc[train_idx, pc_col].to_numpy()
    pd.DataFrame(train_covar_data).to_csv(work_dir / "train.covar", sep=' ', index=False, header=False)

    test_covar_data = {"FID": [iid_to_fid[i] for i in test_iid], "IID": test_iid}
    for pc_col in pc_cols_sorted:
        test_covar_data[pc_col] = df.loc[test_idx, pc_col].to_numpy()
    pd.DataFrame(test_covar_data).to_csv(work_dir / "test.covar", sep=' ', index=False, header=False)

    print(f"Created covariate files with {len(pc_cols_sorted)} PCs: {pc_cols_sorted}")

    print(f"Split complete. Train: {n_train_out}, Test: {n_test_out}")
    print(f"Outputs in {work_dir}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python split_data.py <sim_name>")
        sys.exit(1)
    
    sim_arg = sys.argv[1]
    work_dir = setup_directories(sim_arg)
    split_data(sim_arg, work_dir)

if __name__ == "__main__":
    main()
