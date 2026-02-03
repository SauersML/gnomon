
"""
Train a specific PRS model and generate scores for the test set.
Usage: python train_model.py <sim_id> <method_name>
"""
import sys
import os
import shutil
import pandas as pd
from pathlib import Path

# Add current directory to path to allow imports
sys.path.append(str(Path(__file__).parent))

from prs_tools import BayesR, LDpred2, PRScsx

SIM_NAME_MAP = {
    1: "confounding",
    2: "portability",
    3: "sample_imbalance",
}


def _resolve_sim_prefix(sim_arg: str) -> str:
    # Accept sim id (1/2/3) or sim name
    sim_arg = str(sim_arg).strip()
    if sim_arg.isdigit():
        sim_id = int(sim_arg)
        prefix = SIM_NAME_MAP.get(sim_id, f"sim{sim_id}")
        if os.path.exists(f"{prefix}_work") or os.path.exists(f"{prefix}.tsv"):
            return prefix
        return f"sim{sim_id}"
    return sim_arg

def train_and_score(sim_arg, method_name):
    sim_prefix = _resolve_sim_prefix(sim_arg)
    work_dir = Path(f"{sim_prefix}_work")
    if not work_dir.exists():
        raise FileNotFoundError(
            f"Work directory {work_dir} not found. Did you run split_data.py?"
        )

    train_prefix = work_dir / "train"
    test_prefix = work_dir / "test"
    pheno_file = work_dir / "train.phen"

    if not train_prefix.with_suffix(".bed").exists():
         raise FileNotFoundError(f"Training data {train_prefix}.bed not found.")

    print(f"--- Training {method_name} for Sim {sim_prefix} ---")

    scores = None

    if method_name == 'BayesR':
        br = BayesR()
        covar_file = work_dir / "train.covar"
        if not covar_file.exists():
            raise FileNotFoundError(f"REQUIRED: Covariate file missing: {covar_file}. BayesR requires PC covariates.")
        eff_file = br.fit(str(train_prefix), str(pheno_file), str(work_dir / "bayesr"), covar_file=str(covar_file))
        res = br.predict(str(test_prefix), eff_file, str(work_dir / "bayesr_pred"))
        scores = res

    elif method_name == 'LDpred2':
        ld = LDpred2()
        # LDpred2 often uses validation set for tuning. We use training set as "val" if no separate val provided?
        # The wrapper signature is fit(train, pheno, val, out).
        # We'll pass train as val for simplicity if strictly required by wrapper structure,
        # or the wrapper handles it.
        eff_file = ld.fit(str(train_prefix), str(pheno_file), str(train_prefix), str(work_dir / "ldpred2"))
        res = ld.predict(str(test_prefix), eff_file, str(work_dir / "ldpred2_pred"))
        scores = res

    elif method_name == 'PRS-CSx':
        ref_path_env = os.environ.get("PRSCSX_REF")
        if not ref_path_env:
             raise RuntimeError("PRSCSX_REF environment variable not set. Cannot run PRS-CSx.")

        cs = PRScsx()
        eff_file = cs.fit(str(train_prefix), str(pheno_file), str(work_dir / "prscsx"),
                          ref_dir=ref_path_env)
        res = cs.predict(str(test_prefix), eff_file, str(work_dir / "prscsx_pred"))
        scores = res

    else:
        raise ValueError(f"Unknown method: {method_name}")
    
    if scores is None or scores.empty:
        raise RuntimeError(f"{method_name} produced no scores!")
        
    # Save standard output
    # IID, PRS
    out_file = work_dir / f"{method_name}.sscore"
    scores.to_csv(out_file, sep='\t', index=False)
    print(f"Scores saved to {out_file}")

def main():
    if len(sys.argv) != 3:
        print("Usage: python train_model.py <sim_id> <method_name>")
        sys.exit(1)
        
    sim_arg = sys.argv[1]
    method_name = sys.argv[2]
    
    try:
        train_and_score(sim_arg, method_name)
    except Exception as e:
        print(f"CRITICAL FAILURE in {method_name}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
