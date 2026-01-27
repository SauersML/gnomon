"""
BayesR Mix: Multi-ancestry stacking approach (as used in PolyPred-S+).

Trains separate BayesR models on each ancestry, then combines predictions
using non-negative least squares (NNLS) weights learned on a tuning set.

References:
- PolyPred-S+: https://pmc.ncbi.nlm.nih.gov/articles/PMC9009299/
- Multi-ancestry PRS: https://pmc.ncbi.nlm.nih.gov/articles/PMC10923245/
"""
import subprocess
import os
import shutil
import tempfile
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.optimize import nnls
from sklearn.model_selection import train_test_split

from .bayesr import BayesR


class BayesRMix:
    """
    Multi-ancestry BayesR stacking method.

    Workflow:
    1. Split training data by ancestry
    2. Train separate BayesR models for each ancestry
    3. Split training set into fit/tune subsets
    4. Compute component PRS on tuning set
    5. Learn NNLS mixing weights
    6. Apply to test set
    """

    def __init__(self, ancestries=None, gctb_path="gctb"):
        """
        Args:
            ancestries: List of ancestry labels to train on (default: ['EUR', 'AFR', 'ASIA'])
            gctb_path: Path to GCTB executable
        """
        self.ancestries = ancestries or ['EUR', 'AFR', 'ASIA']
        self.gctb_path = gctb_path
        self.bayesr = BayesR(gctb_path=gctb_path)
        self.weights = None
        self.ancestry_models = {}

    def _split_by_ancestry(self, bfile_train, metadata_file):
        """
        Split training data by ancestry using PLINK2.

        Args:
            bfile_train: Path prefix to training PLINK files
            metadata_file: TSV with columns: IID, pop_label

        Returns:
            dict: {ancestry: bfile_prefix} for each ancestry with sufficient samples
        """
        metadata = pd.read_csv(metadata_file, sep='\t')
        metadata['IID'] = metadata['IID'].astype(str)

        # Read FAM to get FID mapping
        fam_path = f"{bfile_train}.fam"
        fam = pd.read_csv(fam_path, sep=r'\s+', header=None,
                         names=['FID', 'IID', 'PID', 'MID', 'SEX', 'PHENO'],
                         dtype={'FID': str, 'IID': str})
        iid_to_fid = dict(zip(fam['IID'].astype(str), fam['FID'].astype(str)))

        ancestry_bfiles = {}
        min_samples = 50  # Minimum samples per ancestry to train a model

        for ancestry in self.ancestries:
            anc_ids = metadata[metadata['pop_label'] == ancestry]['IID'].tolist()

            if len(anc_ids) < min_samples:
                print(f"  BayesR-Mix: Skipping {ancestry} (only {len(anc_ids)} samples, need {min_samples})")
                continue

            print(f"  BayesR-Mix: Found {len(anc_ids)} {ancestry} samples")

            # Create temporary keep file
            tmp_keep = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.keep')
            for iid in anc_ids:
                if iid in iid_to_fid:
                    tmp_keep.write(f"{iid_to_fid[iid]}\t{iid}\n")
            tmp_keep.close()

            # Create ancestry-specific PLINK files
            out_prefix = f"{bfile_train}_{ancestry}"
            cmd = [
                "plink2",
                "--bfile", bfile_train,
                "--keep", tmp_keep.name,
                "--make-bed",
                "--out", out_prefix
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)
            os.unlink(tmp_keep.name)

            if result.returncode != 0:
                print(f"  BayesR-Mix: Failed to create {ancestry} subset: {result.stderr}")
                continue

            # Verify output
            if not os.path.exists(f"{out_prefix}.bed"):
                print(f"  BayesR-Mix: {ancestry} subset creation failed (no .bed file)")
                continue

            ancestry_bfiles[ancestry] = out_prefix

        return ancestry_bfiles

    def _create_ancestry_pheno_covar(self, bfile_ancestry, pheno_file, covar_file):
        """
        Create ancestry-specific phenotype and covariate files.

        Args:
            bfile_ancestry: Path to ancestry-specific PLINK files
            pheno_file: Original phenotype file (FID IID Pheno)
            covar_file: Original covariate file (FID IID PC1 PC2 ...)

        Returns:
            tuple: (ancestry_pheno_file, ancestry_covar_file)
        """
        # Read ancestry-specific FAM to get IIDs
        fam = pd.read_csv(f"{bfile_ancestry}.fam", sep=r'\s+', header=None,
                         names=['FID', 'IID', 'PID', 'MID', 'SEX', 'PHENO'],
                         dtype={'FID': str, 'IID': str})
        ancestry_iids = set(fam['IID'].astype(str))

        # Filter phenotype file
        pheno = pd.read_csv(pheno_file, sep=r'\s+', header=None,
                           names=['FID', 'IID', 'Pheno'],
                           dtype={'FID': str, 'IID': str})
        pheno['IID'] = pheno['IID'].astype(str)
        pheno_filtered = pheno[pheno['IID'].isin(ancestry_iids)]

        ancestry_pheno = f"{bfile_ancestry}.phen"
        pheno_filtered.to_csv(ancestry_pheno, sep=' ', index=False, header=False)

        # Filter covariate file
        covar = pd.read_csv(covar_file, sep=r'\s+', header=None,
                           dtype={0: str, 1: str})
        covar.iloc[:, 1] = covar.iloc[:, 1].astype(str)
        covar_filtered = covar[covar.iloc[:, 1].isin(ancestry_iids)]

        ancestry_covar = f"{bfile_ancestry}.covar"
        covar_filtered.to_csv(ancestry_covar, sep=' ', index=False, header=False)

        return ancestry_pheno, ancestry_covar

    def fit(self, bfile_train, pheno_file, out_prefix, covar_file, metadata_file):
        """
        Train multi-ancestry BayesR stacking model.

        Args:
            bfile_train: Path prefix to training PLINK files
            pheno_file: Path to phenotype file (FID IID Pheno)
            out_prefix: Output prefix for results
            covar_file: Path to covariate file (FID IID PC1 PC2 ...)
            metadata_file: Path to metadata TSV with columns: IID, pop_label

        Returns:
            Path to saved model file containing effect files and weights
        """
        print("=== BayesR-Mix: Multi-ancestry stacking ===")

        # 1. Split training data by ancestry
        print("Step 1: Splitting training data by ancestry...")
        ancestry_bfiles = self._split_by_ancestry(bfile_train, metadata_file)

        if len(ancestry_bfiles) == 0:
            raise RuntimeError("BayesR-Mix: No ancestry subsets created. Cannot proceed.")

        print(f"Step 1 complete: {len(ancestry_bfiles)} ancestries available: {list(ancestry_bfiles.keys())}")

        # 2. Train BayesR model for each ancestry
        print("\nStep 2: Training ancestry-specific BayesR models...")
        effect_files = {}

        for ancestry, anc_bfile in ancestry_bfiles.items():
            print(f"\n  Training {ancestry} BayesR model...")

            # Create ancestry-specific pheno and covar files
            anc_pheno, anc_covar = self._create_ancestry_pheno_covar(
                anc_bfile, pheno_file, covar_file
            )

            # Train BayesR on this ancestry
            try:
                anc_out = f"{out_prefix}_{ancestry}"
                eff_file = self.bayesr.fit(anc_bfile, anc_pheno, anc_out, anc_covar)
                effect_files[ancestry] = eff_file
                self.ancestry_models[ancestry] = anc_out
                print(f"  {ancestry} model trained: {eff_file}")
            except Exception as e:
                print(f"  WARNING: {ancestry} model training failed: {e}")
                # Continue with other ancestries

        if len(effect_files) == 0:
            raise RuntimeError("BayesR-Mix: All ancestry models failed. Cannot proceed.")

        print(f"\nStep 2 complete: {len(effect_files)} models trained")

        # 3. Split training set into fit/tune (use the original training set)
        # We'll use 50% for tuning the weights
        print("\nStep 3: Creating tuning set for weight learning...")

        # Read metadata to stratify by ancestry
        metadata = pd.read_csv(metadata_file, sep='\t')
        metadata['IID'] = metadata['IID'].astype(str)

        # Read FAM
        fam = pd.read_csv(f"{bfile_train}.fam", sep=r'\s+', header=None,
                         names=['FID', 'IID', 'PID', 'MID', 'SEX', 'PHENO'],
                         dtype={'FID': str, 'IID': str})

        # Merge to get pop_label for train samples
        train_meta = fam.merge(metadata[['IID', 'pop_label']], on='IID', how='left')

        # Split 50/50 for fit (not used) and tune, stratified by ancestry
        indices = np.arange(len(train_meta))
        try:
            _, tune_idx = train_test_split(
                indices, test_size=0.5, random_state=42,
                stratify=train_meta['pop_label'].fillna('UNKNOWN')
            )
        except:
            # If stratification fails, use random split
            _, tune_idx = train_test_split(indices, test_size=0.5, random_state=42)

        tune_ids = train_meta.iloc[tune_idx][['FID', 'IID']]

        # Create tuning set PLINK files
        tune_keep = f"{out_prefix}_tune.keep"
        tune_ids.to_csv(tune_keep, sep='\t', index=False, header=False)

        tune_bfile = f"{out_prefix}_tune"
        cmd = [
            "plink2",
            "--bfile", bfile_train,
            "--keep", tune_keep,
            "--make-bed",
            "--out", tune_bfile
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to create tuning set: {result.stderr}")

        print(f"  Tuning set created: {len(tune_idx)} samples")

        # 4. Compute component PRS on tuning set
        print("\nStep 4: Computing component PRS on tuning set...")
        component_prs = {}

        for ancestry, eff_file in effect_files.items():
            print(f"  Computing {ancestry} PRS...")
            try:
                scores = self.bayesr.predict(tune_bfile, eff_file, f"{tune_bfile}_{ancestry}")
                scores['IID'] = scores['IID'].astype(str)
                component_prs[ancestry] = scores.set_index('IID')['PRS']
            except Exception as e:
                print(f"  WARNING: {ancestry} scoring failed: {e}")

        if len(component_prs) == 0:
            raise RuntimeError("BayesR-Mix: All component PRS calculations failed.")

        # 5. Learn NNLS mixing weights
        print("\nStep 5: Learning NNLS mixing weights...")

        # Read tuning phenotypes
        pheno = pd.read_csv(pheno_file, sep=r'\s+', header=None,
                           names=['FID', 'IID', 'Pheno'],
                           dtype={'FID': str, 'IID': str})
        pheno['IID'] = pheno['IID'].astype(str)
        pheno = pheno.set_index('IID')

        # Combine component PRS into matrix
        prs_df = pd.DataFrame(component_prs)

        # Inner join with phenotypes (to handle any missing)
        combined = prs_df.join(pheno[['Pheno']], how='inner')

        if len(combined) == 0:
            raise RuntimeError("BayesR-Mix: No overlapping samples between PRS and phenotypes.")

        X = combined[list(component_prs.keys())].values
        y = combined['Pheno'].values

        # Fit NNLS: min ||y - X*w||^2 subject to w >= 0
        weights, residual = nnls(X, y)

        # Normalize weights to sum to 1
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            # If all weights are zero, use uniform weights
            weights = np.ones(len(weights)) / len(weights)

        self.weights = dict(zip(component_prs.keys(), weights))

        print(f"\n  Learned weights:")
        for ancestry, weight in self.weights.items():
            print(f"    {ancestry}: {weight:.4f}")

        # Save model (effect files + weights)
        model_file = f"{out_prefix}_mix_model.txt"
        with open(model_file, 'w') as f:
            f.write("# BayesR-Mix Model\n")
            f.write("# Ancestry\tEffectFile\tWeight\n")
            for ancestry in effect_files.keys():
                weight = self.weights.get(ancestry, 0.0)
                f.write(f"{ancestry}\t{effect_files[ancestry]}\t{weight:.6f}\n")

        print(f"\nModel saved to: {model_file}")
        return model_file

    def predict(self, bfile_test, model_file, out_prefix):
        """
        Generate predictions using the trained BayesR-Mix model.

        Args:
            bfile_test: Path prefix to test PLINK files
            model_file: Path to saved model file
            out_prefix: Output prefix for results

        Returns:
            DataFrame with columns: IID, PRS
        """
        print("=== BayesR-Mix: Generating predictions ===")

        # Load model
        model_data = []
        with open(model_file, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.strip().split('\t')
                if len(parts) == 3:
                    ancestry, eff_file, weight = parts
                    model_data.append((ancestry, eff_file, float(weight)))

        if len(model_data) == 0:
            raise RuntimeError(f"No model data found in {model_file}")

        # Compute component PRS
        print("Computing component PRS...")
        component_prs = {}

        for ancestry, eff_file, weight in model_data:
            if weight == 0:
                print(f"  Skipping {ancestry} (weight=0)")
                continue

            print(f"  Computing {ancestry} PRS (weight={weight:.4f})...")
            try:
                scores = self.bayesr.predict(bfile_test, eff_file, f"{out_prefix}_{ancestry}")
                scores['IID'] = scores['IID'].astype(str)
                component_prs[ancestry] = (scores.set_index('IID')['PRS'], weight)
            except Exception as e:
                print(f"  WARNING: {ancestry} scoring failed: {e}")

        if len(component_prs) == 0:
            raise RuntimeError("BayesR-Mix: All component PRS calculations failed.")

        # Combine with learned weights
        print("\nCombining predictions with learned weights...")

        # Start with first ancestry
        first_ancestry = list(component_prs.keys())[0]
        prs_series, weight = component_prs[first_ancestry]
        combined_prs = prs_series * weight

        # Add remaining ancestries
        for ancestry in list(component_prs.keys())[1:]:
            prs_series, weight = component_prs[ancestry]
            combined_prs = combined_prs.add(prs_series * weight, fill_value=0)

        # Format output
        result = pd.DataFrame({
            'IID': combined_prs.index,
            'PRS': combined_prs.values
        })

        print(f"\nPredictions complete: {len(result)} samples")
        return result
