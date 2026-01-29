"""
Wrapper for MUSSEL (multi-ancestry PRS; summary stats + LDref + tuning genotypes).

This wrapper prepares MUSSEL-compatible inputs from the local simulation data,
then runs the MUSSEL pipeline steps:
  - LDpred2_jobs.R
  - LDpred2_tuning.R
  - MUSS_jobs.R
  - MUSSEL.R

Requirements (from MUSSEL README):
  - MUSSEL repo cloned locally
  - LD reference directory (UKBB or 1000G)
  - PLINK2 in PATH
  - R + MUSSEL required R packages
"""
import os
import random
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


class MUSSEL:
    def __init__(
        self,
        mussel_path="MUSSEL",
        ldref_dir="mussel_ldref",
        rscript="Rscript",
        plink2_path="plink2",
        pops=None,
        target_pop=None,
        chroms="22",
        trait_type=None,
        ncores_ldpred2="11",
        ncores_muss="5",
        ncores_mussel="1",
        score_file=None,
        skip_ldpred2=False,
        skip_ldpred2_tuning=False,
        skip_muss=False,
        skip_mussel=False,
    ):
        self.mussel_path = mussel_path
        self.ldref_dir = ldref_dir
        self.rscript = rscript
        self.plink2_path = plink2_path
        self.pops = pops
        self.target_pop = target_pop
        self.chroms = chroms
        self.trait_type = trait_type
        self.ncores_ldpred2 = ncores_ldpred2
        self.ncores_muss = ncores_muss
        self.ncores_mussel = ncores_mussel
        self.score_file = score_file
        self.skip_ldpred2 = skip_ldpred2
        self.skip_ldpred2_tuning = skip_ldpred2_tuning
        self.skip_muss = skip_muss
        self.skip_mussel = skip_mussel

    def _run(self, cmd, cwd=None):
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"MUSSEL command failed:\n{' '.join(cmd)}\n\nSTDERR:\n{result.stderr}\n\nSTDOUT:\n{result.stdout}"
            )
        return result

    def _split_ids(self, ids, tuning_frac=0.8, seed=42):
        ids = list(ids)
        random.Random(seed).shuffle(ids)
        cut = max(1, int(len(ids) * tuning_frac))
        return ids[:cut], ids[cut:]

    def _write_keep(self, fam_df, iids, out_path):
        keep = fam_df[fam_df["IID"].isin(iids)][["FID", "IID"]]
        keep.to_csv(out_path, sep="\t", index=False, header=False)
        return out_path

    def _plink_subset(self, bfile, keep_file, out_prefix, chrom=None):
        cmd = [
            self.plink2_path,
            "--bfile", bfile,
            "--keep", str(keep_file),
            "--make-bed",
            "--out", str(out_prefix),
        ]
        if chrom is not None:
            cmd.extend(["--chr", str(chrom)])
        self._run(cmd)
        return out_prefix

    def _write_pheno_covar(self, fam_df, tsv_df, out_dir):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        tsv_df = tsv_df.copy()
        tsv_df["IID"] = tsv_df["individual_id"].astype(str)

        fam_df = fam_df.copy()
        fam_df["IID"] = fam_df["IID"].astype(str)

        merged = fam_df.merge(tsv_df, on="IID", how="left")
        if merged["y"].isna().any():
            missing = merged[merged["y"].isna()][["FID", "IID"]].head(5).to_string(index=False)
            raise RuntimeError(f"MUSSEL: missing phenotype for some IIDs. Example:\n{missing}")

        pheno_path = out_dir / "pheno.txt"
        pheno_df = merged[["FID", "IID", "y"]]
        pheno_df.to_csv(pheno_path, sep="\t", index=False, header=False)

        pc_cols = [c for c in tsv_df.columns if c.startswith("pc")]
        if not pc_cols:
            raise RuntimeError("MUSSEL: no PC columns found in TSV (expected columns starting with 'pc').")

        covar_path = out_dir / "covar.txt"
        covar_df = merged[["FID", "IID"] + pc_cols]
        covar_df.to_csv(covar_path, sep="\t", index=False, header=False)

        return pheno_path, covar_path

    def _run_gwas(self, bfile, pheno_file, out_prefix, n_eff_override=None):
        cmd = [
            self.plink2_path,
            "--bfile", bfile,
            "--pheno", pheno_file,
            "--1",
            "--glm", "allow-no-covars", "hide-covar",
            "--out", out_prefix,
        ]
        self._run(cmd)

        glm_files = list(Path(".").glob(f"{out_prefix}*.glm.*"))
        if not glm_files:
            raise RuntimeError(f"MUSSEL: no GWAS output found for {out_prefix}")

        def _glm_rank(p: Path) -> int:
            name = p.name.lower()
            if "glm.logistic" in name:
                return 0
            if "glm.firth" in name:
                return 1
            if "glm.linear" in name:
                return 2
            return 3

        glm_files = sorted(glm_files, key=_glm_rank)
        gwas_file = glm_files[0]
        df = pd.read_csv(gwas_file, sep="\t")
        if "TEST" in df.columns:
            df = df[df["TEST"].astype(str) == "ADD"].copy()

        head_preview = df.head(3).to_string(index=False)

        cols_upper = {c.upper(): c for c in df.columns}

        def _find_col(candidates):
            for cand in candidates:
                key = cand.upper()
                if key in cols_upper:
                    return cols_upper[key]
            return None

        def _find_col_prefix(prefixes):
            for key, orig in cols_upper.items():
                for p in prefixes:
                    if key.startswith(p):
                        return orig
            return None

        chrom_col = _find_col(["#CHROM", "CHROM"])
        pos_col = _find_col(["POS", "BP"])
        id_col = _find_col(["ID", "SNP", "RSID"])
        a1_col = _find_col(["A1", "ALT1"])
        ref_col = _find_col(["REF", "A0"])
        alt_col = _find_col(["ALT", "ALT1"])
        a2_col = _find_col(["A2"])

        if not all([chrom_col, pos_col, id_col, a1_col]):
            raise RuntimeError(
                "MUSSEL: GWAS output missing required columns for CHROM/POS/ID/A1. "
                f"File={gwas_file} Columns={list(df.columns)} Head:\n{head_preview}"
            )

        beta_col = _find_col(["BETA"]) or _find_col_prefix(["BETA"])
        or_col = _find_col(["OR"]) or _find_col_prefix(["OR"])
        if beta_col is not None:
            beta = pd.to_numeric(df[beta_col], errors="coerce")
        elif or_col is not None:
            beta = np.log(pd.to_numeric(df[or_col], errors="coerce"))
        else:
            raise RuntimeError(
                f"MUSSEL: GWAS output missing BETA/OR columns. File={gwas_file} Columns={list(df.columns)}"
            )

        se_col = _find_col(["SE", "LOG(OR)_SE", "LOGOR_SE", "SE_LOGOR", "SE_BETA"])
        if se_col is None:
            # Try generic *_SE columns
            for key, orig in cols_upper.items():
                if key.endswith("_SE"):
                    se_col = orig
                    break
        if se_col is not None:
            beta_se = pd.to_numeric(df[se_col], errors="coerce")
        else:
            tstat_col = _find_col(["T_STAT", "Z_STAT", "Z_OR_STAT", "Z_F_STAT"])
            if tstat_col is None:
                tstat_col = _find_col_prefix(["Z_"])
            if tstat_col is None:
                raise RuntimeError(
                    "MUSSEL: GWAS output missing SE (no SE or T_STAT columns). "
                    f"File={gwas_file} Columns={list(df.columns)} Head:\n{head_preview}"
                )
            # Fallback: SE = beta / z if test-stat provided
            tstat = pd.to_numeric(df[tstat_col], errors="coerce")
            beta_se = pd.to_numeric(beta, errors="coerce") / tstat.replace(0, np.nan)

        if a2_col is not None:
            a0 = df[a2_col]
        elif ref_col is not None and alt_col is not None:
            a0 = np.where(df[a1_col] == df[alt_col], df[ref_col], df[alt_col])
        else:
            raise RuntimeError(
                "MUSSEL: GWAS output missing REF/ALT (or A2) to construct a0. "
                f"File={gwas_file} Columns={list(df.columns)} Head:\n{head_preview}"
            )

        if n_eff_override is not None:
            n_eff = np.full(len(df), n_eff_override, dtype=float)
        else:
            n_eff = pd.to_numeric(df.get("OBS_CT", np.nan), errors="coerce")

        out_df = pd.DataFrame({
            "rsid": df[id_col],
            "chr": df[chrom_col],
            "pos": df[pos_col],
            "a0": a0,
            "a1": df[a1_col],
            "beta": beta,
            "beta_se": beta_se,
            "n_eff": n_eff,
        }).dropna()

        sst_path = f"{out_prefix}.mussel.sst"
        out_df.to_csv(sst_path, sep="\t", index=False)
        return sst_path

    def fit(self, sim_id=None, sim_prefix=None, out_prefix=None):
        """
        Prepare MUSSEL inputs from sim{sim_id} and run the MUSSEL pipeline.
        Returns a model directory (PATH_out).
        """
        if not self.mussel_path:
            raise RuntimeError("MUSSEL path is required (path to cloned MUSSEL repo).")

        ldref_dir = self.ldref_dir
        if not ldref_dir:
            raise RuntimeError("LD reference directory is required for MUSSEL.")

        if sim_prefix:
            prefix = sim_prefix
        elif sim_id is not None:
            prefix = f"sim{sim_id}"
        else:
            raise RuntimeError("MUSSEL requires sim_prefix or sim_id.")

        if out_prefix is None:
            raise RuntimeError("MUSSEL requires out_prefix.")

        bfile_full = prefix
        tsv_path = f"{prefix}.tsv"
        if not Path(f"{bfile_full}.bed").exists():
            raise RuntimeError(f"MUSSEL: full bfile not found: {bfile_full}.bed")
        if not Path(tsv_path).exists():
            raise RuntimeError(f"MUSSEL: TSV not found: {tsv_path}")

        chrom_arg = self.chroms
        chrom_list = [c.strip() for c in self.chroms.split(",")]

        df = pd.read_csv(tsv_path, sep="\t")
        pops = self.pops or sorted(df["pop_label"].unique())
        target_pop = self.target_pop or pops[0]

        trait_type = self.trait_type
        if not trait_type:
            unique_y = set(df["y"].dropna().unique().tolist())
            trait_type = "binary" if unique_y.issubset({0, 1}) else "continuous"

        out_prefix = Path(out_prefix)
        path_data = out_prefix.parent / f"{out_prefix.name}_data"
        path_out = out_prefix.parent / f"{out_prefix.name}_out"
        path_data.mkdir(parents=True, exist_ok=True)
        path_out.mkdir(parents=True, exist_ok=True)

        summdata_dir = path_data / "summdata"
        summdata_dir.mkdir(exist_ok=True)

        sample_data_dir = path_data / "sample_data"
        sample_data_dir.mkdir(exist_ok=True)

        fam_full = pd.read_csv(f"{bfile_full}.fam", sep=r"\s+", header=None,
                               names=["FID", "IID", "PID", "MID", "SEX", "PHENO"],
                               dtype={"FID": str, "IID": str})
        df["IID"] = df["individual_id"].astype(str)

        sst_files = []
        bfile_tuning_list = []
        bfile_testing_list = []
        pheno_tuning_list = []
        pheno_testing_list = []
        covar_tuning_list = []
        covar_testing_list = []

        for pop in pops:
            pop_dir = sample_data_dir / pop
            pop_dir.mkdir(parents=True, exist_ok=True)

            pop_df = df[df["pop_label"] == pop].copy()
            if pop_df.empty:
                raise RuntimeError(f"MUSSEL: no individuals found for pop={pop}")

            tuning_ids, testing_ids = self._split_ids(pop_df["IID"], tuning_frac=0.8, seed=42)

            keep_all = pop_dir / f"{pop}.keep"
            self._write_keep(fam_full, pop_df["IID"], keep_all)

            keep_tune = pop_dir / f"{pop}.tuning.keep"
            keep_test = pop_dir / f"{pop}.testing.keep"
            self._write_keep(fam_full, tuning_ids, keep_tune)
            self._write_keep(fam_full, testing_ids, keep_test)

            # Create bfiles: full prefix + per-chrom prefixes
            tuning_prefix = pop_dir / "tuning_geno"
            testing_prefix = pop_dir / "testing_geno"

            self._plink_subset(bfile_full, keep_tune, tuning_prefix)
            self._plink_subset(bfile_full, keep_test, testing_prefix)

            for chrom in chrom_list:
                self._plink_subset(
                    bfile_full,
                    keep_tune,
                    f"{tuning_prefix}_chr{chrom}",
                    chrom=chrom,
                )
                self._plink_subset(
                    bfile_full,
                    keep_test,
                    f"{testing_prefix}_chr{chrom}",
                    chrom=chrom,
                )

            fam_pop = fam_full[fam_full["IID"].isin(pop_df["IID"])].copy()
            pheno_path, covar_path = self._write_pheno_covar(fam_pop, pop_df, pop_dir)

            # Use the same pheno/covar file paths for tuning/testing
            pheno_tuning_list.append(str(pheno_path))
            pheno_testing_list.append(str(pheno_path))
            covar_tuning_list.append(str(covar_path))
            covar_testing_list.append(str(covar_path))

            # GWAS summary stats
            n_eff = None
            yvals = pop_df["y"].dropna()
            n_case = int((yvals == 1).sum())
            n_ctrl = int((yvals == 0).sum())
            if n_case > 0 and n_ctrl > 0:
                n_eff = 4.0 / (1.0 / n_ctrl + 1.0 / n_case)

            sst_path = self._run_gwas(
                bfile=str(tuning_prefix),
                pheno_file=str(pheno_path),
                out_prefix=str(summdata_dir / f"{pop}_gwas"),
                n_eff_override=n_eff,
            )
            sst_files.append(sst_path)

            bfile_tuning_list.append(str(tuning_prefix))
            bfile_testing_list.append(str(testing_prefix))

        pop_arg = ",".join(pops)
        sst_arg = ",".join(sst_files)
        bfile_tuning_arg = ",".join(bfile_tuning_list)
        bfile_testing_arg = ",".join(bfile_testing_list)
        pheno_tuning_arg = ",".join(pheno_tuning_list)
        pheno_testing_arg = ",".join(pheno_testing_list)
        covar_tuning_arg = ",".join(covar_tuning_list)
        covar_testing_arg = ",".join(covar_testing_list)

        path_plink = shutil.which(self.plink2_path) or self.plink2_path

        base_args = [
            f"--PATH_package", self.mussel_path,
            f"--PATH_data", str(path_data),
            f"--PATH_LDref", ldref_dir,
            f"--PATH_out", str(path_out),
            f"--FILE_sst", sst_arg,
            f"--pop", pop_arg,
        ]

        ncores_ldpred2 = self.ncores_ldpred2
        ncores_muss = self.ncores_muss
        ncores_mussel = self.ncores_mussel

        if not self.skip_ldpred2:
            self._run([
                self.rscript,
                str(Path(self.mussel_path) / "R/LDpred2_jobs.R"),
                *base_args,
                "--bfile_tuning", bfile_tuning_arg,
                "--NCORES", str(ncores_ldpred2),
            ])

        if not self.skip_ldpred2_tuning:
            self._run([
                self.rscript,
                str(Path(self.mussel_path) / "R/LDpred2_tuning.R"),
                "--PATH_package", self.mussel_path,
                "--PATH_out", str(path_out),
                "--PATH_plink", str(path_plink),
                "--FILE_sst", sst_arg,
                "--pop", pop_arg,
                "--chrom", chrom_arg,
                "--bfile_tuning", bfile_tuning_arg,
                "--pheno_tuning", pheno_tuning_arg,
                "--covar_tuning", covar_tuning_arg,
                "--bfile_testing", bfile_testing_arg,
                "--pheno_testing", pheno_testing_arg,
                "--covar_testing", covar_testing_arg,
                "--trait_type", trait_type,
                "--testing", "TRUE",
                "--NCORES", str(ncores_mussel),
            ])

        ldpred2_params = ",".join([str(path_out / "LDpred2" / f"{p}_optim_params.txt") for p in pops])

        if not self.skip_muss:
            self._run([
                self.rscript,
                str(Path(self.mussel_path) / "R/MUSS_jobs.R"),
                *base_args,
                "--LDpred2_params", ldpred2_params,
                "--chrom", "1-22",
                "--bfile_tuning", bfile_tuning_arg,
                "--NCORES", str(ncores_muss),
            ])

        if not self.skip_mussel:
            self._run([
                self.rscript,
                str(Path(self.mussel_path) / "R/MUSSEL.R"),
                "--PATH_package", self.mussel_path,
                "--PATH_out", str(path_out),
                "--PATH_plink", str(path_plink),
                "--pop", pop_arg,
                "--target_pop", target_pop,
                "--chrom", chrom_arg,
                "--bfile_tuning", bfile_tuning_arg,
                "--pheno_tuning", pheno_tuning_arg,
                "--covar_tuning", covar_tuning_arg,
                "--bfile_testing", bfile_testing_arg,
                "--pheno_testing", pheno_testing_arg,
                "--covar_testing", covar_testing_arg,
                "--trait_type", trait_type,
                "--testing", "TRUE",
                "--NCORES", str(ncores_mussel),
            ])

        return str(path_out)

    def predict(self, path_out, out_prefix):
        """
        Load MUSSEL-generated scores into a dataframe with columns: IID, PRS.
        """
        score_file = self.score_file
        if score_file:
            score_path = Path(score_file)
            if not score_path.is_absolute():
                score_path = Path(path_out) / score_path
        else:
            score_path = None
            candidates = list(Path(path_out).rglob("*"))
            for cand in candidates:
                if not cand.is_file():
                    continue
                name = cand.name.lower()
                if "mussel" in name or "superlearner" in name or "sl" in name:
                    score_path = cand
                    break

        if not score_path or not score_path.exists():
            files = [str(p) for p in Path(path_out).rglob("*") if p.is_file()]
            preview = "\n".join(files[:40])
            raise RuntimeError(
                "MUSSEL: could not find a score file. "
                "Set score_file in MUSSEL(...) to the output score file (relative to PATH_out or absolute).\n"
                f"PATH_out={path_out}\nFiles:\n{preview}"
            )

        df = pd.read_csv(score_path, sep=r"\s+|,|\t", engine="python")
        cols = [c.lower() for c in df.columns]

        iid_col = None
        prs_col = None
        for c in df.columns:
            if c.lower() in ["iid", "id", "sample", "sampleid"]:
                iid_col = c
            if c.lower() in ["prs", "score", "pred", "prediction", "pgs"]:
                prs_col = c

        if iid_col is None:
            if "iid" in cols:
                iid_col = df.columns[cols.index("iid")]
        if prs_col is None:
            # fallback to last numeric column
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if num_cols:
                prs_col = num_cols[-1]

        if iid_col is None or prs_col is None:
            raise RuntimeError(
                f"MUSSEL: could not infer IID/PRS columns from {score_path}. "
                f"Columns={list(df.columns)}"
            )

        out_df = pd.DataFrame({
            "IID": df[iid_col].astype(str),
            "PRS": pd.to_numeric(df[prs_col], errors="coerce"),
        }).dropna()

        if out_df.empty:
            raise RuntimeError(f"MUSSEL: score file {score_path} produced no valid rows.")

        out_df.to_csv(f"{out_prefix}.sscore", sep="\t", index=False)
        return out_df
