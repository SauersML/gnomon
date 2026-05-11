#!/usr/bin/env python3
"""Fit Bernoulli marginal-slope models on All of Us microarray data.

For each of COPD, hypertension, and obesity:
  * pull cases from the AoU CDR with ICD-10 / ICD-9 codes,
  * pick the best single PGS by AUROC out of the consolidated 533-score bulk,
  * fit `case ~ duchon(PC1..PCk) + sex` with the chosen PGS feeding the
    log-slope channel through a joint Duchon smooth over the same PCs,
  * report AUROC, Nagelkerke R^2, and Lee-2012 liability-scale R^2.

Inputs are read from the gnomon outputs already on disk:
  ~/.aou_cache/sex_terms/sex_<hash>.tsv         (sex)
  ~/aou-gpu-baremetal/arrays.projection_scores.bin + .metadata.json  (PCs)
  ~/aou-gpu-baremetal/arrays_pgs533_<hash>.sscore                    (PGS)
"""

from __future__ import annotations

import json
import os
import struct
from pathlib import Path

import numpy as np
import pandas as pd
import gamfit
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

WORKDIR = Path.home() / "aou-gpu-baremetal"
SEX_CACHE = Path.home() / ".aou_cache" / "sex_terms"
NUM_PCS = 10

DISEASES = {
    "copd": {
        "icd10": ["J44.0", "J44.1", "J44.9"],
        "icd9": ["491", "492", "496"],
        "prevalence": 0.06,
        "pgs": "PGS004536",  # Jung et al. metaPRS for J44; OR/SD=1.488 in UKB EUR.
    },
    "hypertension": {
        "icd10": ["I10", "I11", "I12", "I13", "I15"],
        "icd9": ["401", "402", "403", "404", "405"],
        "prevalence": 0.45,
        "pgs": "PGSXXXXXX",  # TODO: pick a disease-only HTN PRS from the Catalog.
    },
    "obesity": {
        "icd10": [
            "E66.0", "E66.01", "E66.09", "E66.1", "E66.2",
            "E66.8", "E66.811", "E66.812", "E66.813", "E66.89", "E66.9",
        ],
        "icd9": ["278.00", "278.01", "278.03"],
        "prevalence": 0.42,
        "pgs": "PGSXXXXXX",  # TODO: pick an obesity PRS from the Catalog.
    },
}


# --- loaders ----------------------------------------------------------------

def _latest(pattern_dir: Path, glob: str) -> Path:
    hits = sorted(pattern_dir.glob(glob), key=lambda p: p.stat().st_mtime)
    if not hits:
        raise FileNotFoundError(f"no match for {pattern_dir}/{glob}")
    return hits[-1]


def load_sex() -> pd.DataFrame:
    path = _latest(SEX_CACHE, "sex_*.tsv")
    df = pd.read_csv(path, sep="\t", dtype=str)
    id_col = next(c for c in df.columns if c.lower() in {"research_id", "sample_id", "iid", "person_id", "#iid"})
    sex_col = next(c for c in df.columns if "sex" in c.lower())
    out = pd.DataFrame({
        "person_id": df[id_col].astype(str),
        "sex": pd.to_numeric(df[sex_col], errors="coerce"),
    }).dropna()
    out["sex"] = out["sex"].astype(int).clip(0, 1)
    return out


def load_pcs(num_pcs: int) -> pd.DataFrame:
    """Read gnomon's `projection_scores.bin` (GNPRJ001) into a DataFrame."""
    bin_path = WORKDIR / "arrays.projection_scores.bin"
    meta_path = WORKDIR / "arrays.projection_scores.metadata.json"
    meta = json.loads(meta_path.read_text())
    rows, cols = int(meta["rows"]), int(meta["cols"])
    assert meta["dtype"] == "f64_le" and meta["layout"] == "column_major"
    with bin_path.open("rb") as fh:
        magic = fh.read(8)
        assert magic == b"GNPRJ001", f"unexpected magic {magic!r}"
        fh.read(4 + 8 + 8 + 4)  # version, rows, cols, padding (we trust metadata)
        data = np.fromfile(fh, dtype="<f8", count=rows * cols).reshape(cols, rows).T
        # row-id section
        assert fh.read(8) == b"GNPSID01"
        fh.read(4 + 4)  # version, padding
        count = struct.unpack("<Q", fh.read(8))[0]
        string_bytes = struct.unpack("<Q", fh.read(8))[0]
        offsets = np.frombuffer(fh.read(8 * (count + 1)), dtype="<u8")
        blob = fh.read(string_bytes)
    ids = [blob[offsets[i]:offsets[i + 1]].decode() for i in range(count)]
    df = pd.DataFrame(data[:, :num_pcs], columns=[f"PC{i+1}" for i in range(num_pcs)])
    df.insert(0, "person_id", ids)
    return df


def load_pgs_bulk() -> pd.DataFrame:
    path = _latest(WORKDIR, "arrays_pgs533_*.sscore")
    df = pd.read_csv(path, sep="\t", dtype={0: str}, low_memory=False)
    df = df.rename(columns={df.columns[0]: "person_id"})
    df["person_id"] = df["person_id"].astype(str)
    return df


# --- cases ------------------------------------------------------------------

def fetch_cases(icd10: list[str], icd9: list[str]) -> set[str]:
    from google.cloud import bigquery

    cdr = os.environ["WORKSPACE_CDR"]
    codes = {c.upper() for c in icd10 + icd9}
    codes |= {c.replace(".", "") for c in codes}
    client = bigquery.Client()
    job = client.query(
        f"""
        SELECT DISTINCT CAST(person_id AS STRING) AS person_id
        FROM `{cdr}.condition_occurrence`
        WHERE condition_source_value IS NOT NULL
          AND UPPER(TRIM(condition_source_value)) IN UNNEST(@codes)
        """,
        job_config=bigquery.QueryJobConfig(query_parameters=[
            bigquery.ArrayQueryParameter("codes", "STRING", sorted(codes))
        ]),
    )
    return set(job.to_dataframe()["person_id"].astype(str))


# --- model ------------------------------------------------------------------

def fit_marginal_slope(df: pd.DataFrame, pgs_col: str, num_pcs: int) -> gamfit.Model:
    pcs = ", ".join(f"PC{i+1}" for i in range(num_pcs))
    duchon = f"duchon({pcs}, centers={num_pcs + 1}, order=1, power=2, length_scale=1.0)"
    z = (df[pgs_col] - df[pgs_col].mean()) / df[pgs_col].std(ddof=0)
    table = df[["case", "sex"] + [f"PC{i+1}" for i in range(num_pcs)]].copy()
    table["prs_z"] = z.to_numpy()
    return gamfit.fit(
        table,
        f"case ~ {duchon} + sex",
        link="probit",
        z_column="prs_z",
        logslope_formula=duchon,
        scale_dimensions=True,
    )


def metrics(y: np.ndarray, p: np.ndarray, K: float) -> dict[str, float]:
    eps = 1e-12
    p = np.clip(p, eps, 1 - eps)
    ll_full = float(np.sum(y * np.log(p) + (1 - y) * np.log1p(-p)))
    P = float(y.mean())
    ll_null = float(np.sum(y * np.log(P) + (1 - y) * np.log1p(-P)))
    n = len(y)
    cox_snell = 1.0 - np.exp(2 * (ll_null - ll_full) / n)
    nagelkerke = cox_snell / (1.0 - np.exp(2 * ll_null / n))
    z = norm.pdf(norm.ppf(K))
    liability_r2 = nagelkerke * K * (1 - K) / (z ** 2 * P * (1 - P))  # Lee 2012, ascertained
    return {
        "n": n,
        "cases": int(y.sum()),
        "sample_prevalence": P,
        "auroc": float(roc_auc_score(y, p)),
        "nagelkerke_r2": float(nagelkerke),
        "liability_r2": float(liability_r2),
    }


# --- main -------------------------------------------------------------------

def main() -> None:
    print("loading PCs, sex, and the 533-score bulk PGS file ...")
    pcs = load_pcs(NUM_PCS)
    sex = load_sex()
    pgs = load_pgs_bulk()
    cohort = pcs.merge(sex, on="person_id").merge(pgs, on="person_id")
    print(f"cohort: n={len(cohort):,}, pcs={NUM_PCS}")

    for name, cfg in DISEASES.items():
        print(f"\n=== {name.upper()} ===")
        pgs_id = cfg["pgs"]
        col = f"{pgs_id}_AVG"
        if col not in cohort.columns:
            print(f"  [skip] {pgs_id} not present in the bulk score file")
            continue

        cases = fetch_cases(cfg["icd10"], cfg["icd9"])
        y = cohort["person_id"].isin(cases).astype(int).to_numpy()
        df = cohort.copy()
        df["case"] = y
        model = fit_marginal_slope(df, col, NUM_PCS)
        p_hat = np.asarray(model.predict(df.drop(columns=["case"])), dtype=float)
        m = metrics(y, p_hat, cfg["prevalence"])
        print(
            f"  PGS={pgs_id}  n={m['n']:,}  cases={m['cases']:,}  "
            f"P={m['sample_prevalence']:.4f}  K={cfg['prevalence']:.3f}"
        )
        print(
            f"  AUROC={m['auroc']:.4f}  Nagelkerke R^2={m['nagelkerke_r2']:.4f}  "
            f"liability R^2={m['liability_r2']:.4f}"
        )


if __name__ == "__main__":
    main()
