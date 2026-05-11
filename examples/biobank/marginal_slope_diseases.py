#!/usr/bin/env python3
"""Fit Bernoulli marginal-slope GAMs on All of Us microarray data.

For each disease the script:
  1. Looks up the SNOMED standard Condition concept by name in the
     OMOP/OHDSI `concept` table on the AoU CDR.
  2. Pulls everyone whose `condition_occurrence.condition_concept_id`
     descends from that concept via `concept_ancestor`.
  3. Fits `case ~ duchon(PC1..PC10) + sex` with the hard-coded PGS
     feeding the marginal-slope latent z, and the same joint anisotropic
     Duchon smooth on the log-slope channel.

Reported per disease: AUROC, Nagelkerke R^2, Lee-2012 liability R^2.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import struct
import subprocess
import urllib.request
from pathlib import Path

import gamfit
import numpy as np
import pandas as pd
from google.cloud import bigquery
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

WORKDIR = Path.home() / "aou-gpu-baremetal"
PLINK_PREFIX = WORKDIR / "arrays"
SEX_CACHE = Path.home() / ".aou_cache" / "sex_terms"
NUM_PCS = 10
DUCHON_CENTERS = 4 * NUM_PCS  # > polynomial nullspace dim (d+1) for Linear in d=10
GNOMON_BIN = os.environ.get("GNOMON_BIN", "gnomon")
PGS_ID_PATTERN = re.compile(r"^PGS\d{6}$")

DISEASES = {
    "copd": {
        "snomed_name": "Chronic obstructive lung disease",
        "prevalence": 0.06,
        "pgs": "PGS004536",
    },
    "hypertension": {
        "snomed_name": "Hypertensive disorder, systemic arterial",
        "prevalence": 0.45,
        "pgs": "PGSXXXXXX",
    },
    "obesity": {
        "snomed_name": "Obesity",
        "prevalence": 0.42,
        "pgs": "PGSXXXXXX",
    },
}


# --- loaders ---------------------------------------------------------------

def _latest(d: Path, glob: str) -> Path:
    hits = sorted(d.glob(glob), key=lambda p: p.stat().st_mtime)
    if not hits:
        raise FileNotFoundError(f"no match for {d}/{glob}")
    return hits[-1]


def _canonical_id(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip()


_SEX_MAP = {
    "0": 0, "1": 1, "2": 0,
    "f": 0, "female": 0, "m": 1, "male": 1,
    "xx": 0, "xy": 1,
}


def load_sex() -> pd.DataFrame:
    path = _latest(SEX_CACHE, "sex_*.tsv")
    df = pd.read_csv(path, sep="\t", dtype=str)
    id_col = next(c for c in df.columns if c.lower() in {"research_id", "sample_id", "iid", "person_id", "#iid"})
    sex_col = next(c for c in df.columns if "sex" in c.lower())
    out = pd.DataFrame({
        "person_id": _canonical_id(df[id_col]),
        "sex": df[sex_col].astype(str).str.strip().str.lower().map(_SEX_MAP),
    }).dropna()
    out["sex"] = out["sex"].astype(int)
    print(f"  sex:  file={path.name}  n={len(out):,}  e.g. {out['person_id'].head(3).tolist()}")
    return out


def load_pcs(num_pcs: int) -> pd.DataFrame:
    """Read gnomon's `projection_scores.bin` (GNPRJ001/GNPSID01)."""
    bin_path = WORKDIR / "arrays.projection_scores.bin"
    meta = json.loads((WORKDIR / "arrays.projection_scores.metadata.json").read_text())
    rows, cols = int(meta["rows"]), int(meta["cols"])
    with bin_path.open("rb") as fh:
        assert fh.read(8) == b"GNPRJ001"
        fh.read(4 + 8 + 8 + 4)
        data = np.fromfile(fh, dtype="<f8", count=rows * cols).reshape(cols, rows).T
        assert fh.read(8) == b"GNPSID01"
        fh.read(4 + 4)
        count = struct.unpack("<Q", fh.read(8))[0]
        sb = struct.unpack("<Q", fh.read(8))[0]
        offsets = np.frombuffer(fh.read(8 * (count + 1)), dtype="<u8")
        blob = fh.read(sb)
    ids = [blob[offsets[i]:offsets[i + 1]].decode() for i in range(count)]
    df = pd.DataFrame(data[:, :num_pcs], columns=[f"PC{i+1}" for i in range(num_pcs)])
    df.insert(0, "person_id", _canonical_id(pd.Series(ids)))
    print(f"  pcs:  n={len(df):,}  e.g. {df['person_id'].head(3).tolist()}")
    return df


def ensure_scored(pgs_ids: list[str]) -> None:
    """Run a single `gnomon score` call for any PGS that lacks an sscore file."""
    missing = [p for p in pgs_ids if not (WORKDIR / f"arrays_{p}.sscore").exists()]
    if not missing:
        return
    print(f"[score] missing per-PGS sscore for: {missing}")
    tmp = WORKDIR / ".gamfit_pgs_tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    for pgs in missing:
        url = (
            "https://ftp.ebi.ac.uk/pub/databases/spot/pgs/scores/"
            f"{pgs}/ScoringFiles/Harmonized/{pgs}_hmPOS_GRCh38.txt.gz"
        )
        dst = tmp / f"{pgs}_hmPOS_GRCh38.txt.gz"
        print(f"[score] downloading {url}")
        urllib.request.urlretrieve(url, dst)
    print(f"[score] running {GNOMON_BIN} score {tmp} {PLINK_PREFIX}")
    subprocess.run([GNOMON_BIN, "score", str(tmp), str(PLINK_PREFIX)], check=True)
    shutil.rmtree(tmp, ignore_errors=True)


def load_one_pgs(pgs_id: str) -> pd.DataFrame:
    path = WORKDIR / f"arrays_{pgs_id}.sscore"
    df = pd.read_csv(path, sep="\t", dtype={0: str})
    df = df.rename(columns={df.columns[0]: "person_id"})
    df["person_id"] = _canonical_id(df["person_id"])
    avg_col = next(c for c in df.columns if c.endswith("_AVG"))
    df = df[["person_id", avg_col]].rename(columns={avg_col: "pgs"})
    print(f"  pgs:  file={path.name}  n={len(df):,}")
    return df


# --- cases -----------------------------------------------------------------

def lookup_snomed_concept(client: bigquery.Client, cdr: str, name: str) -> int:
    """Return the OMOP concept_id for a standard SNOMED Condition concept."""
    sql = f"""
    SELECT concept_id
    FROM `{cdr}.concept`
    WHERE vocabulary_id = 'SNOMED'
      AND standard_concept = 'S'
      AND domain_id = 'Condition'
      AND LOWER(concept_name) = LOWER(@name)
    ORDER BY concept_id
    LIMIT 1
    """
    job = client.query(
        sql,
        job_config=bigquery.QueryJobConfig(query_parameters=[
            bigquery.ScalarQueryParameter("name", "STRING", name),
        ]),
    )
    rows = list(job.result())
    if not rows:
        raise ValueError(f"no standard SNOMED Condition concept named {name!r}")
    return int(rows[0]["concept_id"])


def fetch_cases(client: bigquery.Client, cdr: str, ancestor_id: int) -> set[str]:
    """Persons with any condition concept descending from `ancestor_id`."""
    sql = f"""
    SELECT DISTINCT CAST(co.person_id AS STRING) AS person_id
    FROM `{cdr}.condition_occurrence` AS co
    JOIN `{cdr}.concept_ancestor` AS ca
      ON ca.descendant_concept_id = co.condition_concept_id
    WHERE ca.ancestor_concept_id = @ancestor
    """
    job = client.query(
        sql,
        job_config=bigquery.QueryJobConfig(query_parameters=[
            bigquery.ScalarQueryParameter("ancestor", "INT64", ancestor_id),
        ]),
    )
    return set(job.to_dataframe()["person_id"].astype(str))


# --- model -----------------------------------------------------------------

def fit_marginal_slope(df: pd.DataFrame, num_pcs: int) -> gamfit.Model:
    pcs = ", ".join(f"PC{i+1}" for i in range(num_pcs))
    duchon = f"duchon({pcs}, centers={DUCHON_CENTERS}, order=1, power=2, length_scale=1.0)"
    z = (df["pgs"] - df["pgs"].mean()) / df["pgs"].std(ddof=0)
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
    z = float(norm.pdf(norm.ppf(K)))
    liability_r2 = nagelkerke * K * (1 - K) / (z ** 2 * P * (1 - P))
    return {
        "n": n,
        "cases": int(y.sum()),
        "P": P,
        "auroc": float(roc_auc_score(y, p)),
        "nagelkerke_r2": float(nagelkerke),
        "liability_r2": float(liability_r2),
    }


# --- main ------------------------------------------------------------------

def main() -> None:
    diseases = {k: v for k, v in DISEASES.items() if PGS_ID_PATTERN.match(v["pgs"])}
    print(f"diseases with real PGS IDs: {list(diseases)}")

    ensure_scored([cfg["pgs"] for cfg in diseases.values()])

    print("loading PCs and sex ...")
    pcs = load_pcs(NUM_PCS)
    sex = load_sex()
    base = pcs.merge(sex, on="person_id")
    print(f"base: n={len(base):,}")

    cdr = os.environ["WORKSPACE_CDR"]
    client = bigquery.Client()

    for name, cfg in diseases.items():
        print(f"\n=== {name.upper()} ===")
        pgs = load_one_pgs(cfg["pgs"])
        df = base.merge(pgs, on="person_id")
        ancestor = lookup_snomed_concept(client, cdr, cfg["snomed_name"])
        cases = fetch_cases(client, cdr, ancestor)
        print(f"  snomed={cfg['snomed_name']!r}  concept_id={ancestor}  cases={len(cases):,}")
        df["case"] = df["person_id"].isin(cases).astype(int)
        y = df["case"].to_numpy()
        model = fit_marginal_slope(df, NUM_PCS)
        p_hat = np.asarray(model.predict(df.drop(columns=["case"])), dtype=float)
        m = metrics(y, p_hat, cfg["prevalence"])
        print(
            f"  PGS={cfg['pgs']}  n={m['n']:,}  cases={m['cases']:,}  "
            f"P={m['P']:.4f}  K={cfg['prevalence']:.3f}"
        )
        print(
            f"  AUROC={m['auroc']:.4f}  Nagelkerke R^2={m['nagelkerke_r2']:.4f}  "
            f"liability R^2={m['liability_r2']:.4f}"
        )


if __name__ == "__main__":
    main()
