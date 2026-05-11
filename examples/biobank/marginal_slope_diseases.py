#!/usr/bin/env python3
"""Fit Bernoulli marginal-slope GAMs on All of Us microarray data.

Cases for each disease are every person with at least one
`condition_occurrence` row whose `condition_concept_id` is a SNOMED
descendant of the disease's parent concept (resolved via the OMOP/OHDSI
`concept_ancestor` table). No ICD string matching, no count thresholds.

For each disease we fit
    case ~ duchon(PC1..PC10) + sex
with the hard-coded PGS feeding the marginal-slope latent z, and the same
joint anisotropic Duchon smooth on the log-slope channel. Reported:
AUROC, Nagelkerke R^2, and Lee-2012 liability-scale R^2.
"""

from __future__ import annotations

import json
import os
import struct
from pathlib import Path

import gamfit
import numpy as np
import pandas as pd
from google.cloud import bigquery
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

WORKDIR = Path.home() / "aou-gpu-baremetal"
SEX_CACHE = Path.home() / ".aou_cache" / "sex_terms"
NUM_PCS = 10

DISEASES = {
    "copd": {
        # SNOMED 13645005 "Chronic obstructive lung disease"
        "snomed_ancestor": 255573,
        "prevalence": 0.06,
        "pgs": "PGS004536",
    },
    "hypertension": {
        # SNOMED 38341003 "Hypertensive disorder, systemic arterial"
        "snomed_ancestor": 316866,
        "prevalence": 0.45,
        "pgs": "PGSXXXXXX",
    },
    "obesity": {
        # SNOMED 414916001 "Obesity"
        "snomed_ancestor": 433736,
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
    """Normalize PLINK / AoU person identifiers to the bare research_id."""
    out = s.astype(str).str.strip()
    split = out.str.split("_", n=1, expand=True)
    if split.shape[1] == 2:
        same = split[0] == split[1]
        out = out.where(~same, split[1])
    out = out.str.replace(r"\.0$", "", regex=True)
    return out


def load_sex() -> pd.DataFrame:
    path = _latest(SEX_CACHE, "sex_*.tsv")
    df = pd.read_csv(path, sep="\t", dtype=str)
    id_col = next(c for c in df.columns if c.lower() in {"research_id", "sample_id", "iid", "person_id", "#iid"})
    sex_col = next(c for c in df.columns if "sex" in c.lower())
    out = pd.DataFrame({
        "person_id": _canonical_id(df[id_col]),
        "sex": pd.to_numeric(df[sex_col], errors="coerce"),
    }).dropna()
    out["sex"] = out["sex"].astype(int).clip(0, 1)
    print(f"  sex:  n={len(out):,}  e.g. {out['person_id'].head(3).tolist()}")
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


def load_pgs_bulk() -> pd.DataFrame:
    path = _latest(WORKDIR, "arrays_pgs533_*.sscore")
    df = pd.read_csv(path, sep="\t", dtype={0: str}, low_memory=False)
    df = df.rename(columns={df.columns[0]: "person_id"})
    df["person_id"] = _canonical_id(df["person_id"])
    n_scores = sum(c.endswith("_AVG") for c in df.columns)
    print(f"  pgs:  n={len(df):,}  scores={n_scores}  e.g. {df['person_id'].head(3).tolist()}")
    return df


# --- cases -----------------------------------------------------------------

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
    print("loading PCs, sex, and the 533-score bulk PGS file ...")
    pcs = load_pcs(NUM_PCS)
    sex = load_sex()
    pgs = load_pgs_bulk()
    cohort = pcs.merge(sex, on="person_id").merge(pgs, on="person_id")
    if cohort.empty:
        sources = {"pcs": pcs, "sex": sex, "pgs": pgs}
        for a in sources:
            for b in sources:
                if a < b:
                    n = len(set(sources[a]["person_id"]) & set(sources[b]["person_id"]))
                    print(f"  overlap {a}<->{b}: {n:,}")
        raise RuntimeError("cohort merge produced 0 rows — ID formats differ across sources")
    print(f"cohort: n={len(cohort):,}")

    cdr = os.environ["WORKSPACE_CDR"]
    client = bigquery.Client()

    for name, cfg in DISEASES.items():
        print(f"\n=== {name.upper()} ===")
        col = f"{cfg['pgs']}_AVG"
        if col not in cohort.columns:
            print(f"  [skip] {cfg['pgs']} not present in bulk score file")
            continue
        cases = fetch_cases(client, cdr, cfg["snomed_ancestor"])
        print(f"  cases: ancestor_concept_id={cfg['snomed_ancestor']}  n={len(cases):,}")
        y = cohort["person_id"].isin(cases).astype(int).to_numpy()
        df = cohort.copy()
        df["case"] = y
        model = fit_marginal_slope(df, col, NUM_PCS)
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
