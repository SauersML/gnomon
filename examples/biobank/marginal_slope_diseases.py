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
import logging
import os
import re
import struct
import subprocess
import sys
from pathlib import Path

import gamfit
import numpy as np
import pandas as pd
from google.cloud import bigquery
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
    force=True,
)

WORKDIR = Path.home() / "aou-gpu-baremetal"
PLINK_PREFIX = WORKDIR / "arrays"
SEX_CACHE = Path.home() / ".aou_cache" / "sex_terms"
NUM_PCS = 3
DUCHON_CENTERS = 12  # > linear nullspace (d+1=4) in d=3
N_TRAIN_CASES = 200
N_TRAIN_CONTROLS = 200
N_TEST_CASES = 200
N_TEST_CONTROLS = 200
RNG_SEED = 0
GNOMON_BIN = os.environ.get("GNOMON_BIN", "gnomon")
PGS_ID_PATTERN = re.compile(r"^PGS\d{6}$")

DISEASES = {
    "copd": {
        "snomed_name": "Chronic obstructive lung disease",
        # Jung et al. metaPRS for J44; OR/SD = 1.488 in UKB EUR. Not trained in AoU.
        "pgs": "PGS004536",
    },
    "hypertension": {
        # OMOP standard SNOMED concept_name for 38341003 in the AoU CDR.
        "snomed_name": "Hypertensive disorder",
        # Privé et al. 2022 sparse hypertension PRS; PGS-only AUROC 0.629 in
        # held-out UKB EUR. Not trained in AoU.
        "pgs": "PGS001320",
    },
    "obesity": {
        "snomed_name": "Obesity",
        # Kim et al. 2026 O_MetPRS_EUR; LDpred2 over multi-ancestry GWAS of 20
        # metabolic traits. OR=2.47, AUROC=0.728 for BMI>=30. AoU appears only
        # as an evaluation cohort, not training.
        "pgs": "PGS005331",
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


def _find_sscore_for(pgs_id: str) -> Path | None:
    """Find any `arrays_*.sscore` whose header contains `{pgs_id}_AVG`.

    gnomon names per-PGS outputs differently depending on how `--score` is
    invoked:
      - directory of pre-downloaded score files -> `arrays_<PGS_ID>.sscore`
      - inline comma-separated IDs              -> `arrays_pgs<N>_<hash>.sscore`
    Filename matching alone would miss the second form, so we look at column
    headers instead.
    """
    target = f"{pgs_id}_AVG"
    for path in WORKDIR.glob("arrays_*.sscore"):
        try:
            with path.open() as fh:
                header = fh.readline().rstrip("\n").split("\t")
        except OSError:
            continue
        if target in header:
            return path
    return None


def ensure_scored(pgs_ids: list[str]) -> None:
    """Run a single `gnomon score` call for any PGS not yet scored on disk.

    "Already scored" = some `arrays_*.sscore` in WORKDIR has a `<PGS>_AVG`
    column. This catches both gnomon's per-PGS naming and the hashed inline
    naming, so we don't re-score a file that's already there under a
    different filename and trip gnomon's overwrite refusal.
    """
    missing = [p for p in pgs_ids if _find_sscore_for(p) is None]
    if not missing:
        return
    score_arg = ",".join(missing)
    # AoU's CUDA image ships only driver + cudart, so feed gnomon every CUDA
    # runtime shared library shipped by the `nvidia-*-cu12` pip wheels via the
    # score subprocess's LD_LIBRARY_PATH.
    import nvidia  # type: ignore[import-not-found]
    nv_libs = [
        str(child / "lib")
        for parent in nvidia.__path__
        for child in Path(parent).iterdir()
        if (child / "lib").is_dir()
    ]
    env = {
        **os.environ,
        "LD_LIBRARY_PATH": ":".join([*nv_libs, os.environ.get("LD_LIBRARY_PATH", "")]),
    }
    print(f"[score] running {GNOMON_BIN} score {score_arg} {PLINK_PREFIX}")
    print(f"[score] cuda libs: {nv_libs}")
    subprocess.run([GNOMON_BIN, "score", score_arg, str(PLINK_PREFIX)], check=True, env=env)


def load_one_pgs(pgs_id: str) -> pd.DataFrame:
    path = _find_sscore_for(pgs_id)
    if path is None:
        raise FileNotFoundError(
            f"no sscore file in {WORKDIR} carries column {pgs_id}_AVG"
        )
    avg_col = f"{pgs_id}_AVG"
    df = pd.read_csv(path, sep="\t", dtype={0: str}, low_memory=False)
    df = df.rename(columns={df.columns[0]: "person_id"})
    df["person_id"] = _canonical_id(df["person_id"])
    df = df[["person_id", avg_col]].rename(columns={avg_col: "pgs"})
    df["pgs"] = pd.to_numeric(df["pgs"], errors="coerce")
    print(f"  pgs:  file={path.name}  col={avg_col}  n={len(df):,}")
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

def fit_marginal_slope(train_df: pd.DataFrame, num_pcs: int) -> gamfit.Model:
    """Bernoulli marginal-slope probit GAM with joint Duchon over PCs in
    both the location and log-slope channels; sex linear; prs_z is the
    latent score (z_column) so its slope varies in PC space.

    No linkwiggle(...) in either formula -> engine's score-warp / link-dev
    deviation blocks remain inactive in the protocol that this triggers.
    """
    pcs = ", ".join(f"PC{i+1}" for i in range(num_pcs))
    duchon = f"duchon({pcs}, centers={DUCHON_CENTERS}, order=1, power=2, length_scale=1.0)"
    formula = f"case ~ {duchon} + sex"
    cols = ["case", "sex", "prs_z"] + [f"PC{i+1}" for i in range(num_pcs)]
    print(f"  fit_spec: family=bernoulli marginal-slope  link=probit")
    print(f"  fit_spec: formula={formula!r}")
    print(f"  fit_spec: z_column='prs_z'  logslope_formula={duchon!r}")
    print(f"  fit_spec: num_pcs={num_pcs}  duchon_centers={DUCHON_CENTERS}  n_train={len(train_df)}")
    return gamfit.fit(
        train_df[cols],
        formula,
        link="probit",
        z_column="prs_z",
        logslope_formula=duchon,
    )


def metrics(y: np.ndarray, p: np.ndarray, K: float) -> dict[str, float]:
    """Held-out AUROC + Nagelkerke + Lee-2011 liability-scale R^2.

    Lee, Wray, Goddard, Visscher (AJHG 2011, eq. 23) for ascertained case-control:

        R^2_l = R^2_O * K^2 * (1-K)^2 / (z^2 * P * (1-P))

    K is the population prevalence, P is the sample case proportion, and
    z = phi(Phi^{-1}(K)) is the standard normal density at the liability threshold.
    """
    eps = 1e-12
    p = np.clip(p, eps, 1 - eps)
    ll_full = float(np.sum(y * np.log(p) + (1 - y) * np.log1p(-p)))
    P = float(y.mean())
    ll_null = float(np.sum(y * np.log(P) + (1 - y) * np.log1p(-P)))
    n = len(y)
    cox_snell = 1.0 - np.exp(2 * (ll_null - ll_full) / n)
    nagelkerke = cox_snell / (1.0 - np.exp(2 * ll_null / n))
    z = float(norm.pdf(norm.ppf(K)))
    liability_r2 = nagelkerke * K**2 * (1 - K) ** 2 / (z**2 * P * (1 - P))
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
    print(f"gamfit version: {gamfit.__version__}")
    print(f"gamfit build_info: {gamfit.build_info()}")
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

    rng = np.random.default_rng(RNG_SEED)

    for name, cfg in diseases.items():
        print(f"\n=== {name.upper()} ===")
        pgs_df = load_one_pgs(cfg["pgs"])
        df_full = base.merge(pgs_df, on="person_id")
        ancestor = lookup_snomed_concept(client, cdr, cfg["snomed_name"])
        cases = fetch_cases(client, cdr, ancestor)
        df_full["case"] = df_full["person_id"].isin(cases).astype(int)
        case_idx = rng.permutation(df_full.index[df_full["case"] == 1].to_numpy())
        ctrl_idx = rng.permutation(df_full.index[df_full["case"] == 0].to_numpy())
        # Cohort prevalence: inferred from the actual merged cohort (PCs ∩ sex ∩
        # PGS rows on disk), not a hardcoded literature number. Lee 2011 wants
        # the population prevalence; for AoU we use the cohort prevalence as
        # its best in-data estimate.
        K = len(case_idx) / max(1, len(case_idx) + len(ctrl_idx))
        print(
            f"  snomed={cfg['snomed_name']!r}  concept_id={ancestor}  "
            f"cases_in_cohort={len(case_idx):,}  controls_in_cohort={len(ctrl_idx):,}  "
            f"K={K:.6f}"
        )

        n_te_case = min(N_TEST_CASES, max(0, len(case_idx) - N_TRAIN_CASES))
        n_te_ctrl = min(N_TEST_CONTROLS, max(0, len(ctrl_idx) - N_TRAIN_CONTROLS))
        train_pick = np.concatenate([case_idx[:N_TRAIN_CASES], ctrl_idx[:N_TRAIN_CONTROLS]])
        test_pick = np.concatenate([
            case_idx[N_TRAIN_CASES : N_TRAIN_CASES + n_te_case],
            ctrl_idx[N_TRAIN_CONTROLS : N_TRAIN_CONTROLS + n_te_ctrl],
        ])
        train = df_full.loc[train_pick].reset_index(drop=True)
        test = df_full.loc[test_pick].reset_index(drop=True)

        # Standardize PGS on training only, then apply the same shift/scale to test
        # so we never mix test rows into the training statistics.
        pgs_mean = float(train["pgs"].mean())
        pgs_std = float(train["pgs"].std(ddof=0))
        train["prs_z"] = (train["pgs"] - pgs_mean) / pgs_std
        test["prs_z"] = (test["pgs"] - pgs_mean) / pgs_std

        model = fit_marginal_slope(train, NUM_PCS)
        predict_cols = ["sex", "prs_z"] + [f"PC{i+1}" for i in range(NUM_PCS)]
        pred = model.predict(test[predict_cols], return_type="dict")
        p_test = np.asarray(pred["mean"], dtype=float)
        m = metrics(test["case"].to_numpy(), p_test, K)

        print(
            f"  PGS={cfg['pgs']}  train_n={len(train):,}  test_n={m['n']:,}  "
            f"test_cases={m['cases']:,}  P={m['P']:.4f}  K={K:.6f}"
        )
        print(
            f"  held-out  AUROC={m['auroc']:.4f}  "
            f"Nagelkerke R^2={m['nagelkerke_r2']:.4f}  liability R^2={m['liability_r2']:.4f}"
        )


if __name__ == "__main__":
    main()
