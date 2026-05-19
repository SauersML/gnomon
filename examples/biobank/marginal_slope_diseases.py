#!/usr/bin/env python3
"""Fit survival marginal-slope GAMs on All of Us microarray data.

For each disease the script:
  1. Looks up the SNOMED standard Condition concept by name in the
     OMOP/OHDSI `concept` table on the AoU CDR.
  2. Pulls everyone whose `condition_occurrence.condition_concept_id`
     descends from that concept via `concept_ancestor`.
  3. Fits `Surv(entry_age, exit_age, event) ~ duchon(PC1..PC10) + sex`
     with the hard-coded PGS feeding the marginal-slope latent z, and the
     same joint anisotropic Duchon smooth on the log-slope channel.
  4. Compares against a Z_norm2 + Cox PH baseline on the same split.
  5. Runs leave-one-group-out OOD refits by care site, Census region, and
     AoU inferred genetic ancestry category.

Reported per disease: train/test Harrell's C and OOD held-out-group C.
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

import numpy as np
import pandas as pd
from google.cloud import bigquery
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
    force=True,
)

WORKDIR = Path.home() / "aou-gpu-baremetal"
PLINK_PREFIX = WORKDIR / "arrays"
SEX_CACHE = Path.home() / ".aou_cache" / "sex_terms"
FITS_DIR = WORKDIR / "biobank_fits"
NUM_PCS = 3
DUCHON_CENTERS = 10  # > linear nullspace (d+1=4) in d=3
TRAIN_FRACTION = 0.80  # per-class 80/20 split
RNG_SEED = 0
MAX_LOSO_CARE_SITES = 5
MIN_LOSO_TRAIN_EVENTS = 50
MIN_LOSO_TRAIN_CENSORS = 50
MIN_LOSO_TEST_EVENTS = 20
MIN_LOSO_TEST_CENSORS = 20
MIN_LOSO_TEST_N = 500
LOSO_AXES = ("care_site", "census_region", "ancestry")
LOSO_AXIS_TO_COLUMN = {
    "care_site": "care_site_group",
    "census_region": "census_region",
    "ancestry": "ancestry_category",
}
GNOMON_BIN = os.environ.get("GNOMON_BIN", "gnomon")
PGS_ID_PATTERN = re.compile(r"^PGS\d{6}$")
ANCESTRY_PREDS_URI = (
    "gs://fc-aou-datasets-controlled/v8/wgs/short_read/snpindel/aux/ancestry/"
    "ancestry_preds.tsv"
)
ANCESTRY_PREDS_CACHE = WORKDIR / "ancestry_preds.tsv"

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
    # Some hits are the 1+ GB bulk 533-PGS file; only read the id col + the
    # one PGS column so the load is fast regardless of file width.
    with path.open() as fh:
        id_col = fh.readline().rstrip("\n").split("\t")[0]
    print(f"  pgs:  file={path.name}  col={avg_col}  reading ...", flush=True)
    df = pd.read_csv(
        path,
        sep="\t",
        usecols=[id_col, avg_col],
        dtype={id_col: str, avg_col: float},
        low_memory=False,
    )
    df = df.rename(columns={id_col: "person_id", avg_col: "pgs"})
    df["person_id"] = _canonical_id(df["person_id"])
    print(f"  pgs:  file={path.name}  col={avg_col}  n={len(df):,}", flush=True)
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


def fetch_cases(client: bigquery.Client, cdr: str, ancestor_id: int) -> pd.DataFrame:
    """Per-case earliest qualifying condition date.

    Returns columns `person_id` (str) and `event_date` (datetime64). The event
    date is the earliest `condition_start_date` across any descendant concept,
    which is what age-as-time-scale survival wants as the failure time.
    """
    sql = f"""
    SELECT CAST(co.person_id AS STRING) AS person_id,
           MIN(co.condition_start_date) AS event_date
    FROM `{cdr}.condition_occurrence` AS co
    JOIN `{cdr}.concept_ancestor` AS ca
      ON ca.descendant_concept_id = co.condition_concept_id
    WHERE ca.ancestor_concept_id = @ancestor
    GROUP BY co.person_id
    """
    job = client.query(
        sql,
        job_config=bigquery.QueryJobConfig(query_parameters=[
            bigquery.ScalarQueryParameter("ancestor", "INT64", ancestor_id),
        ]),
    )
    df = job.to_dataframe()
    df["person_id"] = _canonical_id(df["person_id"])
    df["event_date"] = pd.to_datetime(df["event_date"])
    return df


def fetch_person_times(client: bigquery.Client, cdr: str) -> pd.DataFrame:
    """Per-person birth date and observation-window bounds.

    Returns columns `person_id`, `birth_datetime`, `obs_start`, `obs_end`.
    `obs_start` is the earliest observation_period start; `obs_end` the latest
    end. Used to build age-as-time-scale entry/exit for survival.
    """
    sql = f"""
    SELECT CAST(p.person_id AS STRING) AS person_id,
           p.birth_datetime,
           MIN(op.observation_period_start_date) AS obs_start,
           MAX(op.observation_period_end_date)   AS obs_end
    FROM `{cdr}.person` AS p
    JOIN `{cdr}.observation_period` AS op USING(person_id)
    GROUP BY p.person_id, p.birth_datetime
    """
    df = client.query(sql).to_dataframe()
    df["person_id"] = _canonical_id(df["person_id"])
    df["birth_datetime"] = pd.to_datetime(df["birth_datetime"], utc=True).dt.tz_convert(None)
    df["obs_start"] = pd.to_datetime(df["obs_start"])
    df["obs_end"] = pd.to_datetime(df["obs_end"])
    print(f"  times: n={len(df):,}  e.g. {df['person_id'].head(3).tolist()}")
    return df


STATE_TO_CENSUS_REGION = {
    "CT": "Northeast", "ME": "Northeast", "MA": "Northeast", "NH": "Northeast",
    "RI": "Northeast", "VT": "Northeast", "NJ": "Northeast", "NY": "Northeast",
    "PA": "Northeast",
    "IL": "Midwest", "IN": "Midwest", "MI": "Midwest", "OH": "Midwest",
    "WI": "Midwest", "IA": "Midwest", "KS": "Midwest", "MN": "Midwest",
    "MO": "Midwest", "NE": "Midwest", "ND": "Midwest", "SD": "Midwest",
    "DE": "South", "FL": "South", "GA": "South", "MD": "South",
    "NC": "South", "SC": "South", "VA": "South", "DC": "South",
    "WV": "South", "AL": "South", "KY": "South", "MS": "South",
    "TN": "South", "AR": "South", "LA": "South", "OK": "South",
    "TX": "South",
    "AZ": "West", "CO": "West", "ID": "West", "MT": "West",
    "NV": "West", "NM": "West", "UT": "West", "WY": "West",
    "AK": "West", "CA": "West", "HI": "West", "OR": "West",
    "WA": "West",
    "AS": "Territory", "GU": "Territory", "MP": "Territory", "PR": "Territory",
    "VI": "Territory",
}


def _state_code(s: pd.Series) -> pd.Series:
    return (
        s.fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
        .str.replace(r"^US-", "", regex=True)
    )


def _clean_group_label(s: pd.Series) -> pd.Series:
    return (
        s.fillna("unknown")
        .astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .replace("", "unknown")
    )


def fetch_person_context(client: bigquery.Client, cdr: str) -> pd.DataFrame:
    """Per-person OOD grouping context.

    Geography comes from `person.location_id -> location.state`. Care-site OOD
    uses the dominant visit-level care site rather than `person.care_site_id`,
    because the latter is sparsely populated in many AoU CDRs.
    """
    sql = f"""
    WITH visit_site_counts AS (
      SELECT
        CAST(vo.person_id AS STRING) AS person_id,
        vo.care_site_id,
        ANY_VALUE(cs.care_site_name) AS care_site_name,
        COUNT(*) AS n_visits
      FROM `{cdr}.visit_occurrence` AS vo
      LEFT JOIN `{cdr}.care_site` AS cs
        ON cs.care_site_id = vo.care_site_id
      WHERE vo.care_site_id IS NOT NULL
      GROUP BY vo.person_id, vo.care_site_id
    ),
    dominant_visit_site AS (
      SELECT person_id, care_site_id, care_site_name
      FROM visit_site_counts
      QUALIFY ROW_NUMBER() OVER (
        PARTITION BY person_id
        ORDER BY n_visits DESC, care_site_id
      ) = 1
    )
    SELECT
      CAST(p.person_id AS STRING) AS person_id,
      l.state AS state,
      CAST(COALESCE(dvs.care_site_id, p.care_site_id) AS STRING) AS care_site_id,
      COALESCE(dvs.care_site_name, pcs.care_site_name) AS care_site_name
    FROM `{cdr}.person` AS p
    LEFT JOIN `{cdr}.location` AS l
      ON l.location_id = p.location_id
    LEFT JOIN `{cdr}.care_site` AS pcs
      ON pcs.care_site_id = p.care_site_id
    LEFT JOIN dominant_visit_site AS dvs
      ON dvs.person_id = CAST(p.person_id AS STRING)
    """
    df = client.query(sql).to_dataframe()
    df["person_id"] = _canonical_id(df["person_id"])
    df["state"] = _state_code(df["state"])
    df["census_region"] = df["state"].map(STATE_TO_CENSUS_REGION).fillna("unknown")
    care_site_name = _clean_group_label(df["care_site_name"])
    care_site_id = _clean_group_label(df["care_site_id"])
    df["care_site_group"] = np.where(
        care_site_id.eq("unknown"),
        "unknown",
        care_site_id + ":" + care_site_name,
    )
    print(
        f"  context: n={len(df):,}  regions={df['census_region'].nunique():,}  "
        f"care_sites={df['care_site_group'].nunique():,}"
    )
    return df[["person_id", "state", "census_region", "care_site_group"]]


def load_genetic_ancestry_labels() -> pd.DataFrame:
    """Load AoU inferred genetic ancestry labels.

    Uses `ANCESTRY_PREDS_CACHE` if present; otherwise reads `ANCESTRY_PREDS_URI`
    directly via fsspec/gcsfs with the requester-pays + workspace-project
    storage options — the canonical AoU pattern (see
    `SauersML/ferromic` `phewas/iox.py:load_ancestry_labels`) — and caches the
    result locally for future runs. Any failure propagates and aborts the run.
    AoU's labels are reference-panel-derived categories (`AFR`, `AMR`, `EAS`,
    `EUR`, `MID`, `SAS`, and sometimes `OTH`), not self-reported race/ethnicity.
    """
    if ANCESTRY_PREDS_CACHE.exists() and ANCESTRY_PREDS_CACHE.stat().st_size > 0:
        df = pd.read_csv(
            ANCESTRY_PREDS_CACHE,
            sep="\t",
            usecols=["research_id", "ancestry_pred"],
            dtype=str,
        )
    else:
        print(f"  gcsfs GET {ANCESTRY_PREDS_URI} -> {ANCESTRY_PREDS_CACHE}")
        df = pd.read_csv(
            ANCESTRY_PREDS_URI,
            sep="\t",
            usecols=["research_id", "ancestry_pred"],
            dtype=str,
            storage_options={
                "project": os.environ["GOOGLE_PROJECT"],
                "requester_pays": True,
            },
        )
        ANCESTRY_PREDS_CACHE.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(ANCESTRY_PREDS_CACHE, sep="\t", index=False)
    out = pd.DataFrame({
        "person_id": _canonical_id(df["research_id"]),
        "ancestry_category": _clean_group_label(df["ancestry_pred"]).str.upper(),
    })
    counts = out["ancestry_category"].value_counts().sort_index()
    print(
        "  ancestry: "
        + " ".join(f"{k}={v:,}" for k, v in counts.items())
    )
    return out


# --- model -----------------------------------------------------------------

def fit_marginal_slope(train_df: pd.DataFrame, num_pcs: int):  # -> gamfit.Model
    """Survival marginal-slope GAM with joint Duchon over PCs in both the
    baseline hazard surface and the log-slope (log-HR) channel; sex linear;
    prs_z is the latent score (z_column) so its hazard ratio varies in PC space.

    Age-as-time-scale: `Surv(entry_age, exit_age, event)` is left-truncated at
    each person's age at AoU observation start.
    """
    import gamfit  # lazy: lets the linear baseline import this module without dragging gamfit in
    pcs = ", ".join(f"PC{i+1}" for i in range(num_pcs))
    # `scale_dims=true` enables per-axis anisotropy on the spatial Duchon, so
    # the engine keeps the per-axis length-scale optimization active rather
    # than collapsing to a pure isotropic Duchon. `length_scale=1.0` keeps the
    # resolver in hybrid (non-pure) mode so it doesn't escalate the nullspace
    # order; at d=NUM_PCS, order=1 the polynomial cols stay (d+1), well below
    # DUCHON_CENTERS. `power` omitted -> auto-escalated to satisfy
    # 2*(p+s) > d (basis.rs:resolve_duchon_orders).
    duchon = (
        f"duchon({pcs}, centers={DUCHON_CENTERS}, order=1, "
        f"length_scale=1.0, scale_dims=true)"
    )
    formula = f"Surv(entry_age, exit_age, event) ~ {duchon} + sex"
    cols = ["entry_age", "exit_age", "event", "sex", "prs_z"] + [
        f"PC{i+1}" for i in range(num_pcs)
    ]
    print(f"  fit_spec: family=survival marginal-slope")
    print(f"  fit_spec: formula={formula!r}")
    print(f"  fit_spec: z_column='prs_z'  logslope_formula={duchon!r}")
    print(f"  fit_spec: num_pcs={num_pcs}  duchon_centers={DUCHON_CENTERS}  n_train={len(train_df)}")
    return gamfit.fit(
        train_df[cols],
        formula,
        survival_likelihood="marginal-slope",
        z_column="prs_z",
        logslope_formula=duchon,
    )


def metrics(exit_: np.ndarray, event: np.ndarray, risk: np.ndarray) -> dict[str, float]:
    """Harrell's C on a (exit, event, risk) triple.

    `risk` is a per-row scalar where higher means greater predicted hazard.
    """
    return {
        "n": int(len(event)),
        "n_events": int(event.sum()),
        "median_exit_age": float(np.median(exit_)),
        "concordance": float(concordance_index(exit_, -risk, event)),
    }


def z_norm2(
    train_pgs: np.ndarray,
    train_pcs: np.ndarray,
    test_pgs: np.ndarray,
    test_pcs: np.ndarray,
    pgs_id: str = "PGS",
) -> tuple[np.ndarray, np.ndarray]:
    """pgscatalog-calc Z_norm2 via `pgs_adjust` (mean+var, 2-step, no pop labels).

    Calls the upstream `pgscatalog.calc.lib._ancestry.tools.pgs_adjust` so the
    adjustment is bit-for-bit identical to the pgsc_calc / eMERGE pipeline.
    Regressions are fit on train (`ref_df`) and applied to both train and test
    (`target_df`) with train coefficients only — no test leakage.

    The pgs_adjust API requires `pop` columns even for continuous-ancestry
    methods that ignore them, so we pass a constant dummy label. The Z_norm2
    math itself does not consult them.
    """
    from pgscatalog.calc.lib._ancestry.tools import pgs_adjust  # lazy: heavy import

    n_pcs = train_pcs.shape[1]
    pc_cols = [f"PC{i+1}" for i in range(n_pcs)]

    def _frame(pgs: np.ndarray, pcs: np.ndarray) -> pd.DataFrame:
        df = pd.DataFrame(pcs, columns=pc_cols)
        df["pop"] = "all"
        df[pgs_id] = pgs
        return df

    ref_df = _frame(train_pgs, train_pcs)
    target_df = _frame(
        np.concatenate([train_pgs, test_pgs]),
        np.vstack([train_pcs, test_pcs]),
    )
    kwargs = dict(
        ref_df=ref_df,
        scorecols=[pgs_id],
        ref_pop_col="pop",
        target_pop_col="pop",
        use_method=["mean+var"],
        norm2_2step=True,
        n_pcs=n_pcs,
    )
    adj_train, adj_target, _ = pgs_adjust(target_df=target_df, **kwargs)
    z_col = f"Z_norm2|{pgs_id}"
    return (
        adj_train[z_col].to_numpy(),
        adj_target[z_col].to_numpy()[len(train_pgs):],
    )


def fit_baseline_cox(train_df: pd.DataFrame) -> CoxPHFitter:
    """Cox PH on `(entry_age, exit_age, event) ~ z_norm2`, left-truncated."""
    cph = CoxPHFitter()
    cph.fit(
        train_df[["entry_age", "exit_age", "event", "z_norm2"]],
        duration_col="exit_age",
        entry_col="entry_age",
        event_col="event",
    )
    return cph


def gam_risk(model, df: pd.DataFrame, num_pcs: int, horizon: float) -> np.ndarray:
    """Per-row scalar risk from a gamfit survival model: cumulative hazard at
    `horizon` evaluated on `df`'s covariates. Higher = greater hazard."""
    predict_cols = ["sex", "prs_z"] + [f"PC{i+1}" for i in range(num_pcs)]
    pred = model.predict(df[predict_cols])
    return np.asarray(pred.cumulative_hazard_at([horizon]), dtype=float).reshape(-1)


def prepare_scores(
    train: pd.DataFrame,
    test: pd.DataFrame,
    pc_cols: list[str],
    pgs_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame, float, float]:
    """Train-only PGS standardization for GAM plus train-only Z_norm2 baseline."""
    train = train.copy()
    test = test.copy()
    pgs_mean = float(train["pgs"].mean())
    pgs_std = float(train["pgs"].std(ddof=0))
    if not np.isfinite(pgs_std) or pgs_std <= 0:
        raise ValueError("training PGS has zero or invalid variance")
    train["prs_z"] = (train["pgs"] - pgs_mean) / pgs_std
    test["prs_z"] = (test["pgs"] - pgs_mean) / pgs_std

    z_tr, z_te = z_norm2(
        train["pgs"].to_numpy(),
        train[pc_cols].to_numpy(),
        test["pgs"].to_numpy(),
        test[pc_cols].to_numpy(),
        pgs_id=pgs_id,
    )
    train["z_norm2"] = z_tr
    test["z_norm2"] = z_te
    return train, test, pgs_mean, pgs_std


def evaluate_model_pair(
    train: pd.DataFrame,
    test: pd.DataFrame,
    pc_cols: list[str],
    pgs_id: str,
    label: str,
    save_info: dict[str, object] | None = None,
    score_train: bool = True,
) -> dict[str, float]:
    """Fit GAM + Z_norm2 Cox baseline and print comparable C-index lines."""
    train, test, pgs_mean, pgs_std = prepare_scores(train, test, pc_cols, pgs_id)

    model = fit_marginal_slope(train, len(pc_cols))
    if save_info is not None:
        FITS_DIR.mkdir(parents=True, exist_ok=True)
        disease = str(save_info["disease"])
        fit_path = FITS_DIR / f"{disease}.gamfit"
        meta_path = FITS_DIR / f"{disease}.meta.json"
        try:
            model.save(str(fit_path))
        except Exception as e:
            print(f"  save: model.save failed ({e}); skipping persist")
        else:
            meta = {
                **save_info,
                "num_pcs": len(pc_cols),
                "duchon_centers": DUCHON_CENTERS,
                "train_fraction": TRAIN_FRACTION,
                "rng_seed": RNG_SEED,
                "pgs_mean": pgs_mean,
                "pgs_std": pgs_std,
                "n_train": int(len(train)),
                "survival_likelihood": "marginal-slope",
            }
            meta_path.write_text(json.dumps(meta, indent=2))
            print(f"  save: model -> {fit_path}  meta -> {meta_path}")

    t_horizon = float(test["exit_age"].max())
    gam_risk_test = gam_risk(model, test, len(pc_cols), t_horizon)
    gam_test_m = metrics(
        test["exit_age"].to_numpy(), test["event"].to_numpy(), gam_risk_test,
    )
    gam_train_m = None
    if score_train:
        gam_risk_train = gam_risk(model, train, len(pc_cols), t_horizon)
        gam_train_m = metrics(
            train["exit_age"].to_numpy(), train["event"].to_numpy(), gam_risk_train,
        )

    print(
        f"  baseline_spec: Cox PH on Z_norm2 (PC-based mean+var adjustment, "
        f"no pop labels)"
    )
    cph = fit_baseline_cox(train)
    coef = float(cph.params_["z_norm2"])
    print(f"  baseline_coef: log_HR={coef:+.4f}  HR/SD={np.exp(coef):.4f}")
    base_risk_test = cph.predict_partial_hazard(test[["z_norm2"]]).to_numpy()
    base_test_m = metrics(
        test["exit_age"].to_numpy(), test["event"].to_numpy(), base_risk_test,
    )
    base_train_m = None
    if score_train:
        base_risk_train = cph.predict_partial_hazard(train[["z_norm2"]]).to_numpy()
        base_train_m = metrics(
            train["exit_age"].to_numpy(), train["event"].to_numpy(), base_risk_train,
        )

    print(
        f"  {label}  train_n={len(train):,}  test_n={gam_test_m['n']:,}  "
        f"test_events={gam_test_m['n_events']:,}  "
        f"median_exit_age={gam_test_m['median_exit_age']:.2f}  horizon={t_horizon:.2f}"
    )
    if score_train:
        assert gam_train_m is not None and base_train_m is not None
        print(
            f"  GAM       train_C={gam_train_m['concordance']:.4f}  "
            f"test_C={gam_test_m['concordance']:.4f}"
        )
        print(
            f"  baseline  train_C={base_train_m['concordance']:.4f}  "
            f"test_C={base_test_m['concordance']:.4f}"
        )
    else:
        print(f"  GAM       test_C={gam_test_m['concordance']:.4f}")
        print(f"  baseline  test_C={base_test_m['concordance']:.4f}")
    delta = gam_test_m["concordance"] - base_test_m["concordance"]
    print(f"  delta     test_C(GAM - baseline)={delta:+.4f}")
    return {
        "gam_train_c": (
            float("nan") if gam_train_m is None else gam_train_m["concordance"]
        ),
        "gam_test_c": gam_test_m["concordance"],
        "baseline_train_c": (
            float("nan") if base_train_m is None else base_train_m["concordance"]
        ),
        "baseline_test_c": base_test_m["concordance"],
        "delta_test_c": delta,
    }


def loso_groups(
    df: pd.DataFrame,
    col: str,
    min_train_events: int,
    min_train_censors: int,
    min_test_events: int,
    min_test_censors: int,
    min_test_n: int,
    max_groups: int | None = None,
) -> list[str]:
    """Eligible held-out groups, sorted by size and event-count constraints."""
    known = df[col].notna() & df[col].astype(str).str.lower().ne("unknown")
    total_events = int(df["event"].sum())
    total_censors = int(len(df) - total_events)
    summary = (
        df[known]
        .groupby(col, sort=False)
        .agg(n=("event", "size"), events=("event", "sum"))
        .reset_index()
    )
    summary["censors"] = summary["n"] - summary["events"]
    summary = summary[
        (summary["n"] >= min_test_n)
        & (summary["events"] >= min_test_events)
        & (summary["censors"] >= min_test_censors)
        & ((total_events - summary["events"]) >= min_train_events)
        & ((total_censors - summary["censors"]) >= min_train_censors)
    ]
    summary = summary.sort_values(["n", "events"], ascending=False, kind="stable")
    if max_groups is not None:
        summary = summary.head(max_groups)
    return [str(x) for x in summary[col].tolist()]


def run_loso_axis(
    df_full: pd.DataFrame,
    axis_name: str,
    group_col: str,
    pc_cols: list[str],
    pgs_id: str,
    min_train_events: int,
    min_train_censors: int,
    min_test_events: int,
    min_test_censors: int,
    min_test_n: int,
    max_groups: int | None = None,
    score_train: bool = False,
) -> None:
    """Leave one group out: refit both models on all other groups."""
    groups = loso_groups(
        df_full,
        group_col,
        min_train_events=min_train_events,
        min_train_censors=min_train_censors,
        min_test_events=min_test_events,
        min_test_censors=min_test_censors,
        min_test_n=min_test_n,
        max_groups=max_groups,
    )
    print(
        f"  LOSO axis={axis_name} col={group_col} groups={len(groups)} "
        f"min_test_n={min_test_n} min_test_events={min_test_events} "
        f"min_test_censors={min_test_censors}"
    )
    if not groups:
        print(f"  LOSO axis={axis_name} skipped=no eligible groups")
        return

    deltas: list[float] = []
    for group in groups:
        holdout_mask = df_full[group_col].eq(group)
        train = df_full.loc[~holdout_mask].reset_index(drop=True)
        test = df_full.loc[holdout_mask].reset_index(drop=True)
        group_short = group[:96]
        print(
            f"  LOSO fold axis={axis_name} holdout={group_short!r}  "
            f"train_n={len(train):,} test_n={len(test):,} "
            f"test_events={int(test['event'].sum()):,}"
        )
        result = evaluate_model_pair(
            train,
            test,
            pc_cols,
            pgs_id,
            label=f"LOSO[{axis_name}] holdout={group_short!r}",
            score_train=score_train,
        )
        deltas.append(result["delta_test_c"])

    print(
        f"  LOSO summary axis={axis_name} folds={len(deltas)}  "
        f"mean_delta={float(np.mean(deltas)):+.4f}  "
        f"worst_delta={float(np.min(deltas)):+.4f}  "
        f"best_delta={float(np.max(deltas)):+.4f}"
    )


# --- main ------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) != 1:
        raise SystemExit("marginal_slope_diseases.py takes no arguments")
    import gamfit
    print(f"gamfit version: {gamfit.__version__}")
    print(f"gamfit build_info: {gamfit.build_info()}")
    diseases = {k: v for k, v in DISEASES.items() if PGS_ID_PATTERN.match(v["pgs"])}
    print(f"diseases with real PGS IDs: {list(diseases)}")
    active_axes = list(LOSO_AXES)
    print(f"loso_axes: {active_axes}")

    ensure_scored([cfg["pgs"] for cfg in diseases.values()])

    print("loading PCs and sex ...")
    pcs = load_pcs(NUM_PCS)
    sex = load_sex()
    base = pcs.merge(sex, on="person_id")
    print(f"base: n={len(base):,}")

    cdr = os.environ["WORKSPACE_CDR"]
    client = bigquery.Client()

    print("loading person times (birth + observation period) ...")
    times = fetch_person_times(client, cdr)
    base = base.merge(times, on="person_id")
    print(f"base (with times): n={len(base):,}")

    print("loading geography and care-site context ...")
    context = fetch_person_context(client, cdr)
    base = base.merge(context, on="person_id", how="left")
    base["census_region"] = _clean_group_label(base["census_region"])
    base["care_site_group"] = _clean_group_label(base["care_site_group"])
    print(f"base (with context): n={len(base):,}")

    print("loading AoU inferred genetic ancestry labels ...")
    try:
        ancestry = load_genetic_ancestry_labels()
    except Exception as exc:
        # Bare-metal VMs sit outside the AoU VPC-SC perimeter, so the controlled
        # bucket is unreachable. If a prior workbench-side `gsutil cp` hasn't
        # staged ANCESTRY_PREDS_CACHE here, drop the `ancestry` LOSO axis and
        # finish the rest of the run (random split + care_site + census_region).
        first_line = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
        print(f"  WARNING: ancestry labels unavailable -> dropping 'ancestry' LOSO axis. detail: {first_line}")
        active_axes = [a for a in active_axes if a != "ancestry"]
    else:
        base = base.merge(ancestry, on="person_id", how="left")
        base["ancestry_category"] = _clean_group_label(base["ancestry_category"]).str.upper()
        print(f"base (with ancestry): n={len(base):,}")

    rng = np.random.default_rng(RNG_SEED)
    pc_cols = [f"PC{i+1}" for i in range(NUM_PCS)]

    for name, cfg in diseases.items():
        print(f"\n=== {name.upper()} ===")
        pgs_df = load_one_pgs(cfg["pgs"])
        df_full = base.merge(pgs_df, on="person_id")
        ancestor = lookup_snomed_concept(client, cdr, cfg["snomed_name"])
        case_dates = fetch_cases(client, cdr, ancestor)
        df_full = df_full.merge(case_dates, on="person_id", how="left")
        df_full["event"] = df_full["event_date"].notna().astype(int)

        # Age-as-time-scale (years). Entry: age at AoU obs-period start.
        # Exit: age at first qualifying condition for events; age at obs-period
        # end for censors. Left-truncation respected via `Surv(entry, exit, event)`.
        days_per_year = 365.25
        df_full["entry_age"] = (
            (df_full["obs_start"] - df_full["birth_datetime"]).dt.days / days_per_year
        )
        exit_date = df_full["event_date"].fillna(df_full["obs_end"])
        df_full["exit_age"] = (
            (exit_date - df_full["birth_datetime"]).dt.days / days_per_year
        )

        # Drop rows with non-positive or invalid intervals: missing birth/obs,
        # prevalent cases whose event predates obs_start (entry >= exit), etc.
        before = len(df_full)
        df_full = df_full.dropna(subset=["pgs", "entry_age", "exit_age"]).copy()
        df_full = df_full[df_full["exit_age"] > df_full["entry_age"]].copy()
        df_full = df_full[df_full["entry_age"] >= 0].copy()
        dropped = before - len(df_full)
        n_event = int(df_full["event"].sum())
        n_censor = len(df_full) - n_event
        K = n_event / max(1, len(df_full))
        print(
            f"  snomed={cfg['snomed_name']!r}  concept_id={ancestor}  "
            f"events={n_event:,}  censors={n_censor:,}  K(crude)={K:.6f}  "
            f"dropped_bad_intervals={dropped:,}"
        )

        # Per-class 80/20 split (events vs. censored), like the prior design but
        # without balancing — survival uses censored controls natively. We keep
        # all events and all censors, just split each group 80/20.
        event_idx = rng.permutation(df_full.index[df_full["event"] == 1].to_numpy())
        censor_idx = rng.permutation(df_full.index[df_full["event"] == 0].to_numpy())
        n_train_event = int(round(len(event_idx) * TRAIN_FRACTION))
        n_train_censor = int(round(len(censor_idx) * TRAIN_FRACTION))
        train_pick = np.concatenate([
            event_idx[:n_train_event], censor_idx[:n_train_censor],
        ])
        test_pick = np.concatenate([
            event_idx[n_train_event:], censor_idx[n_train_censor:],
        ])
        print(
            f"  split: n={len(df_full):,}  "
            f"train_events={n_train_event:,} train_censors={n_train_censor:,}  "
            f"test_events={len(event_idx) - n_train_event:,} "
            f"test_censors={len(censor_idx) - n_train_censor:,}"
        )
        train = df_full.loc[train_pick].reset_index(drop=True)
        test = df_full.loc[test_pick].reset_index(drop=True)

        evaluate_model_pair(
            train,
            test,
            pc_cols,
            cfg["pgs"],
            label=f"PGS={cfg['pgs']} random_split K(crude)={K:.6f}",
            save_info={
                "disease": name,
                "pgs": cfg["pgs"],
                "snomed_name": cfg["snomed_name"],
                "concept_id": ancestor,
                "K_crude": K,
            },
        )

        print("  OOD: leave-one-group-out refits")
        for axis in active_axes:
            run_loso_axis(
                df_full,
                axis_name=axis,
                group_col=LOSO_AXIS_TO_COLUMN[axis],
                pc_cols=pc_cols,
                pgs_id=cfg["pgs"],
                min_train_events=MIN_LOSO_TRAIN_EVENTS,
                min_train_censors=MIN_LOSO_TRAIN_CENSORS,
                min_test_events=MIN_LOSO_TEST_EVENTS,
                min_test_censors=MIN_LOSO_TEST_CENSORS,
                min_test_n=MIN_LOSO_TEST_N,
                max_groups=MAX_LOSO_CARE_SITES if axis == "care_site" else None,
                score_train=False,
            )


if __name__ == "__main__":
    main()
