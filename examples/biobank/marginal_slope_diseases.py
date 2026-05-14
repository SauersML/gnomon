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

import numpy as np
import pandas as pd
from google.cloud import bigquery
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.linear_model import LinearRegression

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
NUM_PCS = 10
DUCHON_CENTERS = 30  # > linear nullspace (d+1=11) in d=10
TRAIN_FRACTION = 0.80  # per-class 80/20 split
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
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """pgscatalog two-step PC-based normalization (eMERGE / pgsc_calc spec).

    Step 1 (mean):     mu_hat = OLS(PGS ~ PCs);  resid = PGS - mu_hat
    Step 2 (variance): var_hat = OLS(resid^2 ~ PCs), clipped at `eps`;
                       z      = resid / sqrt(var_hat)

    Regressions fit on train and applied to test using train coefficients
    only. No population labels used — this is the continuous-ancestry form.
    """
    mean_reg = LinearRegression().fit(train_pcs, train_pgs)
    train_resid = train_pgs - mean_reg.predict(train_pcs)
    test_resid = test_pgs - mean_reg.predict(test_pcs)
    var_reg = LinearRegression().fit(train_pcs, train_resid ** 2)
    train_var = np.maximum(var_reg.predict(train_pcs), eps)
    test_var = np.maximum(var_reg.predict(test_pcs), eps)
    return train_resid / np.sqrt(train_var), test_resid / np.sqrt(test_var)


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


# --- main ------------------------------------------------------------------

def main() -> None:
    import gamfit
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

    print("loading person times (birth + observation period) ...")
    times = fetch_person_times(client, cdr)
    base = base.merge(times, on="person_id")
    print(f"base (with times): n={len(base):,}")

    rng = np.random.default_rng(RNG_SEED)

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
        df_full = df_full.dropna(subset=["entry_age", "exit_age"]).copy()
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

        # Standardize PGS on training only, then apply the same shift/scale to test
        # so we never mix test rows into the training statistics. `prs_z` is the
        # GAM's latent z (a plain z-score on training). The baseline uses Z_norm2
        # (PC-based ancestry-adjusted), built separately below.
        pgs_mean = float(train["pgs"].mean())
        pgs_std = float(train["pgs"].std(ddof=0))
        train["prs_z"] = (train["pgs"] - pgs_mean) / pgs_std
        test["prs_z"] = (test["pgs"] - pgs_mean) / pgs_std

        # Z_norm2 baseline score: two-step PC regression (mean + variance),
        # train-only fit, applied to both train and test.
        pc_cols = [f"PC{i+1}" for i in range(NUM_PCS)]
        z_tr, z_te = z_norm2(
            train["pgs"].to_numpy(), train[pc_cols].to_numpy(),
            test["pgs"].to_numpy(), test[pc_cols].to_numpy(),
        )
        train["z_norm2"] = z_tr
        test["z_norm2"] = z_te

        model = fit_marginal_slope(train, NUM_PCS)
        FITS_DIR.mkdir(parents=True, exist_ok=True)
        fit_path = FITS_DIR / f"{name}.gamfit"
        meta_path = FITS_DIR / f"{name}.meta.json"
        try:
            model.save(str(fit_path))
        except Exception as e:
            print(f"  save: model.save failed ({e}); skipping persist")
        else:
            meta_path.write_text(json.dumps({
                "disease": name,
                "pgs": cfg["pgs"],
                "snomed_name": cfg["snomed_name"],
                "concept_id": ancestor,
                "num_pcs": NUM_PCS,
                "duchon_centers": DUCHON_CENTERS,
                "train_fraction": TRAIN_FRACTION,
                "rng_seed": RNG_SEED,
                "pgs_mean": pgs_mean,
                "pgs_std": pgs_std,
                "K_crude": K,
                "n_train": int(len(train)),
                "survival_likelihood": "marginal-slope",
            }, indent=2))
            print(f"  save: model -> {fit_path}  meta -> {meta_path}")

        # Survival predict returns a SurvivalPrediction; use cumulative hazard
        # at a fixed horizon as the scalar risk score for concordance. The
        # horizon is the max test exit age, computed once and reused for train
        # and test so the two C-indices use a common hazard scale.
        t_horizon = float(test["exit_age"].max())
        gam_risk_train = gam_risk(model, train, NUM_PCS, t_horizon)
        gam_risk_test = gam_risk(model, test, NUM_PCS, t_horizon)
        gam_train_m = metrics(
            train["exit_age"].to_numpy(), train["event"].to_numpy(), gam_risk_train,
        )
        gam_test_m = metrics(
            test["exit_age"].to_numpy(), test["event"].to_numpy(), gam_risk_test,
        )

        # Z_norm2 + Cox PH baseline on the SAME train/test split.
        print(
            f"  baseline_spec: Cox PH on Z_norm2 (PC-based mean+var adjustment, "
            f"no pop labels)"
        )
        cph = fit_baseline_cox(train)
        coef = float(cph.params_["z_norm2"])
        print(f"  baseline_coef: log_HR={coef:+.4f}  HR/SD={np.exp(coef):.4f}")
        base_risk_train = cph.predict_partial_hazard(train[["z_norm2"]]).to_numpy()
        base_risk_test = cph.predict_partial_hazard(test[["z_norm2"]]).to_numpy()
        base_train_m = metrics(
            train["exit_age"].to_numpy(), train["event"].to_numpy(), base_risk_train,
        )
        base_test_m = metrics(
            test["exit_age"].to_numpy(), test["event"].to_numpy(), base_risk_test,
        )

        print(
            f"  PGS={cfg['pgs']}  train_n={len(train):,}  test_n={gam_test_m['n']:,}  "
            f"test_events={gam_test_m['n_events']:,}  K(crude)={K:.6f}  "
            f"median_exit_age={gam_test_m['median_exit_age']:.2f}  horizon={t_horizon:.2f}"
        )
        print(
            f"  GAM       train_C={gam_train_m['concordance']:.4f}  "
            f"test_C={gam_test_m['concordance']:.4f}"
        )
        print(
            f"  baseline  train_C={base_train_m['concordance']:.4f}  "
            f"test_C={base_test_m['concordance']:.4f}"
        )
        print(
            f"  delta     test_C(GAM - baseline)={gam_test_m['concordance'] - base_test_m['concordance']:+.4f}"
        )


if __name__ == "__main__":
    main()
