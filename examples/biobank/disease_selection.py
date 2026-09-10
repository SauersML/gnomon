"""Shared AoU disease selection: OHDSI reference roots, CDR ranking, PGS map."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from google.cloud import bigquery

SNOMED_DISEASE_CODE = "64572001"

SNOMED_PGS_MAP: dict[str, dict[str, str]] = {
    # Hypertension -- Privé et al. 2022 sparse hypertension PRS. Not AoU-trained.
    "38341003": {"slug": "hypertension", "pgs": "PGS001320"},
    # Hyperlipidemia -- Jung 2024 RFDiseasemetaPRS meta.E78 (lipoprotein disorders); UKB.
    "55822004": {"slug": "hyperlipidemia", "pgs": "PGS004518"},
    # Osteoarthritis -- INTERVENE MegaPRS knee OA (Jermy/INTERVENE); 1000G dev.
    "396275006": {"slug": "osteoarthritis", "pgs": "PGS004883"},
    # Heart disease (read as CAD) -- Patel 2023 GPS_Mult multi-ancestry; UKB tranche.
    "56265001": {"slug": "heart_disease", "pgs": "PGS003725"},
    # Sleep disorder -- Jung 2024 RFDiseasemetaPRS meta.G47; UKB.
    "39898005": {"slug": "sleep_disorder", "pgs": "PGS004522"},
    # Depressive disorder -- INTERVENE MegaPRS MDD; 1000G dev.
    "35489007": {"slug": "depressive_disorder", "pgs": "PGS004885"},
    # Obesity -- Kim et al. 2026 O_MetPRS_EUR; AoU external-test only.
    "414916001": {"slug": "obesity", "pgs": "PGS005331"},
    # Gastroesophageal reflux disease -- Jung 2024 RFDiseasemetaPRS meta.K21; UKB.
    "235595009": {"slug": "gerd", "pgs": "PGS004538"},
    # Anemia (B12-deficiency anemia) -- Tanigawa snpnet GBE_HC608; UKB.
    "271737000": {"slug": "anemia", "pgs": "PGS001305"},
    # Major depressive disorder -- INTERVENE MegaPRS MDD (same score as depressive_disorder).
    "370143000": {"slug": "major_depressive_disorder", "pgs": "PGS004885"},
    # Drug dependence -- Hatoum PRSaddiction-rf multivariate addiction GWAS. Not AoU-trained.
    "191816009": {"slug": "drug_dependence", "pgs": "PGS003849"},
    # Type 2 diabetes -- D-PRISM D_T2DPRS EUR (PRS-CSx); AoU external-eval only.
    "44054006": {"slug": "type_2_diabetes", "pgs": "PGS005371"},
    # Cardiac arrhythmia (atrial fibrillation) -- Yuan AF PRS-CSx cross-population.
    "698247007": {"slug": "cardiac_arrhythmia", "pgs": "PGS005313"},
    # Allergic rhinitis -- Tanigawa snpnet GBE_HC1021; UKB.
    "61582004": {"slug": "allergic_rhinitis", "pgs": "PGS001109"},
    # Sleep apnea -- PRS_BMIadjOSA (PRS-CSs); FinnGen/MGBB/MVP, AoU only validation
    # (NOT fully paper-verified -- final text was paywalled; flagged for re-check).
    "73430006": {"slug": "sleep_apnea", "pgs": "PGS005219"},
    # Malignant neoplastic disease -- INTERVENE MegaPRS AllCancers; 1000G dev.
    "363346000": {"slug": "malignant_neoplasm", "pgs": "PGS004875"},
    # Asthma -- INTERVENE MegaPRS Asthma; 1000G dev.
    "195967001": {"slug": "asthma", "pgs": "PGS004877"},
    # COPD -- Jung et al. metaPRS for J44. Not AoU-trained.
    "13645005": {"slug": "copd", "pgs": "PGS004536"},
    # No eligible PGS Catalog score (stay SKIP, surfaced in the top-N report):
    # urinary tract infection, sinusitis, pharyngitis.
}

def _load_cohort_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def extract_ohdsi_canonical_disease_concepts(client: bigquery.Client, cdr: str, pl_root: Path) -> set[int]:
    """Return the OMOP concept_id set of canonical OHDSI disease phenotype roots."""
    from google.cloud import bigquery
    json_dir = pl_root / "inst" / "cohorts"
    cohorts_csv = pl_root / "inst" / "Cohorts.csv"

    meta = pd.read_csv(cohorts_csv)
    meta["cohortId"] = meta["cohortId"].astype(int)
    meta["isReferenceCohort"] = meta["isReferenceCohort"].fillna(0).astype(int)
    ref_ids = sorted(meta.loc[meta["isReferenceCohort"] == 1, "cohortId"].tolist())
    print(f"  ohdsi: cohorts={len(meta)} reference={len(ref_ids)}")

    roots: set[int] = set()
    skipped = {"non_condition_primary": 0, "multi_include_or_exclude": 0, "parse": 0}
    for cid in ref_ids:
        path = json_dir / f"{cid}.json"
        try:
            j = _load_cohort_json(path)
        except Exception:
            skipped["parse"] += 1
            continue
        codeset_id = None
        for crit in j.get("PrimaryCriteria", {}).get("CriteriaList", []):
            if "ConditionOccurrence" in crit:
                codeset_id = crit["ConditionOccurrence"].get("CodesetId")
                break
        if codeset_id is None:
            skipped["non_condition_primary"] += 1
            continue
        cs = next((c for c in j.get("ConceptSets", []) if c.get("id") == codeset_id), None)
        if cs is None:
            skipped["parse"] += 1
            continue
        items = cs.get("expression", {}).get("items", [])
        incs = [it for it in items if not it.get("isExcluded")]
        excs = [it for it in items if it.get("isExcluded")]
        if len(incs) != 1 or len(excs) != 0:
            skipped["multi_include_or_exclude"] += 1
            continue
        roots.add(int(incs[0]["concept"]["CONCEPT_ID"]))
    print(f"  ohdsi: single-root condition cohorts={len(roots)} skipped={skipped}")

    # Filter (d): include-root must descend from SNOMED 'Disease (disorder)'.
    disease_root = lookup_snomed_code(client, cdr, SNOMED_DISEASE_CODE)
    roots_arr = sorted(roots)
    df = client.query(
        f"""
        WITH roots AS (SELECT concept_id FROM UNNEST(@roots) AS concept_id),
             dis   AS (SELECT descendant_concept_id AS concept_id
                       FROM `{cdr}.concept_ancestor`
                       WHERE ancestor_concept_id = @disease_root)
        SELECT r.concept_id, dis.concept_id IS NOT NULL AS is_disease
        FROM roots r LEFT JOIN dis USING (concept_id)
        """,
        job_config=bigquery.QueryJobConfig(query_parameters=[
            bigquery.ArrayQueryParameter("roots", "INT64", roots_arr),
            bigquery.ScalarQueryParameter("disease_root", "INT64", disease_root),
        ]),
    ).to_dataframe()
    disease_roots = set(df.loc[df["is_disease"], "concept_id"].astype(int).tolist())
    print(
        f"  ohdsi: canonical disease phenotypes={len(disease_roots)} "
        f"(dropped {len(roots) - len(disease_roots)} symptom-rooted)"
    )
    return disease_roots


def lookup_snomed_code(client: bigquery.Client, cdr: str, code: str) -> int:
    """Return the OMOP concept_id for a SNOMED standard concept by its concept_code."""
    from google.cloud import bigquery
    sql = f"""
    SELECT concept_id
    FROM `{cdr}.concept`
    WHERE vocabulary_id = 'SNOMED' AND standard_concept = 'S'
      AND concept_code = @code
    ORDER BY concept_id
    LIMIT 1
    """
    rows = list(client.query(
        sql,
        job_config=bigquery.QueryJobConfig(query_parameters=[
            bigquery.ScalarQueryParameter("code", "STRING", code),
        ]),
    ).result())
    if not rows:
        raise ValueError(f"no standard SNOMED concept with code {code!r}")
    return int(rows[0]["concept_id"])


def resolve_snomed_codes(client: bigquery.Client, cdr: str, codes: list[str]) -> pd.DataFrame:
    """Resolve SNOMED concept_codes to (concept_id, concept_name)."""
    from google.cloud import bigquery
    df = client.query(
        f"""
        SELECT concept_code, concept_id, concept_name
        FROM `{cdr}.concept`
        WHERE vocabulary_id = 'SNOMED' AND standard_concept = 'S'
          AND concept_code IN UNNEST(@codes)
        """,
        job_config=bigquery.QueryJobConfig(query_parameters=[
            bigquery.ArrayQueryParameter("codes", "STRING", codes),
        ]),
    ).to_dataframe()
    df["concept_id"] = df["concept_id"].astype(int)
    return df


def _concept_names(client: bigquery.Client, cdr: str, concept_ids: list[int]) -> dict[int, str]:
    """Map OMOP concept_id -> concept_name for display (skipped-disease report)."""
    from google.cloud import bigquery
    ids = [int(c) for c in concept_ids]
    if not ids:
        return {}
    df = client.query(
        f"SELECT concept_id, concept_name FROM `{cdr}.concept` "
        f"WHERE concept_id IN UNNEST(@ids)",
        job_config=bigquery.QueryJobConfig(query_parameters=[
            bigquery.ArrayQueryParameter("ids", "INT64", ids),
        ]),
    ).to_dataframe()
    return dict(zip(df["concept_id"].astype(int), df["concept_name"].astype(str)))


def rank_disease_concepts_by_prevalence(
    client: bigquery.Client, cdr: str, concept_ids: list[int]
) -> pd.DataFrame:
    """Distinct case count per ancestor concept_id, descending."""
    from google.cloud import bigquery
    if not concept_ids:
        return pd.DataFrame(columns=["concept_id", "case_count"])
    df = client.query(
        f"""
        SELECT ca.ancestor_concept_id AS concept_id,
               COUNT(DISTINCT co.person_id) AS case_count
        FROM `{cdr}.condition_occurrence` AS co
        JOIN `{cdr}.concept_ancestor` AS ca
          ON ca.descendant_concept_id = co.condition_concept_id
        WHERE ca.ancestor_concept_id IN UNNEST(@ancestors)
        GROUP BY ca.ancestor_concept_id
        """,
        job_config=bigquery.QueryJobConfig(query_parameters=[
            bigquery.ArrayQueryParameter("ancestors", "INT64", concept_ids),
        ]),
    ).to_dataframe()
    df["concept_id"] = df["concept_id"].astype(int)
    df["case_count"] = df["case_count"].astype(int)
    return df.sort_values(["case_count", "concept_id"], ascending=[False, True]).reset_index(drop=True)


def select_runtime_diseases(client: bigquery.Client, cdr: str, top_n: int, pl_root: Path) -> dict[str, dict]:
    """Build the per-run diseases dict by intersecting OHDSI canonical disease
    phenotypes with SNOMED_PGS_MAP and taking the top top_n by case
    prevalence in the active CDR.
    """
    print("\n=== DISEASE SELECTION ===")
    print(f"  snomed_pgs_map_size={len(SNOMED_PGS_MAP)}  top_n={top_n}")
    mapped_codes = sorted(SNOMED_PGS_MAP.keys())
    resolved = resolve_snomed_codes(client, cdr, mapped_codes)
    resolved_codes = set(resolved["concept_code"].astype(str))
    missing = [c for c in mapped_codes if c not in resolved_codes]
    if missing:
        # Non-fatal: one bad/non-standard SNOMED code must not abort the whole
        # run. Skip it (and its disease) and surface it; everything else runs.
        print(f"  WARNING: {len(missing)} SNOMED code(s) not resolvable in "
              f"{cdr}.concept -- skipping: "
              + ", ".join(f"{c} ({SNOMED_PGS_MAP[c]['slug']})" for c in missing))
    code_to_id = dict(zip(resolved["concept_code"].astype(str), resolved["concept_id"]))
    code_to_name = dict(zip(resolved["concept_code"].astype(str), resolved["concept_name"]))
    print(f"  resolved {len(code_to_id)} SNOMED code(s) to OMOP concept_ids")

    canonical = extract_ohdsi_canonical_disease_concepts(client, cdr, pl_root)

    mapped_ids = {int(code_to_id[c]): c for c in mapped_codes if c in code_to_id}
    survivors = {cid: code for cid, code in mapped_ids.items() if cid in canonical}
    dropped = [
        (code, code_to_name[code]) for cid, code in mapped_ids.items() if cid not in canonical
    ]
    print(f"  mapped ∩ OHDSI-canonical: {len(survivors)}/{len(mapped_ids)}")
    for code, name in dropped:
        print(f"    dropped (not OHDSI-canonical disease): {code} {name!r}")

    # Rank ALL OHDSI-canonical diseases by prevalence in one pass, so "top
    # top_n" spans the full disease universe rather than just the
    # PGS-mapped subset. This lets us surface, up front, which of the most
    # prevalent diseases we are NOT scoring because no PGS is mapped.
    ranked = rank_disease_concepts_by_prevalence(client, cdr, sorted(canonical))
    counts = dict(zip(ranked["concept_id"], ranked["case_count"]))

    top_ids = [int(c) for c in ranked["concept_id"].tolist()[:top_n]]
    top_names = _concept_names(client, cdr, top_ids)
    n_run = sum(1 for cid in top_ids if cid in mapped_ids)
    print(
        f"  top-{top_n} most prevalent OHDSI-canonical diseases "
        f"(running {n_run}, skipping {len(top_ids) - n_run} with no PGS):"
    )
    for cid in top_ids:
        nm = top_names.get(cid, f"concept_id={cid}")
        cases = counts.get(cid, 0)
        if cid in mapped_ids:
            cfg = SNOMED_PGS_MAP[mapped_ids[cid]]
            print(f"    RUN   {nm:<42.42} cases={cases:>9,}  pgs={cfg['pgs']}")
        else:
            print(f"    SKIP  {nm:<42.42} cases={cases:>9,}  no PGS available -- not scored")

    # Run rule: a disease must be among the top-N most prevalent overall AND
    # have a PGS (survivor = mapped ∩ OHDSI-canonical). Nothing outside the
    # top-N runs -- the RUN lines above are exactly the chosen set.
    chosen_concept_ids: list[int] = [cid for cid in top_ids if cid in survivors]

    diseases: dict[str, dict] = {}
    print(f"  selected top-{len(chosen_concept_ids)} by prevalence:")
    for cid in chosen_concept_ids:
        code = survivors[cid]
        cfg = SNOMED_PGS_MAP[code]
        slug = cfg["slug"]
        diseases[slug] = {
            "snomed_code": code,
            "concept_id": int(cid),
            "snomed_name": code_to_name[code],
            "pgs": cfg["pgs"],
            "case_count": int(counts.get(cid, 0)),
        }
        print(
            f"    {slug:<24}  concept_id={cid:>8}  cases={counts.get(cid, 0):>9,}  "
            f"pgs={cfg['pgs']}  ({code_to_name[code]})"
        )
    print("=== /DISEASE SELECTION ===\n")
    return diseases


