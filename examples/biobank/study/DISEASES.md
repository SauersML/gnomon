# Study diseases (locked 2026-09-18)

This file is the rationale for the locked disease list. The machine-readable list is `diseases.json` in this
directory. Changes after the lock need the lead's approval (SPEC section 2).

## Rules applied
- **Phenotype.** A qualifying record is a `condition_occurrence` whose concept descends from the root in
  `concept_ancestor`, the root included.
  - A disease can list `excluded_branches`. Records whose concept descends from one (the branch included) are
    not qualifying records. Only CKD uses this.
  - A confirmed case has at least 2 distinct qualifying dates. The event date is the second one.
  - People with a single record are kept.
  - An exclusion is met when a person has at least 2 distinct dates for the exclusion root. When it counts depends
    on the model (SPEC section 2, audit C3):
    - survival: only if met with records dated on or before the landmark. If it is met later, follow-up is
      censored at the date it is met;
    - binary: if met by the CDR cutoff.
- **One PGS Catalog score per disease.** Its trait matches the root.
- **No AoU in development.** A score is ineligible if All of Us participants contributed to its discovery GWAS, to a
  component score, or to its tuning or weight fitting. This covers PRSmix scores such as PGS004787.
  - AoU used only as a later evaluation sample is allowed.
  - Each score was checked twice: in its Catalog record (`samples_variants` and `samples_training` cohorts), and in
    the development paper's full text, by searching for "All of Us".
- **Prefer multi-ancestry scores, then cached ones.** Cached means listed in
  `.aou-workflow/sscore-cache-listing.txt`.
- **No results used.** Choices rest on Data Browser counts and provenance only. No AUC or other result from any earlier
  AoU run was used (audit S8).

## Locked list

"Browser" is the number of participants with at least 1 record of the root or a descendant (AoU Public Data Browser,
CDR 2025q4r5, 747,040 participants, counts rounded to 20). For the two cancers it counts the restricted sex only.

"Array ≤" is browser × 0.5987, where 0.5987 = 447,278 v8 array participants / 747,040. It is an upper bound on
confirmed cases: the 2-date rule, the 365-day lookback and the prune all reduce it. study-cohort measures the real
counts.

| disease | SNOMED root | PGS | trait (score) | multi-ancestry | cached | variants / build | sex | exclusion | browser | array ≤ |
|---|---|---|---|---|---|---|---|---|---|---|
| hypertension | 59621000 Essential hypertension | PGS004236 | Hypertension (PRSsum) | yes | yes | 398,805 / GRCh38 | | | 203,060 | 121,572 |
| type_2_diabetes | 44054006 Type 2 diabetes mellitus | PGS002308 | T2D (PRS-CSx) | yes | yes | 1,259,754 / GRCh37 | | T1D 46635009 | 94,080 | 56,326 |
| atrial_fibrillation | 49436004 Atrial fibrillation | PGS005168 | AF (PRS-CS) | yes | yes | 1,113,668 / GRCh38 | | | 34,920 | 20,907 |
| coronary_artery_disease | 53741008 Coronary arteriosclerosis | PGS003725 | CAD (GPSMult) | yes | **no** | 1,296,172 / GRCh37 | | | 62,020 | 37,131 |
| breast_cancer | 254837009 Malignant tumor of breast | PGS000004 | Breast cancer (PRS313) | no | yes | 313 / GRCh37 | female | | 15,420 | 9,232 |
| prostate_cancer | 399068003 Malignant tumor of prostate | PGS003766 | Prostate cancer (451 SNP) | yes | yes | 451 / GRCh38 | male | | 11,680 | 6,993 |
| asthma | 195967001 Asthma | PGS001782 | Asthma (GBMI PRS-CS-auto) | yes | **no** | 884,043 / GRCh37 | | | 84,840 | 50,794 |
| copd | 13645005 COPD | PGS001783 | COPD (GBMI PRS-CS-auto) | yes | yes | 884,139 / GRCh37 | | | 40,820 | 24,439 |
| major_depressive_disorder | 370143000 Major depressive disorder | PGS004885 | MDD (MegaPRS auto) | no | yes | 801,544 / GRCh37 | | bipolar 13746004 | 111,380 | 66,683 |
| chronic_kidney_disease | 709044004 Chronic kidney disease | PGS002237 | CKD stage ≥3 (GPS, eGFR-based) | partly | yes | 471,316 / GRCh37 | | stage 1-2 branches (records not qualifying) | 50,740* | 30,378* |
| gout | 90560007 Gout | PGS001789 | Gout (GBMI leave-UKB-out) | yes | **no** | 910,151 / GRCh37 | | | 18,320 | 10,968 |
| primary_open_angle_glaucoma | 77075001 Primary open angle glaucoma | PGS001797 | POAG (GBMI PRS-CS-auto) | yes | **no** | 885,417 / GRCh37 | | | 8,040 | 4,814 |

\* The CKD counts include people whose only CKD records are stage 1 or 2. With those branches excluded, the
browser count is between 35,420 and 50,740.

Eight of the 12 scores are cached. PGS003725, PGS001782, PGS001789 and PGS001797 need scoring from the Catalog's
GRCh38 harmonized files.

## Provenance audit (details and URLs in `diseases.json`)

| PGS | discovery | components | tuning | AoU check |
|---|---|---|---|---|
| PGS004236 | MVP SBP, DBP and HTN GWAS (Giri 2019, multi-ancestry) + Pan-UKB | 3 C+T PRSs, unweighted sum | C+T parameters chosen in multi-ethnic TOPMed (Catalog: BioMe, n = 10,314) | 0 hits in PMC9213527 |
| PGS002308 | DIAGRAM EUR, BBJ EAS, MEDIA AA | none | none (PRS-CSx auto + meta) | 0 hits in PMC9241245 |
| PGS005168 | Roselli 2025 multi-ancestry AF meta-analysis, HUNT and UKB left out | none | none (PRS-CS, one weight set) | 0 hits in PMC12094172. AoU appears in the Catalog only in evaluation: 5 of 55 performance records, all from a later external paper (Haydarlou 2026, PGP000788) |
| PGS003725 | CAD + 10 risk-factor GWAS, 5 ancestries (CARDIoGRAMplusC4D, MVP, BBJ, FinnGen, G&H, GBMI, GLGC, GIANT, DIAMANTE, ...) | ancestry- and trait-specific LDpred2 PGSs | integration weights fitted in UKB EUR (n = 116,649) | 2 hits in PMC10353935, both Discussion or reference only. |
| PGS000004 | BCAC EUR GWAS | none | stepwise selection in a BCAC EUR validation set (n = 10,444) | 0 hits in PMC6323553 |
| PGS003766 | Wang 2023 multi-ancestry meta-analysis (PRACTICAL/ELLIPSE, UKB, FinnGen, eMERGE, BioVU, BioMe, MVP, ...) | none | none (fine-mapped variants with meta-analysis betas) | 0 hits in PMC10841479 |
| PGS001782, PGS001783, PGS001789, PGS001797 | GBMI multi-ancestry meta-analyses (no AoU member) | none | none (PRS-CS-auto) | 0 hits in PMC9903818 |
| PGS004885 | Wray 2018 PGC MDD (EUR) | none | none (MegaPRS auto, 1000G reference) | 0 hits in PMC11169548 |
| PGS002237 | Wuttke 2019 trans-ethnic eGFR GWAS | polygenic part only; the paper's APOL1 term is not in the Catalog file | best of 19 P+T scores chosen in 70% of UKB EUR | 0 hits in PMC9329233 |

**CKD and APOL1.** APOL1 G1/G2 is not in this score. It lowers African-ancestry discrimination; its effect on the z
slope is ≤ ~2% for coded CKD, and its mean effect is absorbed by the PC surface, so slope and calibration
heterogeneity remain interpretable. Optional AoU-only sensitivity analysis: add an APOL1 high-risk indicator (two
G1/G2 risk alleles) to the marginal index q, if G1 and G2 are typed.

**Cross-enrollment caveat.** Several development sets are US programs whose members can also be AoU participants,
so individual overlap with the AoU test set cannot be excluded from public cohort names. The programs are MVP,
BioVU, BioMe, MGB, eMERGE and the BCAC US cohorts. This is not AoU involvement in development, and it is recorded
per score. POAG names its six US GBMI cohorts: MGB, BioVU, BioMe, UCLA, CCPM and MGI.
For CKD, overlap is unverified for the CKDGen trans-ethnic eGFR discovery cohorts.

## Root changes from the SPEC candidates
- **Hypertension: 59621000 instead of 38341003.** 38341003 also pulls in obstetric (8,440) and maternal (7,920)
  hypertension, and secondary hypertension (9,740). Essential hypertension keeps 203,060 of 211,140 participants and
  matches the primary-hypertension trait.
- **Glaucoma: 77075001 (POAG) instead of 23986001.** 23986001 includes angle-closure glaucoma (4,280 + 1,020) and
  secondary glaucoma (1,740). Those are different diseases from the POAG score's trait.
- **CAD: 53741008 kept.** Every descendant is coronary atherosclerotic disease. It is not a "heart disease" root.

## Exclusions
- **chronic_kidney_disease excludes the stage 1 (431855005) and stage 2 (431856006) branches.** They are listed in
  `excluded_branches`, because PGS002237 is a stage ≥3 case score.
  - Records under these branches (the branch concepts included) are not qualifying CKD records.
  - This removes records, not people: someone with only stage 1-2 codes has zero qualifying CKD records.
  - Browser: 3,020 participants have at least 1 stage-1 record and 12,300 at least 1 stage-2 record.
- **type_2_diabetes excludes T1D (46635009).** T1D is often miscoded as T2D, and it is a different disease with
  different (HLA) genetics.
  - Known bias: the rule also removes insulin-treated T2D patients whose charts carry T1D (E10-type) codes on 2 or
    more dates. Those are long-duration, likely earlier-onset and higher-PGS cases. In the binary model this
    biases the PGS association toward the null and undercounts cases.
  - The prespecified rule stands. Written sensitivity analysis: exclude only people with more distinct T1D dates
    than T2D dates.
- **major_depressive_disorder excludes bipolar disorder (13746004).** Bipolar depression is often coded as MDD, and
  PGC MDD case definitions exclude bipolar disorder.

## Rejected
- **obesity (414916001).** The code phenotype is badly invalid.
  - Obesity codes are recorded for only part of the people with measured BMI ≥ 30, depending on site and billing,
    so the zero-record controls contain many obese people. The valid phenotype is measured BMI, which the uniform
    condition-code rule cannot express.
  - The multi-ancestry scores are BMI scores (a trait mismatch). The obesity-code scores are UKB-EUR or BBJ only.
- **hypercholesterolemia (13644009).** The code phenotype is badly invalid on the control side.
  - The same lipid picture is coded as hypercholesterolemia (79,340 participants) or as hyperlipidemia 55822004
    (207,520), depending on clinician and site. So at least 128,000 hyperlipidemia-coded people would count as
    controls.
  - No multi-ancestry score exists, and none is cached.
- **osteoarthritis (396275006).** The root pulls in spondylosis (95,000 of 172,760 participants) and every joint
  site.
  - All-site OA scores are UKB-EUR only and not cached.
  - Knee-OA scores (for root 239873007) are EUR-only with weak effects.
  - Dropped to keep the list at 12. POAG takes the slot.

Scores rejected for AoU development: PRSmix/PRSmixPlus (Truong 2024), for example PGS004785 and PGS004787
(hypertension), PGS004767 and PGS004768 (gout), PGS004765 and PGS004766 (glaucoma), and PGS004759 and PGS004760
(MDD).

## Caveats for downstream lanes
- **Descendants were checked in the Data Browser's criteria tree, not in the CDR.** study-cohort should list each
  root's top `concept_ancestor` descendants by count in v8 and confirm that none is unrelated.
- **The array projections assume equal prevalence** in the v8 array subset and the browser CDR. Ancestry-specific
  counts are not public, so per-ancestry case support must be measured in the cohort stage.
- **Some cells may be small.** POAG, prostate and breast cancer have the fewest cases. The survival model's
  incident-case counts will be a fraction of the numbers above, and cells with fewer than 20 events are suppressed.
