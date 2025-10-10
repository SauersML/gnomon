## Running a polygenic score for colorectal cancer on the _All of Us_ cohort
First, let's see which colorectal cancer scores exist in the PGS Catalog:
```
https://www.pgscatalog.org/trait/MONDO_0005575/
```

There are a few types of statistics to consider:
- AUROC / AUC / c-statistic: cases vs. controls
- C-index: like AUROC, but for time-to-event data
- AUPRC: precision vs recall across thresholds
- OR, HR: relative risk (hopefully normalized per-SD of score)
- Nagelkerke’s pseudo-R²: explained variance but for binary

Strong performers seem to include:
- PGS000765
- PGS004904
- PGS003433
- PGS003979
- PGS003386
- PGS003852
- PGS004303

We want to avoid imputation, but we still want lots of samples with many variants.

Let's look at our options:

| Group             | Subset         | PLINK version | File types                | Total size | GCS path                                                                              |
| ----------------- | -------------- | ---------- | ------------------------- | ---------: | ------------------------------------------------------------------------------------- |
| srWGS SNP & Indel | Exome          | 1.9        | `.bed`, `.bim`, `.fam`    |   3.94 TiB | `gs://fc-aou-datasets-controlled/v8/wgs/short_read/snpindel/exome/plink_bed`          |
| srWGS SNP & Indel | Exome          | 2.0        | `.pgen`, `.pvar`, `.psam` |  96.65 GiB | `gs://fc-aou-datasets-controlled/v8/wgs/short_read/snpindel/exome/pgen`               |
| srWGS SNP & Indel | ACAF Threshold | 1.9        | `.bed`, `.bim`, `.fam`    |  10.51 TiB | `gs://fc-aou-datasets-controlled/v8/wgs/short_read/snpindel/acaf_threshold/plink_bed` |
| srWGS SNP & Indel | ACAF Threshold | 2.0        | `.pgen`, `.pvar`, `.psam` |   1.11 TiB | `gs://fc-aou-datasets-controlled/v8/wgs/short_read/snpindel/acaf_threshold/pgen`      |
| srWGS SNP & Indel | ClinVar        | 1.9        | `.bed`, `.bim`, `.fam`    | 204.68 GiB | `gs://fc-aou-datasets-controlled/v8/wgs/short_read/snpindel/clinvar/plink_bed`        |
| srWGS SNP & Indel | ClinVar        | 2.0        | `.pgen`, `.pvar`, `.psam` |    9.3 GiB | `gs://fc-aou-datasets-controlled/v8/wgs/short_read/snpindel/clinvar/pgen`             |
| Genotyping Array  | Array          | 1.9        | `.bed`, `.bim`, `.fam`    |  181.2 GiB | `gs://fc-aou-datasets-controlled/v8/microarray/plink`                                 |

ClinVar and exome subsets will not have enough variants. Array data might work for most scores, and ACAF Threshold will likely work even better. The best option is gs://fc-aou-datasets-controlled/v8/wgs/short_read/snpindel/acaf_threshold/plink_bed, though it will take a very long time to stream 10.51 TiB.

