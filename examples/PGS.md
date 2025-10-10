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

For now, let's download the microarray data. This may impact the variant overlap of our scores.
```
gsutil -u "$GOOGLE_PROJECT" -m cp -r gs://fc-aou-datasets-controlled/v8/microarray/plink/* .
```

Now we can run the scores. It should be faster to run them all at once instead of one at a time.
```
./gnomon/target/release/gnomon score --score "PGS000765,PGS004904,PGS003433,PGS003979,PGS003386,PGS003852,PGS004303" arrays
```

This should take 11 minutes to run, and output a file called:
```
arrays.sscore
```
Let's open a Jupyter analysis notebook.
```
!head ../../arrays.sscore
```

This shows us the columns of the score output:
```
#IID	PGS000765_AVG	PGS000765_MISSING_PCT	PGS003386_AVG	PGS003386_MISSING_PCT	PGS003433_AVG	PGS003433_MISSING_PCT	PGS003852_AVG	PGS003852_MISSING_PCT	PGS003979_AVG	PGS003979_MISSING_PCT	PGS004303_AVG	PGS004303_MISSING_PCT	PGS004904_AVG	PGS004904_MISSING_PCT
```

Let's plot missingness for each score:
```
import pandas as pd, matplotlib.pyplot as plt
d=pd.read_csv('../../arrays.sscore', sep='\t')
for c in d.columns[d.columns.str.endswith('_MISSING_PCT')]: d[c].hist(bins=50); plt.title(c); plt.xlabel(c); plt.ylabel('count'); plt.show()
```

Here's one example:

<img width="597" height="454" alt="image" src="https://github.com/user-attachments/assets/bbe62bb4-6f93-4953-b441-c31ef1859804" />

This is not too bad given that we are using microarray data.
