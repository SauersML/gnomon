Start RStudio first.

This will assume you have a file ending in .sscore already. This example uses scores and phenotype for Alzheimer's disease, but other phenotypes could work too.

You might want to upload it e.g.:
```
import os, subprocess

bucket = os.getenv("WORKSPACE_BUCKET")
src = "arrays.sscore"
subprocess.run(["gsutil", "-m", "cp", src, f"{bucket}/arrays.sscore"], check=True)
```

Or download:
```
bucket <- Sys.getenv("WORKSPACE_BUCKET")
remote <- sprintf("%s/arrays.sscore", bucket)
local  <- "arrays.sscore"
system2("gsutil", c("-m", "cp", remote, local))
```

We want to compare a few models.

- 1.a. Prediction from linear PCs alone
- 1.b. Prediction from spline PCs alone
- 2.a  Prediction from linear scores alone
- 2.b  Prediction from spline scores alone
- 3.a  Predicting accuracy with linear PC
- 3.b  Predicting accuracy with PC splines
- 4.a  Predicting with linear scores and linear PCs
- 4.b  Predicting with spline scores and spline PCs

Let's define the cases and controls:
```
# Alzheimer's case definition in BigQuery + cohort counts (cases/controls)

library(bigrquery)

cdr_id  <- Sys.getenv("WORKSPACE_CDR")
project <- Sys.getenv("GOOGLE_PROJECT")

icd10_alz <- c("G30.0", "G30.1", "G30.8", "G30.9")
icd9_alz  <- "331.0"
codes_raw <- c(icd10_alz, icd9_alz)
codes     <- toupper(unique(c(codes_raw, gsub("\\.", "", codes_raw))))

# Case IDs (distinct person_id with AD codes in condition_source_value)
sql_cases <- sprintf(
  "SELECT DISTINCT CAST(person_id AS STRING) AS person_id
   FROM `%s.condition_occurrence`
   WHERE condition_source_value IS NOT NULL
     AND UPPER(TRIM(condition_source_value)) IN UNNEST(@codes)",
  cdr_id
)

h_cases <- bq_project_query(
  x = project,
  query = sql_cases,
  use_legacy_sql = FALSE,
  parameters = list(codes = codes),
  location = "US"
)

cases <- unique(as.character(bq_table_download(h_cases, bigint = "character")$person_id))

# Counts from BigQuery: total persons, cases, controls (= total - cases)
sql_counts <- sprintf(
  "WITH cases AS (
     SELECT DISTINCT CAST(person_id AS STRING) AS person_id
     FROM `%s.condition_occurrence`
     WHERE condition_source_value IS NOT NULL
       AND UPPER(TRIM(condition_source_value)) IN UNNEST(@codes)
   ),
   tot AS (
     SELECT COUNT(DISTINCT CAST(person_id AS STRING)) AS n_total
     FROM `%s.person`
   )
   SELECT
     (SELECT COUNT(*) FROM cases)                   AS n_cases,
     (SELECT n_total FROM tot)                      AS n_total,
     (SELECT n_total FROM tot) - (SELECT COUNT(*) FROM cases) AS n_controls",
  cdr_id, cdr_id
)

h_counts <- bq_project_query(
  x = project,
  query = sql_counts,
  use_legacy_sql = FALSE,
  parameters = list(codes = codes),
  location = "US"
)

cnt <- bq_table_download(h_counts, bigint = "integer")
n_cases    <- as.integer(cnt$n_cases[1])
n_total    <- as.integer(cnt$n_total[1])
n_controls <- as.integer(cnt$n_controls[1])
prev       <- 100 * n_cases / n_total

cat(sprintf(
  "Total: %s\nCases: %s\nControls: %s\nPrevalence: %.3f%%\n",
  format(n_total, big.mark = ","),
  format(n_cases, big.mark = ","),
  format(n_controls, big.mark = ","),
  prev
))
```
The prevalence is 0.201%.

Let's check per-score accuracies:
```
install.packages("pROC")
# AUROC of each score column in arrays.sscore vs. case/control
library(data.table)
library(pROC)

ss <- fread("arrays.sscore", sep = "\t", showProgress = FALSE)
ids <- as.character(ss[[1]])
y <- as.integer(ids %chin% unique(as.character(cases)))  # 1 = case, 0 = control

score_cols <- grep("_AVG$", names(ss), value = TRUE)

res <- rbindlist(lapply(score_cols, function(cn) {
  x <- suppressWarnings(as.numeric(ss[[cn]]))
  ok <- is.finite(x)
  n1 <- sum(y[ok] == 1L); n0 <- sum(y[ok] == 0L)
  if (n1 == 0L || n0 == 0L) return(data.table())
  r <- roc(response = y[ok], predictor = x[ok], quiet = TRUE)
  data.table(
    score    = sub("_AVG$", "", cn),
    n        = sum(ok),
    cases    = n1,
    controls = n0,
    AUROC    = as.numeric(auc(r))
  )
}))

setorder(res, -AUROC)
print(res)
```

Let's remove all but the top 5 scores:
```
# Keep only the top-5 PGS columns (plus ID) in arrays.sscore

library(data.table)

keep <- c("PGS004146","PGS000015","PGS004898","PGS004869","PGS000332")

dt <- fread("arrays.sscore", sep = "\t", showProgress = FALSE)
idcol <- names(dt)[1]

pat <- paste0("^(", paste(keep, collapse = "|"), ")_")
cols_keep <- c(idcol, grep(pat, names(dt), value = TRUE, perl = TRUE))

dt <- dt[, ..cols_keep]
fwrite(dt, "arrays.sscore", sep = "\t", quote = FALSE)
```


