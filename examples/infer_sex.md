If you haven't already, create a v8 workspace and download the microarray data:
```
!gsutil -u "$GOOGLE_PROJECT" -m cp -r gs://fc-aou-datasets-controlled/v8/microarray/plink/* ../..!gsutil -u "$GOOGLE_PROJECT" -m cp -r gs://fc-aou-datasets-controlled/v8/microarray/plink/* ../..
```

View sex chromosome ploidy counts:
```
import os
import io
import pandas as pd
import gcsfs

SEX_URI = "gs://fc-aou-datasets-controlled/v8/wgs/short_read/snpindel/aux/qc/genomic_metrics.tsv"
SEX_COL = "dragen_sex_ploidy"
FAM_PATH = os.path.join("..", "..", "arrays.fam")

def load_genetic_sex():
    project = os.environ.get("GOOGLE_PROJECT")
    if not project:
        raise RuntimeError("GOOGLE_PROJECT is not set; cannot read requester-pays bucket.")

    fs = gcsfs.GCSFileSystem(project=project, token="cloud", requester_pays=True)
    with fs.open(SEX_URI, "rb") as f:
        df = pd.read_csv(f, sep="\t")
    return df

sex_df = load_genetic_sex()

if SEX_COL not in sex_df.columns:
    raise ValueError(
        f"Expected '{SEX_COL}' in genetic sex table; "
        f"found columns: {list(sex_df.columns)}"
    )

id_candidates = [
    "person_id",
    "participant_id",
    "aou_participant_id",
    "research_id",
    "sample_id",
]
id_col_candidates = [c for c in id_candidates if c in sex_df.columns]

if not id_col_candidates:
    raise ValueError(
        "No suitable participant ID column found in genetic sex table; "
        f"checked {id_candidates}"
    )

id_col = id_col_candidates[0]

sex_df[id_col] = sex_df[id_col].astype(str)
sex_df[SEX_COL] = sex_df[SEX_COL].astype(str)

fam_cols = [
    "family_id",
    "individual_id",
    "father_id",
    "mother_id",
    "sex_code",
    "phenotype",
]
fam_df = pd.read_csv(
    FAM_PATH,
    delim_whitespace=True,
    header=None,
    names=fam_cols,
    dtype=str,
)

array_ids = set(fam_df["individual_id"])

sex_df_with_array = sex_df[sex_df[id_col].isin(array_ids)]

def count_ploidy(values):
    v = values.fillna("NA")
    xx = (v == "XX").sum()
    xy = (v == "XY").sum()
    other = len(v) - xx - xy
    return xx, xy, other

overall_xx, overall_xy, overall_other = count_ploidy(sex_df[SEX_COL])
array_xx, array_xy, array_other = count_ploidy(sex_df_with_array[SEX_COL])

print("Overall genetic sex ploidy counts:")
print(f"  XX: {overall_xx}")
print(f"  XY: {overall_xy}")
print(f"  Other: {overall_other}")

print("\nGenetic sex ploidy counts among participants with microarray data:")
print(f"  XX: {array_xx}")
print(f"  XY: {array_xy}")
print(f"  Other: {array_other}")
```

```
import os
import io
import pandas as pd

SEX_COL = "dragen_sex_ploidy"

# Reconstruct microarray subset if it doesn't exist yet
if "sex_df_with_array" not in globals():
    FAM_PATH = os.path.join("..", "..", "arrays.fam")
    fam_cols = [
        "family_id",
        "individual_id",
        "father_id",
        "mother_id",
        "sex_code",
        "phenotype",
    ]
    fam_df = pd.read_csv(
        FAM_PATH,
        delim_whitespace=True,
        header=None,
        names=fam_cols,
        dtype=str,
    )

    array_ids = set(fam_df["individual_id"].astype(str))

    id_candidates = [
        "person_id",
        "participant_id",
        "aou_participant_id",
        "research_id",
        "sample_id",
    ]
    id_col = None
    for c in id_candidates:
        if c in sex_df.columns:
            id_col = c
            break

    if id_col is None:
        raise ValueError(
            "No suitable participant ID column found in genetic sex table; "
            f"checked {id_candidates}"
        )

    sex_df[id_col] = sex_df[id_col].astype(str)
    sex_df[SEX_COL] = sex_df[SEX_COL].astype(str)

    sex_df_with_array = sex_df[sex_df[id_col].isin(array_ids)]

# Helper: restrict to "Other" = not XX/XY and non-missing
def other_mask(df):
    return (
        df[SEX_COL].notna()
        & (~df[SEX_COL].isin(["XX", "XY"]))
    )

overall_other = sex_df.loc[other_mask(sex_df), SEX_COL]
array_other = sex_df_with_array.loc[other_mask(sex_df_with_array), SEX_COL]

print("Breakdown of 'Other' overall:")
print(overall_other.value_counts().sort_index())

print("\nBreakdown of 'Other' among participants with microarray data:")
print(array_other.value_counts().sort_index())
```

