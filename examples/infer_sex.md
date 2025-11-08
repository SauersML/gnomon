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

Let's run a polygenic score for colorectal cancer:
```
!../../gnomon/target/release/gnomon score "PGS003852" ../../arrays
```

Check missingness by sex chromosome ploidy:
```
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SEX_COL = "dragen_sex_ploidy"
SCORE_COL = "PGS003852_AVG"
MISS_COL = "PGS003852_MISSING_PCT"
SSCORE_PATH = os.path.join("..", "..", "arrays.sscore")

if "sex_df" not in globals():
    raise RuntimeError("sex_df is not defined; run the genetic sex load cell first.")

sscore_df = pd.read_csv(SSCORE_PATH, delim_whitespace=True)
sscore_df = sscore_df.rename(columns={"#IID": "IID"})
sscore_df["IID"] = sscore_df["IID"].astype(str)

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
        "No suitable participant ID column found in sex_df; "
        f"checked {id_candidates}"
    )

sex_df[id_col] = sex_df[id_col].astype(str)

merged = sscore_df.merge(
    sex_df[[id_col, SEX_COL]],
    left_on="IID",
    right_on=id_col,
    how="left",
)

merged["sex_group"] = merged[SEX_COL].where(
    merged[SEX_COL].isin(["XX", "XY"]),
    "Other",
)

positive_missing = merged[MISS_COL] > 0
merged.loc[positive_missing, "log_missing"] = np.log(
    merged.loc[positive_missing, MISS_COL]
)

plt.figure()
for sex, sub in merged.groupby("sex_group"):
    plt.hist(
        sub["log_missing"].dropna(),
        bins=50,
        alpha=0.5,
        label=sex,
    )
plt.xlabel("log(PGS003852_MISSING_PCT)")
plt.ylabel("Count")
plt.title("Distribution of log(missingness) by genetic sex")
plt.legend()
plt.show()

plt.figure()
for sex, sub in merged.groupby("sex_group"):
    plt.hist(
        sub[SCORE_COL].dropna(),
        bins=50,
        alpha=0.5,
        label=sex,
    )
plt.xlabel("PGS003852_AVG")
plt.ylabel("Count")
plt.title("Distribution of PGS003852_AVG by genetic sex")
plt.legend()
plt.show()
```

There is no large visible bias, which is good.

<img width="589" height="455" alt="image" src="https://github.com/user-attachments/assets/18b7f98a-e80d-4858-8b1b-bc61edeefec2" />

Let's check AUC:
```
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

if "sex_df" not in globals():
    raise RuntimeError("sex_df is not defined; load the genetic sex table in a previous cell.")
if "cases" not in globals():
    raise RuntimeError("cases is not defined; construct the colorectal cancer case set (incl. Z85.038, Z85.04x) first.")

sscore = pd.read_csv("../../arrays.sscore", sep="\t")
iid_col = sscore.columns[0]
sscore[iid_col] = sscore[iid_col].astype(str)

if "PGS003852_AVG" not in sscore.columns:
    raise RuntimeError("PGS003852_AVG not found in ../../arrays.sscore.")

id_candidates = [
    "person_id",
    "participant_id",
    "aou_participant_id",
    "research_id",
    "sample_id",
    "person_id_src",
]
iid_values = set(sscore[iid_col])
best_id = None
best_overlap = -1
for c in id_candidates:
    if c in sex_df.columns:
        ids = set(sex_df[c].astype(str))
        ov = len(iid_values & ids)
        if ov > best_overlap:
            best_overlap = ov
            best_id = c

if best_id is None or best_overlap == 0:
    raise RuntimeError("Could not match IIDs in arrays.sscore to any ID column in sex_df.")

sex_use = sex_df[[best_id, "dragen_sex_ploidy"]].copy()
sex_use[best_id] = sex_use[best_id].astype(str)
sex_use = sex_use.rename(columns={best_id: iid_col})

df = sscore.merge(sex_use, on=iid_col, how="left")

case_set = set(pd.Series(cases, dtype=str))
df["case"] = df[iid_col].isin(case_set).astype("i1")

df["sex_group"] = df["dragen_sex_ploidy"].where(
    df["dragen_sex_ploidy"].isin(["XX", "XY"]),
    "Other",
)

rows = []
for g in ["XX", "XY", "Other"]:
    sub = df[(df["sex_group"] == g) & df["PGS003852_AVG"].notna()]
    n = len(sub)
    n_cases = int(sub["case"].sum())
    n_ctrl = n - n_cases
    if n_cases > 0 and n_ctrl > 0:
        auc = float(roc_auc_score(sub["case"], sub["PGS003852_AVG"]))
    else:
        auc = np.nan
    rows.append(
        {
            "sex_group": g,
            "n": n,
            "cases": n_cases,
            "controls": n_ctrl,
            "AUC_PGS003852": auc,
        }
    )

print(pd.DataFrame(rows))
```

AUC looks about the same in each group (0.566504 in XX, 0.563028 in XY).

Let's check prevalence per-group:
```
import pandas as pd
import numpy as np

g = (
    df.groupby("sex_group")["case"]
      .agg(cases="sum", n="count")
      .assign(prevalence_pct=lambda x: 100 * x["cases"] / x["n"])
      .reset_index()
)

for _, row in g.iterrows():
    print(
        f"{row['sex_group']}: "
        f"{row['cases']} / {int(row['n'])} "
        f"({row['prevalence_pct']:.3f}%)"
    )
```
Other: 259 / 33941 (0.763%)
XX: 2089 / 252079 (0.829%)
XY: 1713 / 161258 (1.062%)

Let's check the age distribution in each group:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

data = df_age.copy()

def classify_sex(x):
    if pd.isna(x):
        return "Missing"
    if x == "XX":
        return "XX"
    if x == "XY":
        return "XY"
    return "Other"

data["sex_group4"] = data["dragen_sex_ploidy"].map(classify_sex)

valid = data[data["age"].notna()]
age_min = float(valid["age"].min())
age_max = float(valid["age"].max())
xs = np.linspace(age_min, age_max, 400)

plt.figure(figsize=(8, 5))

for grp in ["XX", "XY", "Other", "Missing"]:
    sub = valid.loc[valid["sex_group4"] == grp, "age"]
    if len(sub) < 2:
        continue
    kde = gaussian_kde(sub.to_numpy())
    ys = kde(xs)
    plt.plot(xs, ys, linewidth=2, label=f"{grp} (n={len(sub)})")
    mean_age = sub.mean()
    plt.axvline(mean_age, linestyle="--", linewidth=1)

plt.xlabel("Age (years)")
plt.ylabel("Density")
plt.title("Age distribution by genetic sex group")
plt.legend()
plt.tight_layout()
plt.show()
```


<img width="790" height="490" alt="image" src="https://github.com/user-attachments/assets/344ee484-9d95-4b6d-93b0-347c8bae6f8e" />


This is interesting. Why do people with non-XY and non-XX ploidy have a much higher mean age?

```
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Specific non-XX/XY ploidy categories to display (including literal "nan" value, excluding true missing)
ploidy_labels = ["X0", "XO", "XXX", "XXY", "XYY", "nan"]
allowed = {p.lower() for p in ploidy_labels}

ploidy = df_age["dragen_sex_ploidy"]
age = df_age["age"]

is_missing_ploidy = ploidy.isna()
ploidy_str = ploidy.astype(str)
mask = age.notna() & (~is_missing_ploidy) & (ploidy_str.str.lower().isin(allowed))

sub = df_age.loc[mask].copy()
sub["ploidy_str"] = sub["dragen_sex_ploidy"].astype(str)

if not sub.empty:
    age_min = float(sub["age"].min())
    age_max = float(sub["age"].max())
    xs = np.linspace(age_min, age_max, 400)

    plt.figure(figsize=(8, 5))

    for p in ploidy_labels:
        grp = sub[sub["ploidy_str"].str.lower() == p.lower()]["age"].dropna()
        if len(grp) < 2:
            continue
        kde = gaussian_kde(grp.to_numpy())
        ys = kde(xs)
        plt.plot(xs, ys, linewidth=2, label=f"{p} (n={len(grp)})")
        mean_age = grp.mean()
        plt.axvline(mean_age, linestyle="--", linewidth=1)

    plt.xlabel("Age (years)")
    plt.ylabel("Density")
    plt.title("Age distribution by specific non-XX/XY ploidy (excluding missing)")
    plt.legend()
    plt.tight_layout()
    plt.show()
```


XO and X0 individuals are driving this trend, with a mean age of around 80.
