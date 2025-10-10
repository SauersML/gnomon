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

We'll assess these scores on three metrics in the test data:
- Odds ratio (per standard deviation of score)
- AUROC
- Nagelkerke's R²

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

Create a standard cloud analysis environment with 8 CPUs and 30 GB RAM (though using lower or higher values should also work similarly).

Install gnomon if you haven't already:
```
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && { for f in ~/.bashrc ~/.profile; do [ -f "$f" ] || touch "$f"; grep -qxF 'source "$HOME/.cargo/env"' "$f" || printf '\n# Rust / Cargo\nsource "$HOME/.cargo/env"\n' >> "$f"; done; } && source "$HOME/.cargo/env" && git clone https://github.com/SauersML/gnomon.git && cd gnomon && rustup override set nightly && cargo build --release && cd ~
```

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

Now let's check the correlation matrix of the scores. We expect moderate correlations.

```
import pandas as pd, matplotlib.pyplot as plt
c=pd.read_csv('../../arrays.sscore',sep='\t').filter(regex='_AVG$').corr()
plt.figure(figsize=(8,8)); im=plt.imshow(c,vmin=-1,vmax=1); plt.colorbar(im); plt.xticks(range(len(c)),c.columns,rotation=90); plt.yticks(range(len(c)),c.columns); plt.tight_layout(); plt.show()
```

<img width="390" height="379" alt="image" src="https://github.com/user-attachments/assets/96f101be-c18f-494f-bdd6-4550364c6644" />

Let's make the case definition.

**ICD-10:**
- C18.0–C18.9 — Malignant neoplasm of colon
- C19 — Malignant neoplasm of rectosigmoid junction. 
- C20 — Malignant neoplasm of rectum. 

**ICD-9:**
- 153.0–153.9 — Malignant neoplasm of colon.
- 154.0 — Malignant neoplasm of rectosigmoid junction.
- 154.1 — Malignant neoplasm of rectum.

```
cdr_id = os.environ["WORKSPACE_CDR"]
from google.cloud import bigquery as bq
codes=[f"C18.{i}" for i in range(10)]+["C19","C20"]+[f"153.{i}" for i in range(10)]+["154.0","154.1"]; codes=[c.upper() for c in (codes+[c.replace(".","") for c in codes])]
sql=f"""SELECT DISTINCT CAST(person_id AS STRING) person_id
        FROM `{cdr_id}.condition_occurrence`
        WHERE condition_source_value IS NOT NULL
          AND UPPER(TRIM(condition_source_value)) IN UNNEST(@codes)"""
cases=bq.Client().query(sql, job_config=bq.QueryJobConfig(query_parameters=[bq.ArrayQueryParameter("codes","STRING",codes)])).to_dataframe()["person_id"].astype(str)
```

For simplicity, we aren't using a time-to-event model. Some of our "controls" will go on to develop colorectal cancer later in life.

Calculate metrics:
```
import pandas as pd, numpy as np, statsmodels.api as sm
from sklearn.metrics import roc_auc_score
from scipy.stats import mannwhitneyu, norm, chi2
d=pd.read_csv('../../arrays.sscore',sep='\t'); idcol=d.columns[0]; y=d[idcol].astype(str).isin(set(cases)).astype(int)
res=[]
for s in d.filter(regex='_AVG$').columns:
    x=pd.to_numeric(d[s],errors='coerce'); t=pd.DataFrame({'y':y,'x':x}).dropna()
    if t['y'].nunique()<2 or t['x'].std(ddof=0)==0: continue
    z=(t['x']-t['x'].mean())/t['x'].std(ddof=0); X=sm.add_constant(pd.DataFrame({'z':z}))
    m=sm.Logit(t['y'],X).fit(disp=False, maxiter=200); beta=float(m.params['z']); se=float(m.bse['z'])
    OR=np.exp(beta); p_or=1-norm.cdf(beta/se)
    auc=roc_auc_score(t['y'],t['x']); p_auc=mannwhitneyu(t.loc[t['y']==1,'x'],t.loc[t['y']==0,'x'],alternative='greater').pvalue
    n=len(t); r2=(1-np.exp((2/n)*(m.llnull-m.llf)))/(1-np.exp(2*m.llnull/n)); p_r2=1-chi2.cdf(2*(m.llf-m.llnull),1)
    res.append([s,n,int(t['y'].sum()),int(n-t['y'].sum()),OR,p_or,auc,p_auc,r2,p_r2])
pd.DataFrame(res,columns=['score','n','cases','controls','OR_perSD','p_one_sided_OR>1','AUROC','p_one_sided_AUC>0.5','Nagelkerke_R2','p_one_sided_R2>0']).sort_values('Nagelkerke_R2',ascending=False)
```

| Score         |       N | Cases | Controls | OR per SD | P (OR>1) | AUROC | P (AUC>0.5) | Nagelkerke R² | P (R²>0) |
| ------------- | ------: | ----: | -------: | --------: | -------: | ----: | ----------: | ------------: | -------: |
| PGS003852_AVG | 447,278 | 3,192 |  444,086 |     1.293 |  <1e-100 | 0.573 |    1.04e-45 |         0.57% |  <1e-100 |
| PGS003979_AVG | 447,278 | 3,192 |  444,086 |     1.294 |  <1e-100 | 0.571 |    5.83e-44 |         0.57% |  <1e-100 |
| PGS003433_AVG | 447,278 | 3,192 |  444,086 |     1.276 |  <1e-100 | 0.569 |    1.27e-41 |         0.52% |  <1e-100 |
| PGS000765_AVG | 447,278 | 3,192 |  444,086 |     1.211 |  <1e-100 | 0.554 |    1.28e-26 |         0.32% |  <1e-100 |
| PGS004904_AVG | 447,278 | 3,192 |  444,086 |     1.176 |  <1e-100 | 0.545 |    5.37e-19 |         0.23% |  <1e-100 |
| PGS003386_AVG | 447,278 | 3,192 |  444,086 |     1.095 | 1.63e-07 | 0.526 |    3.09e-07 |         0.07% | 3.29e-07 |
| PGS004303_AVG | 447,278 | 3,192 |  444,086 |     1.027 |    0.070 | 0.507 |       0.096 |         0.01% |    0.141 |

The GenoBoost method (PGS004303), which uses non-additive models, doesn't perform well relative to the others. The ICD10 codes included in the construction of PGS004303 were similar to ours: C18,C19,C20.

The best performing scores are PGS003852, PGS003979, and PGS003433. One thing common to each of these scores is that the associated study involves colorectal cancer specifically (as opposed to being general and studying many diseases at once).

Let's add 16 principal components and sex to the model as predictors (not controls) and look at the resulting AUC.

```
import os, numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

d = pd.read_csv('../../arrays.sscore', sep='\t')
d['#IID'] = d['#IID'].astype(str)
d = d.set_index('#IID')
score_cols = [c for c in d.columns if c.endswith('_AVG')]

try:
    covars
except NameError:
    NUM_PCS = 16
    PCS_URI = "gs://fc-aou-datasets-controlled/v8/wgs/short_read/snpindel/aux/ancestry/ancestry_preds.tsv"
    SEX_URI = "gs://fc-aou-datasets-controlled/v8/wgs/short_read/snpindel/aux/qc/genomic_metrics.tsv"
    storage_opts = {"project": os.environ.get("GOOGLE_PROJECT"), "requester_pays": True} if os.environ.get("GOOGLE_PROJECT") else {}
    pcs_raw = pd.read_csv(PCS_URI, sep="\t", storage_options=storage_opts, usecols=["research_id","pca_features"])
    parse = lambda s, k=NUM_PCS: ([(float(x) if x!='' else np.nan) for x in str(s).strip()[1:-1].split(",")] + [np.nan]*k)[:k]
    pc_cols = [f"PC{i}" for i in range(1, NUM_PCS+1)]
    pc_df = pd.DataFrame(pcs_raw["pca_features"].apply(parse).to_list(), columns=pc_cols)\
              .assign(person_id=pcs_raw["research_id"].astype(str)).set_index("person_id")
    sex_df = pd.read_csv(SEX_URI, sep="\t", storage_options=storage_opts, usecols=["research_id","dragen_sex_ploidy"])\
              .assign(person_id=lambda x: x["research_id"].astype(str)).set_index("person_id")["dragen_sex_ploidy"]\
              .map({"XX":0,"XY":1}).to_frame("sex").dropna()
    covars = pc_df.join(sex_df, how="inner").filter(pc_cols+["sex"]).dropna()

if 'cases' not in globals():
    raise NameError("Variable 'cases' not found.")
case_set = set(pd.Series(cases, dtype=str))

rows = []
for sc in score_cols:
    tmp = d[[sc]].join(covars, how='inner').dropna()
    if tmp.empty: 
        continue
    y = tmp.index.to_series().isin(case_set).astype('i1').to_numpy()
    X = tmp[["sex"] + pc_cols + [sc]].astype(float).to_numpy()
    mdl = make_pipeline(StandardScaler(), LogisticRegression(solver="lbfgs", max_iter=1000))
    mdl.fit(X, y)
    auc = roc_auc_score(y, mdl.predict_proba(X)[:,1])
    rows.append({"score": sc.replace("_AVG",""), "n": int(len(tmp)), "cases": int(y.sum()), "controls": int(len(y)-y.sum()), "AUROC": float(auc)})

out = pd.DataFrame(rows).sort_values("AUROC", ascending=False).reset_index(drop=True)
display(out)
```


| score     | cases | controls | AUROC    |
| :-------- | ----: | -------: | :------- |
| PGS003852 |  2985 |   410352 | 0.619005 |
| PGS003979 |  2985 |   410352 | 0.617833 |
| PGS003433 |  2985 |   410352 | 0.610325 |
| PGS000765 |  2985 |   410352 | 0.600296 |
| PGS004904 |  2985 |   410352 | 0.599918 |
| PGS003386 |  2985 |   410352 | 0.596961 |
| PGS004303 |  2985 |   410352 | 0.592681 |

This improves the scores somewhat, but there's a few problems. First, we typically want one prediction per individual, but now we have seven (one per score). Further, we're training on the same dataset we're evaluating on.

Now, let's test the same model, except jointly fitting all scores (except for PGS004303) using cross-validation to avoid overfitting.

```
import os, numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

d = pd.read_csv('../../arrays.sscore', sep='\t').set_index('#IID')
d.index = d.index.astype(str)
pgs = [c for c in d.columns if c.endswith('_AVG') and c != 'PGS004303_AVG']

try:
    covars
except NameError:
    NUM_PCS=16; pc_cols=[f"PC{i}" for i in range(1,NUM_PCS+1)]
    so={"project": os.environ.get("GOOGLE_PROJECT"), "requester_pays": True} if os.environ.get("GOOGLE_PROJECT") else {}
    pcs=pd.read_csv("gs://fc-aou-datasets-controlled/v8/wgs/short_read/snpindel/aux/ancestry/ancestry_preds.tsv", sep="\t", storage_options=so, usecols=["research_id","pca_features"])
    parse=lambda s,k=NUM_PCS: ([(float(x) if x!='' else np.nan) for x in str(s).strip()[1:-1].split(",")] + [np.nan]*k)[:k]
    pc_df=pd.DataFrame(pcs["pca_features"].apply(parse).to_list(), columns=pc_cols).assign(person_id=pcs["research_id"].astype(str)).set_index("person_id")
    sex=pd.read_csv("gs://fc-aou-datasets-controlled/v8/wgs/short_read/snpindel/aux/qc/genomic_metrics.tsv", sep="\t", storage_options=so, usecols=["research_id","dragen_sex_ploidy"])
    sex_df=sex.assign(person_id=sex["research_id"].astype(str)).set_index("person_id")["dragen_sex_ploidy"].map({"XX":0,"XY":1}).to_frame("sex").dropna()
    covars=pc_df.join(sex_df, how="inner").filter(pc_cols+["sex"]).dropna()

covars = covars.copy()
covars.index = covars.index.astype(str)

if 'cases' not in globals(): raise NameError("Variable 'cases' not found.")
case_set=set(pd.Series(cases, dtype=str))

Xy = covars.join(d[pgs], how="inner").dropna()
y = Xy.index.to_series().isin(case_set).astype('i1').to_numpy()
X = Xy.to_numpy()

cv = StratifiedKFold(n_splits=15, shuffle=True, random_state=42)
mdl = make_pipeline(StandardScaler(), LogisticRegression(solver="lbfgs", max_iter=1000))
p = cross_val_predict(mdl, X, y, cv=cv, method="predict_proba")[:,1]
auc = roc_auc_score(y, p)

pd.DataFrame([{"n": int(len(y)), "cases": int(y.sum()), "controls": int(len(y)-y.sum()), "predictors": X.shape[1], "AUROC": float(auc)}])
```

This yields an AUROC of 0.617375. (Adding a ridge penalty gives a similar value of 0.617.)
