## Running a polygenic score for obesity on the _All of Us_ cohort

Cognitive decline has a heterogeneous etiology, and may be related to Alzheimer's disease, dementia, or other genetic and environmental causes. This is complicated by the age of individuals in the biobank.

Let's view the polygenic score catalog for dementia:
https://www.pgscatalog.org/trait/MONDO_0001627/

And Alzheimer's disease:
https://www.pgscatalog.org/trait/MONDO_0004975/


Strong performers include:
- PGS004378
- PGS003897
- PGS005199
- PGS005198
- PGS003400
- PGS005235
- PGS005203
- PGS004150

We can use the microarray `gs://fc-aou-datasets-controlled/v8/microarray/plink/` for scoring.

Create a standard cloud analysis environment with 8 CPUs and 30 GB RAM (though using lower or higher values should also work similarly).

Install gnomon if you haven't already:
```
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && { for f in ~/.bashrc ~/.profile; do [ -f "$f" ] || touch "$f"; grep -qxF 'source "$HOME/.cargo/env"' "$f" || printf '\n# Rust / Cargo\nsource "$HOME/.cargo/env"\n' >> "$f"; done; } && source "$HOME/.cargo/env" && git clone https://github.com/SauersML/gnomon.git && cd gnomon && rustup override set nightly && cargo build --release && cd ~
```

Let's download the microarray data:
```
gsutil -u "$GOOGLE_PROJECT" -m cp -r gs://fc-aou-datasets-controlled/v8/microarray/plink/* .
```

Now we can run the scores. It should be faster to run them all at once instead of one at a time.
```
./gnomon/target/release/gnomon score "PGS004378, PGS003897, PGS005199, PGS005198, PGS003400, PGS005235, PGS005203, PGS004150" arrays
```

This should take 23 minutes to run, and output a file called:
```
arrays.sscore
```
Let's open a Jupyter analysis notebook.
```
!head ../../arrays.sscore
```

This shows us the columns of the score output:
```
#IID	PGS003400_AVG	PGS003400_MISSING_PCT	PGS003897_AVG	PGS003897_MISSING_PCT	PGS004150_AVG	PGS004150_MISSING_PCT	PGS004378_AVG	PGS004378_MISSING_PCT	PGS005198_AVG	PGS005198_MISSING_PCT	PGS005199_AVG	PGS005199_MISSING_PCT	PGS005203_AVG	PGS005203_MISSING_PCT	PGS005235_AVG	PGS005235_MISSING_PCT
```

Let's plot missingness for each score:
```
import pandas as pd, matplotlib.pyplot as plt
d=pd.read_csv('../../arrays.sscore', sep='\t')
for c in d.columns[d.columns.str.endswith('_MISSING_PCT')]: d[c].hist(bins=50); plt.title(c); plt.xlabel(c); plt.ylabel('count'); plt.show()
```

Most samples' missingness should be under 2%.

Now let's check the correlation matrix of the scores. We expect moderate correlations.

```
import pandas as pd, matplotlib.pyplot as plt
c=pd.read_csv('../../arrays.sscore',sep='\t').filter(regex='_AVG$').corr()
plt.figure(figsize=(8,8)); im=plt.imshow(c,vmin=-1,vmax=1); plt.colorbar(im); plt.xticks(range(len(c)),c.columns,rotation=90); plt.yticks(range(len(c)),c.columns); plt.tight_layout(); plt.show()
```

<img width="780" height="758" alt="image" src="https://github.com/user-attachments/assets/8598d356-92bd-4ff1-8f7e-f77672f72e37" />


This lets us know that we should drop PGS005199 or PGS005203 since they are highly correlated. Looking at the PGS catalog, PGS005199 is multi-ancestry and better performing.


```
# Read the file
df = pd.read_csv('../../arrays.sscore', sep='\t')

# Drop the PGS005203 columns
df = df.drop(columns=['PGS005203_AVG', 'PGS005203_MISSING_PCT'])

# Save back to file
df.to_csv('../../arrays.sscore', sep='\t', index=False)
```

Let's make the case definition.

ICD-10:

* E66.811
* E66.812
* E66.813
* E66.01
* E66.09
* E66.1
* E66.2
* E66.89
* E66.9
* E88.82

ICD-9:

* 278.00
* 278.01
* 278.03

```
import os
from google.cloud import bigquery as bq

cdr_id = os.environ["WORKSPACE_CDR"]

icd10_obesity = [
    "E66.811", "E66.812", "E66.813",
    "E66.01", "E66.09",
    "E66.1", "E66.2",
    "E66.89", "E66.9",
    "E88.82",
]
icd9_obesity = [
    "278.00", "278.01", "278.03",
]

codes = icd10_obesity + icd9_obesity
codes = [c.upper() for c in (codes + [c.replace(".", "") for c in codes])]

sql = f"""
SELECT DISTINCT CAST(person_id AS STRING) AS person_id
FROM `{cdr_id}.condition_occurrence`
WHERE condition_source_value IS NOT NULL
  AND UPPER(TRIM(condition_source_value)) IN UNNEST(@codes)
"""

cases = (
    bq.Client()
    .query(sql, job_config=bq.QueryJobConfig(
        query_parameters=[bq.ArrayQueryParameter("codes", "STRING", codes)]
    ))
    .to_dataframe()["person_id"]
    .astype(str)
)
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

The best score is PGS005198 with an OR/SD of 1.453851 and AUROC of 0.602658. PGS004378 is in second place, with an AUROC of 0.589763. PGS005235 is in third, with an AUROC of 0.583957.

```
import os, numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, accuracy_score

d = pd.read_csv('../../arrays.sscore', sep='\t')
d['#IID'] = d['#IID'].astype(str)
d = d.set_index('#IID')
score_cols = [c for c in d.columns if c.endswith('_AVG')]

NUM_PCS = 16
PCS_URI = "gs://fc-aou-datasets-controlled/v8/wgs/short_read/snpindel/aux/ancestry/ancestry_preds.tsv"
SEX_URI = "gs://fc-aou-datasets-controlled/v8/wgs/short_read/snpindel/aux/qc/genomic_metrics.tsv"
storage_opts = {"project": os.environ.get("GOOGLE_PROJECT"), "requester_pays": True} if os.environ.get("GOOGLE_PROJECT") else {}

pcs_raw = pd.read_csv(PCS_URI, sep="\t", storage_options=storage_opts, usecols=["research_id","pca_features"])
parse = lambda s, k=NUM_PCS: ([(float(x) if x!='' else np.nan) for x in str(s).strip()[1:-1].split(",")] + [np.nan]*k)[:k]
pc_cols = [f"PC{i}" for i in range(1, NUM_PCS+1)]
pc_df = (pd.DataFrame(pcs_raw["pca_features"].apply(parse).to_list(), columns=pc_cols)
         .assign(person_id=pcs_raw["research_id"].astype(str)).set_index("person_id"))

sex_df = (pd.read_csv(SEX_URI, sep="\t", storage_options=storage_opts, usecols=["research_id","dragen_sex_ploidy"])
          .assign(person_id=lambda x: x["research_id"].astype(str)).set_index("person_id")["dragen_sex_ploidy"]
          .map({"XX":0,"XY":1}).to_frame("sex"))

covars = pc_df.join(sex_df, how="inner").dropna()
case_idx = pd.Index(pd.Series(cases, dtype=str).unique(), name="person_id")

def eval_block(df):
    if df.empty: 
        return {"n": 0, "cases": 0, "controls": 0, "AUROC": np.nan, "ACC": np.nan}
    y = df.index.to_series().isin(case_idx).astype('i1').to_numpy()
    X = df.astype(float).to_numpy()
    mdl = make_pipeline(StandardScaler(), LogisticRegression(solver="lbfgs", max_iter=1000))
    mdl.fit(X, y)
    p = mdl.predict_proba(X)[:,1]
    return {"n": int(len(df)), "cases": int(y.sum()), "controls": int(len(y)-y.sum()),
            "AUROC": float(roc_auc_score(y, p)), "ACC": float(accuracy_score(y, (p>=0.5).astype(int)))}

rows = []
for sc in score_cols:
    s = d[[sc]].dropna()
    s_sex = d[[sc]].join(sex_df, how="inner").dropna()[[sc,"sex"]]
    s_sex_pcs = d[[sc]].join(covars, how="inner").dropna()[["sex"]+pc_cols+[sc]]

    r1 = eval_block(s[[sc]])
    r2 = eval_block(s_sex)
    r3 = eval_block(s_sex_pcs)

    rows.append({
        "score": sc.replace("_AVG",""),
        "n_s": r1["n"], "cases_s": r1["cases"], "AUROC_s": r1["AUROC"], "ACC_s": r1["ACC"],
        "n_sx": r2["n"], "cases_sx": r2["cases"], "AUROC_sx": r2["AUROC"], "ACC_sx": r2["ACC"],
        "n_sxpc": r3["n"], "cases_sxpc": r3["cases"], "AUROC_sxpc": r3["AUROC"], "ACC_sxpc": r3["ACC"],
    })

out = pd.DataFrame(rows).sort_values("AUROC_sxpc", ascending=False).reset_index(drop=True)
print(out.to_string(index=False, float_format=lambda x: f"{x:.3f}" if isinstance(x, float) else str(x)))
```

| Score ID  | AUROC — Score only | AUROC — Score + Sex | AUROC — Score + Sex + PCs |
| :-------- | -----------------: | ------------------: | ------------------------: |
| PGS005199 |              0.588 |               0.587 |                     0.640 |
| PGS005198 |              0.603 |               0.615 |                     0.631 |
| PGS004150 |              0.570 |               0.585 |                     0.615 |
| PGS004378 |              0.590 |               0.604 |                     0.614 |
| PGS003897 |              0.576 |               0.593 |                     0.608 |
| PGS005235 |              0.584 |               0.597 |                     0.607 |
| PGS003400 |              0.546 |               0.569 |                     0.591 |

Everything is helpful: score, sex, and PCs.


Now, let's test the same model, except jointly fitting all scores using cross-validation to avoid overfitting.

```
import os, numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

d = pd.read_csv('../../arrays.sscore', sep='\t').set_index('#IID')
d.index = d.index.astype(str)
pgs = [c for c in d.columns if c.endswith('_AVG')]

NUM_PCS = 16
pc_cols = [f"PC{i}" for i in range(1, NUM_PCS+1)]
so = {"project": os.environ.get("GOOGLE_PROJECT"), "requester_pays": True} if os.environ.get("GOOGLE_PROJECT") else {}

pcs = pd.read_csv("gs://fc-aou-datasets-controlled/v8/wgs/short_read/snpindel/aux/ancestry/ancestry_preds.tsv",
                  sep="\t", storage_options=so, usecols=["research_id","pca_features"])
parse = lambda s, k=NUM_PCS: ([(float(x) if x!='' else np.nan) for x in str(s).strip()[1:-1].split(",")] + [np.nan]*k)[:k]
pc_df = (pd.DataFrame(pcs["pca_features"].apply(parse).to_list(), columns=pc_cols)
           .assign(person_id=pcs["research_id"].astype(str)).set_index("person_id"))

sex_df = (pd.read_csv("gs://fc-aou-datasets-controlled/v8/wgs/short_read/snpindel/aux/qc/genomic_metrics.tsv",
                      sep="\t", storage_options=so, usecols=["research_id","dragen_sex_ploidy"])
           .assign(person_id=lambda x: x["research_id"].astype(str)).set_index("person_id")["dragen_sex_ploidy"]
           .map({"XX":0,"XY":1}).to_frame("sex"))

covars = pc_df.join(sex_df, how="inner").dropna()
case_set = set(pd.Series(cases, dtype=str))

Xy = covars.join(d[pgs], how="inner").dropna()
y = Xy.index.to_series().isin(case_set).astype('i1').to_numpy()
X = Xy.to_numpy()

cv = StratifiedKFold(n_splits=15, shuffle=True, random_state=42)
mdl = make_pipeline(StandardScaler(), LogisticRegression(solver="lbfgs", max_iter=1000))
p = cross_val_predict(mdl, X, y, cv=cv, method="predict_proba")[:, 1]
auc = roc_auc_score(y, p)

pd.DataFrame([{
    "n": int(len(y)),
    "cases": int(y.sum()),
    "controls": int(len(y) - y.sum()),
    "predictors": int(X.shape[1]),
    "AUROC": float(auc)
}])
```

This yields an AUROC of 0.646124.

But do score ensembles really improve the result?

Let's compare:
```
import numpy as np, pandas as pd, math
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

d = pd.read_csv('../../arrays.sscore', sep='\t').set_index('#IID')
d.index = d.index.astype(str)
pgs_cols = [c for c in d.columns if c.endswith('_AVG')]
idx_all = d[pgs_cols].dropna().index
X_all = d.loc[idx_all, pgs_cols].astype(float)
X_507 = d.loc[idx_all, ['PGS005198_AVG']].astype(float)
y = pd.Index(idx_all).to_series().isin(pd.Series(cases, dtype=str)).astype('i1').to_numpy()

cv = StratifiedKFold(n_splits=15, shuffle=True, random_state=42)
pipe = lambda: make_pipeline(StandardScaler(), LogisticRegression(solver="lbfgs", max_iter=1000))

p_all = np.zeros(len(y)); p_507 = np.zeros(len(y))
auc_all_f, auc_507_f = [], []

for tr, te in cv.split(X_all, y):
    m_all = pipe(); m_507 = pipe()
    m_all.fit(X_all.iloc[tr], y[tr]); m_507.fit(X_507.iloc[tr], y[tr])
    p_all[te] = m_all.predict_proba(X_all.iloc[te])[:,1]
    p_507[te] = m_507.predict_proba(X_507.iloc[te])[:,1]
    auc_all_f.append(roc_auc_score(y[te], p_all[te]))
    auc_507_f.append(roc_auc_score(y[te], p_507[te]))

auc_all = roc_auc_score(y, p_all)
auc_507 = roc_auc_score(y, p_507)

perf = pd.DataFrame([
    {"model":"All AVG", "n":len(y), "cases":int(y.sum()), "controls":int(len(y)-y.sum()), "AUROC":float(auc_all)},
    {"model":"PGS005198 only", "n":len(y), "cases":int(y.sum()), "controls":int(len(y)-y.sum()), "AUROC":float(auc_507)},
]).sort_values("AUROC", ascending=False).reset_index(drop=True)

diff =  np.array(auc_all_f) - np.array(auc_507_f)
nz = diff[diff!=0]
s = int((nz>0).sum()); n = int(len(nz))
p_one_sided = sum(math.comb(n,k) for k in range(s, n+1)) / (2**n) if n>0 else np.nan

print(perf.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
print(f"One-sided sign-test p (PGS005198 < All AVG): {p_one_sided:.3g} over {s}/{n} folds; ΔAUROC (OOF) = {auc_all-auc_507:+.3f}")
```

We see that the AUC of the ensemble (0.622) is significantly better than that of the single score (0.603).


Let's plot risk in each decline of the multivariate model.
```
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold

d = pd.read_csv('../../arrays.sscore', sep='\t').set_index('#IID')
d.index = d.index.astype(str)
pgs_cols = [c for c in d.columns if c.endswith('_AVG')]
X_all = d[pgs_cols].dropna().astype(float)
X_507 = d[['PGS000507_AVG']].dropna().astype(float)

NUM_PCS = 16
pc_cols = [f"PC{i}" for i in range(1, NUM_PCS+1)]
so = {"project": os.environ.get("GOOGLE_PROJECT"), "requester_pays": True} if os.environ.get("GOOGLE_PROJECT") else {}
pcs = pd.read_csv("gs://fc-aou-datasets-controlled/v8/wgs/short_read/snpindel/aux/ancestry/ancestry_preds.tsv",
                  sep="\t", storage_options=so, usecols=["research_id","pca_features"])
parse = lambda s, k=NUM_PCS: ([(float(x) if x!='' else np.nan) for x in str(s).strip()[1:-1].split(",")] + [np.nan]*k)[:k]
pc_df = (pd.DataFrame(pcs["pca_features"].apply(parse).to_list(), columns=pc_cols)
           .assign(person_id=pcs["research_id"].astype(str)).set_index("person_id")).dropna()
X_spc = X_all.join(pc_df, how="inner").dropna()

sex = (pd.read_csv("gs://fc-aou-datasets-controlled/v8/wgs/short_read/snpindel/aux/qc/genomic_metrics.tsv",
                   sep="\t", storage_options=so, usecols=["research_id","dragen_sex_ploidy"])
         .assign(person_id=lambda x: x["research_id"].astype(str)).set_index("person_id")["dragen_sex_ploidy"])

case_idx = pd.Index(pd.Series(cases, dtype=str).unique())

def oof(X, y):
    cv = StratifiedKFold(n_splits=15, shuffle=True, random_state=42)
    m = make_pipeline(StandardScaler(), LogisticRegression(solver="lbfgs", max_iter=1000))
    p = np.zeros(len(y))
    for tr, te in cv.split(X, y):
        m.fit(X.iloc[tr], y[tr])
        p[te] = m.predict_proba(X.iloc[te])[:,1]
    return p

def deciles_obs(p, y, q=10):
    df = pd.DataFrame({"p": p, "y": y})
    bins = pd.qcut(df["p"], q=q, labels=False, duplicates="drop")
    g = df.assign(decile=bins).groupby("decile", observed=True)
    r = (g["y"].mean()).to_frame("rate")
    r.index = r.index.astype(int) + 1
    return r

# Women: three lines (observed only)
w_ix = sex[sex=="XX"].index
ix_507 = X_507.index.intersection(w_ix)
ix_all = X_all.index.intersection(w_ix)
ix_spc = X_spc.index.intersection(w_ix)

y_507 = pd.Index(ix_507).to_series().isin(case_idx).astype('i1').to_numpy()
y_all = pd.Index(ix_all).to_series().isin(case_idx).astype('i1').to_numpy()
y_spc = pd.Index(ix_spc).to_series().isin(case_idx).astype('i1').to_numpy()

p_507 = oof(X_507.loc[ix_507], y_507)
p_all = oof(X_all.loc[ix_all], y_all)
p_spc = oof(X_spc.loc[ix_spc], y_spc)

d_507 = deciles_obs(p_507, y_507)
d_all = deciles_obs(p_all, y_all)
d_spc = deciles_obs(p_spc, y_spc)

plt.figure()
plt.plot(d_507.index, d_507["rate"], marker="o", label="PGS000507 only")
plt.plot(d_all.index, d_all["rate"], marker="o", label="All scores")
plt.plot(d_spc.index, d_spc["rate"], marker="o", label="Scores + PCs")
plt.xlabel("Predicted risk decile (women)")
plt.ylabel("Observed case rate")
plt.title(f"Deciles (observed only) — Women  |  n={len(ix_all)}")
plt.legend()
plt.tight_layout()

# Men: bar chart bottom 90% vs top 10% using Scores+PCs ensemble
m_ix = sex[sex=="XY"].index
ix_spc_m = X_spc.index.intersection(m_ix)
y_m = pd.Index(ix_spc_m).to_series().isin(case_idx).astype('i1').to_numpy()
p_spc_m = oof(X_spc.loc[ix_spc_m], y_m)
thr = np.quantile(p_spc_m, 0.9)
top = y_m[p_spc_m >= thr]
bot = y_m[p_spc_m < thr]

rates = [bot.mean() if len(bot) else np.nan, top.mean() if len(top) else np.nan]
counts = [len(bot), len(top)]

plt.figure()
plt.bar(["Bottom 90%", "Top 10%"], rates)
plt.ylabel("Observed case rate")
plt.title(f"Scores + PCs — Men  |  n={len(y_m)}  |  n90={counts[0]}, n10={counts[1]}")
plt.tight_layout()
```

<img width="630" height="470" alt="image" src="https://github.com/user-attachments/assets/d8a3a4b0-63f9-472c-9cdf-c2028780e910" />

<img width="630" height="470" alt="image" src="https://github.com/user-attachments/assets/5e6f0368-a946-4291-ac18-c54f044f25c2" />

We can see that the ensemble helps, and also adding PCs help.

**Note:** we do not include age at all, which adds noise to the results.

Let's check some final metrics.

```
import os, numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# data
d = pd.read_csv('../../arrays.sscore', sep='\t').set_index('#IID')
d.index = d.index.astype(str)
pgs = [c for c in d.columns if c.endswith('_AVG')]

NUM_PCS = 16
pc_cols = [f"PC{i}" for i in range(1, NUM_PCS+1)]
so = {"project": os.environ.get("GOOGLE_PROJECT"), "requester_pays": True} if os.environ.get("GOOGLE_PROJECT") else {}

pcs = pd.read_csv("gs://fc-aou-datasets-controlled/v8/wgs/short_read/snpindel/aux/ancestry/ancestry_preds.tsv",
                  sep="\t", storage_options=so, usecols=["research_id","pca_features"])
parse = lambda s, k=NUM_PCS: ([(float(x) if x!='' else np.nan) for x in str(s).strip()[1:-1].split(",")] + [np.nan]*k)[:k]
pc_df = (pd.DataFrame(pcs["pca_features"].apply(parse).to_list(), columns=pc_cols)
           .assign(person_id=pcs["research_id"].astype(str)).set_index("person_id")).dropna()

sex = (pd.read_csv("gs://fc-aou-datasets-controlled/v8/wgs/short_read/snpindel/aux/qc/genomic_metrics.tsv",
                   sep="\t", storage_options=so, usecols=["research_id","dragen_sex_ploidy"])
         .assign(person_id=lambda x: x["research_id"].astype(str)).set_index("person_id")["dragen_sex_ploidy"])

# women only; predictors = all scores + PCs
X = d[pgs].join(pc_df, how="inner").dropna()
idx_w = X.index.intersection(sex[sex=="XX"].index)
Xw = X.loc[idx_w].astype(float)
y = pd.Index(idx_w).to_series().isin(pd.Series(cases, dtype=str)).astype('i1').to_numpy()

# 10-fold CV, OOF predictions
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
mdl = make_pipeline(StandardScaler(), LogisticRegression(solver="lbfgs", max_iter=1000))
p = np.zeros(len(y))
for tr, te in cv.split(Xw, y):
    mdl.fit(Xw.iloc[tr], y[tr])
    p[te] = mdl.predict_proba(Xw.iloc[te])[:,1]

# metrics
auc = roc_auc_score(y, p)
z = (p - p.mean()) / p.std(ddof=0)
beta = LogisticRegression(solver="lbfgs", max_iter=1000).fit(z.reshape(-1,1), y).coef_[0,0]
or_per_sd = float(np.exp(beta))
base = y.mean()
thr = min(2*base, 0.999)
frac_2x = float((p >= thr).mean())

pd.DataFrame([{
    "n_women": int(len(y)),
    "cases": int(y.sum()),
    "controls": int(len(y)-y.sum()),
    "AUROC": float(auc),
    "OR_per_SD_pred": or_per_sd,
    "fraction_women_at_≥2x_risk": frac_2x,
    "baseline_risk": float(base),
    "2x_threshold": float(thr)
}])
```

In women, PGS ensemble + PCs:
- 0.657404 AUROC
- 1.539613 OR/SD
- 6.2% at 2x risk

Let's check ICD code heterogeneity:

<img width="989" height="390" alt="image" src="https://github.com/user-attachments/assets/93b62d4d-49a0-4301-bc0b-fafd69844fba" />

<img width="989" height="390" alt="image" src="https://github.com/user-attachments/assets/d5397d26-b530-4378-bf8a-7b71bb2cd577" />

Let's check accuracy within each inferred genetic ancestry group:
```
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt, statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# data
d = pd.read_csv('../../arrays.sscore', sep='\t').set_index('#IID'); d.index = d.index.astype(str)
pgs_cols = [c for c in d.columns if c.endswith('_AVG')]

NUM_PCS = 16
pc_cols = [f"PC{i}" for i in range(1, NUM_PCS+1)]
PCS_URI = "gs://fc-aou-datasets-controlled/v8/wgs/short_read/snpindel/aux/ancestry/ancestry_preds.tsv"
SEX_URI = "gs://fc-aou-datasets-controlled/v8/wgs/short_read/snpindel/aux/qc/genomic_metrics.tsv"
so = {"project": os.environ.get("GOOGLE_PROJECT"), "requester_pays": True} if os.environ.get("GOOGLE_PROJECT") else {}

pcs_raw = pd.read_csv(PCS_URI, sep="\t", storage_options=so, usecols=["research_id","pca_features","ancestry_pred"])
parse = lambda s, k=NUM_PCS: ([(float(x) if x!='' else np.nan) for x in str(s).strip()[1:-1].split(",")] + [np.nan]*k)[:k]
pc_df = (pd.DataFrame(pcs_raw["pca_features"].apply(parse).to_list(), columns=pc_cols)
           .assign(person_id=pcs_raw["research_id"].astype(str), ancestry_pred=pcs_raw["ancestry_pred"])
           .set_index("person_id"))

sex = (pd.read_csv(SEX_URI, sep="\t", storage_options=so, usecols=["research_id","dragen_sex_ploidy"])
         .assign(person_id=lambda x: x["research_id"].astype(str))
         .set_index("person_id")["dragen_sex_ploidy"])

women_ids = d.index.intersection(sex[sex=="XX"].index)
case_set = set(pd.Series(cases, dtype=str))

X = d.loc[women_ids, pgs_cols].join(pc_df[pc_cols], how="inner").dropna()
y = pd.Index(X.index).to_series().isin(case_set).astype('i1').to_numpy()
anc = pc_df.loc[X.index, "ancestry_pred"]

# 10-fold OOF probabilities
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
clf = make_pipeline(StandardScaler(), LogisticRegression(solver="lbfgs", C=1e6, max_iter=2000))
p = np.zeros(len(y))
for tr, te in cv.split(X.values, y):
    clf.fit(X.iloc[tr], y[tr]); p[te] = clf.predict_proba(X.iloc[te])[:,1]

def auc_ci_boot(y, p, B=500, rng=123):
    rs = np.random.RandomState(rng)
    y = np.asarray(y); p = np.asarray(p)
    pos, neg = np.where(y==1)[0], np.where(y==0)[0]
    aucs = []
    for _ in range(B):
        idx = np.concatenate([rs.choice(pos, len(pos), True), rs.choice(neg, len(neg), True)])
        aucs.append(roc_auc_score(y[idx], p[idx]))
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return float(roc_auc_score(y, p)), float(lo), float(hi)

def orsd_ci(y, p):
    z = (p - p.mean()) / p.std(ddof=0)
    m = sm.Logit(y, sm.add_constant(z)).fit(disp=False, maxiter=200)
    beta, se = float(m.params[1]), float(m.bse[1])
    or_sd = np.exp(beta)
    return float(or_sd), float(np.exp(beta-1.96*se)), float(np.exp(beta+1.96*se))

rows = []
for anc_val, idx in anc.groupby(anc).groups.items():
    idx = pd.Index(idx)
    y_g = pd.Series(y, index=X.index).loc[idx].to_numpy()
    p_g = pd.Series(p, index=X.index).loc[idx].to_numpy()
    if y_g.sum()==0 or y_g.sum()==len(y_g) or p_g.std(ddof=0)==0: 
        continue
    auc, auc_lo, auc_hi = auc_ci_boot(y_g, p_g)
    orsd, or_lo, or_hi = orsd_ci(y_g, p_g)
    rows.append({"ancestry": str(anc_val), "n_women": int(len(y_g)), "cases": int(y_g.sum()),
                 "AUROC": auc, "AUROC_lo": auc_lo, "AUROC_hi": auc_hi,
                 "OR_per_SD": orsd, "OR_lo": or_lo, "OR_hi": or_hi})

res = pd.DataFrame(rows).sort_values("AUROC", ascending=False).reset_index(drop=True)
print(res.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

# two plots: AUROC and OR/SD with 95% CI
anc_order = res["ancestry"].tolist()
x = np.arange(len(anc_order)); w = 0.6

plt.figure(figsize=(9,5))
yv = res.set_index("ancestry").loc[anc_order, "AUROC"].to_numpy()
lo = yv - res.set_index("ancestry").loc[anc_order, "AUROC_lo"].to_numpy()
hi = res.set_index("ancestry").loc[anc_order, "AUROC_hi"].to_numpy() - yv
plt.bar(x, yv, width=w, yerr=[lo, hi], capsize=4)
plt.xticks(x, anc_order, rotation=25, ha="right"); plt.ylabel("AUROC")
plt.title("Women — AUROC by ancestry (Scores + PCs, 10-fold OOF)"); plt.tight_layout()

plt.figure(figsize=(9,5))
yv = res.set_index("ancestry").loc[anc_order, "OR_per_SD"].to_numpy()
lo = yv - res.set_index("ancestry").loc[anc_order, "OR_lo"].to_numpy()
hi = res.set_index("ancestry").loc[anc_order, "OR_hi"].to_numpy() - yv
plt.bar(x, yv, width=w, yerr=[lo, hi], capsize=4)
plt.xticks(x, anc_order, rotation=25, ha="right"); plt.ylabel("OR per SD of prediction")
plt.title("Women — OR/SD by ancestry (Scores + PCs, 10-fold OOF)"); plt.tight_layout()

res
```

<img width="889" height="490" alt="image" src="https://github.com/user-attachments/assets/7ddcf049-1906-47ba-bf8a-7318b0738e36" />

<img width="889" height="490" alt="image" src="https://github.com/user-attachments/assets/cb2253bf-59b2-4415-a3a2-70ace6713e64" />
