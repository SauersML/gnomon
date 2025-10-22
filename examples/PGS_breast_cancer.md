## Running a polygenic score for breast cancer on the _All of Us_ cohort

First, let's see which colorectal cancer scores exist in the PGS Catalog:

[https://www.pgscatalog.org/trait/MONDO_0005575/
](https://www.pgscatalog.org/trait/MONDO_0007254/)

Strong performers include:
- PGS000332
- PGS000335
- PGS000015
- PGS000508
- PGS000344
- PGS000317
- PGS000007
- PGS000507
- PGS004869

We'll assess these scores on three metrics in the test data:
- Odds ratio (per standard deviation of score)
- AUROC
- Nagelkerke's R²

`gs://fc-aou-datasets-controlled/v8/wgs/short_read/snpindel/acaf_threshold/plink_bed` is 10.51 TiB, so we can use `gs://fc-aou-datasets-controlled/v8/microarray/plink/` instead.

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
./gnomon/target/release/gnomon score "PGS000332,PGS000015,PGS000508,PGS000344,PGS000317,PGS000007,PGS000507,PGS004869" arrays
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
#IID	PGS000007_AVG	PGS000007_MISSING_PCT	PGS000015_AVG	PGS000015_MISSING_PCT	PGS000317_AVG	PGS000317_MISSING_PCT	PGS000332_AVG	PGS000332_MISSING_PCT	PGS000344_AVG	PGS000344_MISSING_PCT	PGS000507_AVG	PGS000507_MISSING_PCT	PGS000508_AVG	PGS000508_MISSING_PCT	PGS004869_AVG	PGS004869_MISSING_PCT
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

<img width="780" height="758" alt="image" src="https://github.com/user-attachments/assets/ecbf0e0e-1e08-4401-8534-0e3879d2f037" />

Let's make the case definition.





```
import os
cdr_id = os.environ["WORKSPACE_CDR"]
from google.cloud import bigquery as bq

# ICD-10: C50.0–C50.9 (malignant), D05.0–D05.9 (in situ), Z85.3 (personal history)
icd10_breast = [f"C50.{i}" for i in range(10)] + [f"D05.{i}" for i in range(10)] + ["Z85.3"]

# ICD-9: 174.0–174.9 (female), 175.0–175.9 (male), 233.0 (in situ), V10.3 (personal history)
icd9_breast = [f"174.{i}" for i in range(10)] + [f"175.{i}" for i in range(10)] + ["233.0", "V10.3"]

codes = icd10_breast + icd9_breast
codes = [c.upper() for c in (codes + [c.replace(".", "") for c in codes])]

sql = f"""
SELECT DISTINCT CAST(person_id AS STRING) AS person_id
FROM `{cdr_id}.condition_occurrence`
WHERE condition_source_value IS NOT NULL
  AND UPPER(TRIM(condition_source_value)) IN UNNEST(@codes)
"""

cases = (
    bq.Client()
    .query(sql, job_config=bq.QueryJobConfig(query_parameters=[bq.ArrayQueryParameter("codes", "STRING", codes)]))
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

| Score         |       N | Cases | Controls | OR per SD | P (OR>1) | AUROC | P (AUC>0.5) | Nagelkerke R² | P (R²>0) |
| ------------- | ------: | ----: | -------: | --------: | -------: | ----: | ----------: | ------------: | -------: |
| PGS000007_AVG | 447,278 | 6,614 |  440,664 |     1.275 |  <1e-10 | 0.569 |    3.49e-82 |         0.61% |  <1e-10 |
| PGS000317_AVG | 447,278 | 6,614 |  440,664 |     1.269 |  <1e-10 | 0.567 |    1.30e-79 |         0.57% |  <1e-10 |
| PGS004869_AVG | 447,278 | 6,614 |  440,664 |     1.201 |  <1e-10 | 0.552 |    1.57e-47 |         0.34% |  <1e-10 |
| PGS000507_AVG | 447,278 | 6,614 |  440,664 |     1.179 |  <1e-10 | 0.547 |    1.68e-40 |         0.28% |  <1e-10 |
| PGS000344_AVG | 447,278 | 6,614 |  440,664 |     1.152 |  <1e-10 | 0.540 |    1.29e-29 |         0.20% |  <1e-10 |
| PGS000508_AVG | 447,278 | 6,614 |  440,664 |     1.151 |  <1e-10 | 0.542 |    2.69e-32 |         0.20% |  <1e-10 |
| PGS000015_AVG | 447,278 | 6,614 |  440,664 |     1.150 |  <1e-10 | 0.539 |    5.96e-28 |         0.20% |  <1e-10 |
| PGS000332_AVG | 447,278 | 6,614 |  440,664 |     1.131 |  <1e-10 | 0.537 |    4.62e-25 |         0.15% |  <1e-10 |

PGS000007 is the best so far. Let's add 16 principal components and sex to the model as predictors (not controls) and look at the resulting AUC.

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


| Score     |      nₛ | casesₛ | AUROCₛ |  ACCₛ |     nₛₓ | casesₛₓ | AUROCₛₓ | ACCₛₓ |   nₛₓₚ꜀ | casesₛₓₚ꜀ | AUROCₛₓₚ꜀ | ACCₛₓₚ꜀ |
| --------- | ------: | -----: | -----: | ----: | ------: | ------: | ------: | ----: | ------: | --------: | --------: | ------: |
| PGS000507 | 447,278 |  6,614 |  0.547 | 0.985 | 413,337 |   6,210 |   0.721 | 0.985 | 413,337 |     6,210 |     0.785 |   0.985 |
| PGS000508 | 447,278 |  6,614 |  0.542 | 0.985 | 413,337 |   6,210 |   0.718 | 0.985 | 413,337 |     6,210 |     0.784 |   0.985 |
| PGS000332 | 447,278 |  6,614 |  0.537 | 0.985 | 413,337 |   6,210 |   0.715 | 0.985 | 413,337 |     6,210 |     0.783 |   0.985 |
| PGS004869 | 447,278 |  6,614 |  0.552 | 0.985 | 413,337 |   6,210 |   0.724 | 0.985 | 413,337 |     6,210 |     0.779 |   0.985 |
| PGS000317 | 447,278 |  6,614 |  0.567 | 0.985 | 413,337 |   6,210 |   0.733 | 0.985 | 413,337 |     6,210 |     0.778 |   0.985 |
| PGS000007 | 447,278 |  6,614 |  0.569 | 0.985 | 413,337 |   6,210 |   0.735 | 0.985 | 413,337 |     6,210 |     0.776 |   0.985 |
| PGS000015 | 447,278 |  6,614 |  0.539 | 0.985 | 413,337 |   6,210 |   0.715 | 0.985 | 413,337 |     6,210 |     0.776 |   0.985 |
| PGS000344 | 447,278 |  6,614 |  0.540 | 0.985 | 413,337 |   6,210 |   0.717 | 0.985 | 413,337 |     6,210 |     0.774 |   0.985 |

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

This yields an AUROC of 0.787221, much better!

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
X_507 = d.loc[idx_all, ['PGS000507_AVG']].astype(float)
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
    {"model":"PGS000507 only", "n":len(y), "cases":int(y.sum()), "controls":int(len(y)-y.sum()), "AUROC":float(auc_507)},
]).sort_values("AUROC", ascending=False).reset_index(drop=True)

diff =  np.array(auc_all_f) - np.array(auc_507_f)
nz = diff[diff!=0]
s = int((nz>0).sum()); n = int(len(nz))
p_one_sided = sum(math.comb(n,k) for k in range(s, n+1)) / (2**n) if n>0 else np.nan

print(perf.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
print(f"One-sided sign-test p (PGS000507 < All AVG): {p_one_sided:.3g} over {s}/{n} folds; ΔAUROC (OOF) = {auc_all-auc_507:+.3f}")
```

We see that the AUC of the ensemble (0.588) is significantly better than that of the single score (0.547).




Let's plot risk in each decline of the multivariate model.
```
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

d=pd.read_csv('../../arrays.sscore',sep='\t').set_index('#IID'); d.index=d.index.astype(str)
pgs=[c for c in d.columns if c.endswith('_AVG') and c!='PGS004303_AVG']

if 'covars' not in globals():
    NUM_PCS=16; pc_cols=[f"PC{i}" for i in range(1,NUM_PCS+1)]
    so={"project": os.environ.get("GOOGLE_PROJECT"), "requester_pays": True} if os.environ.get("GOOGLE_PROJECT") else {}
    pcs=pd.read_csv("gs://fc-aou-datasets-controlled/v8/wgs/short_read/snpindel/aux/ancestry/ancestry_preds.tsv",sep="\t",storage_options=so,usecols=["research_id","pca_features"])
    parse=lambda s,k=NUM_PCS: ([(float(x) if x!='' else np.nan) for x in str(s).strip()[1:-1].split(",")]+[np.nan]*k)[:k]
    pc_df=pd.DataFrame(pcs["pca_features"].apply(parse).to_list(),columns=pc_cols).assign(person_id=pcs["research_id"].astype(str)).set_index("person_id")
    sex=pd.read_csv("gs://fc-aou-datasets-controlled/v8/wgs/short_read/snpindel/aux/qc/genomic_metrics.tsv",sep="\t",storage_options=so,usecols=["research_id","dragen_sex_ploidy"])
    sex_df=sex.assign(person_id=sex["research_id"].astype(str)).set_index("person_id")["dragen_sex_ploidy"].map({"XX":0,"XY":1}).to_frame("sex").dropna()
    covars=pc_df.join(sex_df,how="inner").filter(pc_cols+["sex"]).dropna()
covars=covars.copy(); covars.index=covars.index.astype(str)

if 'cases' not in globals(): raise NameError("Variable 'cases' not found.")
case_set=set(pd.Series(cases,dtype=str))

Xy=covars.join(d[pgs],how="inner").dropna()
y=Xy.index.to_series().isin(case_set).astype('i1').to_numpy()
X=Xy.to_numpy()

cv=StratifiedKFold(n_splits=15,shuffle=True,random_state=42)
mdl=make_pipeline(StandardScaler(),LogisticRegression(solver="lbfgs",max_iter=1000))
p=cross_val_predict(mdl,X,y,cv=cv,method="predict_proba")[:,1]
auc=roc_auc_score(y,p)
print(pd.DataFrame([{"n":int(len(y)),"cases":int(y.sum()),"controls":int(len(y)-y.sum()),"predictors":X.shape[1],"AUROC":float(auc)}]).to_string(index=False))

dec=pd.qcut(p,10,labels=False,duplicates='drop')+1
g=pd.DataFrame({"decile":dec,"y":y,"p":p}).groupby("decile").agg(n=("y","size"),cases=("y","sum"),obs_risk=("y","mean"),pred_risk=("p","mean")).reset_index()
print(g[["decile","n","cases","obs_risk","pred_risk"]].to_string(index=False,float_format=lambda x:f"{x:.4f}"))

plt.figure(figsize=(7,4.5))
plt.bar(g["decile"],g["obs_risk"],width=0.9,label="Observed",color="#4C78A8")
plt.plot(g["decile"],g["pred_risk"],marker="o",linewidth=2,label="Predicted",color="#F58518")
plt.xlabel("Risk decile (low → high)"); plt.ylabel("Event rate"); plt.title("Decile risk: observed vs predicted")
plt.xticks(g["decile"]); plt.legend(); plt.tight_layout(); plt.show()
```

<img width="690" height="440" alt="image" src="https://github.com/user-attachments/assets/d5e0e3ac-2028-4553-be52-5cc328ef6b94" />

Let's get the per-code OR/SD:

```
import os, numpy as np, pandas as pd, statsmodels.api as sm, matplotlib.pyplot as plt
from scipy.stats import norm
from google.cloud import bigquery as bq

cdr_id=os.environ["WORKSPACE_CDR"]
pgs_col="PGS003852_AVG"

cond_codes=[f"C18.{i}" for i in range(10)]+["C19","C20"]+[f"153.{i}" for i in range(10)]+["154.0","154.1"]
cond_codes=[c.upper() for c in (cond_codes+[c.replace(".","") for c in cond_codes])]
cond_codes_n=[c.replace(".","") for c in cond_codes]
codes_exact=["Z85.038","V10.05"]; codes_exact=[c.upper() for c in (codes_exact+[c.replace(".","") for c in codes_exact])]

sql=f"""
WITH obs_raw AS (
  SELECT DISTINCT CAST(person_id AS STRING) person_id,
         REGEXP_REPLACE(UPPER(TRIM(observation_source_value)),'[^A-Z0-9]','') AS code_n
  FROM `{cdr_id}.observation`
  WHERE observation_source_value IS NOT NULL
    AND (
      UPPER(TRIM(observation_source_value)) IN UNNEST(@codes_exact)
      OR REGEXP_REPLACE(UPPER(TRIM(observation_source_value)),'[^A-Z0-9]','') IN UNNEST(@codes_exact)
      OR STARTS_WITH(REGEXP_REPLACE(UPPER(TRIM(observation_source_value)),'[^A-Z0-9]',''),'Z8503')
    )
),
obs AS (
  SELECT person_id,
         CASE
           WHEN code_n='V1005' THEN 'V10.05'
           WHEN code_n='Z85038' THEN 'Z85.038'
           WHEN REGEXP_CONTAINS(code_n, r'^Z8503[0-9]$') THEN CONCAT('Z85.03', SUBSTR(code_n,6,1))
           ELSE code_n
         END AS code
  FROM obs_raw
),
cond_raw AS (
  SELECT DISTINCT CAST(person_id AS STRING) person_id,
         REGEXP_REPLACE(UPPER(TRIM(condition_source_value)),'[^A-Z0-9]','') AS code_n
  FROM `{cdr_id}.condition_occurrence`
  WHERE condition_source_value IS NOT NULL
    AND (
      UPPER(TRIM(condition_source_value)) IN UNNEST(@cond_codes)
      OR REGEXP_REPLACE(UPPER(TRIM(condition_source_value)),'[^A-Z0-9]','') IN UNNEST(@cond_codes_n)
    )
),
cond AS (
  SELECT person_id,
         CASE
           WHEN REGEXP_CONTAINS(code_n, r'^C18[0-9]$') THEN CONCAT('C18.', SUBSTR(code_n,4,1))
           WHEN code_n='C19' THEN 'C19'
           WHEN code_n='C20' THEN 'C20'
           WHEN REGEXP_CONTAINS(code_n, r'^153[0-9]$') THEN CONCAT('153.', SUBSTR(code_n,4,1))
           WHEN code_n IN ('1540','1541') THEN CONCAT('154.', SUBSTR(code_n,4,1))
           ELSE code_n
         END AS code
  FROM cond_raw
),
all_codes AS (
  SELECT 'OBS' AS src, code, person_id FROM obs WHERE code IS NOT NULL
  UNION ALL
  SELECT 'COND' AS src, code, person_id FROM cond WHERE code IS NOT NULL
)
SELECT src, code, person_id FROM all_codes
"""
cp=bq.Client().query(sql, job_config=bq.QueryJobConfig(query_parameters=[
    bq.ArrayQueryParameter("codes_exact","STRING",codes_exact),
    bq.ArrayQueryParameter("cond_codes","STRING",cond_codes),
    bq.ArrayQueryParameter("cond_codes_n","STRING",cond_codes_n),
])).to_dataframe()

df=pd.read_csv('../../arrays.sscore', sep='\t')
idcol=next((c for c in ['#IID','IID','person_id','research_id','sample_id','ID'] if c in df.columns), None)
if idcol is None: raise ValueError("ID column not found in ../../arrays.sscore")
df[idcol]=df[idcol].astype(str); df=df.set_index(idcol)
x=pd.to_numeric(df[pgs_col], errors='coerce'); x=x.dropna(); ids=pd.Index(x.index.astype(str))

rows=[]
for code in sorted(cp['code'].unique()):
    pid=set(cp.loc[cp['code']==code, 'person_id'].astype(str))
    idx=ids
    y=idx.to_series().isin(pid).astype(int).to_numpy()
    xv=x.reindex(idx).to_numpy()
    m=float(np.nanmean(xv)); s=float(np.nanstd(xv))
    if not np.isfinite(s) or s==0: continue
    z=(xv-m)/s
    n=len(z); n1=int(y.sum())
    if n1==0 or n1==n: continue
    try:
        X=sm.add_constant(pd.DataFrame({'z':z}))
        fit=sm.Logit(y, X).fit(disp=0, maxiter=200)
        beta=float(fit.params['z']); se=float(fit.bse['z'])
    except Exception:
        try:
            fit=sm.GLM(y, X, family=sm.families.Binomial()).fit()
            beta=float(fit.params['z']); se=float(fit.bse['z'])
        except Exception:
            continue
    OR=float(np.exp(beta))
    zstat=beta/se if se>0 else np.nan
    p=float(norm.sf(zstat)) if np.isfinite(zstat) else np.nan
    rows.append({'code':code,'n':n,'cases':n1,'OR_perSD':OR,'p_one_sided':p})

res=pd.DataFrame(rows)
res=res[res['n']>=30].dropna(subset=['OR_perSD']).sort_values('OR_perSD', ascending=True)

plt.figure(figsize=(9, max(3, 0.5*len(res))))
bars=plt.barh(res['code'], res['OR_perSD'])
xmax=float(res['OR_perSD'].max()*1.15) if len(res) else 1.0
plt.xlim(0, xmax)
for rect,pv in zip(bars, res['p_one_sided']):
    txt = f"p={pv:.1e}" if np.isfinite(pv) else "p=n/a"
    plt.text(rect.get_width()*1.01, rect.get_y()+rect.get_height()/2, txt, va="center")
plt.xlabel("OR per SD (PGS003852)"); plt.ylabel("ICD code"); plt.title("PGS003852 OR/SD by ICD code (n≥30)")
plt.tight_layout(); plt.show()
```

<img width="889" height="1339" alt="image" src="https://github.com/user-attachments/assets/3f9a820e-1632-471a-b2da-ac802d009401" />


