```
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from google.cloud import bigquery as bq
from sklearn.metrics import roc_auc_score
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

DISEASES = {
    'alzheimers': {
        'icd10': ['G30.0', 'G30.1', 'G30.8', 'G30.9'],
        'icd9': ['331.0'],
        'pgs': ['PGS004146', 'PGS004898', 'PGS003334']
    },
    'colorectal': {
        'icd10': [f'C18.{i}' for i in range(10)] + ['C19', 'C20'],
        'icd9': [f'153.{i}' for i in range(10)] + ['154.0', '154.1'],
        'obs_exact': ['Z85.038', 'V10.05'],
        'obs_prefix': 'Z8504',  # Z85.04x codes
        'pgs': ['PGS003852']
    },
    'obesity': {
        'icd10': ['E66.811', 'E66.812', 'E66.813', 'E66.01', 'E66.09', 
                  'E66.1', 'E66.2', 'E66.89', 'E66.9', 'E88.82'],
        'icd9': ['278.00', '278.01', '278.03'],
        'pgs': ['PGS005199', 'PGS004150']
    },
    'breast': {
        'icd10': [f'C50.{i}' for i in range(10)] + [f'D05.{i}' for i in range(10)] + ['Z85.3'],
        'icd9': [f'174.{i}' for i in range(10)] + [f'175.{i}' for i in range(10)] + ['233.0', 'V10.3'],
        'pgs': ['PGS000007', 'PGS000508', 'PGS000332']
    }
}

cdr_id = os.environ["WORKSPACE_CDR"]
google_project = os.getenv('GOOGLE_PROJECT')

print("="*80)
print("MULTI-DISEASE POLYGENIC RISK MODEL TRAINING AND EVALUATION")
print("="*80)

# ============================================================================
# LOAD BASE DATA
# ============================================================================

print("\n[1/7] Loading base genomic data...")

# Load genetic sex and metrics
import gcsfs
fs = gcsfs.GCSFileSystem(project=google_project, token="cloud", requester_pays=True)
with fs.open("gs://fc-aou-datasets-controlled/v8/wgs/short_read/snpindel/aux/qc/genomic_metrics.tsv", "rb") as f:
    sex_df = pd.read_csv(f, sep="\t")

# Load gnomon sex predictions
gnomon = pd.read_csv("../../arrays.sex.tsv", sep="\t")
gnomon['IID'] = gnomon['IID'].astype(str)
gnomon = gnomon.rename(columns={'Sex': 'gnomon_sex'})

# Load ancestry
with fs.open("gs://fc-aou-datasets-controlled/v8/wgs/short_read/snpindel/aux/ancestry/ancestry_preds.tsv", "rb") as f:
    ancestry_df = pd.read_csv(f, sep="\t")
ancestry_df = ancestry_df[["research_id", "ancestry_pred"]].copy()
ancestry_df.columns = ["person_id", "ancestry"]
ancestry_df["person_id"] = ancestry_df["person_id"].astype(str)

# Load age data
client = bq.Client()
yob = client.query(f"SELECT person_id, year_of_birth FROM `{cdr_id}.person`").to_dataframe()
yob["person_id"] = yob["person_id"].astype(str)

obs = client.query(f"""
    SELECT person_id, EXTRACT(YEAR FROM MAX(observation_period_end_date)) AS obs_end_year
    FROM `{cdr_id}.observation_period` GROUP BY person_id
""").to_dataframe()
obs["person_id"] = obs["person_id"].astype(str)

demo = yob.merge(obs, on="person_id", how="inner")
demo["year_of_birth"] = pd.to_numeric(demo["year_of_birth"], errors="coerce")
demo["age"] = (demo["obs_end_year"] - demo["year_of_birth"]).clip(lower=0, upper=120)

# Load polygenic scores
sscore = pd.read_csv("../../arrays.sscore", sep="\t")
iid_col = sscore.columns[0]
sscore = sscore.rename(columns={iid_col: 'IID'})
sscore['IID'] = sscore['IID'].astype(str)

# Find ID column for merging
id_col = next((c for c in ["research_id", "person_id", "sample_id"] if c in sex_df.columns), None)
sex_df[id_col] = sex_df[id_col].astype(str)

# Merge base data
base_df = sex_df[[id_col]].copy()
base_df = base_df.merge(gnomon, left_on=id_col, right_on='IID', how='inner')
base_df = base_df.merge(ancestry_df, left_on=id_col, right_on='person_id', how='left')
base_df = base_df.merge(demo[['person_id', 'age']], left_on=id_col, right_on='person_id', how='left')
base_df = base_df.merge(sscore, on='IID', how='left')
base_df['sex_binary'] = (base_df['gnomon_sex'].str.lower() == 'male').astype(int)
base_df['age_sq'] = base_df['age'] ** 2

print(f"  Base cohort: {len(base_df):,} individuals")
print(f"  With age: {base_df['age'].notna().sum():,}")
print(f"  With gnomon sex: {base_df['gnomon_sex'].notna().sum():,}")
print(f"  With ancestry: {base_df['ancestry'].notna().sum():,}")

# ============================================================================
# DEFINE CASES FOR EACH DISEASE
# ============================================================================

print("\n[2/7] Defining disease cases from EHR...")

def get_cases(disease_config):
    """Extract cases from BigQuery based on ICD codes"""
    
    # Handle standard condition codes
    if 'icd10' in disease_config or 'icd9' in disease_config:
        cond_codes = disease_config.get('icd10', []) + disease_config.get('icd9', [])
        cond_codes = [c.upper() for c in cond_codes]
        cond_codes_n = sorted(set([c.replace(".", "") for c in cond_codes]))
        
        cond_sql = f"""
        SELECT DISTINCT CAST(person_id AS STRING) AS person_id
        FROM `{cdr_id}.condition_occurrence`
        WHERE REGEXP_REPLACE(UPPER(TRIM(condition_source_value)), '[^A-Z0-9]', '') IN UNNEST(@codes_n)
        """
        
        cases_cond = client.query(
            cond_sql,
            job_config=bq.QueryJobConfig(
                query_parameters=[bq.ArrayQueryParameter("codes_n", "STRING", cond_codes_n)]
            )
        ).to_dataframe()['person_id'].astype(str)
    else:
        cases_cond = pd.Series([], dtype=str)
    
    # Handle observation codes (for colorectal)
    if 'obs_exact' in disease_config:
        obs_codes = disease_config['obs_exact']
        obs_codes = [c.upper() for c in obs_codes]
        obs_codes_n = sorted(set([c.replace(".", "") for c in obs_codes]))
        obs_prefix = disease_config.get('obs_prefix', '')
        
        obs_sql = f"""
        WITH obs_raw AS (
          SELECT DISTINCT CAST(person_id AS STRING) AS person_id,
                 REGEXP_REPLACE(UPPER(TRIM(observation_source_value)), '[^A-Z0-9]', '') AS code_n
          FROM `{cdr_id}.observation`
          WHERE observation_source_value IS NOT NULL
        )
        SELECT DISTINCT person_id
        FROM obs_raw
        WHERE code_n IN UNNEST(@obs_codes_n)
           OR STARTS_WITH(code_n, @obs_prefix)
        """
        
        cases_obs = client.query(
            obs_sql,
            job_config=bq.QueryJobConfig(
                query_parameters=[
                    bq.ArrayQueryParameter("obs_codes_n", "STRING", obs_codes_n),
                    bq.ScalarQueryParameter("obs_prefix", "STRING", obs_prefix)
                ]
            )
        ).to_dataframe()['person_id'].astype(str)
    else:
        cases_obs = pd.Series([], dtype=str)
    
    # Combine
    all_cases = pd.concat([cases_cond, cases_obs]).unique()
    return set(all_cases)

disease_cases = {}
for disease_name, config in DISEASES.items():
    cases = get_cases(config)
    disease_cases[disease_name] = cases
    print(f"  {disease_name}: {len(cases):,} cases")

# ============================================================================
# CREATE DISEASE-SPECIFIC DATAFRAMES
# ============================================================================

print("\n[3/7] Creating disease-specific datasets...")

disease_dfs = {}
for disease_name, config in DISEASES.items():
    df = base_df.copy()
    
    # Add case status
    df['case'] = df[id_col].isin(disease_cases[disease_name]).astype(int)
    
    # Add PGS columns (check if they exist)
    pgs_cols = [f"{pgs}_AVG" for pgs in config['pgs']]
    available_pgs = [col for col in pgs_cols if col in df.columns]
    
    if not available_pgs:
        print(f"  WARNING: No PGS columns found for {disease_name}")
        continue
    
    # Keep only necessary columns
    keep_cols = [id_col, 'IID', 'gnomon_sex', 'sex_binary', 'ancestry', 'age', 'age_sq', 'case'] + available_pgs
    df = df[keep_cols].copy()
    
    # Remove rows with missing critical data
    df = df.dropna(subset=['age', 'sex_binary'] + available_pgs)
    
    disease_dfs[disease_name] = {
        'data': df,
        'pgs_cols': available_pgs,
        'n_pgs': len(available_pgs)
    }
    
    print(f"  {disease_name}: {len(df):,} samples, {df['case'].sum():,} cases ({100*df['case'].mean():.2f}%)")

# ============================================================================
# TRAIN MODELS
# ============================================================================

print("\n[4/7] Training models...")

results = []

for disease_name, disease_info in disease_dfs.items():
    print(f"\n  {disease_name.upper()}")
    print("  " + "-"*60)
    
    df = disease_info['data']
    pgs_cols = disease_info['pgs_cols']
    
    # Split by ancestry
    df_eur = df[df['ancestry'] == 'eur'].copy()
    df_all = df.copy()
    
    for dataset_name, dataset in [('EUR', df_eur), ('ALL', df_all)]:
        print(f"    {dataset_name}: n={len(dataset):,}, cases={dataset['case'].sum():,}")
        
        # Standardize PGS scores using this dataset
        pgs_means = {}
        pgs_stds = {}
        for pgs_col in pgs_cols:
            pgs_means[pgs_col] = dataset[pgs_col].mean()
            pgs_stds[pgs_col] = dataset[pgs_col].std()
            dataset[f'{pgs_col}_std'] = (dataset[pgs_col] - pgs_means[pgs_col]) / pgs_stds[pgs_col]
        
        pgs_std_cols = [f'{col}_std' for col in pgs_cols]
        
        # Train baseline model (age + age² + sex)
        X_base = sm.add_constant(dataset[['age', 'age_sq', 'sex_binary']])
        y = dataset['case']
        model_base = sm.Logit(y, X_base).fit(disp=0)
        
        # Train full model (baseline + PGS)
        X_full = sm.add_constant(dataset[['age', 'age_sq', 'sex_binary'] + pgs_std_cols])
        model_full = sm.Logit(y, X_full).fit(disp=0)
        
        # Store results
        results.append({
            'disease': disease_name,
            'dataset': dataset_name,
            'model_type': 'baseline',
            'model': model_base,
            'pgs_means': pgs_means,
            'pgs_stds': pgs_stds,
            'pgs_cols': pgs_cols,
            'train_n': len(dataset),
            'train_cases': int(dataset['case'].sum())
        })
        
        results.append({
            'disease': disease_name,
            'dataset': dataset_name,
            'model_type': 'full',
            'model': model_full,
            'pgs_means': pgs_means,
            'pgs_stds': pgs_stds,
            'pgs_cols': pgs_cols,
            'train_n': len(dataset),
            'train_cases': int(dataset['case'].sum())
        })

print(f"\n  Trained {len(results)} models total")

# ============================================================================
# EVALUATE MODELS ON EACH ANCESTRY
# ============================================================================

print("\n[5/7] Evaluating models on each ancestry...")

eval_results = []

for result in results:
    disease_name = result['disease']
    dataset_name = result['dataset']
    model_type = result['model_type']
    model = result['model']
    pgs_means = result['pgs_means']
    pgs_stds = result['pgs_stds']
    pgs_cols = result['pgs_cols']
    
    df = disease_dfs[disease_name]['data']
    
    # Evaluate on each ancestry
    for ancestry in ['eur', 'afr', 'amr', 'eas', 'sas']:
        df_anc = df[df['ancestry'] == ancestry].copy()
        
        if len(df_anc) < 10 or df_anc['case'].sum() < 2:
            continue
        
        # Standardize PGS using training parameters
        for pgs_col in pgs_cols:
            df_anc[f'{pgs_col}_std'] = (df_anc[pgs_col] - pgs_means[pgs_col]) / pgs_stds[pgs_col]
        
        pgs_std_cols = [f'{col}_std' for col in pgs_cols]
        
        # Prepare features
        if model_type == 'baseline':
            X_eval = sm.add_constant(df_anc[['age', 'age_sq', 'sex_binary']])
        else:
            X_eval = sm.add_constant(df_anc[['age', 'age_sq', 'sex_binary'] + pgs_std_cols])
        
        # Predict
        y_true = df_anc['case']
        y_pred = model.predict(X_eval)
        
        # Calculate metrics
        if y_true.sum() > 0 and y_true.sum() < len(y_true):
            auc = roc_auc_score(y_true, y_pred)
        else:
            auc = np.nan
        
        # Calculate OR per SD for PGS (only for full models)
        ors_per_sd = {}
        if model_type == 'full':
            for pgs_std_col in pgs_std_cols:
                if pgs_std_col in model.params:
                    beta = model.params[pgs_std_col]
                    or_per_sd = np.exp(beta)
                    ors_per_sd[pgs_std_col] = or_per_sd
        
        eval_results.append({
            'disease': disease_name,
            'train_dataset': dataset_name,
            'model_type': model_type,
            'eval_ancestry': ancestry,
            'n': len(df_anc),
            'cases': int(y_true.sum()),
            'prevalence': y_true.mean(),
            'auc': auc,
            'ors_per_sd': ors_per_sd
        })

eval_df = pd.DataFrame(eval_results)

# ============================================================================
# PRINT COMPREHENSIVE STATISTICS
# ============================================================================

print("\n[6/7] Model performance summary...")
print("="*80)

for disease_name in disease_dfs.keys():
    print(f"\n{disease_name.upper()}")
    print("-"*80)
    
    disease_evals = eval_df[eval_df['disease'] == disease_name]
    
    for dataset_name in ['EUR', 'ALL']:
        for model_type in ['baseline', 'full']:
            print(f"\n  {dataset_name} {model_type.upper()} MODEL:")
            
            subset = disease_evals[
                (disease_evals['train_dataset'] == dataset_name) & 
                (disease_evals['model_type'] == model_type)
            ]
            
            for _, row in subset.iterrows():
                anc = row['eval_ancestry'].upper()
                n = row['n']
                cases = row['cases']
                prev = row['prevalence']
                auc = row['auc']
                
                print(f"    {anc:4s}: n={n:7,} cases={cases:6,} ({100*prev:5.2f}%) AUC={auc:.4f}")
                
                if model_type == 'full' and row['ors_per_sd']:
                    for pgs, or_val in row['ors_per_sd'].items():
                        print(f"          {pgs}: OR/SD={or_val:.4f}")

# Print model coefficients
print("\n" + "="*80)
print("MODEL COEFFICIENTS")
print("="*80)

for result in results:
    disease = result['disease']
    dataset = result['dataset']
    model_type = result['model_type']
    model = result['model']
    
    print(f"\n{disease.upper()} - {dataset} {model_type.upper()}")
    print("-"*60)
    
    for param, coef in model.params.items():
        pval = model.pvalues[param]
        print(f"  {param:20s}: β={coef:12.6f}  p={pval:.2e}  OR={np.exp(coef):.4f}")

# ============================================================================
# CREATE DECILE RISK PLOTS FOR EUR MODELS
# ============================================================================

print("\n[7/7] Creating decile risk plots for EUR models...")

n_diseases = len(disease_dfs)
fig, axes = plt.subplots(n_diseases, 2, figsize=(14, 4*n_diseases))
if n_diseases == 1:
    axes = axes.reshape(1, -1)

for idx, (disease_name, disease_info) in enumerate(disease_dfs.items()):
    df = disease_info['data']
    pgs_cols = disease_info['pgs_cols']
    
    df_eur = df[df['ancestry'] == 'eur'].copy()
    
    # Get EUR models
    baseline_result = [r for r in results if r['disease']==disease_name and r['dataset']=='EUR' and r['model_type']=='baseline'][0]
    full_result = [r for r in results if r['disease']==disease_name and r['dataset']=='EUR' and r['model_type']=='full'][0]
    
    for col_idx, (model_result, model_name) in enumerate([(baseline_result, 'Baseline'), (full_result, 'Full')]):
        model = model_result['model']
        pgs_means = model_result['pgs_means']
        pgs_stds = model_result['pgs_stds']
        
        # Standardize PGS
        for pgs_col in pgs_cols:
            df_eur[f'{pgs_col}_std'] = (df_eur[pgs_col] - pgs_means[pgs_col]) / pgs_stds[pgs_col]
        
        pgs_std_cols = [f'{col}_std' for col in pgs_cols]
        
        # Predict
        if model_name == 'Baseline':
            X = sm.add_constant(df_eur[['age', 'age_sq', 'sex_binary']])
        else:
            X = sm.add_constant(df_eur[['age', 'age_sq', 'sex_binary'] + pgs_std_cols])
        
        df_eur['risk'] = model.predict(X)
        
        # Create deciles
        df_eur['decile'] = pd.qcut(df_eur['risk'], 10, labels=False, duplicates='drop') + 1
        
        # Calculate risk per decile
        decile_stats = df_eur.groupby('decile').agg({
            'case': ['sum', 'count', 'mean']
        }).reset_index()
        decile_stats.columns = ['decile', 'cases', 'n', 'risk']
        
        # Plot
        ax = axes[idx, col_idx]
        ax.plot(decile_stats['decile'], decile_stats['risk']*100, 'o-', linewidth=2, markersize=8)
        ax.set_xlabel('Risk decile')
        ax.set_ylabel('Disease prevalence (%)')
        ax.set_title(f'{disease_name.upper()} - EUR {model_name}')
        ax.grid(alpha=0.3)
        ax.set_xticks(range(1, 11))

plt.tight_layout()
plt.savefig('../../disease_risk_deciles.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print(f"Trained and evaluated {len(results)} models across {len(disease_dfs)} diseases")
print(f"Results saved to: ../../disease_risk_deciles.png")
```

