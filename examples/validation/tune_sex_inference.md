Download data and run Gnomon to generate sex inference metrics:

```
%%bash
# Download microarray PLINK data if not present
if [ ! -f "../../arrays.bed" ]; then
    echo "Downloading microarray data..."
    gsutil -u "$GOOGLE_PROJECT" -m cp -r gs://fc-aou-datasets-controlled/v8/microarray/plink/* ../..
fi
```

Build `gnomon`:
```
%%bash
# Install Rust nightly
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && { for f in ~/.bashrc ~/.profile; do [ -f "$f" ] || touch "$f"; grep -qxF 'source "$HOME/.cargo/env"' "$f" || printf '\n# Rust / Cargo\nsource "$HOME/.cargo/env"\n' >> "$f"; done; } && source "$HOME/.cargo/env" && rustup toolchain install nightly && rustup default nightly
git clone https://github.com/SauersML/gnomon.git
cd gnomon
rustup override set nightly
cargo build --release
```

Run Gnomon sex inference to generate arrays.sex.tsv with metric columns:
```
!echo "Running Gnomon sex inference..."
!./gnomon/target/release/gnomon terms --sex ../../arrays.bed
```

Load data and visualize specific inference metrics against Self-Reported Sex and DRAGEN Ploidy:

```
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import gcsfs

# --- Configuration & Styling ---
sns.set_theme(style="whitegrid", context="notebook", font_scale=1.1)

# 1. Load Gnomon Results (Local)
gnomon_path = "../../arrays.sex.tsv"
if not os.path.exists(gnomon_path):
    raise FileNotFoundError(f"Could not find {gnomon_path}. Please run the Gnomon inference first.")

gnomon_df = pd.read_csv(gnomon_path, sep="\t")
gnomon_df["IID"] = gnomon_df["IID"].astype(str)

# 2. Load Truth Data (GCS)
project_id = os.environ.get("GOOGLE_PROJECT")
metrics_uri = "gs://fc-aou-datasets-controlled/v8/wgs/short_read/snpindel/aux/qc/genomic_metrics.tsv"

print(f"Loading metrics from {metrics_uri}...")
fs = gcsfs.GCSFileSystem(project=project_id, token="cloud", requester_pays=True)
with fs.open(metrics_uri, "rb") as f:
    metrics_df = pd.read_csv(f, sep="\t")

# 3. Merge Datasets
possible_ids = ["person_id", "sample_id", "participant_id"]
id_col = next((c for c in possible_ids if c in metrics_df.columns), None)
if not id_col:
    raise ValueError(f"Could not find ID column in metrics. Checked: {possible_ids}")

metrics_df[id_col] = metrics_df[id_col].astype(str)
merged = gnomon_df.merge(metrics_df, left_on="IID", right_on=id_col, how="inner")
print(f"Merged dataset size: {len(merged)} samples.")

# 4. Prepare for Plotting
# Filter for visualization clarity
plot_df = merged.dropna(subset=["sex_at_birth", "dragen_sex_ploidy"]).copy()

metric_cols = [
    "X_Het_Ratio", 
    "Y_Non_PAR", 
    "Y_PAR", 
    "SRY_Count",         
    "PAR_NonPAR_Ratio", 
    "Male_Votes", 
    "Female_Votes"
]

targets = ["sex_at_birth", "dragen_sex_ploidy"]

# Pre-convert metrics to numeric, coercing errors to NaN for clean plotting
for col in metric_cols:
    if col in plot_df.columns:
        plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")

# 5. Generate Plots
n_metrics = len(metric_cols)
n_targets = len(targets)

fig, axes = plt.subplots(
    n_metrics, 
    n_targets, 
    figsize=(15, 4.5 * n_metrics), 
    constrained_layout=True
)

for i, metric in enumerate(metric_cols):
    if metric not in plot_df.columns:
        # Placeholder if column is missing from Gnomon output
        for j in range(n_targets):
            axes[i, j].text(0.5, 0.5, f"Column '{metric}' not found", ha='center')
        continue

    for j, target in enumerate(targets):
        ax = axes[i, j]
        
        # Use boxenplot for detailed tail distribution
        sns.boxenplot(
            data=plot_df, 
            x=target, 
            y=metric, 
            ax=ax, 
            showfliers=False,
            palette="mako" if j == 0 else "rocket"  # Different palettes for different targets
        )
        
        clean_metric_name = metric.replace("_", " ")
        clean_target_name = target.replace("_", " ").title()
        
        ax.set_title(f"{clean_metric_name} by {clean_target_name}", fontweight='bold')
        ax.set_ylabel(clean_metric_name)
        ax.set_xlabel("")
        ax.tick_params(axis='x', rotation=30)
        
        sns.despine(ax=ax, trim=True)
        ax.grid(True, axis='y', linestyle=':', alpha=0.6)

plt.suptitle(f"Sex Inference Metrics Validation (N={len(plot_df)})", fontsize=16, y=1.02)
plt.show()
```
