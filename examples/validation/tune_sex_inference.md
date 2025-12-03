Download data and run Gnomon to generate sex inference metrics:

```
# Download microarray PLINK data if not present
if [ ! -f "../../arrays.bed" ]; then
    echo "Downloading microarray data..."
    gsutil -u "$GOOGLE_PROJECT" -m cp -r gs://fc-aou-datasets-controlled/v8/microarray/plink/* ../..
fi
```

Build `gnomon`:
```
# Install Rust nightly
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && { for f in ~/.bashrc ~/.profile; do [ -f "$f" ] || touch "$f"; grep -qxF 'source "$HOME/.cargo/env"' "$f" || printf '\n# Rust / Cargo\nsource "$HOME/.cargo/env"\n' >> "$f"; done; } && source "$HOME/.cargo/env" && rustup toolchain install nightly && rustup default nightly
git clone https://github.com/SauersML/gnomon.git
cd gnomon
rustup override set nightly
cargo build --release
```

Run Gnomon sex inference to generate arrays.sex.tsv with metric columns:
```
echo "Running Gnomon sex inference..."
../../gnomon/target/release/gnomon terms --sex ../../arrays.bed
```

Load data and visualize specific inference metrics against Self-Reported Sex and DRAGEN Ploidy:

```
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import gcsfs

# 1. Load Gnomon Results (Local)
gnomon_path = "../../arrays.sex.tsv"
if not os.path.exists(gnomon_path):
    raise FileNotFoundError(f"Could not find {gnomon_path}. Run the cell above.")

gnomon_df = pd.read_csv(gnomon_path, sep="\t")
# Ensure ID is string for merging
gnomon_df["IID"] = gnomon_df["IID"].astype(str)

# 2. Load Truth Data (GCS)
project_id = os.environ.get("GOOGLE_PROJECT")
metrics_uri = "gs://fc-aou-datasets-controlled/v8/wgs/short_read/snpindel/aux/qc/genomic_metrics.tsv"

fs = gcsfs.GCSFileSystem(project=project_id, token="cloud", requester_pays=True)
with fs.open(metrics_uri, "rb") as f:
    metrics_df = pd.read_csv(f, sep="\t")

# 3. Merge Datasets
# Identify correct ID column in metrics file
possible_ids = ["person_id", "sample_id", "participant_id"]
id_col = next((c for c in possible_ids if c in metrics_df.columns), None)
if not id_col:
    raise ValueError(f"Could not find ID column in metrics. Checked: {possible_ids}")

metrics_df[id_col] = metrics_df[id_col].astype(str)
merged = gnomon_df.merge(metrics_df, left_on="IID", right_on=id_col, how="inner")

# 4. Prepare for Plotting
# Filter for visualization clarity (remove NAs in targets)
plot_df = merged.dropna(subset=["sex_at_birth", "dragen_sex_ploidy"]).copy()

# List of new quantitative metrics from Gnomon to plot
# Note: Adjust column names if they differ slightly in the final build output
metric_cols = [
    "X_Het_Ratio", 
    "Y_Non_PAR", 
    "Y_PAR", 
    "PAR_Ratio", 
    "Male_Votes", 
    "Female_Votes"
]

targets = ["sex_at_birth", "dragen_sex_ploidy"]

# Set up grid
n_metrics = len(metric_cols)
n_targets = len(targets)
fig, axes = plt.subplots(n_metrics, n_targets, figsize=(14, 4 * n_metrics))

# 5. Generate Plots
for i, metric in enumerate(metric_cols):
    # Check if metric exists in output
    if metric not in plot_df.columns:
        for j in range(n_targets):
            axes[i, j].text(0.5, 0.5, f"Column '{metric}' not found", ha='center')
        continue

    # Ensure metric is numeric (coerce errors to NaN)
    plot_df[metric] = pd.to_numeric(plot_df[metric], errors="coerce")

    for j, target in enumerate(targets):
        ax = axes[i, j]
        
        # Create Boxen plot for detailed distribution view
        sns.boxenplot(
            data=plot_df, 
            x=target, 
            y=metric, 
            ax=ax, 
            showfliers=False,
            palette="viridis"
        )
        
        ax.set_title(f"{metric} vs {target}")
        ax.set_ylabel(metric)
        ax.set_xlabel("")
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, linestyle='--', alpha=0.3)

plt.tight_layout()
plt.show()
```
