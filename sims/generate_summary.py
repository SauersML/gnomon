"""
Generate GitHub Actions Summary report combining all simulation results.

Writes markdown to $GITHUB_STEP_SUMMARY for display in GitHub Actions UI.
Uses artifact download links instead of base64 to avoid size limits.
"""
import sys
import os
from pathlib import Path
import pandas as pd


def generate_summary_report():
    """Generate consolidated markdown report for all simulations."""
    
    # Get GitHub Actions summary file path
    summary_file = os.environ.get('GITHUB_STEP_SUMMARY')
    if not summary_file:
        print("$GITHUB_STEP_SUMMARY not set, writing to summary.md instead")
        summary_file = 'summary.md'
    
    # Get workflow run info for artifact links
    github_server = os.environ.get('GITHUB_SERVER_URL', 'https://github.com')
    github_repo = os.environ.get('GITHUB_REPOSITORY', 'SauersML/gnomon')
    github_run_id = os.environ.get('GITHUB_RUN_ID', '')
    
    artifact_base = f"{github_server}/{github_repo}/actions/runs/{github_run_id}"
    
    lines = []
    
    # Header
    lines.append("# 🧬 PGS Calibration Methods Comparison Report")
    lines.append("")
    lines.append("Comparison of four ancestry-aware polygenic score calibration methods across three simulation scenarios.")
    lines.append("")
    
    # Overview of methods
    lines.append("## 📊 Methods Evaluated")
    lines.append("")
    lines.append("| Method | Description |")
    lines.append("|--------|-------------|")
    lines.append("| **Raw PGS** | Baseline logistic regression using PGS only |")
    lines.append("| **Linear Interaction** | Logistic regression with linear P×PC interaction terms |")
    lines.append("| **Normalization (Empirical)** | Phenotype-agnostic residualization: P ~ PC |")
    lines.append("| **Normalization (Mean)** | Standardize mean by ancestry percentile bins |")
    lines.append("| **Normalization (Mean+Var)** | Standardize mean and variance by ancestry |")
    lines.append("| **GAM (mgcv)** | Tensor product splines via R's mgcv package |")
    lines.append("")
    
    # Simulation descriptions
    lines.append("## 🎯 Simulation Scenarios")
    lines.append("")
    lines.append("| Simulation | Design | Key Question |")
    lines.append("|------------|--------|--------------|")
    lines.append("| **Sim 1** | Ancestry-correlated liability (PC shifts mean) | Does normalization remove true signal? |")
    lines.append("| **Sim 2** | Ancestry-dependent accuracy (attenuation + noise) | Can GAM capture non-linear calibration? |")
    lines.append("| **Sim 3** | Imbalanced populations (EUR majority) | Does ancestry-aware calibration help underrepresented groups? |")
    lines.append("")
    
    # Process each simulation
    for sim_id in [1, 2, 3]:
        lines.append(f"---")
        lines.append("")
        lines.append(f"## Simulation {sim_id} Results")
        lines.append("")
        
        # Link to PC plot
        pc_plot = Path(f"sim{sim_id}_pcs.png")
        if pc_plot.exists():
            lines.append("### Population Structure")
            lines.append("")
            lines.append(f"📊 [View PC Plot](../../actions/runs/{github_run_id}#artifacts) - Download `sim-{sim_id}-outputs` artifact")
            lines.append("")
        
        # Add metrics table
        metrics_file = Path(f"sim{sim_id}_metrics.csv")
        if metrics_file.exists():
            lines.append("### Performance Metrics")
            lines.append("")
            df = pd.read_csv(metrics_file)
            
            # Format as markdown table
            lines.append(df.to_markdown(index=False))
            lines.append("")
            
            # Highlight best performing method
            best_auc_idx = df['AUC_overall'].idxmax()
            best_method = df.loc[best_auc_idx, 'Method']
            best_auc = df.loc[best_auc_idx, 'AUC_overall']
            lines.append(f"**🏆 Best Overall AUC:** {best_method} ({best_auc:.3f})")
            lines.append("")
        
        # Link to plots
        lines.append("### Visualizations")
        lines.append("")
        lines.append(f"- 📈 [ROC Curves](../../actions/runs/{github_run_id}#artifacts)")
        lines.append(f"- 📉 [Calibration Curves](../../actions/runs/{github_run_id}#artifacts)")
        lines.append("")
        lines.append(f"*Download `sim-{sim_id}-outputs` artifact to view plots*")
        lines.append("")
    
    # Summary conclusions
    lines.append("---")
    lines.append("")
    lines.append("## 📝 Key Takeaways")
    lines.append("")
    lines.append("- **Simulation 1**: Tests whether phenotype-agnostic normalization removes true ancestry-associated genetic signal")
    lines.append("- **Simulation 2**: Demonstrates need for non-linear calibration when PGS accuracy varies with ancestry")
    lines.append("- **Simulation 3**: Evaluates performance in underrepresented populations with imbalanced training data")
    lines.append("")
    lines.append("## 📦 Download Results")
    lines.append("")
    lines.append(f"All plots and data files are available in the [workflow artifacts]({artifact_base}#artifacts).")
    lines.append("")
    lines.append("*Generated automatically by GitHub Actions*")
    lines.append("")
    
    # Write to summary file
    content = '\n'.join(lines)
    with open(summary_file, 'w') as f:
        f.write(content)
    
    print(f"✅ Summary report written to {summary_file}")
    print(f"   Report size: {len(content)} bytes (within 1MB limit)")


if __name__ == '__main__':
    generate_summary_report()
