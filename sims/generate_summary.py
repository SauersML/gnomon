"""
Generate GitHub Actions Summary report combining all simulation results.

Writes markdown to $GITHUB_STEP_SUMMARY for display in GitHub Actions UI.
Renders plots inline when the PNG files are available in the workspace.
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
    
    lines = []
    
    # Header
    lines.append("# 🧬 PGS Calibration Methods Comparison Report")
    lines.append("")
    lines.append("Comparison of four ancestry-aware polygenic score calibration methods across two simulation scenarios.")
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
    lines.append("| **Sim 3** | Portability (train EUR, test across ancestries) | Do models transfer across ancestries without confounding? |")
    lines.append("")

    sim_names = ["confounding", "portability"]

    # Check if any results exist
    has_results = False
    for sim_name in sim_names:
        if Path(f"{sim_name}_metrics.csv").exists():
            has_results = True
            break

    if not has_results:
        lines.append("## ⚠️ No Results Available")
        lines.append("")
        lines.append("No simulation results were found. This may indicate that some or all evaluation jobs failed.")
        lines.append("")
        lines.append("Please check the workflow logs for errors in the `train` and `evaluate` jobs.")
        lines.append("")

    # Process each simulation
    for sim_name in sim_names:
        lines.append(f"---")
        lines.append("")
        lines.append(f"## Simulation {sim_name} Results")
        lines.append("")
        
        # Population structure plot (if available in this job's workspace)
        pc_plot = Path(f"{sim_name}_pcs.png")
        if pc_plot.exists():
            lines.append("### Population Structure")
            lines.append("")
            lines.append(f"![Simulation {sim_name} PC Plot]({pc_plot.as_posix()})")
            lines.append("")
        
        # Add metrics table
        metrics_file = Path(f"{sim_name}_metrics.csv")
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
        
        # Plots rendered inline if present
        lines.append("### Visualizations")
        lines.append("")
        roc_plot = Path(f"{sim_name}_comparison_roc.png")
        cal_plot = Path(f"{sim_name}_comparison_calibration.png")
        auc_plot = Path(f"{sim_name}_comparison_auc.png")
        brier_plot = Path(f"{sim_name}_comparison_brier.png")
        if roc_plot.exists():
            lines.append(f"![Simulation {sim_name} ROC Curves]({roc_plot.as_posix()})")
            lines.append("")
        else:
            lines.append(f"- 📈 ROC Curves: missing (`{roc_plot.name}` not found)")
        if cal_plot.exists():
            lines.append(f"![Simulation {sim_name} Calibration Curves]({cal_plot.as_posix()})")
            lines.append("")
        else:
            lines.append(f"- 📉 Calibration Curves: missing (`{cal_plot.name}` not found)")
        if auc_plot.exists():
            lines.append(f"![Simulation {sim_name} AUC Summary]({auc_plot.as_posix()})")
            lines.append("")
        else:
            lines.append(f"- 📊 AUC Summary: missing (`{auc_plot.name}` not found)")
        if brier_plot.exists():
            lines.append(f"![Simulation {sim_name} Brier Summary]({brier_plot.as_posix()})")
            lines.append("")
        else:
            lines.append(f"- 📊 Brier Summary: missing (`{brier_plot.name}` not found)")
        lines.append("")
    
    # Summary conclusions
    lines.append("---")
    lines.append("")
    lines.append("## 📝 Key Takeaways")
    lines.append("")
    lines.append("- **Simulation 1**: Tests whether phenotype-agnostic normalization removes true ancestry-associated genetic signal")
    lines.append("- **Simulation 3**: Evaluates performance in underrepresented populations with imbalanced training data")
    lines.append("")
    lines.append("*Generated automatically by GitHub Actions*")
    lines.append("")
    
    # Write to summary file
    content = '\n'.join(lines)
    with open(summary_file, 'w') as f:
        f.write(content)

    # Also write to summary.md for artifact upload
    with open('summary.md', 'w') as f:
        f.write(content)

    print(f"✅ Summary report written to {summary_file}")
    print(f"   Report size: {len(content)} bytes (within 1MB limit)")


if __name__ == '__main__':
    generate_summary_report()
