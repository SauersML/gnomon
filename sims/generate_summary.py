"""
Generate GitHub Actions Summary report combining all simulation results.

Writes markdown to $GITHUB_STEP_SUMMARY for display in GitHub Actions UI.
"""
import sys
import os
import base64
from pathlib import Path
import pandas as pd


def encode_image_base64(image_path: str) -> str:
    """Encode image to base64 for embedding in markdown."""
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


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
        
        # Add PC plot
        pc_plot = Path(f"sim{sim_id}_pcs.png")
        if pc_plot.exists():
            lines.append("### Population Structure")
            lines.append("")
            pc_base64 = encode_image_base64(str(pc_plot))
            lines.append(f'<img src="data:image/png;base64,{pc_base64}" width="600" />')
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
        
        # Add ROC curves
        roc_plot = Path(f"sim{sim_id}_comparison_roc.png")
        if roc_plot.exists():
            lines.append("### ROC Curves")
            lines.append("")
            roc_base64 = encode_image_base64(str(roc_plot))
            lines.append(f'<img src="data:image/png;base64,{roc_base64}" width="700" />')
            lines.append("")
        
        # Add calibration curves
        cal_plot = Path(f"sim{sim_id}_comparison_calibration.png")
        if cal_plot.exists():
            lines.append("### Calibration Curves")
            lines.append("")
            cal_base64 = encode_image_base64(str(cal_plot))
            lines.append(f'<img src="data:image/png;base64,{cal_base64}" width="700" />')
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
    lines.append("*Generated automatically by GitHub Actions*")
    lines.append("")
    
    # Write to summary file
    with open(summary_file, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"✅ Summary report written to {summary_file}")


if __name__ == '__main__':
    generate_summary_report()
