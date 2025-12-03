#!/usr/bin/env python3
"""Generate comprehensive comparison graphs for Cordon benchmark analysis.

This script creates visualizations comparing:
1. Sample size scaling (50k -> 1M)
2. Parameter tuning impact
3. Threshold sensitivity
4. Variance/stability analysis
"""

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 11


def load_aggregated_results(config_path: Path) -> dict[str, Any]:
    """Load aggregated results from a test configuration directory."""
    results_file = config_path / "aggregated_results.json"
    if not results_file.exists():
        # Try nested run directory structure
        run_dirs = list(config_path.glob("run_*/aggregated_results.json"))
        if run_dirs:
            results_file = run_dirs[0]
        else:
            raise FileNotFoundError(f"No aggregated_results.json found in {config_path}")
    
    with open(results_file) as f:
        return json.load(f)


def load_individual_runs(config_path: Path) -> list[dict[str, Any]]:
    """Load individual run results for detailed analysis."""
    runs = []
    
    # Check for run_N subdirectories
    run_dirs = sorted(config_path.glob("run_*"))
    if run_dirs:
        for run_dir in run_dirs:
            result_files = list(run_dir.glob("results_run*.json"))
            if result_files:
                with open(result_files[0]) as f:
                    runs.append(json.load(f))
    else:
        # Check for nested run directory structure
        nested_dirs = list(config_path.glob("run_*/run_*"))
        for run_dir in sorted(nested_dirs):
            result_files = list(run_dir.glob("results_run*.json"))
            if result_files:
                with open(result_files[0]) as f:
                    runs.append(json.load(f))
    
    return runs


def plot_sample_size_scaling(results_dir: Path, output_dir: Path):
    """Generate sample size scaling comparison graphs."""
    # Load baseline test results
    configs = {
        "50k": "50k/50k_w5_k10_p02_bge",
        "100k": "100k/100k_w5_k10_p02_bge",
        "250k": "250k/250k_w5_k10_p02_bge",
        "500k": "500k/500k_w5_k10_p02_bge",
        "1M": "1M/1M_w5_k10_p02_bge",
    }
    
    data = {}
    for label, path in configs.items():
        try:
            data[label] = load_aggregated_results(results_dir / path)
        except FileNotFoundError as e:
            print(f"Warning: Could not load {path}: {e}")
    
    if not data:
        print("Error: No baseline test data found")
        return
    
    # Extract data for plotting
    sample_sizes = []
    template_recalls = []
    template_recall_stds = []
    rare_recalls = []
    rare_recall_stds = []
    freq_weighted = []
    freq_weighted_stds = []
    cvs = []
    
    for label in ["50k", "100k", "250k", "500k", "1M"]:
        if label not in data:
            continue
        
        size_num = {"50k": 50, "100k": 100, "250k": 250, "500k": 500, "1M": 1000}[label]
        sample_sizes.append(size_num)
        
        d = data[label]
        template_recalls.append(d["template_recall_mean"])
        template_recall_stds.append(d["template_recall_std"])
        rare_recalls.append(d["rare_recall_mean"])
        rare_recall_stds.append(d["rare_recall_std"])
        freq_weighted.append(d["freq_weighted_mean"])
        freq_weighted_stds.append(d["freq_weighted_std"])
        
        # Calculate CV%
        cv = (d["template_recall_std"] / d["template_recall_mean"] * 100) if d["template_recall_mean"] > 0 else 0
        cvs.append(cv)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Sample Size Scaling Analysis (Baseline: w5_k10_p02)", fontsize=16, fontweight="bold")
    
    # 1. Template Recall vs Sample Size
    ax = axes[0, 0]
    ax.errorbar(sample_sizes, template_recalls, yerr=template_recall_stds, 
                marker='o', linewidth=2, markersize=8, capsize=5, capthick=2)
    ax.set_xlabel("Sample Size (thousands of lines)")
    ax.set_ylabel("Template Recall")
    ax.set_title("Template Recall vs Sample Size")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.0)
    
    # Add CV annotations
    for i, (x, y, cv) in enumerate(zip(sample_sizes, template_recalls, cvs)):
        ax.annotate(f"CV: {cv:.1f}%", xy=(x, y), xytext=(0, 10), 
                   textcoords='offset points', ha='center', fontsize=9,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))
    
    # 2. Rare Template Recall vs Sample Size
    ax = axes[0, 1]
    ax.errorbar(sample_sizes, rare_recalls, yerr=rare_recall_stds,
                marker='s', linewidth=2, markersize=8, capsize=5, capthick=2, color='orange')
    ax.set_xlabel("Sample Size (thousands of lines)")
    ax.set_ylabel("Rare Template Recall")
    ax.set_title("Rare Template Recall vs Sample Size\n(Templates with < 100 occurrences)")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.0)
    
    # 3. Frequency-Weighted Recall vs Sample Size
    ax = axes[1, 0]
    ax.errorbar(sample_sizes, freq_weighted, yerr=freq_weighted_stds,
                marker='^', linewidth=2, markersize=8, capsize=5, capthick=2, color='green')
    ax.set_xlabel("Sample Size (thousands of lines)")
    ax.set_ylabel("Frequency-Weighted Recall")
    ax.set_title("Frequency-Weighted Recall vs Sample Size\n(Higher = More Bias Toward Rare Patterns)")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.0)
    
    # 4. Coefficient of Variation vs Sample Size
    ax = axes[1, 1]
    colors = ['red' if cv > 15 else 'green' for cv in cvs]
    bars = ax.bar(range(len(sample_sizes)), cvs, color=colors, alpha=0.7)
    ax.axhline(y=15, color='red', linestyle='--', linewidth=2, label='15% Threshold (Stable)')
    ax.set_xlabel("Sample Size")
    ax.set_ylabel("Coefficient of Variation (%)")
    ax.set_title("Stability Analysis: CV% vs Sample Size\n(Lower = More Stable)")
    ax.set_xticks(range(len(sample_sizes)))
    ax.set_xticklabels([f"{s}k" for s in [50, 100, 250, 500] + [1000]])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, cv) in enumerate(zip(bars, cvs)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
               f'{cv:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    output_file = output_dir / "sample_size_scaling.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_parameter_tuning(results_dir: Path, output_dir: Path):
    """Generate parameter tuning comparison graphs (50k baseline vs tuned)."""
    configs = {
        "Baseline\n(w5_k10_p02)": "50k/50k_w5_k10_p02_bge",
        "Tuned\n(w2_k5_p02)": "50k/50k_w2_k5_p02_bge",
    }
    
    data = {}
    for label, path in configs.items():
        try:
            data[label] = load_aggregated_results(results_dir / path)
        except FileNotFoundError as e:
            print(f"Warning: Could not load {path}: {e}")
            return
    
    # Create comparison bar chart
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle("Parameter Tuning Impact (50k Sample)", fontsize=16, fontweight="bold")
    
    metrics = [
        ("template_recall_mean", "template_recall_std", "Template Recall"),
        ("rare_recall_mean", "rare_recall_std", "Rare Template Recall"),
        ("freq_weighted_mean", "freq_weighted_std", "Frequency-Weighted Recall"),
    ]
    
    x = np.arange(len(configs))
    width = 0.6
    
    for i, (mean_key, std_key, title) in enumerate(metrics):
        ax = axes[i]
        
        means = [data[label][mean_key] for label in configs.keys()]
        stds = [data[label][std_key] for label in configs.keys()]
        
        bars = ax.bar(x, means, width, yerr=stds, capsize=5, alpha=0.8)
        
        # Color bars
        bars[0].set_color('steelblue')
        bars[1].set_color('coral')
        
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(configs.keys())
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, mean, std in zip(bars, means, stds):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.02,
                   f'{mean:.3f}\n±{std:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    output_file = output_dir / "parameter_tuning_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_threshold_sensitivity(results_dir: Path, output_dir: Path):
    """Generate threshold sensitivity analysis graphs (50k tests)."""
    configs = {
        "p=0.02 (2%)": "50k/50k_w2_k5_p02_bge",
        "p=0.05 (5%)": "50k/50k_w2_k5_p05_bge",
        "p=0.10 (10%)": "50k/50k_w2_k5_p10_bge",
    }
    
    data = {}
    for label, path in configs.items():
        try:
            data[label] = load_aggregated_results(results_dir / path)
        except FileNotFoundError as e:
            print(f"Warning: Could not load {path}: {e}")
    
    if len(data) < 2:
        print("Warning: Not enough threshold sensitivity data found")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Threshold Sensitivity Analysis (50k, w2_k5)", fontsize=16, fontweight="bold")
    
    x = np.arange(len(data))
    width = 0.6
    
    # 1. Template-Level Metrics
    ax = axes[0]
    template_recalls = [data[label]["template_recall_mean"] for label in data.keys()]
    template_stds = [data[label]["template_recall_std"] for label in data.keys()]
    
    bars = ax.bar(x, template_recalls, width, yerr=template_stds, capsize=5, alpha=0.8)
    ax.set_ylabel("Template Recall")
    ax.set_title("Template Recall vs Anomaly Threshold")
    ax.set_xticks(x)
    ax.set_xticklabels(data.keys())
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, mean, std in zip(bars, template_recalls, template_stds):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.02,
               f'{mean:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 2. Traditional Precision/Recall Tradeoff
    ax = axes[1]
    precisions = [data[label]["traditional_precision_mean"] for label in data.keys()]
    recalls = [data[label]["traditional_recall_mean"] for label in data.keys()]
    
    # Plot as line with markers
    ax.plot(recalls, precisions, marker='o', linewidth=2, markersize=10, label='P-R Curve')
    
    # Annotate points
    for i, label in enumerate(data.keys()):
        ax.annotate(label, xy=(recalls[i], precisions[i]), 
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=10, bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.5),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    ax.set_xlabel("Traditional Recall (Line-Level)")
    ax.set_ylabel("Traditional Precision (Line-Level)")
    ax.set_title("Precision-Recall Tradeoff")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlim(0, max(recalls) * 1.2)
    ax.set_ylim(0, max(precisions) * 1.2)
    
    plt.tight_layout()
    output_file = output_dir / "threshold_sensitivity.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_stability_boxplots(results_dir: Path, output_dir: Path):
    """Generate box plots showing distribution of template recall across runs."""
    configs = {
        "50k": "50k/50k_w5_k10_p02_bge",
        "100k": "100k/100k_w5_k10_p02_bge",
        "250k": "250k/250k_w5_k10_p02_bge",
        "500k": "500k/500k_w5_k10_p02_bge",
        "1M": "1M/1M_w5_k10_p02_bge",
    }
    
    all_recalls = []
    labels = []
    
    for label, path in configs.items():
        try:
            runs = load_individual_runs(results_dir / path)
            if runs:
                recalls = [run["template_coverage"]["template_recall"] for run in runs]
                all_recalls.append(recalls)
                labels.append(label)
        except Exception as e:
            print(f"Warning: Could not load runs for {path}: {e}")
    
    if not all_recalls:
        print("Warning: No individual run data found for stability analysis")
        return
    
    # Create box plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    bp = ax.boxplot(all_recalls, labels=labels, patch_artist=True, widths=0.6,
                     medianprops=dict(color='red', linewidth=2),
                     boxprops=dict(facecolor='lightblue', alpha=0.7),
                     whiskerprops=dict(linewidth=1.5),
                     capprops=dict(linewidth=1.5))
    
    ax.set_xlabel("Sample Size", fontsize=13, fontweight='bold')
    ax.set_ylabel("Template Recall", fontsize=13, fontweight='bold')
    ax.set_title("Template Recall Distribution Across 10 Runs\n(Baseline: w5_k10_p02)", 
                fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.0)
    
    # Add mean markers
    for i, recalls in enumerate(all_recalls):
        mean_val = np.mean(recalls)
        ax.plot(i + 1, mean_val, marker='D', color='darkgreen', markersize=8, 
               label='Mean' if i == 0 else '')
    
    ax.legend(loc='lower right', fontsize=11)
    
    plt.tight_layout()
    output_file = output_dir / "stability_boxplots.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_traditional_vs_template_metrics(results_dir: Path, output_dir: Path):
    """Generate scatter plot comparing traditional F1 vs template recall."""
    configs = {
        "50k": "50k/50k_w5_k10_p02_bge",
        "100k": "100k/100k_w5_k10_p02_bge",
        "250k": "250k/250k_w5_k10_p02_bge",
        "500k": "500k/500k_w5_k10_p02_bge",
        "1M": "1M/1M_w5_k10_p02_bge",
    }
    
    template_recalls = []
    traditional_f1s = []
    labels = []
    sizes = []
    
    for label, path in configs.items():
        try:
            data = load_aggregated_results(results_dir / path)
            template_recalls.append(data["template_recall_mean"])
            traditional_f1s.append(data["traditional_f1_mean"])
            labels.append(label)
            sizes.append({"50k": 50, "100k": 100, "250k": 250, "500k": 500, "1M": 1000}[label])
        except Exception as e:
            print(f"Warning: Could not load {path}: {e}")
    
    if not template_recalls:
        print("Warning: No data found for metrics comparison")
        return
    
    # Create scatter plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot points with size representing sample size
    scatter = ax.scatter(traditional_f1s, template_recalls, s=[s*0.5 for s in sizes], 
                        alpha=0.6, c=range(len(labels)), cmap='viridis', edgecolors='black', linewidth=2)
    
    # Add labels
    for i, label in enumerate(labels):
        ax.annotate(label, xy=(traditional_f1s[i], template_recalls[i]),
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3'))
    
    ax.set_xlabel("Traditional F1 Score (Line-Level)", fontsize=13, fontweight='bold')
    ax.set_ylabel("Template Recall (Template-Level)", fontsize=13, fontweight='bold')
    ax.set_title("Traditional vs Template-Based Metrics\nShowing Why Traditional Metrics Fail for Semantic Uniqueness Detection",
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add explanation text
    textstr = ("Traditional F1 is low because Cordon ignores repetitive patterns.\n"
              "Template Recall measures diversity of anomaly types detected.\n"
              "Bubble size represents sample size (50k to 1M lines).")
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    output_file = output_dir / "traditional_vs_template_metrics.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_model_comparison(results_dir: Path, output_dir: Path):
    """Generate model comparison graphs (bge-large vs minilm)."""
    # BGE-large results (w5, original runs)
    bge_configs = {
        "50k": "50k/50k_w5_k10_p02_bge",
        "100k": "100k/100k_w5_k10_p02_bge",
        "250k": "250k/250k_w5_k10_p02_bge",
        "500k": "500k/500k_w5_k10_p02_bge",
        "1M": "1M/1M_w5_k10_p02_bge",
    }
    
    # MiniLM results (w4, new runs)
    minilm_configs = {
        "50k": "50k/50k_w4_k10_p02_minilm/run_20251130_044439",
        "100k": "100k/100k_w4_k10_p02_minilm/run_20251130_045705",
        "250k": "250k/250k_w4_k10_p02_minilm/run_20251130_051954",
        "500k": "500k/500k_w4_k10_p02_minilm/run_20251130_062157",
        "1M": "1M/1M_w4_k10_p02_minilm/run_20251130_155602",
    }
    
    bge_data = {}
    minilm_data = {}
    
    for label, path in bge_configs.items():
        try:
            bge_data[label] = load_aggregated_results(results_dir / path)
        except Exception as e:
            print(f"Warning: Could not load BGE {path}: {e}")
    
    for label, path in minilm_configs.items():
        try:
            minilm_data[label] = load_aggregated_results(results_dir / path)
        except Exception as e:
            print(f"Warning: Could not load MiniLM {path}: {e}")
    
    if not bge_data or not minilm_data:
        print("Warning: Insufficient model comparison data")
        return
    
    # Extract data
    sample_sizes = []
    bge_recalls = []
    bge_stds = []
    minilm_recalls = []
    minilm_stds = []
    
    for label in ["50k", "100k", "250k", "500k", "1M"]:
        if label in bge_data and label in minilm_data:
            size_num = {"50k": 50, "100k": 100, "250k": 250, "500k": 500, "1M": 1000}[label]
            sample_sizes.append(size_num)
            
            bge_recalls.append(bge_data[label]["template_recall_mean"])
            bge_stds.append(bge_data[label]["template_recall_std"])
            minilm_recalls.append(minilm_data[label]["template_recall_mean"])
            minilm_stds.append(minilm_data[label]["template_recall_std"])
    
    # Create comparison plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(sample_sizes))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, bge_recalls, width, yerr=bge_stds,
                   label='BGE-Large (w=5, 1024-dim)', capsize=5, alpha=0.8, color='steelblue')
    bars2 = ax.bar(x + width/2, minilm_recalls, width, yerr=minilm_stds,
                   label='MiniLM (w=4, 384-dim)', capsize=5, alpha=0.8, color='coral')
    
    ax.set_xlabel("Sample Size (thousands of lines)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Template Recall", fontsize=12, fontweight='bold')
    ax.set_title("Model Comparison: BGE-Large vs MiniLM\n(Both using k=10, p=0.02)",
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}k" if s < 1000 else f"{s//1000}M" for s in sample_sizes])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.0)
    
    # Add value labels
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        height1 = bar1.get_height()
        height2 = bar2.get_height()
        ax.text(bar1.get_x() + bar1.get_width()/2., height1 + bge_stds[i] + 0.02,
               f'{bge_recalls[i]:.3f}', ha='center', va='bottom', fontsize=9)
        ax.text(bar2.get_x() + bar2.get_width()/2., height2 + minilm_stds[i] + 0.02,
               f'{minilm_recalls[i]:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Add explanation text
    textstr = ("BGE-Large: BAAI/bge-large-en-v1.5 (1024-dim embeddings, 512 tokens, w=5)\n"
              "MiniLM: all-MiniLM-L6-v2 (384-dim embeddings, 256 tokens, w=4)\n"
              "Both models show nearly identical performance across all sample sizes.")
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    output_file = output_dir / "model_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def main():
    """Generate all analysis graphs."""
    results_dir = Path(__file__).parent
    output_dir = results_dir / "analysis_graphs"
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("GENERATING BENCHMARK ANALYSIS GRAPHS")
    print("=" * 80)
    print(f"Results directory: {results_dir}")
    print(f"Output directory: {output_dir}")
    print()
    
    print("1. Sample Size Scaling Analysis...")
    plot_sample_size_scaling(results_dir, output_dir)
    print()
    
    print("2. Parameter Tuning Comparison...")
    plot_parameter_tuning(results_dir, output_dir)
    print()
    
    print("3. Threshold Sensitivity Analysis...")
    plot_threshold_sensitivity(results_dir, output_dir)
    print()
    
    print("4. Stability Box Plots...")
    plot_stability_boxplots(results_dir, output_dir)
    print()
    
    print("5. Traditional vs Template Metrics...")
    plot_traditional_vs_template_metrics(results_dir, output_dir)
    print()
    
    print("6. Model Comparison (BGE-Large vs MiniLM)...")
    plot_model_comparison(results_dir, output_dir)
    print()
    
    print("=" * 80)
    print("ALL GRAPHS GENERATED SUCCESSFULLY")
    print(f"Output: {output_dir}/")
    print("=" * 80)


if __name__ == "__main__":
    main()

