#!/usr/bin/env python3
"""Visualization tools for embedding space and score distributions."""

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

# try GPU-accelerated UMAP first (cuML), fall back to CPU
try:
    from cuml.manifold import UMAP

    HAS_GPU_UMAP = True
except ImportError:
    try:
        import umap

        UMAP = umap.UMAP
        HAS_GPU_UMAP = False
    except ImportError:
        # final fallback to sklearn UMAP if available
        try:
            from sklearn.manifold import UMAP

            HAS_GPU_UMAP = False
        except ImportError:
            UMAP = None
            HAS_GPU_UMAP = False

# Suppress tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from evaluate import (
    extract_session_id,
    load_dataset_config,
    load_ground_truth,
    read_log_sample,
)

from cordon import AnalysisConfig
from cordon.analysis.scorer import DensityAnomalyScorer
from cordon.embedding import create_vectorizer
from cordon.ingestion.reader import LogFileReader
from cordon.segmentation.windower import SlidingWindowSegmenter


def get_embeddings(
    log_lines: list[str],
    config: AnalysisConfig,
    sample_size: int | None = None,
) -> tuple[npt.NDArray[np.floating[Any]], list[tuple[int, int]]]:
    """Get embeddings from Cordon pipeline without scoring.

    Args:
        log_lines: Pre-loaded log lines
        config: Cordon configuration
        sample_size: Number of lines to analyze

    Returns:
        Tuple of (embeddings_matrix, window_line_ranges)
        - embeddings_matrix: N x D matrix of embeddings
        - window_line_ranges: List of (start_line, end_line) for each window
    """
    # Create temp file
    temp_log = Path(f"/tmp/cordon_visualize_{sample_size or 'full'}.log")

    with open(temp_log, "w") as fout:
        for line in log_lines:
            fout.write(line + "\n")

    analysis_path = temp_log

    # Read logs
    reader = LogFileReader()
    lines = reader.read_lines(analysis_path)

    # Segment
    segmenter = SlidingWindowSegmenter()
    windows = list(segmenter.segment(lines, config))

    # Extract line ranges
    window_ranges = [(w.start_line, w.end_line) for w in windows]

    # Embed
    vectorizer = create_vectorizer(config)
    embedded = list(vectorizer.embed_windows(windows))

    # Extract embeddings matrix
    embeddings = np.array([emb for _, emb in embedded])

    # Clean up
    if sample_size and temp_log.exists():
        temp_log.unlink()

    return embeddings, window_ranges


def get_window_labels(
    window_ranges: list[tuple[int, int]],
    log_lines: list[str],
    ground_truth: dict[str, str],
    id_pattern: str,
) -> list[str]:
    """Get ground truth labels for each window (session-level for HDFS).

    Args:
        window_ranges: Line ranges for each window
        log_lines: All log lines
        ground_truth: Session ID -> label mapping
        id_pattern: Regex to extract session IDs

    Returns:
        List of labels ("Normal" or "Anomaly") for each window
    """
    window_labels = []

    for start, end in window_ranges:
        # Session-level labels (HDFS)
        window_session_ids = set()
        for line_idx in range(start - 1, end):  # Convert to 0-indexed
            if line_idx < len(log_lines):
                session_id = extract_session_id(log_lines[line_idx], id_pattern)
                if session_id and session_id in ground_truth:
                    window_session_ids.add(session_id)

        is_anomalous = any(ground_truth.get(sid) == "Anomaly" for sid in window_session_ids)

        window_labels.append("Anomaly" if is_anomalous else "Normal")

    return window_labels


def compute_knn_scores(
    embeddings: npt.NDArray[np.floating[Any]],
    config: AnalysisConfig,
) -> npt.NDArray[np.floating[Any]]:
    """Compute k-NN distance scores for embeddings.

    Args:
        embeddings: N x D embedding matrix
        config: Cordon configuration with k_neighbors

    Returns:
        Array of k-NN distance scores
    """
    from cordon.core.types import TextWindow

    # Create dummy windows for scorer
    embedded_windows = [
        (
            TextWindow(content="", start_line=i + 1, end_line=i + 1, window_id=i),
            emb,
        )
        for i, emb in enumerate(embeddings)
    ]

    # Score
    scorer = DensityAnomalyScorer()
    scored = scorer.score_windows(embedded_windows, config)

    return np.array([sw.score for sw in scored])


def plot_umap_embeddings(
    embeddings: npt.NDArray[np.floating[Any]],
    scores: npt.NDArray[np.floating[Any]],
    labels: list[str],
    output_path: Path,
    title: str = "Embedding Space Visualization",
):
    """Create UMAP visualization colored by score and ground truth.

    Args:
        embeddings: N x D embedding matrix
        scores: Anomaly scores for each embedding
        labels: Ground truth labels for each embedding
        output_path: Where to save the figure
        title: Figure title
    """
    if UMAP is None:
        print("WARNING: UMAP not available. Install with: uv pip install umap-learn")
        print("Skipping UMAP visualization.")
        return

    # ensure embeddings are on CPU for UMAP (avoid GPU memory issues)
    if hasattr(embeddings, "get"):
        # cupy array
        embeddings = embeddings.get()
    elif hasattr(embeddings, "cpu"):
        # torch tensor
        embeddings = embeddings.cpu().numpy()

    coords = None
    if HAS_GPU_UMAP:
        try:
            print("Computing UMAP projection using GPU (cuML)...")
            reducer = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
            coords = reducer.fit_transform(embeddings)
        except (MemoryError, RuntimeError) as e:
            print(f"GPU UMAP failed ({e.__class__.__name__}), falling back to CPU...")
        except Exception as e:
            print(f"GPU UMAP failed ({e}), falling back to CPU...")

    if coords is None:
        # fallback to CPU UMAP
        print("Computing UMAP projection using CPU (umap-learn)...")
        reducer = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1, n_jobs=-1)
        coords = reducer.fit_transform(embeddings)

    _, axes = plt.subplots(1, 2, figsize=(16, 6))

    # plot 1: colored by k-NN distance score
    scatter1 = axes[0].scatter(
        coords[:, 0],
        coords[:, 1],
        c=scores,
        cmap="plasma",  # better contrast than viridis for narrow ranges
        alpha=0.7,
        s=15,  # slightly larger points
        edgecolors="none",
    )
    axes[0].set_title("Colored by k-NN Distance Score", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("UMAP Dimension 1")
    axes[0].set_ylabel("UMAP Dimension 2")
    cbar1 = plt.colorbar(scatter1, ax=axes[0])
    cbar1.set_label("Anomaly Score (Higher = More Anomalous)", rotation=270, labelpad=20)

    # plot 2: colored by ground truth
    colors = ["green" if label == "Normal" else "red" for label in labels]
    axes[1].scatter(
        coords[:, 0],
        coords[:, 1],
        c=colors,
        alpha=0.7,
        s=15,  # match left panel size
        edgecolors="none",
    )
    axes[1].set_title("Colored by Ground Truth", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("UMAP Dimension 1")
    axes[1].set_ylabel("UMAP Dimension 2")

    # Add legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="green", label="Normal"),
        Patch(facecolor="red", label="Anomaly"),
    ]
    axes[1].legend(handles=legend_elements, loc="upper right")

    plt.suptitle(title, fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved UMAP plot to {output_path}")
    plt.close()


def plot_score_distributions(
    scores: npt.NDArray[np.floating[Any]],
    labels: list[str],
    output_path: Path,
    title: str = "Score Distribution",
):
    """Plot histogram of scores for normal vs anomalous windows.

    Args:
        scores: Anomaly scores
        labels: Ground truth labels
        output_path: Where to save the figure
        title: Figure title
    """
    normal_scores = [s for s, label in zip(scores, labels, strict=False) if label == "Normal"]
    anomaly_scores = [s for s, label in zip(scores, labels, strict=False) if label == "Anomaly"]

    _, ax = plt.subplots(figsize=(10, 6))

    # Plot histograms
    bins = np.linspace(min(scores), max(scores), 50)
    ax.hist(
        normal_scores,
        bins=bins,
        alpha=0.6,
        label=f"Normal (n={len(normal_scores):,})",
        color="green",
        edgecolor="black",
    )
    ax.hist(
        anomaly_scores,
        bins=bins,
        alpha=0.6,
        label=f"Anomaly (n={len(anomaly_scores):,})",
        color="red",
        edgecolor="black",
    )

    # Add statistics
    normal_mean = np.mean(normal_scores)
    anomaly_mean = np.mean(anomaly_scores)

    ax.axvline(
        normal_mean,
        color="darkgreen",
        linestyle="--",
        linewidth=2,
        label=f"Normal mean: {normal_mean:.4f}",
    )
    ax.axvline(
        anomaly_mean,
        color="darkred",
        linestyle="--",
        linewidth=2,
        label=f"Anomaly mean: {anomaly_mean:.4f}",
    )

    ax.set_xlabel("k-NN Distance Score", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved score distribution to {output_path}")
    plt.close()


def plot_score_vs_label_boxplot(
    scores: npt.NDArray[np.floating[Any]],
    labels: list[str],
    output_path: Path,
):
    """Create box plot comparing scores for normal vs anomalous windows.

    Args:
        scores: Anomaly scores
        labels: Ground truth labels
        output_path: Where to save figure
    """
    normal_scores = [s for s, label in zip(scores, labels, strict=False) if label == "Normal"]
    anomaly_scores = [s for s, label in zip(scores, labels, strict=False) if label == "Anomaly"]

    _, ax = plt.subplots(figsize=(8, 6))

    bp = ax.boxplot(
        [normal_scores, anomaly_scores],
        labels=["Normal", "Anomaly"],
        patch_artist=True,
        widths=0.6,
    )

    # Color the boxes
    bp["boxes"][0].set_facecolor("lightgreen")
    bp["boxes"][1].set_facecolor("lightcoral")

    ax.set_ylabel("k-NN Distance Score", fontsize=12)
    ax.set_title("Score Distribution by Label", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    # Add statistics
    stats_text = (
        f"Normal: median={np.median(normal_scores):.4f}, mean={np.mean(normal_scores):.4f}\n"
        f"Anomaly: median={np.median(anomaly_scores):.4f}, mean={np.mean(anomaly_scores):.4f}"
    )
    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved box plot to {output_path}")
    plt.close()


def plot_template_coverage(
    template_counts: dict[str, int],
    found_templates: set[str],
    missed_templates: set[str],
    output_path: Path,
    title: str = "Template Coverage: Detected vs Missed",
):
    """Create bar chart showing which templates were detected vs missed.

    Visualizes detection results against ground truth templates.
    Shows rare template prioritization without circular reasoning.

    Args:
        template_counts: Dict mapping template_id -> occurrence count
        found_templates: Set of template IDs that were detected
        missed_templates: Set of template IDs that were missed
        output_path: Where to save the figure
        title: Figure title
    """
    # sort templates by frequency (descending - most common first)
    # matplotlib barh plots from bottom to top, so first item appears at bottom
    # we want rarest at top, so reverse the sort
    all_templates = sorted(template_counts.items(), key=lambda x: x[1], reverse=True)
    template_ids = [t[0] for t in all_templates]
    counts = [t[1] for t in all_templates]

    # Determine colors: green = detected, red = missed
    colors = ["#2ecc71" if tid in found_templates else "#e74c3c" for tid in template_ids]

    # Create figure
    _, ax = plt.subplots(figsize=(12, 8))

    # Create bars
    ax.barh(range(len(template_ids)), counts, color=colors, edgecolor="black", linewidth=0.5)

    # Add template IDs as y-axis labels
    ax.set_yticks(range(len(template_ids)))
    ax.set_yticklabels(template_ids, fontsize=10)

    # Log scale for x-axis (makes rare vs common more visible)
    ax.set_xscale("log")
    ax.set_xlabel("Occurrence Count (log scale)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Template ID", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

    # Add grid
    ax.grid(True, alpha=0.3, axis="x", which="both")

    # Add legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#2ecc71", edgecolor="black", label=f"Detected ({len(found_templates)})"),
        Patch(facecolor="#e74c3c", edgecolor="black", label=f"Missed ({len(missed_templates)})"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=11, framealpha=0.9)

    # Add statistics box
    total_templates = len(template_ids)
    detection_rate = len(found_templates) / total_templates if total_templates > 0 else 0

    # Identify rare templates (< 100 occurrences)
    rare_templates = {tid for tid, count in template_counts.items() if count < 100}
    rare_found = found_templates & rare_templates
    rare_rate = len(rare_found) / len(rare_templates) if rare_templates else 0

    stats_text = (
        f"Overall Detection: {len(found_templates)}/{total_templates} ({detection_rate:.1%})\n"
        f"Rare Templates (<100): {len(rare_found)}/{len(rare_templates)} ({rare_rate:.1%})\n"
        f"Median Frequency: {np.median(counts):.0f} occurrences"
    )
    ax.text(
        0.98,
        0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox={"boxstyle": "round", "facecolor": "lightyellow", "alpha": 0.9, "edgecolor": "black"},
    )

    # Add explanation
    explanation = (
        "Green = Detected (found these anomaly types)\n"
        "Red = Missed (didn't find these anomaly types)\n"
        "Bars show frequency (rarer templates at top)"
    )
    ax.text(
        0.02,
        0.02,
        explanation,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="bottom",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8, "edgecolor": "black"},
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved template coverage chart to {output_path}")
    plt.close()


def main():
    """Main visualization CLI."""
    parser = argparse.ArgumentParser(
        description="Visualize embedding space and score distributions"
    )
    parser.add_argument(
        "dataset",
        default="hdfs_v1",
        nargs="?",
        help="Dataset ID from manifest",
    )
    parser.add_argument(
        "--sample-size",
        type=str,
        default="100000",
        help="Number of log lines (smaller = faster, e.g., 100000)",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=4,
        help="Window size (non-overlapping windows)",
    )
    parser.add_argument(
        "--k-neighbors",
        type=int,
        default=5,
        help="K-neighbors for scoring",
    )
    parser.add_argument(
        "--model",
        default="all-MiniLM-L6-v2",
        help="Embedding model",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark/results"),
        help="Output directory for plots",
    )

    args = parser.parse_args()

    # Load dataset config
    dataset_config = load_dataset_config(args.dataset)
    sample_size = None if args.sample_size.lower() == "full" else int(args.sample_size)

    # Paths
    data_dir = Path(dataset_config["extract_to"])
    log_path = data_dir / dataset_config["log_file"]

    if not log_path.exists():
        print(f"ERROR: Dataset not found. Run: python benchmark/download.py {args.dataset}")
        return 1

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("EMBEDDING SPACE VISUALIZATION")
    print("=" * 80)
    print(f"Dataset: {dataset_config['name']}")
    print(f"Sample size: {sample_size or 'FULL'}")
    print(f"Output directory: {args.output_dir}")
    print()

    # Load log lines
    print("Loading log file...")
    log_lines = read_log_sample(
        log_path,
        sample_size,
        total_lines=dataset_config.get("total_lines"),
    )
    print(f"  Loaded {len(log_lines):,} lines")
    print()

    # Load ground truth (HDFS session-level labels)
    labels_path = data_dir / dataset_config["labels_file"]
    print("Loading ground truth...")
    full_ground_truth = load_ground_truth(
        labels_path,
        dataset_config["id_column"],
        dataset_config["label_column"],
    )

    # Filter to sessions in sample
    print("Filtering ground truth to sampled sessions...")
    sampled_session_ids = set()
    for line in log_lines:
        session_id = extract_session_id(line, dataset_config["id_pattern"])
        if session_id:
            sampled_session_ids.add(session_id)

    ground_truth = {
        sid: label for sid, label in full_ground_truth.items() if sid in sampled_session_ids
    }
    print(f"  Sessions in sample: {len(ground_truth):,}")
    print()

    # Get embeddings
    print("Generating embeddings...")
    config = AnalysisConfig(
        window_size=args.window_size,
        k_neighbors=args.k_neighbors,
        model_name=args.model,
    )
    embeddings, window_ranges = get_embeddings(log_lines, config, sample_size)
    print(f"  Created {len(embeddings):,} windows")
    print()

    # Get window labels
    print("Mapping windows to ground truth labels...")
    window_labels = get_window_labels(
        window_ranges,
        log_lines,
        ground_truth,
        dataset_config["id_pattern"],
    )
    anomaly_window_count = sum(1 for label in window_labels if label == "Anomaly")
    print(
        f"  Anomalous windows: {anomaly_window_count:,} ({anomaly_window_count/len(window_labels):.2%})"
    )
    print()

    # Compute k-NN scores
    print("Computing k-NN distance scores...")
    scores = compute_knn_scores(embeddings, config)
    print(f"  Score range: [{scores.min():.4f}, {scores.max():.4f}]")
    print()

    # Generate visualizations
    sample_name = f"{sample_size//1000}k" if sample_size else "full"

    print("Generating visualizations...")
    print()

    # UMAP plot
    plot_umap_embeddings(
        embeddings,
        scores,
        window_labels,
        args.output_dir / f"umap_{args.dataset}_{sample_name}.png",
        title=f"{dataset_config['name']} - UMAP Projection ({sample_name} lines)",
    )

    # template coverage chart - shows which anomaly types were detected
    # note: this requires running evaluation first to get template data
    # for standalone visualization, this plot is skipped
    # use the integrated evaluate.py script with --generate-plots for all visualizations including template coverage

    print()
    print("=" * 80)
    print("Visualization complete!")
    print(f"Plots saved to: {args.output_dir}")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    exit(main())
