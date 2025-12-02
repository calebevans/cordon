#!/usr/bin/env python3
"""Evaluate Cordon on HDFS dataset using template-based metrics.

This evaluation measures semantic uniqueness detection - the core of what Cordon does.
Rather than counting error lines, we count unique error TYPES (templates) detected.

This script can run standalone evaluation or full benchmark (evaluation + visualizations + structured output).
"""

import argparse
import csv
import json
import os
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# suppress tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from cordon import AnalysisConfig, SemanticLogAnalyzer


def load_dataset_config(dataset_id: str) -> dict[str, Any]:
    """Load dataset configuration from manifest."""
    with open("benchmark/datasets.yaml") as f:
        manifest = yaml.safe_load(f)

    if dataset_id not in manifest["datasets"]:
        available = ", ".join(manifest["datasets"].keys())
        raise ValueError(f"Unknown dataset '{dataset_id}'. Available: {available}")

    return manifest["datasets"][dataset_id]


def extract_session_id(line: str, pattern: str) -> str | None:
    """Extract session ID from log line using regex pattern."""
    match = re.search(pattern, line)
    return match.group(0) if match else None


def read_log_sample(
    log_path: Path,
    sample_size: int | None,
    total_lines: int | None = None,
) -> list[str]:
    """Read log file, sampling N lines from random offset."""
    import random

    # determine starting offset (random if sampling)
    if sample_size and total_lines:
        max_start = max(0, total_lines - sample_size)
        start_offset = random.randint(0, max_start)
        print(f"  Random sampling: offset {start_offset:,} to {start_offset + sample_size:,}")
    else:
        start_offset = 0

    lines = []
    with open(log_path) as f:
        # skip to starting offset
        for _ in range(start_offset):
            next(f, None)

        # read sample_size lines
        for i, line in enumerate(f):
            if sample_size and i >= sample_size:
                break
            lines.append(line.rstrip("\n"))

    return lines


def load_ground_truth(labels_path: Path, id_column: str, label_column: str) -> dict[str, str]:
    """Load ground truth labels from CSV.

    Returns:
        Dict mapping session_id -> label ('Normal' or 'Anomaly')
    """
    labels = {}
    with open(labels_path) as f:
        header = f.readline().strip().split(",")
        id_idx = header.index(id_column)
        label_idx = header.index(label_column)

        for line in f:
            parts = line.strip().split(",")
            session_id = parts[id_idx]
            label = parts[label_idx]
            labels[session_id] = label

    return labels


def load_hdfs_session_templates(
    traces_path: Path,
    labels_path: Path,
) -> tuple[dict[str, list[str]], dict[str, str]]:
    """Load HDFS session templates and labels.

    Returns:
        Tuple of (session_templates, session_labels)
        - session_templates: Dict mapping BlockId -> list of template IDs
        - session_labels: Dict mapping BlockId -> "Normal" or "Anomaly"
    """
    session_templates = {}
    with open(traces_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            block_id = row["BlockId"]
            # parse features field which contains template list
            features = row["Features"].strip('[]"').split(",")
            templates = [t.strip() for t in features if t.strip().startswith("E")]
            session_templates[block_id] = templates

    session_labels = {}
    with open(labels_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            block_id = row["BlockId"]
            label = row["Label"]
            session_labels[block_id] = label

    return session_templates, session_labels


def load_template_patterns(templates_path: Path) -> dict[str, re.Pattern[str]]:
    """Load HDFS template patterns for line-level matching.

    Returns:
        Dict mapping template ID -> compiled regex pattern
    """
    patterns = {}
    with open(templates_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            template_id = row["EventId"]
            template_pattern = row["EventTemplate"]

            # convert template wildcards [*] to regex pattern
            # escape special regex chars, then replace [*] with .*?
            regex_pattern = re.escape(template_pattern)
            regex_pattern = regex_pattern.replace(r"\[\*\]", ".*?")

            # match from start of log content (after timestamp)
            # hdfs logs start with timestamp, we match the rest
            patterns[template_id] = re.compile(regex_pattern)

    return patterns


def match_line_to_template(line: str, template_patterns: dict[str, re.Pattern[str]]) -> str | None:
    """Match a log line to its template ID.

    Args:
        line: Log line to match
        template_patterns: Dict of template_id -> regex pattern

    Returns:
        Template ID if matched, None otherwise
    """
    # hdfs log format: "YYMMDD HHMMSS microseconds LEVEL message"
    # skip timestamp and match message content
    parts = line.split(maxsplit=4)
    if len(parts) < 5:
        return None

    message = parts[4]

    # try each template pattern
    for template_id, pattern in template_patterns.items():
        if pattern.search(message):
            return template_id

    return None


def run_cordon_analysis(
    log_lines: list[str],
    config: AnalysisConfig,
    return_intermediates: bool = False,
) -> tuple[list[tuple[int, int]], dict[str, Any] | None]:
    """Run Cordon on log file and extract flagged line ranges.

    Args:
        log_lines: Log lines to analyze
        config: Cordon configuration
        return_intermediates: If True, return embeddings/scores for visualization (avoids recomputation)

    Returns:
        Tuple of (flagged_ranges, intermediates_dict or None)
        - flagged_ranges: List of (start_line, end_line) tuples
        - intermediates: Dict with embeddings, scores, windows (if return_intermediates=True)
    """
    from cordon.analysis.scorer import DensityAnomalyScorer
    from cordon.embedding import create_vectorizer
    from cordon.ingestion.reader import LogFileReader
    from cordon.segmentation.windower import SlidingWindowSegmenter

    # create temporary file
    temp_log = Path(f"/tmp/cordon_eval_{os.getpid()}.log")

    with open(temp_log, "w") as fout:
        for line in log_lines:
            fout.write(line + "\n")

    if return_intermediates:
        # run pipeline manually to capture intermediates
        reader = LogFileReader()
        lines = reader.read_lines(temp_log)

        segmenter = SlidingWindowSegmenter()
        windows = list(segmenter.segment(lines, config))

        vectorizer = create_vectorizer(config)
        embedded = list(vectorizer.embed_windows(windows))

        scorer = DensityAnomalyScorer()
        scored = scorer.score_windows(embedded, config)

        # extract embeddings and scores for later use
        embeddings = np.array([sw.embedding for sw in scored])
        scores = np.array([sw.score for sw in scored])
        window_ranges = [(w.window.start_line, w.window.end_line) for w in scored]

        # get flagged ranges from scored windows
        from cordon.analysis.thresholder import Thresholder

        thresholder = Thresholder()
        significant = thresholder.select_significant(scored, config)

        # extract line ranges from significant windows
        flagged_ranges = []
        for sw in significant:
            flagged_ranges.append((sw.window.start_line, sw.window.end_line))

        intermediates = {
            "embeddings": embeddings,
            "scores": scores,
            "window_ranges": window_ranges,
        }
    else:
        # run standard pipeline
        analyzer = SemanticLogAnalyzer(config)
        result = analyzer.analyze_file_detailed(temp_log)

        # parse output to extract line ranges
        flagged_ranges = []
        output_lines = result.output.split("\n")

        for line in output_lines:
            # look for <block lines="X-Y" score="Z">
            match = re.search(r'<block lines="(\d+)-(\d+)" score="([0-9.]+)">', line)
            if match:
                start = int(match.group(1))
                end = int(match.group(2))
                flagged_ranges.append((start, end))

        intermediates = None

    # clean up temp file
    if temp_log.exists():
        temp_log.unlink()

    return flagged_ranges, intermediates


def evaluate_traditional_metrics(
    flagged_ranges: list[tuple[int, int]],
    anomaly_lines: set[int],
    total_lines: int,
) -> dict[str, float]:
    """Compute traditional line-level precision, recall, F1."""
    # extract flagged line numbers
    flagged_lines = set()
    for start, end in flagged_ranges:
        for line_num in range(start, end + 1):
            flagged_lines.add(line_num)

    # calculate metrics
    true_positives = len(flagged_lines & anomaly_lines)
    false_positives = len(flagged_lines - anomaly_lines)
    false_negatives = len(anomaly_lines - flagged_lines)

    precision = true_positives / len(flagged_lines) if flagged_lines else 0
    recall = true_positives / len(anomaly_lines) if anomaly_lines else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "flagged_lines": len(flagged_lines),
        "anomaly_lines": len(anomaly_lines),
        "total_lines": total_lines,
    }


def evaluate_template_coverage(
    flagged_templates: set[str],
    anomaly_templates: set[str],
) -> dict[str, Any]:
    """Evaluate template coverage metrics."""
    found_templates = flagged_templates & anomaly_templates
    missed_templates = anomaly_templates - flagged_templates

    recall = len(found_templates) / len(anomaly_templates) if anomaly_templates else 0
    precision = len(found_templates) / len(flagged_templates) if flagged_templates else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "unique_templates_found": len(found_templates),
        "unique_templates_total": len(anomaly_templates),
        "template_recall": recall,
        "template_precision": precision,
        "template_f1": f1,
        "found_templates": sorted(found_templates),
        "missed_templates": sorted(missed_templates),
    }


def evaluate_rare_template_detection(
    flagged_templates: set[str],
    template_counts: dict[str, int],
    anomaly_templates: set[str],
    threshold: int = 100,
) -> dict[str, Any]:
    """Evaluate detection of rare templates (< threshold occurrences)."""
    # identify rare anomaly templates
    rare_templates = {t for t in anomaly_templates if template_counts.get(t, 0) < threshold}

    # check which rare templates were found
    found_rare = flagged_templates & rare_templates

    recall = len(found_rare) / len(rare_templates) if rare_templates else 0

    return {
        "rare_templates_found": len(found_rare),
        "rare_templates_total": len(rare_templates),
        "rare_template_recall": recall,
        "found_rare": sorted(found_rare),
        "missed_rare": sorted(rare_templates - found_rare),
        "threshold": threshold,
    }


def evaluate_frequency_weighted_recall(
    flagged_templates: set[str],
    template_counts: dict[str, int],
    anomaly_templates: set[str],
) -> float:
    """Compute frequency-weighted recall (rarer = higher weight)."""
    if not anomaly_templates:
        return 0.0

    # weight = 1 / count (rarer templates have higher weight)
    total_weight = sum(1.0 / template_counts.get(t, 1) for t in anomaly_templates)
    found_weight = sum(
        1.0 / template_counts.get(t, 1) for t in flagged_templates & anomaly_templates
    )

    return found_weight / total_weight if total_weight > 0 else 0.0


def evaluate_hdfs(
    dataset_config: dict[str, Any],
    log_lines: list[str],
    flagged_ranges: list[tuple[int, int]],
) -> dict[str, Any]:
    """Evaluate HDFS dataset using LINE-LEVEL template matching.

    This function uses proper line-level template attribution:
    - Only templates that appear IN THE FLAGGED LINES are credited as found
    - Fixes the previous session-level attribution flaw
    """
    data_dir = Path(dataset_config["extract_to"])

    # load templates and labels
    traces_path = data_dir / dataset_config["traces_file"]
    labels_path = data_dir / dataset_config["labels_file"]
    templates_path = data_dir / dataset_config["templates_file"]

    session_templates, session_labels = load_hdfs_session_templates(traces_path, labels_path)
    template_patterns = load_template_patterns(templates_path)

    # extract session IDs from sampled log lines
    id_pattern = dataset_config["id_pattern"]
    sampled_sessions = set()
    for line in log_lines:
        session_id = extract_session_id(line, id_pattern)
        if session_id:
            sampled_sessions.add(session_id)

    # filter to sampled sessions
    sampled_templates = {
        sid: templates for sid, templates in session_templates.items() if sid in sampled_sessions
    }
    sampled_labels = {
        sid: label for sid, label in session_labels.items() if sid in sampled_sessions
    }

    # get anomaly sessions and their templates
    anomaly_sessions = {sid for sid, label in sampled_labels.items() if label == "Anomaly"}
    anomaly_templates = set()
    template_counts = Counter()

    for sid in anomaly_sessions:
        templates = sampled_templates.get(sid, [])
        anomaly_templates.update(templates)
        template_counts.update(templates)

    # line-level template matching (fixed):
    # only credit templates that actually appear in the flagged lines
    flagged_templates = set()
    for start, end in flagged_ranges:
        for line_idx in range(start - 1, end):  # convert to 0-indexed
            if line_idx < len(log_lines):
                line = log_lines[line_idx]
                template_id = match_line_to_template(line, template_patterns)
                if template_id and template_id in anomaly_templates:
                    flagged_templates.add(template_id)

    # build anomaly line set for traditional metrics
    anomaly_lines = set()
    for line_idx, line in enumerate(log_lines, start=1):
        session_id = extract_session_id(line, id_pattern)
        if session_id and sampled_labels.get(session_id) == "Anomaly":
            anomaly_lines.add(line_idx)

    # compute metrics
    traditional = evaluate_traditional_metrics(flagged_ranges, anomaly_lines, len(log_lines))
    template_coverage = evaluate_template_coverage(flagged_templates, anomaly_templates)
    rare_detection = evaluate_rare_template_detection(
        flagged_templates, template_counts, anomaly_templates
    )
    freq_weighted = evaluate_frequency_weighted_recall(
        flagged_templates, template_counts, anomaly_templates
    )

    return {
        "traditional": traditional,
        "template_coverage": template_coverage,
        "rare_detection": rare_detection,
        "frequency_weighted_recall": freq_weighted,
        "template_counts": template_counts,
        "anomaly_templates": anomaly_templates,
        "sampled_sessions": len(sampled_sessions),
        "anomaly_sessions": len(anomaly_sessions),
        "flagged_templates_detail": sorted(flagged_templates),  # for debugging
    }


def aggregate_results(all_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate results from multiple runs into mean and std."""
    import numpy as np

    # extract key metrics across all runs
    template_recalls = [r["template_coverage"]["template_recall"] for r in all_results]
    rare_recalls = [r["rare_detection"]["rare_template_recall"] for r in all_results]
    freq_weighted = [r["frequency_weighted_recall"] for r in all_results]
    traditional_f1s = [r["traditional"]["f1"] for r in all_results]
    traditional_precisions = [r["traditional"]["precision"] for r in all_results]
    traditional_recalls = [r["traditional"]["recall"] for r in all_results]

    return {
        "template_recall_mean": np.mean(template_recalls),
        "template_recall_std": np.std(template_recalls),
        "rare_recall_mean": np.mean(rare_recalls),
        "rare_recall_std": np.std(rare_recalls),
        "freq_weighted_mean": np.mean(freq_weighted),
        "freq_weighted_std": np.std(freq_weighted),
        "traditional_f1_mean": np.mean(traditional_f1s),
        "traditional_f1_std": np.std(traditional_f1s),
        "traditional_precision_mean": np.mean(traditional_precisions),
        "traditional_precision_std": np.std(traditional_precisions),
        "traditional_recall_mean": np.mean(traditional_recalls),
        "traditional_recall_std": np.std(traditional_recalls),
        "num_runs": len(all_results),
    }


def print_aggregated_results(
    dataset_name: str, sample_size: int | None, aggregated: dict[str, Any]
):
    """Print aggregated results from multiple runs."""
    print()
    print("=" * 80)
    print(f"CORDON EVALUATION (AGGREGATED): {dataset_name}")
    print("=" * 80)
    print(f"Sample: {sample_size:,} lines" if sample_size else "Sample: FULL dataset")
    print(f"Runs: {aggregated['num_runs']}")
    print()

    print("AGGREGATED METRICS (Mean ± Std):")
    print()

    print("  Traditional (Line-Level):")
    print(
        f"    Precision: {aggregated['traditional_precision_mean']:.4f} ± {aggregated['traditional_precision_std']:.4f}"
    )
    print(
        f"    Recall:    {aggregated['traditional_recall_mean']:.4f} ± {aggregated['traditional_recall_std']:.4f}"
    )
    print(
        f"    F1 Score:  {aggregated['traditional_f1_mean']:.4f} ± {aggregated['traditional_f1_std']:.4f}"
    )
    print("    Note: Low F1 expected - Cordon ignores repetitive errors")
    print()

    print("  Template Coverage (PRIMARY METRIC):")
    print(
        f"    Template Recall: {aggregated['template_recall_mean']:.4f} ± {aggregated['template_recall_std']:.4f}"
    )
    print(
        f"                     ({aggregated['template_recall_mean']*100:.1f}% ± {aggregated['template_recall_std']*100:.1f}%)"
    )
    print()

    print("  Rare Template Detection:")
    print(
        f"    Rare Recall: {aggregated['rare_recall_mean']:.4f} ± {aggregated['rare_recall_std']:.4f}"
    )
    print(
        f"                 ({aggregated['rare_recall_mean']*100:.1f}% ± {aggregated['rare_recall_std']*100:.1f}%)"
    )
    print()

    print(
        f"  Frequency-Weighted Recall: {aggregated['freq_weighted_mean']:.4f} ± {aggregated['freq_weighted_std']:.4f}"
    )
    print()

    # Interpretation
    print("INTERPRETATION:")
    variance_pct = (
        (aggregated["template_recall_std"] / aggregated["template_recall_mean"] * 100)
        if aggregated["template_recall_mean"] > 0
        else 0
    )
    print(
        f"  Template Recall: {aggregated['template_recall_mean']*100:.1f}% (CV: {variance_pct:.1f}%)"
    )
    if variance_pct > 15:
        print("  WARNING: High variance detected (CV > 15%). Results may be sensitive to sampling.")
    else:
        print("  Low variance (CV < 15%). Results are stable.")
    print("=" * 80)


def create_run_directory(base_dir: Path, run_name: str | None = None) -> Path:
    """Create structured directory for benchmark run."""
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"run_{timestamp}"

    run_dir = base_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # create plots subdirectory
    (run_dir / "plots").mkdir(exist_ok=True)

    return run_dir


def save_parameters(output_dir: Path, params: dict[str, Any]):
    """Save evaluation parameters to YAML file."""
    param_file = output_dir / "parameters.yaml"
    with open(param_file, "w") as f:
        yaml.dump(params, f, default_flow_style=False, sort_keys=False)
    print(f"Saved parameters to {param_file}")


def save_results_json(output_dir: Path, results: dict[str, Any], run_idx: int | None = None):
    """Save evaluation results to JSON."""
    if run_idx is not None:
        result_file = output_dir / f"results_run{run_idx + 1}.json"
    else:
        result_file = output_dir / "results.json"

    # convert to JSON-serializable format
    json_results = {}
    for key, value in results.items():
        if isinstance(value, dict):
            json_results[key] = {
                k: (list(v) if isinstance(v, set) else v) for k, v in value.items()
            }
        elif isinstance(value, set):
            json_results[key] = sorted(value)
        elif isinstance(value, Counter):
            json_results[key] = dict(value)
        elif isinstance(value, np.integer | np.floating):
            json_results[key] = float(value)
        else:
            json_results[key] = value

    with open(result_file, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"Saved results to {result_file}")


def generate_visualizations(
    log_lines: list[str],
    config: AnalysisConfig,
    dataset_config: dict[str, Any],
    results: dict[str, Any],
    output_dir: Path,
    sample_size: int | None = None,
    cached_intermediates: dict[str, Any] | None = None,
):
    """Generate all benchmark visualizations.

    Args:
        log_lines: Log lines
        config: Cordon configuration
        dataset_config: Dataset configuration
        results: Evaluation results
        output_dir: Output directory
        sample_size: Sample size
        cached_intermediates: Optional pre-computed embeddings/scores to avoid recomputation
    """
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)

    # import visualization functions
    from visualize import (
        compute_knn_scores,
        get_embeddings,
        get_window_labels,
        plot_template_coverage,
        plot_umap_embeddings,
    )

    plots_dir = output_dir / "plots"

    # Use cached embeddings if available, otherwise generate
    if cached_intermediates:
        print("Using cached embeddings from analysis...")
        embeddings = cached_intermediates["embeddings"]
        scores = cached_intermediates["scores"]
        window_ranges = cached_intermediates["window_ranges"]
        print(f"  Loaded {len(embeddings):,} windows")
    else:
        # get embeddings
        print("Generating embeddings...")
        embeddings, window_ranges = get_embeddings(log_lines, config, sample_size)
        print(f"  Created {len(embeddings):,} windows")

        # compute k-NN scores
        print("Computing k-NN scores...")
        scores = compute_knn_scores(embeddings, config)

    # get window labels
    print("Mapping windows to ground truth...")
    data_dir = Path(dataset_config["extract_to"])
    labels_path = data_dir / dataset_config["labels_file"]
    full_ground_truth = load_ground_truth(
        labels_path,
        dataset_config["id_column"],
        dataset_config["label_column"],
    )

    sampled_session_ids = set()
    for line in log_lines:
        session_id = extract_session_id(line, dataset_config["id_pattern"])
        if session_id:
            sampled_session_ids.add(session_id)

    ground_truth = {
        sid: label for sid, label in full_ground_truth.items() if sid in sampled_session_ids
    }

    window_labels = get_window_labels(
        window_ranges,
        log_lines,
        ground_truth,
        dataset_config["id_pattern"],
    )

    sample_name = f"{sample_size//1000}k" if sample_size else "full"

    # generate plots
    print("\nGenerating plots...")

    # 1. UMAP projection
    plot_umap_embeddings(
        embeddings,
        scores,
        window_labels,
        plots_dir / f"umap_{sample_name}.png",
        title=f"{dataset_config['name']} - UMAP Projection",
    )

    # 2. template coverage chart
    if "template_counts" in results and "template_coverage" in results:
        template_counts = dict(results["template_counts"])
        found_templates = set(results["template_coverage"]["found_templates"])
        missed_templates = set(results["template_coverage"]["missed_templates"])

        plot_template_coverage(
            template_counts,
            found_templates,
            missed_templates,
            plots_dir / f"template_coverage_{sample_name}.png",
            title=f"{dataset_config['name']} - Template Detection Coverage",
        )

    print("\nAll visualizations generated")


def print_results(
    dataset_name: str,
    sample_size: int | None,
    results: dict[str, Any],
    flagged_ranges: list[tuple[int, int]],
):
    """Print evaluation results in a clear format."""
    print()
    print("=" * 80)
    print(f"CORDON EVALUATION: {dataset_name}")
    print("=" * 80)
    print(f"Sample: {sample_size:,} lines" if sample_size else "Sample: FULL dataset")
    print()

    # Template analysis
    print("TEMPLATE ANALYSIS (LINE-LEVEL MATCHING):")
    print(
        f"  Anomalous sessions: {results['anomaly_sessions']:,} of {results['sampled_sessions']:,}"
    )
    print(f"  Unique anomaly templates: {len(results['anomaly_templates'])}")
    print("  Note: Using line-level template matching (only credits templates in flagged lines)")

    # Show rarest templates
    if results["template_counts"]:
        rare_items = sorted(results["template_counts"].items(), key=lambda x: x[1])[:5]
        print("  Rarest templates:")
        for template, count in rare_items:
            print(f"    {template}: {count:,} occurrences")
    print()

    # Cordon flagged
    print("CORDON FLAGGED:")
    print(f"  Blocks: {len(flagged_ranges)}")
    total_flagged_lines = sum(end - start + 1 for start, end in flagged_ranges)
    print(f"  Lines covered: {total_flagged_lines:,}")
    print()

    # Metrics
    print("METRICS:")
    print()

    trad = results["traditional"]
    print("  Traditional (Line-Level):")
    print(f"    Precision: {trad['precision']:.4f}")
    print(f"    Recall:    {trad['recall']:.4f}")
    print(f"    F1 Score:  {trad['f1']:.4f}")
    print("    Note: Low F1 expected - Cordon ignores repetitive errors")
    print()

    tmpl = results["template_coverage"]
    print("  Template Coverage:")
    print(
        f"    Unique templates found: {tmpl['unique_templates_found']} of {tmpl['unique_templates_total']} ({tmpl['template_recall']:.1%})"
    )
    print(f"    Template Precision: {tmpl['template_precision']:.4f}")
    print(f"    Template Recall:    {tmpl['template_recall']:.4f}")
    print(f"    Template F1:        {tmpl['template_f1']:.4f}")
    if tmpl["found_templates"]:
        print(f"    Templates found: {', '.join(tmpl['found_templates'][:10])}")
        if len(tmpl["found_templates"]) > 10:
            print(f"      ... and {len(tmpl['found_templates']) - 10} more")
    print()

    rare = results["rare_detection"]
    print(f"  Rare Template Detection (< {rare['threshold']} occurrences):")
    print(
        f"    Found: {rare['rare_templates_found']} of {rare['rare_templates_total']} ({rare['rare_template_recall']:.1%})"
    )
    if rare["found_rare"]:
        print(f"    Templates found: {', '.join(rare['found_rare'])}")
    if rare["missed_rare"]:
        print(f"    Templates missed: {', '.join(rare['missed_rare'])}")
    print()

    print(f"  Frequency-Weighted Recall: {results['frequency_weighted_recall']:.4f}")
    print("    (Higher = found rarer patterns)")
    print()

    # Interpretation
    print("INTERPRETATION:")
    tmpl_recall_pct = tmpl["template_recall"] * 100
    line_coverage_pct = (
        (total_flagged_lines / trad["total_lines"]) * 100 if trad["total_lines"] > 0 else 0
    )
    print(
        f"  Cordon found {tmpl_recall_pct:.1f}% of unique anomaly types while flagging only {line_coverage_pct:.2f}% of lines."
    )
    print("  Traditional metrics penalize ignoring repetitive errors (by design).")
    print("=" * 80)


def main():
    """Main evaluation CLI."""
    parser = argparse.ArgumentParser(
        description="Evaluate Cordon using template-based metrics on HDFS dataset"
    )
    parser.add_argument(
        "dataset",
        default="hdfs_v1",
        nargs="?",
        help="Dataset ID from manifest (default: hdfs_v1)",
    )
    parser.add_argument(
        "--sample-size",
        type=str,
        default="100000",
        help="Number of log lines to analyze (e.g., 100000 or 'full')",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=4,
        help="Cordon window size (non-overlapping)",
    )
    parser.add_argument(
        "--k-neighbors",
        type=int,
        default=5,
        help="Cordon k-neighbors",
    )
    parser.add_argument(
        "--anomaly-percentile",
        type=float,
        default=0.1,
        help="Cordon anomaly percentile (0.0-1.0)",
    )
    parser.add_argument(
        "--model",
        default="all-MiniLM-L6-v2",
        help="Embedding model name",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "mps", "cpu"],
        help="Device to use for embedding and scoring (default: auto-detect)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of runs for statistical reporting (default: 1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: random)",
    )
    parser.add_argument(
        "--generate-plots",
        action="store_true",
        help="Generate visualizations (UMAP, template coverage)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Create structured output directory with results and plots",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Custom name for run directory (default: auto timestamp)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for embedding generation (default: 32, increase for faster processing on high-VRAM GPUs)",
    )
    parser.add_argument(
        "--scoring-batch-size",
        type=int,
        default=None,
        help="Batch size for k-NN scoring queries (default: auto-detect based on GPU memory)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing run directory, continuing until --runs target is reached",
    )

    args = parser.parse_args()

    # Load dataset config
    dataset_config = load_dataset_config(args.dataset)
    sample_size = None if args.sample_size.lower() == "full" else int(args.sample_size)

    # Dataset paths
    data_dir = Path(dataset_config["extract_to"])
    log_path = data_dir / dataset_config["log_file"]

    # Check if dataset exists
    if not log_path.exists():
        print(f"ERROR: Dataset not found at {data_dir}")
        print(f"       Run: python benchmark/download.py {args.dataset}")
        return 1

    # create output directory if requested
    output_dir = None
    start_run_idx = 0
    existing_seed = None

    if args.output_dir or args.generate_plots:
        base_dir = args.output_dir if args.output_dir else Path("benchmark/runs")

        # check for resume
        if args.resume:
            if not args.run_name:
                print("ERROR: --resume requires --run-name to specify which run to resume")
                return 1
            output_dir = base_dir / args.run_name
            if not output_dir.exists():
                print(f"ERROR: Resume directory not found: {output_dir}")
                return 1

            # count existing runs
            existing_runs = sorted(output_dir.glob("run_*/results_run*.json"))
            start_run_idx = len(existing_runs)

            if start_run_idx >= args.runs:
                print(
                    f"Resume: Already have {start_run_idx} runs (target: {args.runs}). Nothing to do."
                )
                return 0

            # load seed from parameters
            params_file = output_dir / "parameters.yaml"
            if params_file.exists():
                import yaml

                with open(params_file) as f:
                    params = yaml.safe_load(f)
                    existing_seed = params.get("seed")

            print("=" * 80)
            print("CORDON INTEGRATED BENCHMARK (RESUME)")
            print("=" * 80)
            print(f"Output directory: {output_dir}")
            print(f"Resuming from run {start_run_idx + 1}/{args.runs}")
        else:
            output_dir = create_run_directory(base_dir, args.run_name)
            print("=" * 80)
            print("CORDON INTEGRATED BENCHMARK")
            print("=" * 80)
            print(f"Output directory: {output_dir}")
    else:
        print("=" * 80)
        print("CORDON SEMANTIC UNIQUENESS BENCHMARK")
        print("=" * 80)

    print(f"Dataset: {args.dataset}")
    print(f"Log file: {log_path}")
    print(f"Sample size: {sample_size or 'FULL'}")
    print(f"Runs: {args.runs}")
    if args.seed is not None:
        print(f"Random seed: {args.seed}")
    print()

    # configure Cordon
    config = AnalysisConfig(
        window_size=args.window_size,
        k_neighbors=args.k_neighbors,
        anomaly_percentile=args.anomaly_percentile,
        model_name=args.model,
        device=args.device,
        batch_size=args.batch_size,
        scoring_batch_size=args.scoring_batch_size,
    )

    print("Cordon Configuration:")
    print(f"  Window size: {config.window_size}")
    print(f"  K-neighbors: {config.k_neighbors}")
    print(f"  Anomaly percentile: {config.anomaly_percentile:.2%}")
    print(f"  Model: {config.model_name}")
    print(f"  Device: {config.device or 'auto-detect'}")
    print(f"  Batch size: {config.batch_size}")
    print()

    # save parameters if output directory specified (skip if resuming)
    if output_dir and not args.resume:
        parameters = {
            "dataset": args.dataset,
            "sample_size": sample_size,
            "window_size": args.window_size,
            "k_neighbors": args.k_neighbors,
            "anomaly_percentile": args.anomaly_percentile,
            "model": args.model,
            "device": args.device or "auto-detect",
            "batch_size": args.batch_size,
            "runs": args.runs,
            "seed": args.seed,
            "timestamp": datetime.now().isoformat(),
        }
        save_parameters(output_dir, parameters)

    # Multiple runs for statistical reporting
    all_results = []
    all_flagged_ranges = []
    first_run_log_lines = None  # Save for visualization
    first_run_intermediates = None  # Save cached embeddings

    import random

    # determine seed (use existing if resuming)
    if args.resume and existing_seed is not None:
        base_seed = existing_seed
        print(f"Using original seed: {base_seed}")
    else:
        base_seed = args.seed if args.seed is not None else random.randint(0, 1000000)

    for run_idx in range(start_run_idx, args.runs):
        current_seed = base_seed + run_idx
        random.seed(current_seed)

        if args.runs > 1:
            print(f"--- Run {run_idx + 1}/{args.runs} (seed: {current_seed}) ---")

        # load log sample (with different random offset each run)
        print("Loading log file...")
        log_lines = read_log_sample(
            log_path,
            sample_size,
            total_lines=dataset_config.get("total_lines"),
        )
        print(f"  Loaded {len(log_lines):,} lines")
        print()

        # run Cordon (cache intermediates if generating plots)
        print("Running Cordon analysis...")
        print("  (This may take several minutes)")
        flagged_ranges, intermediates = run_cordon_analysis(
            log_lines, config, return_intermediates=args.generate_plots
        )
        print(f"  Flagged {len(flagged_ranges)} blocks")
        print()

        # save first run's data for summary visualization
        if run_idx == 0:
            first_run_log_lines = log_lines
            first_run_intermediates = intermediates

        # evaluate
        print("Computing template-based metrics...")
        results = evaluate_hdfs(dataset_config, log_lines, flagged_ranges)

        all_results.append(results)
        all_flagged_ranges.append(flagged_ranges)

        # save intermediate aggregated results after each run (crash recovery)
        if output_dir and args.runs > 1 and len(all_results) > 1:
            partial_agg = aggregate_results(all_results)
            partial_agg["num_runs_completed"] = len(all_results)
            partial_file = output_dir / "aggregated_results_partial.json"
            with open(partial_file, "w") as f:
                json.dump(partial_agg, f, indent=2)

        # save results if output directory specified
        if output_dir and args.runs > 1:
            run_dir = output_dir / f"run_{run_idx + 1}"
            run_dir.mkdir(exist_ok=True)
            (run_dir / "plots").mkdir(exist_ok=True)  # create plots subdirectory
            save_results_json(run_dir, results, run_idx)

            # generate per-run visualizations using cached embeddings
            if args.generate_plots:
                try:
                    print(f"\nGenerating visualizations for Run {run_idx + 1}...")
                    generate_visualizations(
                        log_lines,
                        config,
                        dataset_config,
                        results,
                        run_dir,
                        sample_size,
                        cached_intermediates=intermediates,
                    )
                except Exception as e:
                    print(f"WARNING: Visualization failed for Run {run_idx + 1}: {e}")
                    print("Continuing with remaining runs...")

        # print individual run results if multiple runs
        if args.runs > 1:
            print(f"Run {run_idx + 1} Results:")
            print(f"  Template Recall: {results['template_coverage']['template_recall']:.1%}")
            print(f"  Rare Recall: {results['rare_detection']['rare_template_recall']:.1%}")
            print(f"  Freq-Weighted: {results['frequency_weighted_recall']:.4f}")
            print(f"  Traditional F1: {results['traditional']['f1']:.4f}")
            print()

        # cleanup between runs to prevent memory accumulation
        del log_lines, flagged_ranges, intermediates

        # force GPU memory cleanup if using CUDA
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except ImportError:
            pass

    # print final results
    if args.runs == 1:
        # single run - print detailed results
        print_results(dataset_config["name"], sample_size, all_results[0], all_flagged_ranges[0])

        # save results if output directory specified
        if output_dir:
            save_results_json(output_dir, all_results[0])

            # generate visualizations for single run using cached embeddings
            if args.generate_plots and first_run_intermediates:
                generate_visualizations(
                    first_run_log_lines,
                    config,
                    dataset_config,
                    all_results[0],
                    output_dir,
                    sample_size,
                    cached_intermediates=first_run_intermediates,
                )
    else:
        # multiple runs - print aggregated statistics
        # if resuming, load existing results and combine
        if args.resume and output_dir:
            print("Loading existing results for aggregation...")
            all_results_combined = []
            for i in range(args.runs):
                result_file = output_dir / f"run_{i + 1}" / f"results_run{i + 1}.json"
                if result_file.exists():
                    with open(result_file) as f:
                        all_results_combined.append(json.load(f))
            aggregated = aggregate_results(all_results_combined)
        else:
            aggregated = aggregate_results(all_results)

        print_aggregated_results(dataset_config["name"], sample_size, aggregated)
        print()
        print("Individual run details available above.")

        # save aggregated results if output directory specified
        if output_dir:
            agg_file = output_dir / "aggregated_results.json"
            with open(agg_file, "w") as f:
                json.dump(aggregated, f, indent=2)
            print(f"Saved aggregated results to {agg_file}")

            # cleanup partial results file now that run completed successfully
            partial_file = output_dir / "aggregated_results_partial.json"
            if partial_file.exists():
                partial_file.unlink()

            # generate summary visualizations from Run 1 using cached embeddings
            if args.generate_plots and first_run_intermediates:
                print("\nGenerating summary visualizations (from Run 1)...")
                generate_visualizations(
                    first_run_log_lines,
                    config,
                    dataset_config,
                    all_results[0],
                    output_dir,
                    sample_size,
                    cached_intermediates=first_run_intermediates,
                )
                print("Note: Per-run visualizations are in run_N/plots/ directories")

    # final output message
    if output_dir:
        print("\n" + "=" * 80)
        print("BENCHMARK COMPLETE")
        print(f"Results saved to: {output_dir}")
        print("=" * 80)

    return 0


if __name__ == "__main__":
    exit(main())
