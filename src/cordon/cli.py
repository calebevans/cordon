#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

from cordon import AnalysisConfig, SemanticLogAnalyzer


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        prog="cordon",
        description="Analyze log files for anomalous patterns using semantic similarity",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # positional arguments
    parser.add_argument(
        "logfiles",
        type=Path,
        nargs="+",
        help="Path(s) to log file(s) to analyze",
    )

    # configuration options
    config_group = parser.add_argument_group("analysis configuration")
    config_group.add_argument(
        "--window-size",
        type=int,
        default=10,
        help="Number of lines per window (default: 10)",
    )
    config_group.add_argument(
        "--stride",
        type=int,
        default=5,
        help="Step size for sliding window in lines (default: 5)",
    )
    config_group.add_argument(
        "--k-neighbors",
        type=int,
        default=5,
        help="Number of neighbors for k-NN density calculation (default: 5)",
    )
    config_group.add_argument(
        "--anomaly-percentile",
        type=float,
        default=0.1,
        help="Percentile of windows to retain, e.g., 0.1 = top 10%% (default: 0.1)",
    )
    config_group.add_argument(
        "--model-name",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Sentence-transformers model name (default: all-MiniLM-L6-v2)",
    )
    config_group.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for embedding operations (default: 32)",
    )
    config_group.add_argument(
        "--device",
        type=str,
        choices=["cuda", "mps", "cpu"],
        default=None,
        help="Device for model inference (default: auto-detect)",
    )
    config_group.add_argument(
        "--use-faiss",
        action="store_true",
        help="Use FAISS for k-NN search (faster for large logs, requires faiss-cpu or faiss-gpu)",
    )

    # output options
    output_group = parser.add_argument_group("output options")
    output_group.add_argument(
        "--detailed",
        action="store_true",
        help="Show detailed statistics in addition to anomalous blocks",
    )

    return parser.parse_args()


def analyze_file(log_path: Path, analyzer: SemanticLogAnalyzer, detailed: bool) -> None:
    """Analyze a single log file and print results.

    Args:
        log_path: Path to the log file
        analyzer: Configured SemanticLogAnalyzer instance
        detailed: Whether to show detailed statistics
    """
    # verify file exists and is readable
    if not log_path.exists():
        print(f"Error: File not found: {log_path}", file=sys.stderr)
        return
    if not log_path.is_file():
        print(f"Error: Not a file: {log_path}", file=sys.stderr)
        return

    # count lines in file
    with open(log_path) as log_file:
        line_count = sum(1 for _ in log_file)

    print("=" * 80)
    print(f"Analyzing: {log_path}")
    print(f"Total lines: {line_count:,}")
    print("=" * 80)

    if detailed:
        # run detailed analysis
        result = analyzer.analyze_file_detailed(log_path)

        print("\nAnalysis Statistics:")
        print(f"  Total windows created: {result.total_windows:,}")
        print(f"  Significant windows: {result.significant_windows:,}")
        print(f"  Merged blocks: {result.merged_blocks}")
        print(f"  Processing time: {result.processing_time:.2f}s")
        print("\nScore Distribution:")
        print(f"  Min:    {result.score_distribution['min']:.4f}")
        print(f"  Mean:   {result.score_distribution['mean']:.4f}")
        print(f"  Median: {result.score_distribution['median']:.4f}")
        print(f"  P90:    {result.score_distribution['p90']:.4f}")
        print(f"  Max:    {result.score_distribution['max']:.4f}")

        print(f"\n{'Significant Blocks':^80}")
        print("=" * 80)
        print(result.output)
    else:
        # run simple analysis
        output = analyzer.analyze_file(log_path)
        print(output)

    print()


def main() -> None:
    """Main entry point for the CLI."""
    args = parse_args()

    # create configuration from arguments
    try:
        config = AnalysisConfig(
            window_size=args.window_size,
            stride=args.stride,
            k_neighbors=args.k_neighbors,
            anomaly_percentile=args.anomaly_percentile,
            model_name=args.model_name,
            batch_size=args.batch_size,
            device=args.device,
            use_faiss_threshold=0 if args.use_faiss else None,
        )
    except ValueError as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        sys.exit(1)

    # create analyzer
    print("Initializing analyzer...")
    analyzer = SemanticLogAnalyzer(config)
    print(f"Using model: {config.model_name}")
    print(f"Device: {config.device or 'auto'}")
    print()

    # analyze each log file
    for log_path in args.logfiles:
        analyze_file(log_path, analyzer, args.detailed)


if __name__ == "__main__":
    main()
