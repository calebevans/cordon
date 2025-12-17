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

    # embedding backend selection
    backend_group = parser.add_argument_group("embedding backend")
    backend_group.add_argument(
        "--backend",
        type=str,
        choices=["sentence-transformers", "llama-cpp", "remote"],
        default="sentence-transformers",
        help="Embedding backend to use (default: sentence-transformers)",
    )
    backend_group.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="GGUF model path (auto-downloads default if omitted)",
    )
    backend_group.add_argument(
        "--n-gpu-layers",
        type=int,
        default=0,
        help="Number of layers to offload to GPU (llama-cpp only, default: 0)",
    )
    backend_group.add_argument(
        "--n-threads",
        type=int,
        default=None,
        help="Thread count for llama.cpp (default: auto-detect)",
    )
    backend_group.add_argument(
        "--n-ctx",
        type=int,
        default=2048,
        help="Context size for llama.cpp (default: 2048)",
    )
    backend_group.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for remote embeddings (remote backend only, falls back to env vars)",
    )
    backend_group.add_argument(
        "--endpoint",
        type=str,
        default=None,
        help="Custom API endpoint URL (remote backend only)",
    )

    # configuration options
    config_group = parser.add_argument_group("analysis configuration")
    config_group.add_argument(
        "--window-size",
        type=int,
        default=4,
        help="Number of lines per window (default: 4)",
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
        "--anomaly-range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=None,
        help="Percentile range window, e.g., '0.05 0.15' excludes top 5%%, keeps next 10%%",
    )
    config_group.add_argument(
        "--model-name",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Model name: HuggingFace for sentence-transformers, provider/model for remote (e.g., openai/text-embedding-3-small) (default: all-MiniLM-L6-v2)",
    )
    config_group.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for embeddings (default: 32)",
    )
    config_group.add_argument(
        "--device",
        type=str,
        choices=["cuda", "mps", "cpu"],
        default=None,
        help="Device for embedding and scoring (default: auto-detect)",
    )
    config_group.add_argument(
        "--scoring-batch-size",
        type=int,
        default=None,
        help="Batch size for k-NN scoring queries (default: auto-detect based on GPU memory)",
    )

    # output options
    output_group = parser.add_argument_group("output options")
    output_group.add_argument(
        "--detailed",
        action="store_true",
        help="Show detailed statistics in addition to anomalous blocks",
    )
    output_group.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Save anomalous blocks to file (default: print to stdout)",
    )

    return parser.parse_args()


def analyze_file(
    log_path: Path, analyzer: SemanticLogAnalyzer, detailed: bool, output_path: Path | None = None
) -> None:
    """Analyze a single log file and print results.

    Args:
        log_path: Path to the log file
        analyzer: Configured SemanticLogAnalyzer instance
        detailed: Whether to show detailed statistics
        output_path: Optional path to save anomalous blocks (None = stdout)
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

        # write output to file or stdout
        if output_path:
            output_path.write_text(result.output)
            print(f"Anomalous blocks written to: {output_path}")
        else:
            print(result.output)
    else:
        # run simple analysis
        output = analyzer.analyze_file(log_path)

        # write output to file or stdout
        if output_path:
            output_path.write_text(output)
            print(f"Anomalous blocks written to: {output_path}")
        else:
            print(output)

    print()


def main() -> None:
    """Main entry point for the CLI."""
    args = parse_args()

    # handle anomaly range vs percentile mutual exclusivity
    anomaly_range_min = None
    anomaly_range_max = None
    anomaly_percentile = args.anomaly_percentile

    if args.anomaly_range is not None:
        # Using range mode
        anomaly_range_min = args.anomaly_range[0]
        anomaly_range_max = args.anomaly_range[1]
        # Keep default percentile value (not used in range mode)
        if args.anomaly_percentile != 0.1:
            print(
                "Warning: --anomaly-percentile is ignored when using --anomaly-range",
                file=sys.stderr,
            )

    # create configuration from arguments
    try:
        config = AnalysisConfig(
            window_size=args.window_size,
            k_neighbors=args.k_neighbors,
            anomaly_percentile=anomaly_percentile,
            anomaly_range_min=anomaly_range_min,
            anomaly_range_max=anomaly_range_max,
            model_name=args.model_name,
            batch_size=args.batch_size,
            device=args.device,
            scoring_batch_size=args.scoring_batch_size,
            backend=args.backend,
            model_path=str(args.model_path) if args.model_path else None,
            n_gpu_layers=args.n_gpu_layers,
            n_threads=args.n_threads,
            n_ctx=args.n_ctx,
            api_key=args.api_key,
            endpoint=args.endpoint,
        )
    except ValueError as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        sys.exit(1)

    # create analyzer
    print("Initializing analyzer...")
    print(f"Backend: {config.backend}")
    if config.backend == "sentence-transformers":
        print(f"Model: {config.model_name}")
        print(f"Device: {config.device or 'auto'}")
    elif config.backend == "llama-cpp":
        print(f"Model path: {config.model_path}")
        print(f"GPU layers: {config.n_gpu_layers}")
        if config.n_threads:
            print(f"Threads: {config.n_threads}")
    elif config.backend == "remote":
        print(f"Model: {config.model_name}")
        if config.endpoint:
            print(f"Endpoint: {config.endpoint}")
        print(f"Timeout: {config.request_timeout}s")

    # Display filtering mode
    if config.anomaly_range_min is not None:
        # Type narrowing: if min is not None, max is also not None (enforced in config)
        assert config.anomaly_range_max is not None
        print(
            f"Filtering mode: Range (exclude top {config.anomaly_range_min*100:.1f}%, keep up to {config.anomaly_range_max*100:.1f}%)"
        )
    else:
        print(f"Filtering mode: Percentile (top {config.anomaly_percentile*100:.1f}%)")
    print()

    try:
        analyzer = SemanticLogAnalyzer(config)
    except ImportError as error:
        print(f"Import error: {error}", file=sys.stderr)
        print("\nTo install llama.cpp support:", file=sys.stderr)
        print("  uv pip install 'cordon[llama-cpp]'", file=sys.stderr)
        print("  or: pip install llama-cpp-python", file=sys.stderr)
        sys.exit(1)
    except Exception as error:
        print(f"Initialization error: {error}", file=sys.stderr)
        sys.exit(1)
    print()

    # analyze each log file
    for log_path in args.logfiles:
        analyze_file(log_path, analyzer, args.detailed, args.output)


if __name__ == "__main__":
    main()
