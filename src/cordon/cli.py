#!/usr/bin/env python3
import argparse
import sys
from math import isclose
from pathlib import Path

from cordon import AnalysisConfig, AnalysisResult, SemanticLogAnalyzer


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
        help="Path(s) to log file(s) to analyze (use '-' for stdin)",
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
        "--max-line-length",
        type=int,
        default=None,
        help="Maximum characters per line before splitting into virtual lines (default: no limit)",
    )
    config_group.add_argument(
        "--scoring-batch-size",
        type=int,
        default=None,
        help="Batch size for k-NN scoring queries (default: auto-detect based on GPU memory)",
    )
    config_group.add_argument(
        "--token-budget",
        type=int,
        default=None,
        help="Maximum token budget for output; dynamically adjusts percentile to fit (overrides --anomaly-percentile)",
    )
    config_group.add_argument(
        "--tokenizer-encoding",
        type=str,
        default="cl100k_base",
        help="tiktoken encoding for token counting (default: cl100k_base)",
    )
    config_group.add_argument(
        "--max-blocks",
        type=int,
        default=None,
        help="Maximum number of anomaly blocks to output (keeps highest scoring)",
    )
    config_group.add_argument(
        "--min-score",
        type=float,
        default=None,
        help="Minimum anomaly score threshold for output blocks",
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
    output_group.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output file if it exists",
    )
    output_group.add_argument(
        "--format",
        type=str,
        choices=["xml", "json"],
        default="xml",
        dest="output_format",
        help="Output format for anomaly blocks (default: xml)",
    )
    output_group.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress all human-readable banners and progress bars, keeping only formatted output on stdout",
    )
    output_group.add_argument(
        "--fail-if-anomalies",
        action="store_true",
        help="Exit with code 2 if anomalies are found (useful for CI gating)",
    )

    return parser.parse_args()


def _validate_file(log_path: Path) -> bool:
    """Check that a log file exists and is a regular file.

    Args:
        log_path: Path to the log file to validate.

    Returns:
        True if the file is valid, False otherwise.
    """
    if not log_path.exists():
        print(f"Error: File not found: {log_path}", file=sys.stderr)
        return False
    if not log_path.is_file():
        print(f"Error: Not a file: {log_path}", file=sys.stderr)
        return False
    return True


def _write_output(content: str, output_path: Path, force: bool) -> None:
    """Write analysis output to a file with overwrite protection.

    Args:
        content: The content to write.
        output_path: Destination file path.
        force: If True, overwrite an existing file.
    """
    if output_path.exists() and not force:
        print(
            f"Error: Output file already exists: {output_path}",
            file=sys.stderr,
        )
        print("Use --force to overwrite.", file=sys.stderr)
        return

    try:
        output_path.write_text(content)
        print(f"Anomalous blocks written to: {output_path}")
    except OSError as error:
        print(f"Error writing output file: {error}", file=sys.stderr)


def _display_results(
    result: AnalysisResult,
    detailed: bool,
    output_path: Path | None,
    force: bool,
    quiet: bool,
) -> None:
    """Display analysis results, optionally with detailed statistics.

    Args:
        result: The analysis result to display.
        detailed: Whether to print detailed statistics before the output.
        output_path: Optional path to save anomalous blocks (None prints to stdout).
        force: If True, overwrite an existing output file.
        quiet: If True, suppress human-readable banners and stats.
    """
    if detailed and not quiet:
        print(f"Total lines: {result.total_lines:,}")
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

    if output_path:
        _write_output(result.output, output_path, force)
    else:
        print(result.output)

    if not quiet:
        print()


def analyze_file(
    log_path: Path,
    analyzer: SemanticLogAnalyzer,
    detailed: bool,
    output_path: Path | None = None,
    force: bool = False,
    quiet: bool = False,
) -> bool:
    """Analyze a single log file and print results.

    Args:
        log_path: Path to the log file.
        analyzer: Configured SemanticLogAnalyzer instance.
        detailed: Whether to show detailed statistics.
        output_path: Optional path to save anomalous blocks (None prints to stdout).
        force: If True, overwrite an existing output file.
        quiet: If True, suppress human-readable banners and stats.

    Returns:
        True if anomalies were found, False otherwise.
    """
    if not _validate_file(log_path):
        return False

    if not quiet:
        print("=" * 80)
        print(f"Analyzing: {log_path}")
        print("=" * 80)

    try:
        result = analyzer.analyze_file_detailed(log_path)
    except Exception as error:
        print(f"Error analyzing {log_path}: {error}", file=sys.stderr)
        return False

    _display_results(result, detailed, output_path, force, quiet)

    return result.merged_blocks > 0


def analyze_stdin(
    analyzer: SemanticLogAnalyzer,
    detailed: bool,
    output_path: Path | None = None,
    force: bool = False,
    quiet: bool = False,
) -> bool:
    """Analyze log data from stdin and print results.

    Args:
        analyzer: Configured SemanticLogAnalyzer instance.
        detailed: Whether to show detailed statistics.
        output_path: Optional path to save anomalous blocks (None prints to stdout).
        force: If True, overwrite an existing output file.
        quiet: If True, suppress human-readable banners and stats.

    Returns:
        True if anomalies were found, False otherwise.
    """
    text = sys.stdin.read()

    if not quiet:
        print("=" * 80)
        print("Analyzing: <stdin>")
        print("=" * 80)

    try:
        result = analyzer.analyze_text_detailed(text)
    except Exception as error:
        print(f"Error analyzing <stdin>: {error}", file=sys.stderr)
        return False

    _display_results(result, detailed, output_path, force, quiet)

    return result.merged_blocks > 0


def _print_backend_info(config: AnalysisConfig) -> None:
    """Print backend configuration details."""
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


def _print_filtering_mode(config: AnalysisConfig) -> None:
    """Print filtering mode configuration."""
    if config.anomaly_range_min is not None:
        if config.anomaly_range_max is None:
            raise ValueError("anomaly_range_max must be set when anomaly_range_min is set")
        print(
            f"Filtering mode: Range (exclude top {config.anomaly_range_min*100:.1f}%, keep up to {config.anomaly_range_max*100:.1f}%)"
        )
    else:
        print(f"Filtering mode: Percentile (top {config.anomaly_percentile*100:.1f}%)")


def _main_impl() -> None:
    """Implementation of the main CLI entry point."""
    args = parse_args()

    # handle anomaly range vs percentile mutual exclusivity
    anomaly_range_min = None
    anomaly_range_max = None
    anomaly_percentile = args.anomaly_percentile

    if args.anomaly_range is not None:
        anomaly_range_min = args.anomaly_range[0]
        anomaly_range_max = args.anomaly_range[1]
        if not isclose(args.anomaly_percentile, 0.1):
            print(
                "Warning: --anomaly-percentile is ignored when using --anomaly-range",
                file=sys.stderr,
            )

    if args.token_budget is not None and not isclose(args.anomaly_percentile, 0.1):
        print(
            "Warning: --anomaly-percentile is overridden by --token-budget",
            file=sys.stderr,
        )

    # create configuration from arguments
    try:
        config = AnalysisConfig(
            window_size=args.window_size,
            max_line_length=args.max_line_length,
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
            show_progress=not args.quiet,
            token_budget=args.token_budget,
            tokenizer_encoding=args.tokenizer_encoding,
            output_format=args.output_format,
            max_blocks=args.max_blocks,
            min_score=args.min_score,
        )
    except ValueError as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        sys.exit(1)

    if not args.quiet:
        print("Initializing analyzer...")
        _print_backend_info(config)
        _print_filtering_mode(config)
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

    if not args.quiet:
        print()

    # analyze each log file
    any_anomalies_found = False
    for log_path in args.logfiles:
        if str(log_path) == "-":
            found = analyze_stdin(
                analyzer, args.detailed, args.output, args.force, quiet=args.quiet
            )
        else:
            found = analyze_file(
                log_path, analyzer, args.detailed, args.output, args.force, quiet=args.quiet
            )
        if found:
            any_anomalies_found = True

    if args.fail_if_anomalies and any_anomalies_found:
        sys.exit(2)


def main() -> None:
    """Main entry point for the CLI."""
    try:
        _main_impl()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
