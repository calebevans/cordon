#!/usr/bin/env python3
from pathlib import Path

from cordon import AnalysisConfig, SemanticLogAnalyzer


def main() -> None:
    """Demonstrate basic library usage."""
    # Example 1: Percentile mode (default) - keep top 10% most anomalous
    config = AnalysisConfig(
        window_size=4,
        k_neighbors=5,
        anomaly_percentile=0.1,
        model_name="all-MiniLM-L6-v2",
        batch_size=32,
        device="cpu",  # or "cuda", "mps", or None for auto-detect
    )

    # create analyzer instance
    analyzer = SemanticLogAnalyzer(config)

    # analyze a log file (simple API)
    log_path = Path(__file__).parent / "apache_sample.log"
    output = analyzer.analyze_file(log_path)
    print("Anomalous blocks (percentile mode):")
    print(output)

    # or use detailed API for statistics
    result = analyzer.analyze_file_detailed(log_path)
    print("\nStatistics:")
    print(f"  Total windows: {result.total_windows}")
    print(f"  Significant windows: {result.significant_windows}")
    print(f"  Processing time: {result.processing_time:.2f}s")
    print("\nScore distribution:")
    print(f"  Mean: {result.score_distribution['mean']:.4f}")
    print(f"  Max: {result.score_distribution['max']:.4f}")
    print("\nOutput:")
    print(result.output)

    # Example 2: Range mode - exclude top 5%, keep next 10%
    print("\n" + "=" * 60)
    print("Range mode: exclude top 5%, keep next 10%")
    print("=" * 60)
    config_range = AnalysisConfig(
        window_size=4,
        k_neighbors=5,
        anomaly_range_min=0.05,  # exclude top 5% (most extreme)
        anomaly_range_max=0.15,  # include up to 15% (keep next 10%)
        device="cpu",
    )
    analyzer_range = SemanticLogAnalyzer(config_range)
    result_range = analyzer_range.analyze_file_detailed(log_path)
    print(f"  Significant windows: {result_range.significant_windows}")
    print("  (Excludes the most extreme anomalies, focuses on moderate ones)")


if __name__ == "__main__":
    main()
