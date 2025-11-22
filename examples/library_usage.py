#!/usr/bin/env python3
from pathlib import Path

from cordon import AnalysisConfig, SemanticLogAnalyzer


def main() -> None:
    """Demonstrate basic library usage."""
    # create a custom configuration
    config = AnalysisConfig(
        window_size=10,
        stride=5,
        k_neighbors=5,
        anomaly_percentile=0.1,
        model_name="all-MiniLM-L6-v2",
        device="cpu",  # or "cuda", "mps", or None for auto-detect
    )

    # create analyzer instance
    analyzer = SemanticLogAnalyzer(config)

    # analyze a log file (simple API)
    log_path = Path("sample.log")
    output = analyzer.analyze_file(log_path)
    print("Anomalous blocks:")
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


if __name__ == "__main__":
    main()
