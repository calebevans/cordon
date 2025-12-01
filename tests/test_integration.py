from pathlib import Path
from tempfile import NamedTemporaryFile

import pytest

from cordon import AnalysisConfig, SemanticLogAnalyzer


class TestIntegration:
    """Integration tests for the complete analysis pipeline."""

    def test_analyze_simple_log(self) -> None:
        """Test analyzing a simple log file."""
        with NamedTemporaryFile(mode="w", delete=False, suffix=".log") as f:
            # create a log with repetitive and unique content
            for i in range(20):
                f.write(f"INFO: Normal operation {i}\n")
            f.write("ERROR: Critical failure in subsystem A\n")
            f.write("TRACE: Stack trace follows\n")
            f.write("at module.function (file.py:123)\n")
            for i in range(20):
                f.write(f"INFO: Normal operation {i}\n")
            temp_path = Path(f.name)

        try:
            config = AnalysisConfig(
                window_size=5,
                k_neighbors=3,
                anomaly_percentile=0.1,
                device="cpu",
            )
            analyzer = SemanticLogAnalyzer(config)
            result = analyzer.analyze_file_detailed(temp_path)

            # verify result structure
            assert result.total_windows > 0
            assert result.significant_windows > 0
            assert result.merged_blocks >= 0
            assert result.processing_time > 0
            assert "min" in result.score_distribution
            assert "max" in result.score_distribution
            assert result.output is not None
        finally:
            temp_path.unlink()

    def test_analyze_empty_log(self) -> None:
        """Test analyzing an empty log file."""
        with NamedTemporaryFile(mode="w", delete=False, suffix=".log") as f:
            temp_path = Path(f.name)

        try:
            config = AnalysisConfig(device="cpu")
            analyzer = SemanticLogAnalyzer(config)
            result = analyzer.analyze_file_detailed(temp_path)

            assert result.total_windows == 0
            assert result.significant_windows == 0
            assert result.merged_blocks == 0
            assert result.output == ""
        finally:
            temp_path.unlink()

    def test_analyze_single_line_log(self) -> None:
        """Test analyzing a single-line log file."""
        with NamedTemporaryFile(mode="w", delete=False, suffix=".log") as f:
            f.write("Single log line\n")
            temp_path = Path(f.name)

        try:
            config = AnalysisConfig(device="cpu")
            analyzer = SemanticLogAnalyzer(config)
            result = analyzer.analyze_file_detailed(temp_path)

            assert result.total_windows == 1
            # single window gets score 0.0, might not be selected
            assert result.processing_time > 0
        finally:
            temp_path.unlink()

    def test_analyze_with_different_configs(self) -> None:
        """Test analyzing with different configurations."""
        with NamedTemporaryFile(mode="w", delete=False, suffix=".log") as f:
            for i in range(50):
                f.write(f"Log line {i}\n")
            temp_path = Path(f.name)

        try:
            # test with high anomaly percentile
            config1 = AnalysisConfig(anomaly_percentile=0.5, device="cpu")
            analyzer1 = SemanticLogAnalyzer(config1)
            result1 = analyzer1.analyze_file_detailed(temp_path)

            # test with low anomaly percentile
            config2 = AnalysisConfig(anomaly_percentile=0.05, device="cpu")
            analyzer2 = SemanticLogAnalyzer(config2)
            result2 = analyzer2.analyze_file_detailed(temp_path)

            # higher anomaly percentile should select more windows
            assert result1.significant_windows >= result2.significant_windows
        finally:
            temp_path.unlink()

    def test_analyze_file_simple_api(self) -> None:
        """Test the simple analyze_file API."""
        with NamedTemporaryFile(mode="w", delete=False, suffix=".log") as f:
            for i in range(10):
                f.write(f"Log line {i}\n")
            temp_path = Path(f.name)

        try:
            config = AnalysisConfig(device="cpu")
            analyzer = SemanticLogAnalyzer(config)
            output = analyzer.analyze_file(temp_path)

            assert isinstance(output, str)
        finally:
            temp_path.unlink()

    def test_nonexistent_file_raises_error(self) -> None:
        """Test that analyzing a nonexistent file raises an error."""
        config = AnalysisConfig(device="cpu")
        analyzer = SemanticLogAnalyzer(config)

        with pytest.raises(FileNotFoundError):
            analyzer.analyze_file(Path("/nonexistent/file.log"))
