"""Tests for the CLI module."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from cordon.cli import analyze_file, analyze_stdin, parse_args


class TestParseArgs:
    """Tests for argument parsing."""

    def test_basic_args(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test parsing basic positional arguments."""
        monkeypatch.setattr(sys, "argv", ["cordon", "test.log"])
        args = parse_args()
        assert args.logfiles == [Path("test.log")]
        assert args.backend == "sentence-transformers"
        assert args.window_size == 4

    def test_multiple_files(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test parsing multiple log files."""
        monkeypatch.setattr(sys, "argv", ["cordon", "a.log", "b.log"])
        args = parse_args()
        assert len(args.logfiles) == 2

    def test_backend_choice(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test backend selection."""
        monkeypatch.setattr(sys, "argv", ["cordon", "--backend", "remote", "test.log"])
        args = parse_args()
        assert args.backend == "remote"

    def test_device_choice(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test device selection."""
        monkeypatch.setattr(sys, "argv", ["cordon", "--device", "cpu", "test.log"])
        args = parse_args()
        assert args.device == "cpu"

    def test_output_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test output file flag."""
        monkeypatch.setattr(sys, "argv", ["cordon", "-o", "out.xml", "test.log"])
        args = parse_args()
        assert args.output == Path("out.xml")

    def test_force_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test --force flag."""
        monkeypatch.setattr(sys, "argv", ["cordon", "--force", "-o", "out.xml", "test.log"])
        args = parse_args()
        assert args.force is True

    def test_force_flag_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test --force defaults to False."""
        monkeypatch.setattr(sys, "argv", ["cordon", "test.log"])
        args = parse_args()
        assert args.force is False

    def test_anomaly_range(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test anomaly range parsing."""
        monkeypatch.setattr(
            sys,
            "argv",
            ["cordon", "--anomaly-range", "0.05", "0.15", "test.log"],
        )
        args = parse_args()
        assert args.anomaly_range == [0.05, 0.15]

    def test_detailed_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test --detailed flag."""
        monkeypatch.setattr(sys, "argv", ["cordon", "--detailed", "test.log"])
        args = parse_args()
        assert args.detailed is True

    def test_format_json(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test --format json flag."""
        monkeypatch.setattr(sys, "argv", ["cordon", "--format", "json", "test.log"])
        args = parse_args()
        assert args.output_format == "json"

    def test_format_default_xml(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that --format defaults to xml."""
        monkeypatch.setattr(sys, "argv", ["cordon", "test.log"])
        args = parse_args()
        assert args.output_format == "xml"

    def test_stdin_argument(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that '-' is accepted as a logfile argument for stdin."""
        monkeypatch.setattr(sys, "argv", ["cordon", "-"])
        args = parse_args()
        assert str(args.logfiles[0]) == "-"

    def test_token_budget(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test --token-budget parsing."""
        monkeypatch.setattr(sys, "argv", ["cordon", "--token-budget", "500", "test.log"])
        args = parse_args()
        assert args.token_budget == 500

    def test_token_budget_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that --token-budget defaults to None."""
        monkeypatch.setattr(sys, "argv", ["cordon", "test.log"])
        args = parse_args()
        assert args.token_budget is None

    def test_tokenizer_encoding(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test --tokenizer-encoding parsing."""
        monkeypatch.setattr(
            sys, "argv", ["cordon", "--tokenizer-encoding", "p50k_base", "test.log"]
        )
        args = parse_args()
        assert args.tokenizer_encoding == "p50k_base"

    def test_tokenizer_encoding_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that --tokenizer-encoding defaults to cl100k_base."""
        monkeypatch.setattr(sys, "argv", ["cordon", "test.log"])
        args = parse_args()
        assert args.tokenizer_encoding == "cl100k_base"


class TestAnalyzeFile:
    """Tests for analyze_file function."""

    def test_nonexistent_file(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test handling of nonexistent file."""
        analyzer = MagicMock()
        analyze_file(Path("/nonexistent/file.log"), analyzer, detailed=False)
        captured = capsys.readouterr()
        assert "Error: File not found" in captured.err

    def test_not_a_file(self, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
        """Test handling of a path that exists but is not a regular file."""
        analyzer = MagicMock()
        analyze_file(tmp_path, analyzer, detailed=False)
        captured = capsys.readouterr()
        assert "Error: Not a file" in captured.err

    def test_analysis_error_continues(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        """Test that analysis errors are caught and reported."""
        log_file = tmp_path / "test.log"
        log_file.write_text("line 1\nline 2\n")

        analyzer = MagicMock()
        analyzer.analyze_file_detailed.side_effect = RuntimeError("Model failed")

        analyze_file(log_file, analyzer, detailed=True)
        captured = capsys.readouterr()
        assert "Error analyzing" in captured.err
        assert "Model failed" in captured.err

    def test_analysis_error_non_detailed(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        """Test that analysis errors in non-detailed mode are caught."""
        log_file = tmp_path / "test.log"
        log_file.write_text("line 1\nline 2\n")

        analyzer = MagicMock()
        analyzer.analyze_file_detailed.side_effect = ValueError("Bad input")

        analyze_file(log_file, analyzer, detailed=False)
        captured = capsys.readouterr()
        assert "Error analyzing" in captured.err
        assert "Bad input" in captured.err

    def test_write_failure(self, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
        """Test handling of output write failure."""
        log_file = tmp_path / "test.log"
        log_file.write_text("line 1\n")

        analyzer = MagicMock()
        mock_result = MagicMock()
        mock_result.output = "<anomalies></anomalies>"
        mock_result.total_lines = 1
        mock_result.total_windows = 0
        mock_result.significant_windows = 0
        mock_result.merged_blocks = 0
        mock_result.processing_time = 0.1
        mock_result.score_distribution = {
            "min": 0,
            "max": 0,
            "mean": 0,
            "median": 0,
            "p90": 0,
        }
        analyzer.analyze_file_detailed.return_value = mock_result

        bad_output = Path("/nonexistent/dir/output.xml")
        analyze_file(log_file, analyzer, detailed=True, output_path=bad_output, force=True)
        captured = capsys.readouterr()
        assert "Error writing output file" in captured.err

    def test_overwrite_protection(self, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
        """Test that existing output files are not overwritten without --force."""
        log_file = tmp_path / "test.log"
        log_file.write_text("line 1\n")
        output_file = tmp_path / "output.xml"
        output_file.write_text("existing content")

        analyzer = MagicMock()
        mock_result = MagicMock()
        mock_result.output = "<anomalies/>"
        mock_result.merged_blocks = 0
        analyzer.analyze_file_detailed.return_value = mock_result

        analyze_file(
            log_file,
            analyzer,
            detailed=False,
            output_path=output_file,
            force=False,
        )

        assert output_file.read_text() == "existing content"
        captured = capsys.readouterr()
        assert "already exists" in captured.err
        assert "--force" in captured.err

    def test_force_overwrite(self, tmp_path: Path) -> None:
        """Test that --force allows overwriting."""
        log_file = tmp_path / "test.log"
        log_file.write_text("line 1\n")
        output_file = tmp_path / "output.xml"
        output_file.write_text("existing content")

        analyzer = MagicMock()
        mock_result = MagicMock()
        mock_result.output = "<anomalies>new</anomalies>"
        mock_result.merged_blocks = 1
        analyzer.analyze_file_detailed.return_value = mock_result

        analyze_file(
            log_file,
            analyzer,
            detailed=False,
            output_path=output_file,
            force=True,
        )
        assert output_file.read_text() == "<anomalies>new</anomalies>"

    def test_stdout_output(self, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
        """Test that output is printed to stdout when no output path is given."""
        log_file = tmp_path / "test.log"
        log_file.write_text("line 1\n")

        analyzer = MagicMock()
        mock_result = MagicMock()
        mock_result.output = "<anomalies>results</anomalies>"
        mock_result.merged_blocks = 0
        analyzer.analyze_file_detailed.return_value = mock_result

        analyze_file(log_file, analyzer, detailed=False)
        captured = capsys.readouterr()
        assert "<anomalies>results</anomalies>" in captured.out

    def test_detailed_output(self, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
        """Test that detailed mode prints analysis statistics."""
        log_file = tmp_path / "test.log"
        log_file.write_text("line 1\n")

        analyzer = MagicMock()
        mock_result = MagicMock()
        mock_result.output = "<anomalies></anomalies>"
        mock_result.total_lines = 100
        mock_result.total_windows = 25
        mock_result.significant_windows = 3
        mock_result.merged_blocks = 2
        mock_result.processing_time = 1.23
        mock_result.score_distribution = {
            "min": 0.1,
            "max": 0.9,
            "mean": 0.5,
            "median": 0.45,
            "p90": 0.8,
        }
        analyzer.analyze_file_detailed.return_value = mock_result

        analyze_file(log_file, analyzer, detailed=True)
        captured = capsys.readouterr()
        assert "Total lines: 100" in captured.out
        assert "Total windows created: 25" in captured.out
        assert "Significant windows: 3" in captured.out
        assert "Processing time: 1.23s" in captured.out
        assert "Score Distribution:" in captured.out


class TestQuietMode:
    """Tests for --quiet banner suppression."""

    def test_quiet_suppresses_banners(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        """Test that --quiet suppresses all human-readable banners."""
        log_file = tmp_path / "test.log"
        log_file.write_text("line 1\n")

        analyzer = MagicMock()
        mock_result = MagicMock()
        mock_result.output = "<anomalies/>"
        mock_result.merged_blocks = 0
        analyzer.analyze_file_detailed.return_value = mock_result

        analyze_file(log_file, analyzer, detailed=True, quiet=True)
        captured = capsys.readouterr()
        assert "Analyzing:" not in captured.out
        assert "=" * 80 not in captured.out
        assert "Total lines" not in captured.out
        assert "Score Distribution" not in captured.out
        assert "<anomalies/>" in captured.out

    def test_quiet_suppresses_stdin_banners(
        self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that --quiet suppresses banners for stdin analysis."""
        monkeypatch.setattr("sys.stdin", MagicMock(read=MagicMock(return_value="line 1\n")))

        analyzer = MagicMock()
        mock_result = MagicMock()
        mock_result.output = "<anomalies/>"
        mock_result.merged_blocks = 0
        analyzer.analyze_text_detailed.return_value = mock_result

        analyze_stdin(analyzer, detailed=True, quiet=True)
        captured = capsys.readouterr()
        assert "Analyzing:" not in captured.out
        assert "=" * 80 not in captured.out
        assert "<anomalies/>" in captured.out


class TestNewFlags:
    """Tests for --fail-if-anomalies, --max-blocks, and --min-score flags."""

    def test_fail_if_anomalies_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that --fail-if-anomalies is parsed correctly."""
        monkeypatch.setattr(sys, "argv", ["cordon", "--fail-if-anomalies", "test.log"])
        args = parse_args()
        assert args.fail_if_anomalies is True

    def test_fail_if_anomalies_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that --fail-if-anomalies defaults to False."""
        monkeypatch.setattr(sys, "argv", ["cordon", "test.log"])
        args = parse_args()
        assert args.fail_if_anomalies is False

    def test_max_blocks_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that --max-blocks is parsed correctly."""
        monkeypatch.setattr(sys, "argv", ["cordon", "--max-blocks", "10", "test.log"])
        args = parse_args()
        assert args.max_blocks == 10

    def test_max_blocks_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that --max-blocks defaults to None."""
        monkeypatch.setattr(sys, "argv", ["cordon", "test.log"])
        args = parse_args()
        assert args.max_blocks is None

    def test_min_score_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that --min-score is parsed correctly."""
        monkeypatch.setattr(sys, "argv", ["cordon", "--min-score", "0.5", "test.log"])
        args = parse_args()
        assert args.min_score == 0.5

    def test_min_score_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that --min-score defaults to None."""
        monkeypatch.setattr(sys, "argv", ["cordon", "test.log"])
        args = parse_args()
        assert args.min_score is None

    def test_analyze_file_returns_true_when_anomalies(self, tmp_path: Path) -> None:
        """Test that analyze_file returns True when anomalies are found."""
        log_file = tmp_path / "test.log"
        log_file.write_text("line 1\n")

        analyzer = MagicMock()
        mock_result = MagicMock()
        mock_result.output = "<anomalies>block</anomalies>"
        mock_result.merged_blocks = 3
        analyzer.analyze_file_detailed.return_value = mock_result

        result = analyze_file(log_file, analyzer, detailed=False, quiet=True)
        assert result is True

    def test_analyze_file_returns_false_when_no_anomalies(self, tmp_path: Path) -> None:
        """Test that analyze_file returns False when no anomalies are found."""
        log_file = tmp_path / "test.log"
        log_file.write_text("line 1\n")

        analyzer = MagicMock()
        mock_result = MagicMock()
        mock_result.output = "<anomalies/>"
        mock_result.merged_blocks = 0
        analyzer.analyze_file_detailed.return_value = mock_result

        result = analyze_file(log_file, analyzer, detailed=False, quiet=True)
        assert result is False


class TestMainEntryPoint:
    """Tests for the main() entry point."""

    def test_keyboard_interrupt(self) -> None:
        """Test that KeyboardInterrupt results in exit code 130."""
        from cordon.cli import main

        with pytest.raises(SystemExit) as exc_info, pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                "cordon.cli._main_impl",
                MagicMock(side_effect=KeyboardInterrupt),
            )
            main()
        assert exc_info.value.code == 130
