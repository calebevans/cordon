"""Unit tests for pipeline with mocked components."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from cordon.core.config import AnalysisConfig
from cordon.pipeline import SemanticLogAnalyzer


class TestPipelineDI:
    """Test dependency injection in SemanticLogAnalyzer."""

    @patch("cordon.pipeline.create_embedder")
    def test_custom_reader(self, mock_create: MagicMock, default_config: AnalysisConfig) -> None:
        """Test that a custom reader is used when provided."""
        mock_embedder = MagicMock()
        mock_create.return_value = mock_embedder
        mock_embedder.embed_windows.return_value = iter([])

        mock_reader = MagicMock()
        mock_reader.read_lines.return_value = iter([(1, "test line")])

        analyzer = SemanticLogAnalyzer(default_config, reader=mock_reader)
        analyzer.analyze_file_detailed(Path("dummy.log"))

        mock_reader.read_lines.assert_called_once()

    @patch("cordon.pipeline.create_embedder")
    def test_custom_formatter(self, mock_create: MagicMock, default_config: AnalysisConfig) -> None:
        """Test that a custom formatter is used when provided."""
        mock_embedder = MagicMock()
        mock_create.return_value = mock_embedder
        mock_embedder.embed_windows.return_value = iter([])

        mock_reader = MagicMock()
        mock_reader.read_lines.return_value = iter([(1, "line")])

        mock_formatter = MagicMock()
        mock_formatter.format_blocks.return_value = "<custom/>"

        analyzer = SemanticLogAnalyzer(default_config, reader=mock_reader, formatter=mock_formatter)
        result = analyzer.analyze_file_detailed(Path("dummy.log"))

        mock_formatter.format_blocks.assert_called_once()
        assert result.output == "<custom/>"
        assert isinstance(result.blocks, list)

    @patch("cordon.pipeline.create_embedder")
    def test_default_components_used_when_none_provided(
        self, mock_create: MagicMock, default_config: AnalysisConfig
    ) -> None:
        """Test that default concrete classes are used when no custom components given."""
        from cordon.analysis.scorer import DensityAnomalyScorer
        from cordon.analysis.thresholder import Thresholder as ThresholderImpl
        from cordon.ingestion.reader import LogFileReader
        from cordon.postprocess.formatter import XmlFormatter
        from cordon.postprocess.merger import IntervalMerger
        from cordon.segmentation.windower import SlidingWindowSegmenter

        mock_create.return_value = MagicMock()

        analyzer = SemanticLogAnalyzer(default_config)

        assert isinstance(analyzer._reader, LogFileReader)
        assert isinstance(analyzer._segmenter, SlidingWindowSegmenter)
        assert isinstance(analyzer._scorer, DensityAnomalyScorer)
        assert isinstance(analyzer._thresholder, ThresholderImpl)
        assert isinstance(analyzer._merger, IntervalMerger)
        assert isinstance(analyzer._formatter, XmlFormatter)

    @patch("cordon.pipeline.create_embedder")
    def test_custom_scorer(self, mock_create: MagicMock, default_config: AnalysisConfig) -> None:
        """Test that a custom scorer is used when provided."""
        mock_embedder = MagicMock()
        mock_create.return_value = mock_embedder
        mock_embedder.embed_windows.return_value = iter([])

        mock_reader = MagicMock()
        mock_reader.read_lines.return_value = iter([(1, "line")])

        mock_scorer = MagicMock()
        mock_scorer.score_windows.return_value = []

        analyzer = SemanticLogAnalyzer(default_config, reader=mock_reader, scorer=mock_scorer)
        analyzer.analyze_file_detailed(Path("dummy.log"))

        mock_scorer.score_windows.assert_called_once()

    @patch("cordon.pipeline.create_embedder")
    def test_result_blocks_populated(
        self, mock_create: MagicMock, default_config: AnalysisConfig
    ) -> None:
        """Test that result.blocks contains the merged blocks."""
        from cordon.core.types import MergedBlock, ScoredWindow, TextWindow

        mock_embedder = MagicMock()
        mock_create.return_value = mock_embedder
        mock_embedder.embed_windows.return_value = iter([])

        mock_reader = MagicMock()
        mock_reader.read_lines.return_value = iter([(1, "line 1"), (2, "line 2")])

        expected_blocks = [
            MergedBlock(start_line=1, end_line=2, original_windows=(0,), max_score=0.8),
        ]
        mock_merger = MagicMock()
        mock_merger.merge_windows.return_value = expected_blocks

        mock_thresholder = MagicMock()
        window = TextWindow(content="line 1\nline 2", start_line=1, end_line=2, window_id=0)
        mock_thresholder.select_significant.return_value = [ScoredWindow(window=window, score=0.8)]

        analyzer = SemanticLogAnalyzer(
            default_config,
            reader=mock_reader,
            merger=mock_merger,
            thresholder=mock_thresholder,
        )
        result = analyzer.analyze_file_detailed(Path("dummy.log"))

        assert result.blocks == expected_blocks
        assert len(result.blocks) == 1
        assert result.blocks[0].start_line == 1

    @patch("cordon.pipeline.create_embedder")
    def test_json_formatter_selected(self, mock_create: MagicMock) -> None:
        """Test that JSON formatter is selected when output_format is json."""
        from cordon.postprocess.json_formatter import JsonFormatter

        mock_create.return_value = MagicMock()
        config = AnalysisConfig(device="cpu", output_format="json")
        analyzer = SemanticLogAnalyzer(config)
        assert isinstance(analyzer._formatter, JsonFormatter)
