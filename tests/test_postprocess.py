from cordon.core.types import MergedBlock, ScoredWindow, TextWindow
from cordon.postprocess.formatter import XmlFormatter
from cordon.postprocess.merger import IntervalMerger


class TestIntervalMerger:
    """Tests for IntervalMerger class."""

    def test_merge_no_overlap(self) -> None:
        """Test merging windows with no overlap."""
        scored = [
            ScoredWindow(
                window=TextWindow(content="w1", start_line=1, end_line=3, window_id=0),
                score=0.5,
            ),
            ScoredWindow(
                window=TextWindow(content="w2", start_line=10, end_line=12, window_id=1),
                score=0.5,
            ),
        ]

        merger = IntervalMerger()
        merged = merger.merge_windows(scored)

        assert len(merged) == 2
        assert merged[0].start_line == 1
        assert merged[0].end_line == 3
        assert merged[1].start_line == 10
        assert merged[1].end_line == 12

    def test_merge_overlapping_windows(self) -> None:
        """Test merging overlapping windows."""
        scored = [
            ScoredWindow(
                window=TextWindow(content="w1", start_line=1, end_line=5, window_id=0),
                score=0.5,
            ),
            ScoredWindow(
                window=TextWindow(content="w2", start_line=3, end_line=7, window_id=1),
                score=0.5,
            ),
            ScoredWindow(
                window=TextWindow(content="w3", start_line=6, end_line=10, window_id=2),
                score=0.5,
            ),
        ]

        merger = IntervalMerger()
        merged = merger.merge_windows(scored)

        assert len(merged) == 1
        assert merged[0].start_line == 1
        assert merged[0].end_line == 10
        assert merged[0].original_windows == (0, 1, 2)

    def test_merge_adjacent_windows(self) -> None:
        """Test merging adjacent windows (lines N and N+1)."""
        scored = [
            ScoredWindow(
                window=TextWindow(content="w1", start_line=1, end_line=5, window_id=0),
                score=0.5,
            ),
            ScoredWindow(
                window=TextWindow(content="w2", start_line=6, end_line=10, window_id=1),
                score=0.5,
            ),
        ]

        merger = IntervalMerger()
        merged = merger.merge_windows(scored)

        # adjacent windows should merge
        assert len(merged) == 1
        assert merged[0].start_line == 1
        assert merged[0].end_line == 10

    def test_merge_preserves_max_score(self) -> None:
        """Test that merging preserves the maximum score."""
        scored = [
            ScoredWindow(
                window=TextWindow(content="w1", start_line=1, end_line=5, window_id=0),
                score=0.8,
            ),
            ScoredWindow(
                window=TextWindow(content="w2", start_line=3, end_line=7, window_id=1),
                score=0.5,
            ),
        ]

        merger = IntervalMerger()
        merged = merger.merge_windows(scored)

        assert len(merged) == 1
        assert merged[0].max_score == 0.8

    def test_merge_empty_windows(self) -> None:
        """Test merging with no windows."""
        scored: list[ScoredWindow] = []

        merger = IntervalMerger()
        merged = merger.merge_windows(scored)

        assert len(merged) == 0

    def test_merge_single_window(self) -> None:
        """Test merging with a single window."""
        scored = [
            ScoredWindow(
                window=TextWindow(content="w1", start_line=1, end_line=5, window_id=0),
                score=0.5,
            ),
        ]

        merger = IntervalMerger()
        merged = merger.merge_windows(scored)

        assert len(merged) == 1
        assert merged[0].start_line == 1
        assert merged[0].end_line == 5


class TestXmlFormatter:
    """Tests for XmlFormatter class."""

    def test_format_single_block(self) -> None:
        """Test formatting a single block."""
        lines = [(1, "line 1"), (2, "line 2"), (3, "line 3")]
        blocks = [MergedBlock(start_line=1, end_line=2, original_windows=(0,), max_score=0.8)]

        formatter = XmlFormatter()
        output = formatter.format_blocks(blocks, lines)

        assert '<?xml version="1.0" encoding="UTF-8"?>' in output
        assert "<anomalies>" in output
        assert "</anomalies>" in output
        assert '<block lines="1-2" score="0.8000">' in output
        assert "line 1" in output
        assert "line 2" in output
        assert "</block>" in output

    def test_format_multiple_blocks(self) -> None:
        """Test formatting multiple blocks."""
        lines = [(i, f"line {i}") for i in range(1, 11)]
        blocks = [
            MergedBlock(start_line=1, end_line=2, original_windows=(0,), max_score=0.8),
            MergedBlock(start_line=5, end_line=7, original_windows=(1,), max_score=0.9),
        ]

        formatter = XmlFormatter()
        output = formatter.format_blocks(blocks, lines)

        assert '<?xml version="1.0" encoding="UTF-8"?>' in output
        assert "<anomalies>" in output
        assert "</anomalies>" in output
        assert '<block lines="1-2" score="0.8000">' in output
        assert '<block lines="5-7" score="0.9000">' in output
        assert output.count("</block>") == 2

    def test_format_empty_blocks(self) -> None:
        """Test formatting with no blocks."""
        lines = [(1, "line 1")]
        blocks: list[MergedBlock] = []

        formatter = XmlFormatter()
        output = formatter.format_blocks(blocks, lines)

        assert output == '<?xml version="1.0" encoding="UTF-8"?>\n<anomalies></anomalies>'

    def test_format_escapes_xml_special_chars(self) -> None:
        """Test that XML special characters are properly escaped."""
        lines = [
            (1, "command: test |& tee file.txt"),
            (2, "error: x < y && z > 10"),
            (3, "message: \"quoted\" & 'single'"),
        ]
        blocks = [MergedBlock(start_line=1, end_line=3, original_windows=(0,), max_score=0.8)]

        formatter = XmlFormatter()
        output = formatter.format_blocks(blocks, lines)

        assert '<?xml version="1.0" encoding="UTF-8"?>' in output
        assert "<anomalies>" in output
        assert "</anomalies>" in output
        assert "&amp;" in output
        assert "&lt;" in output
        assert "&gt;" in output
        assert "command: test |&amp; tee file.txt" in output
        assert "error: x &lt; y &amp;&amp; z &gt; 10" in output
        assert "|& tee" not in output
        assert "x < y" not in output
        assert "z > 10" not in output

    def test_format_multi_block_content_correctness(self) -> None:
        """Test that multi-block formatting extracts correct content for each block."""
        lines = [(i, f"line {i}") for i in range(1, 11)]
        blocks = [
            MergedBlock(start_line=2, end_line=3, original_windows=(0,), max_score=0.7),
            MergedBlock(start_line=7, end_line=8, original_windows=(1,), max_score=0.9),
        ]

        formatter = XmlFormatter()
        output = formatter.format_blocks(blocks, lines)

        assert "line 7" in output
        assert "line 8" in output
        assert '<block lines="7-8" score="0.9000">' in output
        assert "line 2" in output
        assert "line 3" in output
        assert '<block lines="2-3" score="0.7000">' in output

    def test_format_truncated_file(self) -> None:
        """Test formatting when lines don't cover block end_line."""
        lines = [(1, "line 1"), (2, "line 2"), (3, "line 3")]
        blocks = [
            MergedBlock(start_line=2, end_line=10, original_windows=(0,), max_score=0.8),
        ]

        formatter = XmlFormatter()
        output = formatter.format_blocks(blocks, lines)

        assert "line 2" in output
        assert "line 3" in output
        assert "</anomalies>" in output

    def test_format_unsorted_block_input(self) -> None:
        """Test that blocks passed in unsorted order are output in correct order."""
        lines = [(i, f"line {i}") for i in range(1, 11)]
        blocks = [
            MergedBlock(start_line=7, end_line=8, original_windows=(1,), max_score=0.9),
            MergedBlock(start_line=2, end_line=3, original_windows=(0,), max_score=0.7),
        ]

        formatter = XmlFormatter()
        output = formatter.format_blocks(blocks, lines)

        block_2_pos = output.index('<block lines="2-3"')
        block_7_pos = output.index('<block lines="7-8"')
        assert block_2_pos < block_7_pos
        assert "line 2" in output
        assert "line 7" in output
