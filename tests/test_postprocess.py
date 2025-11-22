from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from cordon.core.types import ScoredWindow, TextWindow
from cordon.postprocess.formatter import OutputFormatter
from cordon.postprocess.merger import IntervalMerger


class TestIntervalMerger:
    """Tests for IntervalMerger class."""

    def test_merge_no_overlap(self) -> None:
        """Test merging windows with no overlap."""
        windows = [
            TextWindow(content="w1", start_line=1, end_line=3, window_id=0),
            TextWindow(content="w2", start_line=10, end_line=12, window_id=1),
        ]
        embeddings = [np.array([0.1]), np.array([0.2])]
        scored = [
            ScoredWindow(window=w, score=0.5, embedding=e)
            for w, e in zip(windows, embeddings, strict=False)
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
        windows = [
            TextWindow(content="w1", start_line=1, end_line=5, window_id=0),
            TextWindow(content="w2", start_line=3, end_line=7, window_id=1),
            TextWindow(content="w3", start_line=6, end_line=10, window_id=2),
        ]
        embeddings = [np.array([0.1]), np.array([0.2]), np.array([0.3])]
        scored = [
            ScoredWindow(window=w, score=0.5, embedding=e)
            for w, e in zip(windows, embeddings, strict=False)
        ]

        merger = IntervalMerger()
        merged = merger.merge_windows(scored)

        assert len(merged) == 1
        assert merged[0].start_line == 1
        assert merged[0].end_line == 10
        assert merged[0].original_windows == (0, 1, 2)

    def test_merge_adjacent_windows(self) -> None:
        """Test merging adjacent windows (lines N and N+1)."""
        windows = [
            TextWindow(content="w1", start_line=1, end_line=5, window_id=0),
            TextWindow(content="w2", start_line=6, end_line=10, window_id=1),
        ]
        embeddings = [np.array([0.1]), np.array([0.2])]
        scored = [
            ScoredWindow(window=w, score=0.5, embedding=e)
            for w, e in zip(windows, embeddings, strict=False)
        ]

        merger = IntervalMerger()
        merged = merger.merge_windows(scored)

        # adjacent windows should merge
        assert len(merged) == 1
        assert merged[0].start_line == 1
        assert merged[0].end_line == 10

    def test_merge_preserves_max_score(self) -> None:
        """Test that merging preserves the maximum score."""
        windows = [
            TextWindow(content="w1", start_line=1, end_line=5, window_id=0),
            TextWindow(content="w2", start_line=3, end_line=7, window_id=1),
        ]
        embeddings = [np.array([0.1]), np.array([0.2])]
        scored = [
            ScoredWindow(window=windows[0], score=0.8, embedding=embeddings[0]),
            ScoredWindow(window=windows[1], score=0.5, embedding=embeddings[1]),
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
        windows = [TextWindow(content="w1", start_line=1, end_line=5, window_id=0)]
        embeddings = [np.array([0.1])]
        scored = [
            ScoredWindow(window=w, score=0.5, embedding=e)
            for w, e in zip(windows, embeddings, strict=False)
        ]

        merger = IntervalMerger()
        merged = merger.merge_windows(scored)

        assert len(merged) == 1
        assert merged[0].start_line == 1
        assert merged[0].end_line == 5


class TestOutputFormatter:
    """Tests for OutputFormatter class."""

    def test_format_single_block(self) -> None:
        """Test formatting a single block."""
        with NamedTemporaryFile(mode="w", delete=False, suffix=".log") as f:
            f.write("line 1\n")
            f.write("line 2\n")
            f.write("line 3\n")
            temp_path = Path(f.name)

        try:
            from cordon.core.types import MergedBlock

            blocks = [MergedBlock(start_line=1, end_line=2, original_windows=(0,), max_score=0.8)]

            formatter = OutputFormatter()
            output = formatter.format_blocks(blocks, temp_path)

            assert '<block lines="1-2" score="0.8000">' in output
            assert "line 1\n" in output
            assert "line 2\n" in output
            assert "</block>" in output
        finally:
            temp_path.unlink()

    def test_format_multiple_blocks(self) -> None:
        """Test formatting multiple blocks."""
        with NamedTemporaryFile(mode="w", delete=False, suffix=".log") as f:
            for i in range(1, 11):
                f.write(f"line {i}\n")
            temp_path = Path(f.name)

        try:
            from cordon.core.types import MergedBlock

            blocks = [
                MergedBlock(start_line=1, end_line=2, original_windows=(0,), max_score=0.8),
                MergedBlock(start_line=5, end_line=7, original_windows=(1,), max_score=0.9),
            ]

            formatter = OutputFormatter()
            output = formatter.format_blocks(blocks, temp_path)

            assert '<block lines="1-2" score="0.8000">' in output
            assert '<block lines="5-7" score="0.9000">' in output
            assert output.count("</block>") == 2
        finally:
            temp_path.unlink()

    def test_format_empty_blocks(self) -> None:
        """Test formatting with no blocks."""
        with NamedTemporaryFile(mode="w", delete=False, suffix=".log") as f:
            f.write("line 1\n")
            temp_path = Path(f.name)

        try:
            from cordon.core.types import MergedBlock

            blocks: list[MergedBlock] = []

            formatter = OutputFormatter()
            output = formatter.format_blocks(blocks, temp_path)

            assert output == ""
        finally:
            temp_path.unlink()
