from cordon.core.config import AnalysisConfig
from cordon.segmentation.windower import SlidingWindowSegmenter


class TestSlidingWindowSegmenter:
    """Tests for SlidingWindowSegmenter class."""

    def test_basic_segmentation(self) -> None:
        """Test basic non-overlapping window segmentation."""
        lines = [(1, "line1"), (2, "line2"), (3, "line3"), (4, "line4"), (5, "line5"), (6, "line6")]
        config = AnalysisConfig(window_size=3)

        segmenter = SlidingWindowSegmenter()
        windows = list(segmenter.segment(iter(lines), config))

        assert len(windows) == 2
        assert windows[0].start_line == 1
        assert windows[0].end_line == 3
        assert windows[0].content == "line1\nline2\nline3"
        assert windows[1].start_line == 4
        assert windows[1].end_line == 6
        assert windows[1].content == "line4\nline5\nline6"

    def test_non_overlapping_windows(self) -> None:
        """Test segmentation creates non-overlapping windows."""
        lines = [(1, "line1"), (2, "line2"), (3, "line3"), (4, "line4")]
        config = AnalysisConfig(window_size=2)

        segmenter = SlidingWindowSegmenter()
        windows = list(segmenter.segment(iter(lines), config))

        assert len(windows) == 2
        assert windows[0].content == "line1\nline2"
        assert windows[1].content == "line3\nline4"

    def test_partial_final_window(self) -> None:
        """Test that partial final window is included."""
        lines = [(1, "line1"), (2, "line2"), (3, "line3")]
        config = AnalysisConfig(window_size=2)

        segmenter = SlidingWindowSegmenter()
        windows = list(segmenter.segment(iter(lines), config))

        assert len(windows) == 2
        assert windows[0].content == "line1\nline2"
        assert windows[1].content == "line3"
        assert windows[1].start_line == 3
        assert windows[1].end_line == 3

    def test_empty_input(self) -> None:
        """Test segmentation with empty input."""
        lines: list[tuple[int, str]] = []
        config = AnalysisConfig(window_size=3)

        segmenter = SlidingWindowSegmenter()
        windows = list(segmenter.segment(iter(lines), config))

        assert len(windows) == 0

    def test_single_line(self) -> None:
        """Test segmentation with single line."""
        lines = [(1, "line1")]
        config = AnalysisConfig(window_size=3)

        segmenter = SlidingWindowSegmenter()
        windows = list(segmenter.segment(iter(lines), config))

        assert len(windows) == 1
        assert windows[0].content == "line1"
        assert windows[0].start_line == 1
        assert windows[0].end_line == 1

    def test_exact_multiple_windows(self) -> None:
        """Test segmentation when lines are exact multiple of window size."""
        lines = [(1, "line1"), (2, "line2"), (3, "line3"), (4, "line4"), (5, "line5"), (6, "line6")]
        config = AnalysisConfig(window_size=3)

        segmenter = SlidingWindowSegmenter()
        windows = list(segmenter.segment(iter(lines), config))

        assert len(windows) == 2
        assert windows[0].content == "line1\nline2\nline3"
        assert windows[1].content == "line4\nline5\nline6"

    def test_window_ids_incremental(self) -> None:
        """Test that window IDs are incremental."""
        lines = [(1, "line1"), (2, "line2"), (3, "line3"), (4, "line4")]
        config = AnalysisConfig(window_size=2)

        segmenter = SlidingWindowSegmenter()
        windows = list(segmenter.segment(iter(lines), config))

        for i, window in enumerate(windows):
            assert window.window_id == i
