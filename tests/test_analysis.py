"""Tests for analysis module."""

import numpy as np

from cordon.analysis.scorer import DensityAnomalyScorer
from cordon.analysis.thresholder import Thresholder
from cordon.core.config import AnalysisConfig
from cordon.core.types import ScoredWindow, TextWindow


class TestDensityAnomalyScorer:
    def test_score_single_window(self) -> None:
        """Test scoring with a single window."""
        window = TextWindow(content="test", start_line=1, end_line=1, window_id=0)
        embedding = np.array([0.1, 0.2, 0.3])
        embedded = [(window, embedding)]
        config = AnalysisConfig()

        scorer = DensityAnomalyScorer()
        scored = scorer.score_windows(embedded, config)

        assert len(scored) == 1
        assert scored[0].score == 0.0

    def test_score_empty_windows(self) -> None:
        """Test scoring with no windows."""
        embedded: list[tuple[TextWindow, np.ndarray]] = []
        config = AnalysisConfig()

        scorer = DensityAnomalyScorer()
        scored = scorer.score_windows(embedded, config)

        assert len(scored) == 0

    def test_score_similar_windows(self) -> None:
        """Test that similar windows have low scores."""
        windows = [
            TextWindow(content="test", start_line=i, end_line=i, window_id=i - 1)
            for i in range(1, 6)
        ]
        # all embeddings very similar
        embeddings = [np.array([0.1, 0.2, 0.3]) + np.random.randn(3) * 0.01 for _ in range(5)]
        # normalize
        embeddings = [e / np.linalg.norm(e) for e in embeddings]
        embedded = list(zip(windows, embeddings, strict=False))
        config = AnalysisConfig(k_neighbors=3)

        scorer = DensityAnomalyScorer()
        scored = scorer.score_windows(embedded, config)

        # all scores should be relatively small
        for sw in scored:
            assert sw.score < 0.2

    def test_score_diverse_windows(self) -> None:
        """Test that diverse embeddings have higher scores for outliers."""
        windows = [
            TextWindow(content=f"test{i}", start_line=i, end_line=i, window_id=i - 1)
            for i in range(1, 7)
        ]
        # create 5 similar embeddings and 1 outlier
        embeddings = [np.array([0.1, 0.2, 0.3]) for _ in range(5)]
        embeddings.append(np.array([0.9, 0.1, 0.1]))  # outlier
        # normalize
        embeddings = [e / np.linalg.norm(e) for e in embeddings]
        embedded = list(zip(windows, embeddings, strict=False))
        config = AnalysisConfig(k_neighbors=3)

        scorer = DensityAnomalyScorer()
        scored = scorer.score_windows(embedded, config)

        # outlier (last window) should have highest score
        assert scored[-1].score > scored[0].score

    def test_mmap_strategy_consistency(self) -> None:
        """Test that memory-mapped strategy produces consistent results."""
        # create enough windows to trigger mmap (if threshold is set low)
        windows = [
            TextWindow(content=f"test{i}", start_line=i, end_line=i, window_id=i - 1)
            for i in range(1, 101)
        ]
        # create diverse embeddings
        embeddings = [np.random.randn(10) for _ in range(100)]
        embeddings = [e / np.linalg.norm(e) for e in embeddings]
        embedded = list(zip(windows, embeddings, strict=False))

        # test with mmap enabled (low threshold)
        config_mmap = AnalysisConfig(k_neighbors=5, use_mmap_threshold=50)
        scorer = DensityAnomalyScorer()
        scored_mmap = scorer.score_windows(embedded, config_mmap)

        # test with in-memory (high threshold)
        config_mem = AnalysisConfig(k_neighbors=5, use_mmap_threshold=1000000)
        scored_mem = scorer.score_windows(embedded, config_mem)

        # results should be very similar
        assert len(scored_mmap) == len(scored_mem)
        for sw_mmap, sw_mem in zip(scored_mmap, scored_mem, strict=False):
            assert abs(sw_mmap.score - sw_mem.score) < 1e-5


class TestThresholder:
    """Tests for Thresholder class."""

    def test_select_top_10_percent(self) -> None:
        """Test selecting top 10% of windows."""
        windows = [
            TextWindow(content=f"test{i}", start_line=i, end_line=i, window_id=i - 1)
            for i in range(1, 101)
        ]
        embeddings = [np.array([float(i)]) for i in range(100)]
        scored = [
            ScoredWindow(window=w, score=float(i), embedding=e)
            for i, (w, e) in enumerate(zip(windows, embeddings, strict=False))
        ]
        config = AnalysisConfig(anomaly_percentile=0.1)

        thresholder = Thresholder()
        significant = thresholder.select_significant(scored, config)

        # should get approximately 10 windows
        assert len(significant) >= 10
        assert len(significant) <= 11  # allow for ties at threshold

    def test_select_all_windows(self) -> None:
        """Test selecting all windows with ratio=1.0."""
        windows = [
            TextWindow(content=f"test{i}", start_line=i, end_line=i, window_id=i - 1)
            for i in range(1, 11)
        ]
        embeddings = [np.array([float(i)]) for i in range(10)]
        scored = [
            ScoredWindow(window=w, score=float(i), embedding=e)
            for i, (w, e) in enumerate(zip(windows, embeddings, strict=False))
        ]
        config = AnalysisConfig(anomaly_percentile=1.0)

        thresholder = Thresholder()
        significant = thresholder.select_significant(scored, config)

        assert len(significant) == 10

    def test_select_no_windows(self) -> None:
        """Test selecting no windows with ratio=0.0."""
        windows = [
            TextWindow(content=f"test{i}", start_line=i, end_line=i, window_id=i - 1)
            for i in range(1, 11)
        ]
        embeddings = [np.array([float(i)]) for i in range(10)]
        scored = [
            ScoredWindow(window=w, score=float(i), embedding=e)
            for i, (w, e) in enumerate(zip(windows, embeddings, strict=False))
        ]
        config = AnalysisConfig(anomaly_percentile=0.0)

        thresholder = Thresholder()
        significant = thresholder.select_significant(scored, config)

        assert len(significant) == 0

    def test_empty_windows(self) -> None:
        """Test thresholding with no windows."""
        scored: list[ScoredWindow] = []
        config = AnalysisConfig()

        thresholder = Thresholder()
        significant = thresholder.select_significant(scored, config)

        assert len(significant) == 0

    def test_results_sorted_descending(self) -> None:
        """Test that results are sorted by score (descending)."""
        windows = [
            TextWindow(content=f"test{i}", start_line=i, end_line=i, window_id=i - 1)
            for i in range(1, 11)
        ]
        embeddings = [np.array([float(i)]) for i in range(10)]
        # create scores in random order
        scores = [5.0, 2.0, 8.0, 1.0, 9.0, 3.0, 7.0, 4.0, 6.0, 0.0]
        scored = [
            ScoredWindow(window=w, score=s, embedding=e)
            for w, s, e in zip(windows, scores, embeddings, strict=False)
        ]
        config = AnalysisConfig(anomaly_percentile=0.5)

        thresholder = Thresholder()
        significant = thresholder.select_significant(scored, config)

        # verify sorted descending
        for i in range(len(significant) - 1):
            assert significant[i].score >= significant[i + 1].score
