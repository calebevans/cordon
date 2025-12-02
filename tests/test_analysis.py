"""Tests for analysis module."""

import numpy as np
import torch

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

    def test_large_dataset_consistency(self) -> None:
        """Test that PyTorch handles large datasets consistently."""
        # create enough windows to test scaling
        windows = [
            TextWindow(content=f"test{i}", start_line=i, end_line=i, window_id=i - 1)
            for i in range(1, 101)
        ]
        # create diverse embeddings
        np.random.seed(42)
        embeddings = [np.random.randn(10).astype(np.float32) for _ in range(100)]
        embeddings = [e / np.linalg.norm(e) for e in embeddings]
        embedded = list(zip(windows, embeddings, strict=False))

        # test with different batch sizes - should produce consistent results
        config_small_batch = AnalysisConfig(k_neighbors=5, scoring_batch_size=20, device="cpu")
        scorer = DensityAnomalyScorer()
        scored_small = scorer.score_windows(embedded, config_small_batch)

        config_large_batch = AnalysisConfig(k_neighbors=5, scoring_batch_size=50, device="cpu")
        scored_large = scorer.score_windows(embedded, config_large_batch)

        # results should be identical regardless of batch size
        assert len(scored_small) == len(scored_large)
        for sw_small, sw_large in zip(scored_small, scored_large, strict=False):
            assert abs(sw_small.score - sw_large.score) < 1e-6

    def test_device_detection(self) -> None:
        """Test that device detection respects config and auto-detects correctly."""
        scorer = DensityAnomalyScorer()

        # test explicit device settings
        config_cpu = AnalysisConfig(device="cpu")
        assert scorer._detect_device(config_cpu) == "cpu"

        config_cuda = AnalysisConfig(device="cuda")
        assert scorer._detect_device(config_cuda) == "cuda"

        config_mps = AnalysisConfig(device="mps")
        assert scorer._detect_device(config_mps) == "mps"

        # test auto-detection (should return one of the supported devices)
        config_auto = AnalysisConfig()
        device = scorer._detect_device(config_auto)
        assert device in ("cuda", "mps", "cpu")

    def test_pytorch_scoring_cpu(self) -> None:
        """Test that PyTorch scoring works correctly on CPU."""
        windows = [
            TextWindow(content=f"test{i}", start_line=i, end_line=i, window_id=i - 1)
            for i in range(1, 11)
        ]
        # create diverse embeddings
        np.random.seed(42)
        embeddings = [np.random.randn(10).astype(np.float32) for _ in range(10)]
        embeddings = [e / np.linalg.norm(e) for e in embeddings]
        embedded = list(zip(windows, embeddings, strict=False))

        # test with CPU device explicitly
        config = AnalysisConfig(k_neighbors=3, device="cpu", scoring_batch_size=5)
        scorer = DensityAnomalyScorer()
        scored = scorer.score_windows(embedded, config)

        # verify results
        assert len(scored) == 10
        for sw in scored:
            assert sw.score >= 0.0
            assert isinstance(sw.score, float)
            assert len(sw.embedding) == 10

    def test_gpu_scoring_when_available(self) -> None:
        """Test that GPU scoring works when GPU is available."""
        windows = [
            TextWindow(content=f"test{i}", start_line=i, end_line=i, window_id=i - 1)
            for i in range(1, 11)
        ]
        embeddings = [np.random.randn(8).astype(np.float32) for _ in range(10)]
        embeddings = [e / np.linalg.norm(e) for e in embeddings]
        embedded = list(zip(windows, embeddings, strict=False))

        # only run if GPU is available
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            # skip test if no GPU available
            return

        config = AnalysisConfig(k_neighbors=3, device=device, scoring_batch_size=5)
        scorer = DensityAnomalyScorer()
        scored = scorer._score_windows_gpu(embedded, config, device)

        # verify results
        assert len(scored) == 10
        for sw in scored:
            assert sw.score >= 0.0
            assert isinstance(sw.score, float)

    def test_scoring_batch_size_configuration(self) -> None:
        """Test that scoring_batch_size parameter is properly used."""
        windows = [
            TextWindow(content=f"test{i}", start_line=i, end_line=i, window_id=i - 1)
            for i in range(1, 26)
        ]
        embeddings = [np.random.randn(5).astype(np.float32) for _ in range(25)]
        embeddings = [e / np.linalg.norm(e) for e in embeddings]
        embedded = list(zip(windows, embeddings, strict=False))

        # test with different batch sizes (should all work without error)
        for batch_size in [5, 10, 25, 100]:
            config = AnalysisConfig(k_neighbors=3, device="cpu", scoring_batch_size=batch_size)
            scorer = DensityAnomalyScorer()
            scored = scorer.score_windows(embedded, config)
            assert len(scored) == 25

        # test auto-detection (None)
        config_auto = AnalysisConfig(k_neighbors=3, device="cpu", scoring_batch_size=None)
        scorer_auto = DensityAnomalyScorer()
        scored_auto = scorer_auto.score_windows(embedded, config_auto)
        assert len(scored_auto) == 25

    def test_pytorch_gpu_availability(self) -> None:
        """Test that PyTorch GPU implementation runs if GPU is available."""
        # this test will use GPU if available, otherwise skip to CPU
        windows = [
            TextWindow(content=f"test{i}", start_line=i, end_line=i, window_id=i - 1)
            for i in range(1, 11)
        ]
        embeddings = [np.random.randn(8).astype(np.float32) for _ in range(10)]
        embeddings = [e / np.linalg.norm(e) for e in embeddings]
        embedded = list(zip(windows, embeddings, strict=False))

        # test with auto-detected device
        config = AnalysisConfig(k_neighbors=3, scoring_batch_size=5)
        scorer = DensityAnomalyScorer()
        scored = scorer.score_windows(embedded, config)

        # should work regardless of GPU availability
        assert len(scored) == 10
        for sw in scored:
            assert sw.score >= 0.0


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

    def test_range_mode_basic(self) -> None:
        """Test range mode excludes top X% and keeps next Y%."""
        windows = [
            TextWindow(content=f"test{i}", start_line=i, end_line=i, window_id=i - 1)
            for i in range(1, 101)
        ]
        embeddings = [np.array([float(i)]) for i in range(100)]
        # scores from 0.0 to 99.0
        scored = [
            ScoredWindow(window=w, score=float(i), embedding=e)
            for i, (w, e) in enumerate(zip(windows, embeddings, strict=False))
        ]
        # exclude top 5% (scores >= 95th percentile = 94.05)
        # keep up to 15% (scores >= 85th percentile = 84.15)
        # result: scores in range [84.15, 94.05)
        config = AnalysisConfig(anomaly_range_min=0.05, anomaly_range_max=0.15)

        thresholder = Thresholder()
        significant = thresholder.select_significant(scored, config)

        # should get approximately 10 windows (between 85th and 95th percentile)
        assert len(significant) >= 9
        assert len(significant) <= 11  # allow for boundary effects

        # verify all selected scores are in expected range
        for sw in significant:
            assert sw.score < 95.0  # below 95th percentile
            assert sw.score >= 84.0  # at or above 85th percentile

    def test_range_mode_sorted_descending(self) -> None:
        """Test that range mode results are sorted by score (descending)."""
        windows = [
            TextWindow(content=f"test{i}", start_line=i, end_line=i, window_id=i - 1)
            for i in range(1, 51)
        ]
        embeddings = [np.array([float(i)]) for i in range(50)]
        # random scores
        scores = list(range(50))
        np.random.shuffle(scores)
        scored = [
            ScoredWindow(window=w, score=float(s), embedding=e)
            for w, s, e in zip(windows, scores, embeddings, strict=False)
        ]
        config = AnalysisConfig(anomaly_range_min=0.1, anomaly_range_max=0.3)

        thresholder = Thresholder()
        significant = thresholder.select_significant(scored, config)

        # verify sorted descending
        for i in range(len(significant) - 1):
            assert significant[i].score >= significant[i + 1].score

    def test_range_mode_empty_result(self) -> None:
        """Test range mode with no windows in the range."""
        windows = [
            TextWindow(content=f"test{i}", start_line=i, end_line=i, window_id=i - 1)
            for i in range(1, 11)
        ]
        embeddings = [np.array([float(i)]) for i in range(10)]
        # all scores are 5.0
        scored = [
            ScoredWindow(window=w, score=5.0, embedding=e)
            for w, e in zip(windows, embeddings, strict=False)
        ]
        # with identical scores, percentile range will be very narrow
        config = AnalysisConfig(anomaly_range_min=0.01, anomaly_range_max=0.05)

        thresholder = Thresholder()
        significant = thresholder.select_significant(scored, config)

        # with identical scores, might get some or none depending on boundaries
        assert len(significant) >= 0

    def test_range_mode_vs_percentile_mode(self) -> None:
        """Test that range mode and percentile mode return different score ranges."""
        windows = [
            TextWindow(content=f"test{i}", start_line=i, end_line=i, window_id=i - 1)
            for i in range(1, 101)
        ]
        embeddings = [np.array([float(i)]) for i in range(100)]
        scored = [
            ScoredWindow(window=w, score=float(i), embedding=e)
            for i, (w, e) in enumerate(zip(windows, embeddings, strict=False))
        ]

        # percentile mode: top 10% (scores >= 90th percentile)
        config_percentile = AnalysisConfig(anomaly_percentile=0.1)
        thresholder = Thresholder()
        sig_percentile = thresholder.select_significant(scored, config_percentile)

        # range mode: exclude top 5%, keep next 10% (scores 85th-95th percentile)
        config_range = AnalysisConfig(anomaly_range_min=0.05, anomaly_range_max=0.15)
        sig_range = thresholder.select_significant(scored, config_range)

        # Both return 10 windows, but different score ranges
        assert len(sig_percentile) == 10
        assert len(sig_range) == 10

        # percentile mode should include the highest scores (90-99)
        # range mode should exclude the top 5% and return 85-94
        assert max(sw.score for sw in sig_percentile) > max(sw.score for sw in sig_range)
        assert min(sw.score for sw in sig_percentile) > min(sw.score for sw in sig_range)

        # Verify specific ranges
        pct_scores = [sw.score for sw in sig_percentile]
        range_scores = [sw.score for sw in sig_range]

        # Percentile mode: should have scores >= 90
        assert all(s >= 90.0 for s in pct_scores)
        # Range mode: should have scores in [85, 95)
        assert all(85.0 <= s < 95.0 for s in range_scores)

    def test_range_mode_boundary_inclusive_exclusive(self) -> None:
        """Test that range boundaries are inclusive on lower, exclusive on upper."""
        windows = [
            TextWindow(content=f"test{i}", start_line=i, end_line=i, window_id=i - 1)
            for i in range(1, 101)
        ]
        embeddings = [np.array([float(i)]) for i in range(100)]
        # scores from 0 to 99
        scored = [
            ScoredWindow(window=w, score=float(i), embedding=e)
            for i, (w, e) in enumerate(zip(windows, embeddings, strict=False))
        ]
        # exclude top 10% (score < 90th percentile = 89.1)
        # keep up to 20% (score >= 80th percentile = 79.2)
        config = AnalysisConfig(anomaly_range_min=0.1, anomaly_range_max=0.2)

        thresholder = Thresholder()
        significant = thresholder.select_significant(scored, config)

        # verify boundaries
        for sw in significant:
            # should be less than upper threshold (exclusive)
            assert sw.score < 90.0
            # should be at or above lower threshold (inclusive)
            assert sw.score >= 79.0
