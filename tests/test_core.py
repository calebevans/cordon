import numpy as np
import pytest

from cordon.core.config import AnalysisConfig
from cordon.core.types import MergedBlock, ScoredWindow, TextWindow


class TestTextWindow:
    """Tests for TextWindow dataclass."""

    def test_valid_window(self) -> None:
        """Test creating a valid text window."""
        window = TextWindow(content="test", start_line=1, end_line=5, window_id=0)
        assert window.content == "test"
        assert window.start_line == 1
        assert window.end_line == 5
        assert window.window_id == 0

    def test_validation(self) -> None:
        """Test that invalid parameters are rejected."""
        with pytest.raises(ValueError):
            TextWindow(content="test", start_line=0, end_line=5, window_id=0)
        with pytest.raises(ValueError):
            TextWindow(content="test", start_line=5, end_line=3, window_id=0)


class TestScoredWindow:
    """Tests for ScoredWindow dataclass."""

    def test_valid_scored_window(self) -> None:
        """Test creating a valid scored window."""
        window = TextWindow(content="test", start_line=1, end_line=5, window_id=0)
        embedding = np.array([0.1, 0.2, 0.3])
        scored = ScoredWindow(window=window, score=0.5, embedding=embedding)
        assert scored.window == window
        assert scored.score == 0.5
        np.testing.assert_array_equal(scored.embedding, embedding)


class TestMergedBlock:
    """Tests for MergedBlock dataclass."""

    def test_valid_merged_block(self) -> None:
        """Test creating a valid merged block."""
        block = MergedBlock(start_line=1, end_line=10, original_windows=(0, 1, 2), max_score=0.8)
        assert block.start_line == 1
        assert block.end_line == 10
        assert block.original_windows == (0, 1, 2)
        assert block.max_score == 0.8


class TestAnalysisConfig:
    """Tests for AnalysisConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = AnalysisConfig()
        assert config.window_size == 4
        assert config.k_neighbors == 5
        assert config.anomaly_percentile == 0.1
        assert config.model_name == "all-MiniLM-L6-v2"
        assert config.batch_size == 32
        assert config.device is None
        assert config.scoring_batch_size is None  # auto-detect by default

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = AnalysisConfig(
            window_size=20,
            k_neighbors=10,
            anomaly_percentile=0.05,
            device="cpu",
        )
        assert config.window_size == 20
        assert config.k_neighbors == 10
        assert config.anomaly_percentile == 0.05
        assert config.device == "cpu"

    def test_validation(self) -> None:
        """Test that invalid configurations are rejected."""
        with pytest.raises(ValueError):
            AnalysisConfig(window_size=0)
        with pytest.raises(ValueError):
            AnalysisConfig(anomaly_percentile=1.5)
        with pytest.raises(ValueError):
            AnalysisConfig(device="gpu")

    def test_range_mode_valid(self) -> None:
        """Test that valid range configurations are accepted."""
        config = AnalysisConfig(anomaly_range_min=0.05, anomaly_range_max=0.15)
        assert config.anomaly_range_min == 0.05
        assert config.anomaly_range_max == 0.15

    def test_range_mode_both_required(self) -> None:
        """Test that both range parameters must be set together."""
        # only min set - should fail
        with pytest.raises(ValueError, match="must both be set"):
            AnalysisConfig(anomaly_range_min=0.05)

        # only max set - should fail
        with pytest.raises(ValueError, match="must both be set"):
            AnalysisConfig(anomaly_range_max=0.15)

    def test_range_mode_bounds_validation(self) -> None:
        """Test that range bounds are validated."""
        # min out of range
        with pytest.raises(ValueError, match="anomaly_range_min must be between"):
            AnalysisConfig(anomaly_range_min=-0.1, anomaly_range_max=0.15)

        with pytest.raises(ValueError, match="anomaly_range_min must be between"):
            AnalysisConfig(anomaly_range_min=1.5, anomaly_range_max=2.0)

        # max out of range
        with pytest.raises(ValueError, match="anomaly_range_max must be between"):
            AnalysisConfig(anomaly_range_min=0.05, anomaly_range_max=-0.1)

        with pytest.raises(ValueError, match="anomaly_range_max must be between"):
            AnalysisConfig(anomaly_range_min=0.05, anomaly_range_max=1.5)

    def test_range_mode_min_less_than_max(self) -> None:
        """Test that range_min must be less than range_max."""
        # min >= max should fail
        with pytest.raises(ValueError, match="must be less than"):
            AnalysisConfig(anomaly_range_min=0.15, anomaly_range_max=0.05)

        # equal values should fail
        with pytest.raises(ValueError, match="must be less than"):
            AnalysisConfig(anomaly_range_min=0.1, anomaly_range_max=0.1)

    def test_range_mode_with_default_percentile(self) -> None:
        """Test that range mode works with default percentile value."""
        # should be valid - percentile is ignored in range mode
        config = AnalysisConfig(
            anomaly_range_min=0.05,
            anomaly_range_max=0.15,
            anomaly_percentile=0.1,  # default value
        )
        assert config.anomaly_range_min == 0.05
        assert config.anomaly_range_max == 0.15
