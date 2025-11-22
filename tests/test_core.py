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
        assert config.window_size == 10
        assert config.stride == 5
        assert config.k_neighbors == 5
        assert config.anomaly_percentile == 0.1
        assert config.model_name == "all-MiniLM-L6-v2"
        assert config.batch_size == 32
        assert config.device is None

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = AnalysisConfig(
            window_size=20,
            stride=10,
            k_neighbors=10,
            anomaly_percentile=0.05,
            device="cpu",
        )
        assert config.window_size == 20
        assert config.stride == 10
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
