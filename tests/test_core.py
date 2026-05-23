import dataclasses

import pytest

from cordon.core.config import AnalysisConfig
from cordon.core.types import AnalysisResult, MergedBlock, ScoredWindow, TextWindow


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

    def test_empty_content_rejected(self) -> None:
        """Test that empty content is rejected."""
        with pytest.raises(ValueError, match="content must not be empty"):
            TextWindow(content="", start_line=1, end_line=1, window_id=0)

    def test_whitespace_only_content_rejected(self) -> None:
        """Test that whitespace-only content is rejected."""
        with pytest.raises(ValueError, match="content must not be empty"):
            TextWindow(content="   \n\t  ", start_line=1, end_line=1, window_id=0)


class TestScoredWindow:
    """Tests for ScoredWindow dataclass."""

    def test_valid_scored_window(self) -> None:
        """Test creating a valid scored window."""
        window = TextWindow(content="test", start_line=1, end_line=5, window_id=0)
        scored = ScoredWindow(window=window, score=0.5)
        assert scored.window == window
        assert scored.score == 0.5


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
        assert config.scoring_batch_size is None

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
            AnalysisConfig(device="gpu")  # type: ignore[arg-type]

        with pytest.raises(ValueError, match="backend must be one of"):
            AnalysisConfig(backend="openai")  # type: ignore[arg-type]

    def test_empty_model_name_rejected(self) -> None:
        """Test that empty model_name is rejected."""
        with pytest.raises(ValueError, match="model_name must not be empty"):
            AnalysisConfig(model_name="")

        with pytest.raises(ValueError, match="model_name must not be empty"):
            AnalysisConfig(model_name="   ")

    def test_frozen_config(self) -> None:
        """Test that config attributes cannot be mutated after construction."""
        config = AnalysisConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            config.window_size = 10  # type: ignore[misc]

    def test_range_mode_valid(self) -> None:
        """Test that valid range configurations are accepted."""
        config = AnalysisConfig(anomaly_range_min=0.05, anomaly_range_max=0.15)
        assert config.anomaly_range_min == 0.05
        assert config.anomaly_range_max == 0.15

    def test_range_mode_both_required(self) -> None:
        """Test that both range parameters must be set together."""
        with pytest.raises(ValueError, match="must both be set"):
            AnalysisConfig(anomaly_range_min=0.05)

        with pytest.raises(ValueError, match="must both be set"):
            AnalysisConfig(anomaly_range_max=0.15)

    def test_range_mode_bounds_validation(self) -> None:
        """Test that range bounds are validated."""
        with pytest.raises(ValueError, match="anomaly_range_min must be between"):
            AnalysisConfig(anomaly_range_min=-0.1, anomaly_range_max=0.15)

        with pytest.raises(ValueError, match="anomaly_range_min must be between"):
            AnalysisConfig(anomaly_range_min=1.5, anomaly_range_max=2.0)

        with pytest.raises(ValueError, match="anomaly_range_max must be between"):
            AnalysisConfig(anomaly_range_min=0.05, anomaly_range_max=-0.1)

        with pytest.raises(ValueError, match="anomaly_range_max must be between"):
            AnalysisConfig(anomaly_range_min=0.05, anomaly_range_max=1.5)

    def test_range_mode_min_less_than_max(self) -> None:
        """Test that range_min must be less than range_max."""
        with pytest.raises(ValueError, match="must be less than"):
            AnalysisConfig(anomaly_range_min=0.15, anomaly_range_max=0.05)

        with pytest.raises(ValueError, match="must be less than"):
            AnalysisConfig(anomaly_range_min=0.1, anomaly_range_max=0.1)

    def test_range_mode_with_default_percentile(self) -> None:
        """Test that range mode works with default percentile value."""
        config = AnalysisConfig(
            anomaly_range_min=0.05,
            anomaly_range_max=0.15,
            anomaly_percentile=0.1,
        )
        assert config.anomaly_range_min == 0.05
        assert config.anomaly_range_max == 0.15

    def test_max_line_length_validation(self) -> None:
        """Test that max_line_length must be >= 1 if set."""
        with pytest.raises(ValueError, match="max_line_length"):
            AnalysisConfig(max_line_length=0)
        with pytest.raises(ValueError, match="max_line_length"):
            AnalysisConfig(max_line_length=-1)
        config = AnalysisConfig(max_line_length=None)
        assert config.max_line_length is None
        config = AnalysisConfig(max_line_length=500)
        assert config.max_line_length == 500

    def test_output_format_default(self) -> None:
        """Test that output_format defaults to xml."""
        config = AnalysisConfig()
        assert config.output_format == "xml"

    def test_output_format_json(self) -> None:
        """Test that output_format can be set to json."""
        config = AnalysisConfig(output_format="json")
        assert config.output_format == "json"

    def test_token_budget_validation(self) -> None:
        """Test that invalid token_budget values are rejected."""
        with pytest.raises(ValueError, match="token_budget"):
            AnalysisConfig(token_budget=0)
        with pytest.raises(ValueError, match="token_budget"):
            AnalysisConfig(token_budget=-1)
        config = AnalysisConfig(token_budget=1000)
        assert config.token_budget == 1000

    def test_max_blocks_validation(self) -> None:
        """Test that invalid max_blocks values are rejected."""
        with pytest.raises(ValueError, match="max_blocks"):
            AnalysisConfig(max_blocks=0)
        with pytest.raises(ValueError, match="max_blocks"):
            AnalysisConfig(max_blocks=-1)
        config = AnalysisConfig(max_blocks=5)
        assert config.max_blocks == 5
        config_none = AnalysisConfig(max_blocks=None)
        assert config_none.max_blocks is None

    def test_min_score_validation(self) -> None:
        """Test that invalid min_score values are rejected."""
        with pytest.raises(ValueError, match="min_score"):
            AnalysisConfig(min_score=-0.1)
        config = AnalysisConfig(min_score=0.0)
        assert config.min_score == 0.0
        config_pos = AnalysisConfig(min_score=0.5)
        assert config_pos.min_score == 0.5
        config_none = AnalysisConfig(min_score=None)
        assert config_none.min_score is None


class TestAnalysisResult:
    """Tests for AnalysisResult dataclass."""

    def test_blocks_field_populated(self) -> None:
        """Test that blocks field stores MergedBlock instances."""
        blocks = [
            MergedBlock(start_line=1, end_line=4, original_windows=(0,), max_score=0.5),
            MergedBlock(start_line=10, end_line=16, original_windows=(2, 3), max_score=0.9),
        ]
        result = AnalysisResult(
            output="<output>",
            blocks=blocks,
            total_lines=100,
            total_windows=25,
            significant_windows=3,
            merged_blocks=2,
            score_distribution={"min": 0.0, "max": 1.0, "mean": 0.5, "median": 0.5, "p90": 0.9},
            processing_time=1.0,
        )
        assert result.blocks == blocks
        assert len(result.blocks) == 2
        assert result.blocks[0].start_line == 1
        assert result.blocks[1].max_score == 0.9

    def test_empty_blocks(self) -> None:
        """Test that blocks field can be empty."""
        result = AnalysisResult(
            output="",
            blocks=[],
            total_lines=0,
            total_windows=0,
            significant_windows=0,
            merged_blocks=0,
            score_distribution={"min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0, "p90": 0.0},
            processing_time=0.0,
        )
        assert result.blocks == []
