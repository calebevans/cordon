"""Shared test fixtures for the cordon test suite."""

from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from cordon.core.config import AnalysisConfig
from cordon.core.types import MergedBlock, ScoredWindow, TextWindow


@pytest.fixture
def default_config() -> AnalysisConfig:
    """Create a default AnalysisConfig for testing."""
    return AnalysisConfig(device="cpu")


@pytest.fixture
def sample_windows() -> list[TextWindow]:
    """Create a list of sample TextWindow instances."""
    return [
        TextWindow(
            content=f"content for window {i}",
            start_line=i * 4 + 1,
            end_line=(i + 1) * 4,
            window_id=i,
        )
        for i in range(5)
    ]


@pytest.fixture
def sample_embeddings() -> npt.NDArray[np.floating[Any]]:
    """Create sample normalized embedding vectors."""
    rng = np.random.default_rng(42)
    embeddings = rng.standard_normal((5, 384)).astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / norms


@pytest.fixture
def sample_scored_windows(sample_windows: list[TextWindow]) -> list[ScoredWindow]:
    """Create sample scored windows."""
    scores = [0.1, 0.5, 0.9, 0.3, 0.7]
    return [ScoredWindow(window=w, score=s) for w, s in zip(sample_windows, scores, strict=True)]


@pytest.fixture
def sample_merged_blocks() -> list[MergedBlock]:
    """Create sample merged blocks."""
    return [
        MergedBlock(start_line=1, end_line=4, original_windows=(0,), max_score=0.5),
        MergedBlock(start_line=10, end_line=16, original_windows=(2, 3), max_score=0.9),
    ]
