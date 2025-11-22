from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from cordon.core.config import AnalysisConfig


@dataclass(frozen=True)
class TextWindow:
    """Immutable representation of a text window with line tracking.

    Attributes:
        content: The text content of the window
        start_line: Starting line number (1-indexed)
        end_line: Ending line number (1-indexed, inclusive)
        window_id: Unique identifier for this window
    """

    content: str
    start_line: int
    end_line: int
    window_id: int

    def __post_init__(self) -> None:
        """Validate window invariants."""
        if self.start_line < 1:
            raise ValueError("start_line must be >= 1")
        if self.end_line < self.start_line:
            raise ValueError("end_line must be >= start_line")
        if self.window_id < 0:
            raise ValueError("window_id must be >= 0")


@dataclass(frozen=True)
class ScoredWindow:
    """Window with its anomaly score.

    Attributes:
        window: The text window
        score: Anomaly score (higher = more anomalous)
        embedding: Vector embedding for downstream use
    """

    window: TextWindow
    score: float
    embedding: npt.NDArray[np.floating[Any]]

    def __post_init__(self) -> None:
        """Validate score is non-negative."""
        if self.score < 0:
            raise ValueError("score must be >= 0")


@dataclass(frozen=True)
class MergedBlock:
    """Contiguous block of significant lines after merging.

    Attributes:
        start_line: Starting line number (1-indexed)
        end_line: Ending line number (1-indexed, inclusive)
        original_windows: Window IDs that contributed to this block
        max_score: Highest score among contributing windows
    """

    start_line: int
    end_line: int
    original_windows: tuple[int, ...]
    max_score: float

    def __post_init__(self) -> None:
        """Validate merged block invariants."""
        if self.start_line < 1:
            raise ValueError("start_line must be >= 1")
        if self.end_line < self.start_line:
            raise ValueError("end_line must be >= start_line")
        if not self.original_windows:
            raise ValueError("original_windows cannot be empty")
        if self.max_score < 0:
            raise ValueError("max_score must be >= 0")


@dataclass
class AnalysisResult:
    """Complete analysis result with metadata.

    Attributes:
        output: Formatted output string with XML tags
        total_windows: Total number of windows created
        significant_windows: Number of windows above threshold
        merged_blocks: Number of merged blocks in output
        score_distribution: Statistical summary of scores
        processing_time: Total processing time in seconds
    """

    output: str
    total_windows: int
    significant_windows: int
    merged_blocks: int
    score_distribution: dict[str, float]
    processing_time: float


class Embedder(Protocol):
    """Protocol for embedding implementations."""

    def embed_windows(
        self, windows: Iterable[TextWindow]
    ) -> Iterator[tuple[TextWindow, npt.NDArray[np.floating[Any]]]]:
        """Embed text windows into vector representations.

        Args:
            windows: Iterable of text windows to embed

        Yields:
            Tuples of (window, embedding) pairs
        """
        ...


class Scorer(Protocol):
    """Protocol for scoring implementations."""

    def score_windows(
        self,
        embedded_windows: Sequence[tuple[TextWindow, npt.NDArray[np.floating[Any]]]],
        config: "AnalysisConfig",
    ) -> list[ScoredWindow]:
        """Score windows based on their embeddings.

        Args:
            embedded_windows: Sequence of (window, embedding) pairs
            config: Analysis configuration

        Returns:
            List of scored windows
        """
        ...
