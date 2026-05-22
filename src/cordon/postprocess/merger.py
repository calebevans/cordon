from collections.abc import Sequence
from typing import NamedTuple

from cordon.core.types import MergedBlock, ScoredWindow


class _WindowInterval(NamedTuple):
    """Internal representation of a window's line interval for merging."""

    start: int
    end: int
    window_id: int
    score: float


class IntervalMerger:
    """Merge overlapping line ranges into contiguous blocks.

    This merger uses a sweep line algorithm to efficiently combine
    overlapping or adjacent windows into single contiguous blocks,
    preventing duplicate content in the output.
    """

    def merge_windows(self, scored_windows: Sequence[ScoredWindow]) -> list[MergedBlock]:
        """Merge overlapping windows into contiguous blocks.

        Args:
            scored_windows: Sequence of scored windows to merge.

        Returns:
            List of merged blocks with no overlaps.
        """
        if not scored_windows:
            return []

        intervals = [
            _WindowInterval(
                start=sw.window.start_line,
                end=sw.window.end_line,
                window_id=sw.window.window_id,
                score=sw.score,
            )
            for sw in scored_windows
        ]
        intervals.sort(key=lambda interval: interval.start)

        merged: list[MergedBlock] = []
        first = intervals[0]
        current_start = first.start
        current_end = first.end
        contributing_ids = [first.window_id]
        max_score = first.score

        for interval in intervals[1:]:
            if interval.start <= current_end + 1:
                current_end = max(current_end, interval.end)
                contributing_ids.append(interval.window_id)
                max_score = max(max_score, interval.score)
            else:
                merged.append(
                    MergedBlock(
                        start_line=current_start,
                        end_line=current_end,
                        original_windows=tuple(contributing_ids),
                        max_score=max_score,
                    )
                )
                current_start = interval.start
                current_end = interval.end
                contributing_ids = [interval.window_id]
                max_score = interval.score

        merged.append(
            MergedBlock(
                start_line=current_start,
                end_line=current_end,
                original_windows=tuple(contributing_ids),
                max_score=max_score,
            )
        )

        return merged
