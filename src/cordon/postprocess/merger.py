from collections.abc import Sequence

from cordon.core.types import MergedBlock, ScoredWindow


class IntervalMerger:
    """Merge overlapping line ranges into contiguous blocks.

    This merger uses a sweep line algorithm to efficiently combine
    overlapping or adjacent windows into single contiguous blocks,
    preventing duplicate content in the output.
    """

    def merge_windows(self, scored_windows: Sequence[ScoredWindow]) -> list[MergedBlock]:
        """Merge overlapping windows into contiguous blocks.

        Args:
            scored_windows: Sequence of scored windows to merge

        Returns:
            List of merged blocks with no overlaps
        """
        if not scored_windows:
            return []

        # convert to intervals: (start, end, window_id, score)
        intervals = [
            (
                sw.window.start_line,
                sw.window.end_line,
                sw.window.window_id,
                sw.score,
            )
            for sw in scored_windows
        ]
        intervals.sort(key=lambda interval: interval[0])

        # initialize merge state with first interval
        merged: list[MergedBlock] = []
        current_start, current_end, first_id, first_score = intervals[0]
        contributing_ids = [first_id]
        max_score = first_score

        # sweep through remaining intervals
        for start, end, window_id, score in intervals[1:]:
            # check if overlapping or adjacent (lines N and N+1 are adjacent)
            if start <= current_end + 1:
                # extend current block
                current_end = max(current_end, end)
                contributing_ids.append(window_id)
                max_score = max(max_score, score)
            else:
                # gap found - save current block and start new one
                merged.append(
                    MergedBlock(
                        start_line=current_start,
                        end_line=current_end,
                        original_windows=tuple(contributing_ids),
                        max_score=max_score,
                    )
                )
                current_start = start
                current_end = end
                contributing_ids = [window_id]
                max_score = score

        # append final block
        merged.append(
            MergedBlock(
                start_line=current_start,
                end_line=current_end,
                original_windows=tuple(contributing_ids),
                max_score=max_score,
            )
        )

        return merged
