import warnings
from collections import deque
from collections.abc import Iterator

from cordon.core.config import AnalysisConfig
from cordon.core.types import TextWindow


class SlidingWindowSegmenter:
    """Convert line stream into overlapping text windows with line tracking.

    This segmenter uses a sliding window approach to create overlapping chunks
    of text from a stream of lines. Each window maintains references to its
    original line numbers for downstream processing.
    """

    def segment(
        self, lines: Iterator[tuple[int, str]], config: AnalysisConfig
    ) -> Iterator[TextWindow]:
        """Segment lines into overlapping text windows.

        Args:
            lines: Iterator of (line_number, line_content) tuples
            config: Analysis configuration with window_size and stride

        Yields:
            TextWindow instances with content and line tracking

        Warnings:
            Issues a warning if stride > window_size (creates gaps)
        """
        window_size = config.window_size
        stride = config.stride

        # warn about gaps
        if stride > window_size:
            warnings.warn(
                f"stride ({stride}) > window_size ({window_size}) creates gaps " "between windows",
                UserWarning,
                stacklevel=2,
            )

        # use deque without maxlen to handle variable-length buffers
        buffer: deque[tuple[int, str]] = deque()
        window_id = 0

        for line_num, line_text in lines:
            buffer.append((line_num, line_text))

            # when buffer reaches window size, yield a window
            if len(buffer) == window_size:
                start_line = buffer[0][0]
                end_line = buffer[-1][0]
                content = "\n".join(text for _, text in buffer)

                yield TextWindow(
                    content=content,
                    start_line=start_line,
                    end_line=end_line,
                    window_id=window_id,
                )

                window_id += 1

                # remove first 'stride' items for next window
                for _ in range(min(stride, len(buffer))):
                    buffer.popleft()

        # handle final partial window
        if len(buffer) > 0:
            start_line = buffer[0][0]
            end_line = buffer[-1][0]
            content = "\n".join(text for _, text in buffer)

            yield TextWindow(
                content=content,
                start_line=start_line,
                end_line=end_line,
                window_id=window_id,
            )
