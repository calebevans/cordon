from collections.abc import Iterator

from cordon.core.config import AnalysisConfig
from cordon.core.types import TextWindow


class SlidingWindowSegmenter:
    """Convert line stream into non-overlapping text windows with line tracking.

    This segmenter creates non-overlapping chunks of text from a stream of lines.
    Each window maintains references to its original line numbers for downstream
    processing. Optionally splits long lines into virtual lines based on a
    character limit.
    """

    def segment(
        self, lines: Iterator[tuple[int, str]], config: AnalysisConfig
    ) -> Iterator[TextWindow]:
        """Segment lines into non-overlapping text windows.

        When max_line_length is set, lines exceeding that limit are split into
        multiple virtual lines, each counting toward window_size. The window's
        start_line/end_line still reference real file line numbers.

        Args:
            lines: Iterator of (line_number, line_content) tuples.
            config: Analysis configuration with window_size and
                optional max_line_length.

        Yields:
            TextWindow instances with content and line tracking.
        """
        window_size = config.window_size
        max_line_length = config.max_line_length

        buffer: list[tuple[int, str]] = []
        window_id = 0

        for line_num, line_text in lines:
            chunks = self._split_line(line_num, line_text, max_line_length)

            for chunk_line_num, chunk_text in chunks:
                buffer.append((chunk_line_num, chunk_text))

                if len(buffer) == window_size:
                    yield self._flush_buffer(buffer, window_id)
                    window_id += 1
                    buffer = []

        if buffer:
            yield self._flush_buffer(buffer, window_id)

    @staticmethod
    def _flush_buffer(buffer: list[tuple[int, str]], window_id: int) -> TextWindow:
        """Create a TextWindow from buffered lines.

        Args:
            buffer: List of (line_number, line_content) tuples.
            window_id: Unique identifier for this window.

        Returns:
            A TextWindow constructed from the buffer contents.
        """
        start_line = buffer[0][0]
        end_line = buffer[-1][0]
        content = "\n".join(text for _, text in buffer)
        return TextWindow(
            content=content,
            start_line=start_line,
            end_line=end_line,
            window_id=window_id,
        )

    @staticmethod
    def _split_line(line_num: int, line_text: str, max_length: int | None) -> list[tuple[int, str]]:
        """Split a line into chunks if it exceeds max_length.

        Args:
            line_num: The real file line number.
            line_text: The line content.
            max_length: Maximum characters per chunk, or None to disable.

        Returns:
            List of (line_number, chunk_text) tuples. All chunks share
            the same line_number since they originate from the same file line.
        """
        if max_length is None or len(line_text) <= max_length:
            return [(line_num, line_text)]

        return [
            (line_num, line_text[i : i + max_length]) for i in range(0, len(line_text), max_length)
        ]
