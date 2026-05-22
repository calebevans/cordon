from collections.abc import Iterator
from pathlib import Path


class LogFileReader:
    """Read files line-by-line with minimal memory footprint.

    This reader yields (line_number, line_content) tuples where line numbers
    are 1-indexed. Non-UTF-8 bytes are replaced with the Unicode replacement
    character (U+FFFD) rather than raising an encoding error.
    """

    def read_lines(self, file_path: Path) -> Iterator[tuple[int, str]]:
        """Read lines from a file with line number tracking.

        Args:
            file_path: Path to the file to read.

        Yields:
            Tuples of (line_number, line_content) where line_number is 1-indexed
            and line_content has trailing whitespace stripped.

        Raises:
            FileNotFoundError: If the file does not exist.
            PermissionError: If the file cannot be read.
        """
        with open(file_path, encoding="utf-8", errors="replace") as file_handle:
            for line_num, line in enumerate(file_handle, start=1):
                yield line_num, line.rstrip()
