from collections.abc import Iterator
from pathlib import Path


class LogFileReader:
    """Read files line-by-line with minimal memory footprint.

    This reader yields (line_number, line_content) tuples where line numbers
    are 1-indexed. It handles UTF-8 encoding with fallback to latin-1 for
    compatibility with various log formats.
    """

    def _read_with_encoding(self, file_path: Path, encoding: str) -> Iterator[tuple[int, str]]:
        """Read lines from a file with specified encoding.

        Args:
            file_path: Path to the file to read
            encoding: Character encoding to use

        Yields:
            Tuples of (line_number, line_content)
        """
        with open(file_path, encoding=encoding) as file_handle:
            for line_num, line in enumerate(file_handle, start=1):
                yield line_num, line.rstrip()

    def read_lines(self, file_path: Path) -> Iterator[tuple[int, str]]:
        """Read lines from a file with line number tracking.

        Args:
            file_path: Path to the file to read

        Yields:
            Tuples of (line_number, line_content) where line_number is 1-indexed
            and line_content has trailing whitespace stripped

        Raises:
            FileNotFoundError: If the file does not exist
            PermissionError: If the file cannot be read
        """
        # try UTF-8 first, fallback to latin-1
        try:
            yield from self._read_with_encoding(file_path, "utf-8")
        except UnicodeDecodeError:
            yield from self._read_with_encoding(file_path, "latin-1")
