from pathlib import Path
from tempfile import NamedTemporaryFile

import pytest

from cordon.ingestion.reader import LogFileReader


class TestLogFileReader:
    """Tests for LogFileReader class."""

    def test_read_simple_file(self) -> None:
        """Test reading a simple file."""
        with NamedTemporaryFile(mode="w", delete=False, suffix=".log") as f:
            f.write("line 1\n")
            f.write("line 2\n")
            f.write("line 3\n")
            temp_path = Path(f.name)

        try:
            reader = LogFileReader()
            lines = list(reader.read_lines(temp_path))

            assert len(lines) == 3
            assert lines[0] == (1, "line 1")
            assert lines[1] == (2, "line 2")
            assert lines[2] == (3, "line 3")
        finally:
            temp_path.unlink()

    def test_read_empty_file(self) -> None:
        """Test reading an empty file."""
        with NamedTemporaryFile(mode="w", delete=False, suffix=".log") as f:
            temp_path = Path(f.name)

        try:
            reader = LogFileReader()
            lines = list(reader.read_lines(temp_path))
            assert len(lines) == 0
        finally:
            temp_path.unlink()

    def test_read_file_with_trailing_whitespace(self) -> None:
        """Test that trailing whitespace is stripped."""
        with NamedTemporaryFile(mode="w", delete=False, suffix=".log") as f:
            f.write("line 1  \n")
            f.write("line 2\t\n")
            f.write("line 3   \t  \n")
            temp_path = Path(f.name)

        try:
            reader = LogFileReader()
            lines = list(reader.read_lines(temp_path))

            assert lines[0] == (1, "line 1")
            assert lines[1] == (2, "line 2")
            assert lines[2] == (3, "line 3")
        finally:
            temp_path.unlink()

    def test_read_file_with_leading_whitespace(self) -> None:
        """Test that leading whitespace is preserved."""
        with NamedTemporaryFile(mode="w", delete=False, suffix=".log") as f:
            f.write("  line 1\n")
            f.write("\tline 2\n")
            temp_path = Path(f.name)

        try:
            reader = LogFileReader()
            lines = list(reader.read_lines(temp_path))

            assert lines[0] == (1, "  line 1")
            assert lines[1] == (2, "\tline 2")
        finally:
            temp_path.unlink()

    def test_read_nonexistent_file(self) -> None:
        """Test that reading a nonexistent file raises FileNotFoundError."""
        reader = LogFileReader()
        with pytest.raises(FileNotFoundError):
            list(reader.read_lines(Path("/nonexistent/file.log")))

    def test_read_file_with_unicode(self) -> None:
        """Test reading a file with unicode characters."""
        with NamedTemporaryFile(mode="w", delete=False, suffix=".log", encoding="utf-8") as f:
            f.write("line with émojis 🎉\n")
            f.write("line with 中文\n")
            temp_path = Path(f.name)

        try:
            reader = LogFileReader()
            lines = list(reader.read_lines(temp_path))

            assert len(lines) == 2
            assert lines[0] == (1, "line with émojis 🎉")
            assert lines[1] == (2, "line with 中文")
        finally:
            temp_path.unlink()
