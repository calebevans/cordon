"""Tests for JsonFormatter."""

import json

from cordon.core.types import MergedBlock
from cordon.postprocess.json_formatter import JsonFormatter


class TestJsonFormatter:
    """Tests for JsonFormatter output."""

    def test_format_single_block(self) -> None:
        """Test formatting a single anomaly block."""
        lines = [(1, "line 1"), (2, "line 2"), (3, "line 3")]
        blocks = [MergedBlock(start_line=1, end_line=2, original_windows=(0,), max_score=0.8)]
        formatter = JsonFormatter()
        output = formatter.format_blocks(blocks, lines)
        data = json.loads(output)
        assert len(data["anomalies"]) == 1
        assert data["anomalies"][0]["start_line"] == 1
        assert data["anomalies"][0]["end_line"] == 2
        assert data["anomalies"][0]["score"] == 0.8
        assert "line 1" in data["anomalies"][0]["content"]

    def test_format_multiple_blocks(self) -> None:
        """Test formatting multiple anomaly blocks."""
        lines = [(i, f"line {i}") for i in range(1, 11)]
        blocks = [
            MergedBlock(start_line=1, end_line=2, original_windows=(0,), max_score=0.8),
            MergedBlock(start_line=7, end_line=9, original_windows=(1,), max_score=0.9),
        ]
        formatter = JsonFormatter()
        output = formatter.format_blocks(blocks, lines)
        data = json.loads(output)
        assert len(data["anomalies"]) == 2
        assert data["anomalies"][0]["score"] == 0.8
        assert data["anomalies"][1]["score"] == 0.9

    def test_format_empty_blocks(self) -> None:
        """Test formatting with no blocks produces empty anomalies array."""
        lines = [(1, "line 1")]
        formatter = JsonFormatter()
        output = formatter.format_blocks([], lines)
        data = json.loads(output)
        assert data["anomalies"] == []

    def test_no_xml_escaping_in_json(self) -> None:
        """Test that special characters are preserved without XML escaping."""
        lines = [(1, "x < y && z > 10")]
        blocks = [MergedBlock(start_line=1, end_line=1, original_windows=(0,), max_score=0.5)]
        formatter = JsonFormatter()
        output = formatter.format_blocks(blocks, lines)
        data = json.loads(output)
        assert "x < y && z > 10" in data["anomalies"][0]["content"]

    def test_output_is_valid_json(self) -> None:
        """Test that output is always valid JSON."""
        lines = [(1, "test")]
        blocks = [MergedBlock(start_line=1, end_line=1, original_windows=(0,), max_score=0.1234)]
        formatter = JsonFormatter()
        output = formatter.format_blocks(blocks, lines)
        json.loads(output)
