"""JSON output formatter for anomaly blocks."""

import json
from collections.abc import Sequence

from cordon.core.types import MergedBlock


class JsonFormatter:
    """Format merged blocks as a JSON document.

    Produces a JSON object with an 'anomalies' array where each entry
    contains start_line, end_line, score, and the original line content.
    """

    def format_blocks(
        self,
        merged_blocks: Sequence[MergedBlock],
        lines: Sequence[tuple[int, str]],
    ) -> str:
        """Format merged blocks into a JSON string.

        Args:
            merged_blocks: Sequence of merged blocks to format.
            lines: Sequence of (line_number, line_content) tuples.

        Returns:
            JSON string with anomaly blocks.
        """
        sorted_blocks = sorted(merged_blocks, key=lambda b: b.start_line)
        line_map = dict(lines)

        anomalies = []
        for block in sorted_blocks:
            content_lines = []
            for line_num in range(block.start_line, block.end_line + 1):
                content_lines.append(line_map.get(line_num, ""))

            anomalies.append(
                {
                    "start_line": block.start_line,
                    "end_line": block.end_line,
                    "score": round(block.max_score, 4),
                    "content": "\n".join(content_lines),
                }
            )

        return json.dumps({"anomalies": anomalies}, indent=2)
