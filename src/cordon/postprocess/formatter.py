from collections.abc import Sequence
from xml.sax.saxutils import escape

from cordon.core.types import MergedBlock


class OutputFormatter:
    """Generate XML-tagged output with original line content.

    This formatter wraps each merged block in XML tags that specify
    line ranges and scores, making it easy for downstream agents to
    reference specific sections of the original file.
    """

    def format_blocks(
        self,
        merged_blocks: Sequence[MergedBlock],
        lines: Sequence[tuple[int, str]],
    ) -> str:
        """Format merged blocks into XML-tagged output.

        Args:
            merged_blocks: Sequence of merged blocks to format.
            lines: Sequence of (line_number, line_content) tuples from the
                ingestion reader.

        Returns:
            Formatted string with XML tags and original content.
        """
        if not merged_blocks:
            return '<?xml version="1.0" encoding="UTF-8"?>\n<anomalies></anomalies>'

        sorted_blocks = sorted(merged_blocks, key=lambda b: b.start_line)
        output_parts: list[str] = ['<?xml version="1.0" encoding="UTF-8"?>', "<anomalies>", ""]

        line_map = dict(lines)

        for block in sorted_blocks:
            content_lines: list[str] = []
            for line_num in range(block.start_line, block.end_line + 1):
                line_content = line_map.get(line_num, "")
                content_lines.append(line_content + "\n")

            tag = (
                f'  <block lines="{block.start_line}-{block.end_line}" '
                f'score="{block.max_score:.4f}">'
            )
            content = "".join(content_lines)
            escaped_content = escape(content)

            indented_content = "\n".join(
                "    " + content_line if content_line else content_line
                for content_line in escaped_content.splitlines()
            )

            output_parts.append(f"{tag}\n{indented_content}\n  </block>")
            output_parts.append("")

        output_parts.append("</anomalies>")
        return "\n".join(output_parts)
