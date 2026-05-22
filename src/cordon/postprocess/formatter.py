from collections.abc import Sequence
from pathlib import Path
from xml.sax.saxutils import escape

from cordon.core.types import MergedBlock


class OutputFormatter:
    """Generate XML-tagged output with original line content.

    This formatter wraps each merged block in XML tags that specify
    line ranges and scores, making it easy for downstream agents to
    reference specific sections of the original file.
    """

    def _format_block_xml(self, block: MergedBlock, content_lines: list[str]) -> str:
        """Format a single block's content as an XML element.

        Args:
            block: The merged block with line range and score metadata.
            content_lines: Raw lines from the original file for this block.

        Returns:
            XML string for the block with escaped, indented content.
        """
        tag = (
            f'  <block lines="{block.start_line}-{block.end_line}" '
            f'score="{block.max_score:.4f}">'
        )
        content = "".join(content_lines)
        escaped_content = escape(content)
        indented_content = "\n".join(
            "    " + line if line else line for line in escaped_content.splitlines()
        )
        return f"{tag}\n{indented_content}\n  </block>"

    def format_blocks(self, merged_blocks: Sequence[MergedBlock], original_file: Path) -> str:
        """Format merged blocks into XML-tagged output.

        Uses single-pass streaming to efficiently handle large files by only
        keeping anomalous blocks in memory.

        Args:
            merged_blocks: Sequence of merged blocks to format.
            original_file: Path to original file (for extracting content).

        Returns:
            Formatted string with XML tags and original content.
        """
        if not merged_blocks:
            return '<?xml version="1.0" encoding="UTF-8"?>\n<anomalies></anomalies>'

        sorted_blocks = sorted(merged_blocks, key=lambda b: b.start_line)

        output_parts = ['<?xml version="1.0" encoding="UTF-8"?>', "<anomalies>", ""]
        block_idx = 0
        current_line = 1

        with open(original_file, encoding="utf-8", errors="replace") as file_handle:
            for line in file_handle:
                if block_idx >= len(sorted_blocks):
                    break

                block = sorted_blocks[block_idx]

                if current_line == block.start_line:
                    content_lines = [line]

                    while current_line < block.end_line:
                        next_line = next(file_handle, None)
                        if next_line is None:
                            break
                        content_lines.append(next_line)
                        current_line += 1

                    output_parts.append(self._format_block_xml(block, content_lines))
                    output_parts.append("")

                    block_idx += 1
                    current_line = block.end_line + 1
                    continue

                current_line += 1

        output_parts.append("</anomalies>")

        return "\n".join(output_parts)
