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

    def format_blocks(self, merged_blocks: Sequence[MergedBlock], original_file: Path) -> str:
        """Format merged blocks into XML-tagged output.

        Uses single-pass streaming to efficiently handle large files by only
        keeping anomalous blocks in memory.

        Args:
            merged_blocks: Sequence of merged blocks to format (sorted by line number)
            original_file: Path to original file (for extracting content)

        Returns:
            Formatted string with XML tags and original content
        """
        if not merged_blocks:
            return '<?xml version="1.0" encoding="UTF-8"?>\n<anomalies></anomalies>'

        # merged blocks are sorted by start_line from the merger
        output_parts = ['<?xml version="1.0" encoding="UTF-8"?>', "<anomalies>", ""]
        block_idx = 0
        current_line = 1

        with open(original_file, encoding="utf-8", errors="replace") as file_handle:
            for line in file_handle:
                if block_idx >= len(merged_blocks):
                    # all blocks have been processed
                    break

                block = merged_blocks[block_idx]

                if current_line == block.start_line:
                    # start collecting lines for this block
                    content_lines = [line]

                    # read the remaining lines of the block
                    while current_line < block.end_line:
                        next_line = next(file_handle)
                        content_lines.append(next_line)
                        current_line += 1

                    # format the block with indentation
                    tag = (
                        f'  <block lines="{block.start_line}-{block.end_line}" '
                        f'score="{block.max_score:.4f}">'
                    )
                    content = "".join(content_lines)
                    # Escape XML special characters to ensure valid XML output
                    escaped_content = escape(content)

                    # Indent each line of content for pretty printing
                    # Preserve whitespace-only lines from original content
                    indented_content = "\n".join(
                        "    " + line if line else line for line in escaped_content.splitlines()
                    )

                    output_parts.append(f"{tag}\n{indented_content}\n  </block>")
                    output_parts.append("")  # blank line between blocks

                    # move to next block
                    block_idx += 1

                current_line += 1

        # close the root element
        output_parts.append("</anomalies>")

        # join with newlines
        return "\n".join(output_parts)
