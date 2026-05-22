import logging
import warnings
from collections.abc import Iterable, Iterator
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from cordon.core.config import AnalysisConfig
from cordon.core.device import detect_device
from cordon.core.types import TextWindow


class TransformerEmbedder:
    """Convert text windows to dense embeddings with hardware acceleration.

    Uses sentence-transformers models to create semantic embeddings of text
    windows. Automatically detects and utilizes available hardware
    acceleration (CUDA, MPS, or CPU).
    """

    def __init__(self, config: AnalysisConfig) -> None:
        """Initialize the embedder with a sentence-transformer model.

        Args:
            config: Analysis configuration specifying model and device.

        Raises:
            RuntimeError: If the model cannot be loaded.
        """
        self.config = config
        self.device = detect_device(self.config.device)

        try:
            self.model = SentenceTransformer(config.model_name)
        except Exception as error:
            raise RuntimeError(
                f"Failed to load sentence-transformer model '{config.model_name}'. "
                f"Verify the model name is correct and you have network access "
                f"for first-time downloads. Error: {error}"
            ) from error

        self.model.to(self.device)
        self._truncation_warned = False

    def embed_windows(
        self, windows: Iterable[TextWindow]
    ) -> Iterator[tuple[TextWindow, npt.NDArray[np.floating[Any]]]]:
        """Embed text windows into vector representations.

        Args:
            windows: Iterable of text windows to embed.

        Yields:
            Tuples of (window, embedding) where embeddings are L2-normalized
            numpy arrays.
        """
        window_list = list(windows)

        if not window_list:
            return

        if not self._truncation_warned:
            self._check_truncation_warning(window_list)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        batch_size = self.config.batch_size
        total_batches = (len(window_list) + batch_size - 1) // batch_size

        for batch_start_idx in tqdm(
            range(0, len(window_list), batch_size),
            desc="Generating embeddings",
            total=total_batches,
            unit="batch",
            disable=not self.config.show_progress,
        ):
            batch = window_list[batch_start_idx : batch_start_idx + batch_size]
            texts = [window.content for window in batch]

            embeddings = self.model.encode(
                texts,
                batch_size=len(batch),
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )

            yield from zip(batch, embeddings, strict=False)

    def _check_truncation_warning(self, windows: list[TextWindow]) -> None:
        """Check if windows are likely to be truncated and warn user.

        Args:
            windows: List of windows to check.
        """
        if not windows:
            return

        try:
            tokenizer = self.model.tokenizer
            max_seq_length = self.model.max_seq_length
            sample_size = min(10, len(windows))
            sample_windows = windows[:sample_size]

            token_counts = []
            for window in sample_windows:
                tokens = tokenizer.encode(window.content, add_special_tokens=True)
                token_counts.append(len(tokens))

            avg_tokens = sum(token_counts) / len(token_counts)
            max_tokens = max(token_counts)

            if avg_tokens > max_seq_length * 1.2:
                lines_in_window = len(sample_windows[0].content.split("\n"))
                tokens_per_line = avg_tokens / lines_in_window
                lines_that_fit = int(max_seq_length / tokens_per_line)
                coverage_pct = (lines_that_fit / lines_in_window) * 100

                warnings.warn(
                    f"\n{'=' * 70}\n"
                    f"⚠️  TOKEN TRUNCATION WARNING\n"
                    f"{'=' * 70}\n"
                    f"Windows contain ~{avg_tokens:.0f} tokens on average (max: {max_tokens})\n"
                    f"Model '{self.config.model_name}' has a {max_seq_length}-token limit.\n"
                    f"\n"
                    f"Impact:\n"
                    f"  • Only the first ~{lines_that_fit} of {lines_in_window} lines per window are analyzed\n"
                    f"  • Coverage: ~{coverage_pct:.0f}% of each window\n"
                    f"\n"
                    f"Recommendations:\n"
                    f"  1. Reduce window size: --window-size {lines_that_fit}\n"
                    f"  2. Use larger model: --model-name BAAI/bge-base-en-v1.5 (512 tokens)\n"
                    f"  3. Accept partial coverage (non-overlapping windows may miss some context)\n"
                    f"{'=' * 70}\n",
                    UserWarning,
                    stacklevel=3,
                )
                self._truncation_warned = True
        except (AttributeError, TypeError, ValueError):
            logging.getLogger(__name__).debug("Could not check truncation warning", exc_info=True)
