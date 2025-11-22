import warnings
from collections.abc import Iterable, Iterator
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from sentence_transformers import SentenceTransformer

from cordon.core.config import AnalysisConfig
from cordon.core.types import TextWindow


class TransformerVectorizer:
    """Convert text windows to dense embeddings with hardware acceleration.

    This vectorizer uses sentence-transformers models to create semantic
    embeddings of text windows. It automatically detects and utilizes
    available hardware acceleration (CUDA, MPS, or CPU).
    """

    def __init__(self, config: AnalysisConfig) -> None:
        """Initialize the vectorizer with a model.

        Args:
            config: Analysis configuration specifying model and device
        """
        self.config = config
        self.device = self._detect_device()
        self.model = SentenceTransformer(config.model_name)
        self.model.to(self.device)
        self._truncation_warned = False  # Only warn once per analyzer instance

    def _detect_device(self) -> str:
        """Detect the best available device for inference.

        Returns:
            Device string: 'cuda', 'mps', or 'cpu'
        """
        if self.config.device is not None:
            return self.config.device

        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def embed_windows(
        self, windows: Iterable[TextWindow]
    ) -> Iterator[tuple[TextWindow, npt.NDArray[np.floating[Any]]]]:
        """Embed text windows into vector representations.

        Args:
            windows: Iterable of text windows to embed

        Yields:
            Tuples of (window, embedding) where embeddings are normalized
            numpy arrays
        """
        # convert to list for batching
        window_list = list(windows)

        if not window_list:
            return

        # check for token truncation (warn once)
        if not self._truncation_warned:
            self._check_truncation_warning(window_list)

        # process in batches
        batch_size = self.config.batch_size
        for batch_start_idx in range(0, len(window_list), batch_size):
            batch = window_list[batch_start_idx : batch_start_idx + batch_size]
            texts = [window.content for window in batch]

            # encode batch with normalization for cosine distance
            embeddings = self.model.encode(
                texts,
                batch_size=len(batch),
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )

            # yield individual pairs
            yield from zip(batch, embeddings, strict=False)

    def _check_truncation_warning(self, windows: list[TextWindow]) -> None:
        """Check if windows are likely to be truncated and warn user.

        Args:
            windows: List of windows to check
        """
        if not windows:
            return

        try:
            # sample first few windows to estimate token usage
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

            # warn if significant truncation is occurring
            if avg_tokens > max_seq_length * 1.2:  # 20% over limit
                lines_in_window = len(sample_windows[0].content.split("\n"))
                tokens_per_line = avg_tokens / lines_in_window
                lines_that_fit = int(max_seq_length / tokens_per_line)
                coverage_pct = (lines_that_fit / lines_in_window) * 100

                warnings.warn(
                    f"\n{'='*70}\n"
                    f"⚠️  TOKEN TRUNCATION WARNING\n"
                    f"{'='*70}\n"
                    f"Windows contain ~{avg_tokens:.0f} tokens on average (max: {max_tokens})\n"
                    f"Model '{self.config.model_name}' has a {max_seq_length}-token limit.\n"
                    f"\n"
                    f"Impact:\n"
                    f"  • Only the first ~{lines_that_fit} of {lines_in_window} lines per window are analyzed\n"
                    f"  • Coverage: ~{coverage_pct:.0f}% of each window\n"
                    f"\n"
                    f"Recommendations:\n"
                    f"  1. Reduce window size: --window-size {lines_that_fit} --stride {lines_that_fit // 2}\n"
                    f"  2. Use larger model: --model-name BAAI/bge-base-en-v1.5 (512 tokens)\n"
                    f"  3. Accept partial coverage (overlapping windows still capture all lines)\n"
                    f"{'='*70}\n",
                    UserWarning,
                    stacklevel=3,
                )
                self._truncation_warned = True
        except Exception:
            # If tokenizer check fails, silently skip the warning
            # (e.g., during testing or if tokenizer is not available)
            pass
