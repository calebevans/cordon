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
