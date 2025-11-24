from collections.abc import Iterable, Iterator
from typing import Any

import numpy as np
import numpy.typing as npt

from cordon.core.config import AnalysisConfig
from cordon.core.types import TextWindow

DEFAULT_REPO_ID = "second-state/All-MiniLM-L6-v2-Embedding-GGUF"
DEFAULT_FILENAME = "all-MiniLM-L6-v2-Q4_K_M.gguf"


class LlamaCppVectorizer:
    """Convert text windows to embeddings using llama.cpp GGUF models."""

    def __init__(self, config: AnalysisConfig) -> None:
        """Initialize the vectorizer with llama.cpp model.

        Args:
            config: Analysis configuration specifying model and parameters
        """
        self.config = config
        if not config.model_path:
            # Auto-download default model and update config
            config.model_path = self._get_default_model()

        try:
            from llama_cpp import Llama
        except ImportError as error:
            raise ImportError(
                "llama-cpp-python is required. Install it with: pip install 'cordon[llama-cpp]'"
            ) from error

        self.model = Llama(
            model_path=config.model_path,
            embedding=True,
            n_ctx=config.n_ctx,
            n_threads=config.n_threads,
            n_gpu_layers=config.n_gpu_layers,
            n_batch=config.n_ctx,
            verbose=False,
        )

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
        for window in windows:
            result = self.model.create_embedding(window.content)
            embedding_list = result["data"][0]["embedding"]
            embedding_array = np.array(embedding_list, dtype=np.float32)

            norm = np.linalg.norm(embedding_array)
            if norm > 0:
                embedding_array = embedding_array / norm

            yield window, embedding_array

    def _get_default_model(self) -> str:
        """Get path to default GGUF model, downloading if necessary.

        Returns:
            Path to the model file
        """
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as error:
            raise RuntimeError(
                "huggingface_hub is required for auto-downloading GGUF models. "
                "Install with: pip install huggingface-hub"
            ) from error

        try:
            print(f"Downloading default GGUF model: {DEFAULT_FILENAME}")
            model_path = hf_hub_download(
                repo_id=DEFAULT_REPO_ID,
                filename=DEFAULT_FILENAME,
            )
            print(f"Model downloaded to: {model_path}")
            return str(model_path)
        except Exception as error:
            raise RuntimeError(
                f"Failed to download default GGUF model: {error}\n"
                f"You can manually download from: https://huggingface.co/{DEFAULT_REPO_ID}\n"
                f"And specify path with: --model-path /path/to/{DEFAULT_FILENAME}"
            ) from error
