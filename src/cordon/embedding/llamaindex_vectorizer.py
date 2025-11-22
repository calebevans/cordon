"""LlamaIndex-based embedding vectorizer for log analysis.

This module provides a unified implementation of the Embedder protocol using
LlamaIndex's BaseEmbedding interface, supporting both HuggingFace and
llama.cpp backends with consistent behavior.
"""

from collections.abc import Iterable, Iterator
from typing import Any

import numpy as np
import numpy.typing as npt
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.bridge.pydantic import PrivateAttr

from cordon.core.config import AnalysisConfig
from cordon.core.types import TextWindow

# Default model configuration (Q4_K_M for balanced performance/accuracy)
DEFAULT_REPO_ID = "second-state/All-MiniLM-L6-v2-Embedding-GGUF"
DEFAULT_FILENAME = "all-MiniLM-L6-v2-Q4_K_M.gguf"


class LlamaCPPEmbedding(BaseEmbedding):  # type: ignore[misc]
    """LlamaIndex embedding wrapper for llama.cpp."""

    _model: Any = PrivateAttr()

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 2048,
        n_threads: int | None = None,
        n_gpu_layers: int = 0,
        n_batch: int = 512,
        **kwargs: Any,
    ) -> None:
        """Initialize llama.cpp model."""
        super().__init__(**kwargs)
        try:
            from llama_cpp import Llama
        except ImportError as error:
            raise ImportError(
                "llama-cpp-python is required. Install it with: pip install llama-cpp-python"
            ) from error

        self._model = Llama(
            model_path=model_path,
            embedding=True,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            n_batch=n_batch,
            verbose=False,
        )

    def _get_query_embedding(self, query: str) -> list[float]:
        """Get query embedding."""
        return self._get_text_embedding(query)

    def _get_text_embedding(self, text: str) -> list[float]:
        """Get text embedding."""
        # create_embedding returns OpenAI-compatible format
        result = self._model.create_embedding(text)
        return result["data"][0]["embedding"]  # type: ignore[no-any-return]

    async def _aget_query_embedding(self, query: str) -> list[float]:
        """Async get query embedding."""
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> list[float]:
        """Async get text embedding."""
        return self._get_text_embedding(text)

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Get text embeddings in batch."""
        # llama-cpp-python supports batching via list input
        result = self._model.create_embedding(texts)
        # result['data'] is a list of dicts with 'embedding' and 'index'
        # We need to sort by index to ensure order, though usually it preserves order
        data = sorted(result["data"], key=lambda x: x["index"])
        return [item["embedding"] for item in data]

    async def _aget_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Async get text embeddings in batch."""
        return self._get_text_embeddings(texts)


class LlamaIndexVectorizer:
    """Convert text windows to embeddings using LlamaIndex backends.

    This vectorizer wraps LlamaIndex's embedding models to provide a unified
    interface for both standard (HuggingFace) and quantized (llama.cpp)
    models. It leverages LlamaIndex's built-in text chunking and token
    management.
    """

    def __init__(self, config: AnalysisConfig) -> None:
        """Initialize the vectorizer with appropriate LlamaIndex model.

        Args:
            config: Analysis configuration specifying backend and model details
        """
        self.config = config
        self.model = self._initialize_model()

    def _initialize_model(self) -> BaseEmbedding:
        """Initialize the specific LlamaIndex embedding model.

        Returns:
            Configured BaseEmbedding instance
        """
        if self.config.backend == "sentence-transformers":
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding

            # Map 'cuda', 'mps', 'cpu' to LlamaIndex device string
            device = self.config.device
            if device is None:
                # LlamaIndex auto-detection is usually good, but we can be explicit
                # if needed based on config. For now, let it auto-detect if None.
                pass

            return HuggingFaceEmbedding(
                model_name=self.config.model_name,
                device=device,
                embed_batch_size=self.config.batch_size,
            )

        elif self.config.backend == "llama-cpp":
            if not self.config.model_path:
                self.config.model_path = self._get_default_model()

            return LlamaCPPEmbedding(
                model_path=self.config.model_path,
                n_ctx=self.config.n_ctx,
                n_threads=self.config.n_threads,
                n_gpu_layers=self.config.n_gpu_layers,
                n_batch=self.config.n_ctx,  # Set n_batch to n_ctx to allow large batches
                embed_batch_size=self.config.batch_size,
            )
        else:
            raise ValueError(f"Unsupported backend: {self.config.backend}")

    def _get_default_model(self) -> str:
        """Get path to default GGUF model, downloading if necessary.

        Downloads all-MiniLM-L6-v2.F32.gguf from HuggingFace.
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
            print("Using HuggingFace Hub...")
            print("This is a one-time download (~23MB)...")

            model_path = hf_hub_download(
                repo_id=DEFAULT_REPO_ID,
                filename=DEFAULT_FILENAME,
            )

            print("✓ Model downloaded successfully")
            print(f"  Cache: {model_path}")
            return model_path  # type: ignore[no-any-return]

        except Exception as error:
            raise RuntimeError(
                f"Failed to download default GGUF model: {error}\n"
                f"You can manually download from HuggingFace:\n"
                f"  https://huggingface.co/{DEFAULT_REPO_ID}\n"
                f"  And specify path with: --model-path /path/to/{DEFAULT_FILENAME}"
            ) from error

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
        window_list = list(windows)
        if not window_list:
            return

        texts = [w.content for w in window_list]

        # LlamaIndex handles batching internally based on embed_batch_size
        embeddings = self.model.get_text_embedding_batch(texts)

        for window, embedding_list in zip(window_list, embeddings, strict=True):
            # Convert to numpy and ensure float32
            embedding_array = np.array(embedding_list, dtype=np.float32)

            # LlamaIndex embeddings are usually normalized, but we ensure it
            # for consistency with downstream cosine distance
            norm = np.linalg.norm(embedding_array)
            if norm > 0:
                embedding_array = embedding_array / norm

            yield window, embedding_array
