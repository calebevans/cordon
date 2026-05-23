import logging
from collections.abc import Iterable, Iterator
from typing import Any

import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from cordon.core.config import AnalysisConfig
from cordon.core.types import TextWindow
from cordon.embedding.normalize import normalize_embeddings

logger = logging.getLogger(__name__)

DEFAULT_REPO_ID = "second-state/All-MiniLM-L6-v2-Embedding-GGUF"
DEFAULT_FILENAME = "all-MiniLM-L6-v2-Q4_K_M.gguf"


class LlamaCppEmbedder:
    """Convert text windows to embeddings using llama.cpp GGUF models."""

    def __init__(self, config: AnalysisConfig) -> None:
        """Initialize the embedder with llama.cpp model.

        Args:
            config: Analysis configuration specifying model and parameters.
        """
        self.config = config
        self.model_path = config.model_path if config.model_path else self._get_default_model()

        try:
            from llama_cpp import Llama
        except ImportError as error:
            raise ImportError(
                "llama-cpp-python is required. Install it with: pip install 'cordon[llama-cpp]'"
            ) from error

        self.model = Llama(
            model_path=self.model_path,
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
            windows: Iterable of text windows to embed.

        Yields:
            Tuples of (window, embedding) where embeddings are L2-normalized
            numpy arrays.
        """
        window_list = list(windows)
        if not window_list:
            return

        batch_size = self.config.batch_size
        total_batches = (len(window_list) + batch_size - 1) // batch_size

        for batch_start in tqdm(
            range(0, len(window_list), batch_size),
            desc="Generating embeddings",
            total=total_batches,
            unit="batch",
        ):
            batch = window_list[batch_start : batch_start + batch_size]
            texts = [w.content for w in batch]

            try:
                result = self.model.create_embedding(texts)
            except Exception as error:
                raise RuntimeError(
                    f"llama.cpp embedding failed on batch starting at window "
                    f"{batch[0].window_id}: {error}"
                ) from error

            embeddings = np.array(
                [item["embedding"] for item in result["data"]],
                dtype=np.float32,
            )
            embeddings = normalize_embeddings(embeddings)

            yield from zip(batch, embeddings, strict=False)

    def _get_default_model(self) -> str:
        """Get path to default GGUF model, downloading if necessary.

        Returns:
            Path to the model file.
        """
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as error:
            raise RuntimeError(
                "huggingface_hub is required for auto-downloading GGUF models. "
                "Install with: pip install huggingface-hub"
            ) from error

        try:
            logger.info("Downloading default GGUF model: %s", DEFAULT_FILENAME)
            model_path = hf_hub_download(
                repo_id=DEFAULT_REPO_ID,
                filename=DEFAULT_FILENAME,
            )
            logger.info("Model downloaded to: %s", model_path)
            return str(model_path)
        except Exception as error:
            raise RuntimeError(
                f"Failed to download default GGUF model: {error}\n"
                f"You can manually download from: https://huggingface.co/{DEFAULT_REPO_ID}\n"
                f"And specify path with: --model-path /path/to/{DEFAULT_FILENAME}"
            ) from error
