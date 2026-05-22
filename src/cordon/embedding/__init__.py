"""Embedding module for log analysis."""

import importlib
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from cordon.core.config import AnalysisConfig
    from cordon.core.types import Embedder

_BACKEND_REGISTRY: dict[str, str] = {
    "remote": "cordon.embedding.remote.RemoteEmbedder",
    "llama-cpp": "cordon.embedding.llama_cpp.LlamaCppEmbedder",
    "sentence-transformers": "cordon.embedding.transformer.TransformerEmbedder",
}


def create_embedder(config: "AnalysisConfig") -> "Embedder":
    """Factory function to create the appropriate embedder for a config.

    Args:
        config: Analysis configuration with backend selection.

    Returns:
        Embedder instance implementing the Embedder protocol.

    Raises:
        ValueError: If the backend is not recognized.
    """
    class_path = _BACKEND_REGISTRY.get(config.backend)
    if class_path is None:
        raise ValueError(
            f"Unknown backend: {config.backend!r}. "
            f"Available: {', '.join(sorted(_BACKEND_REGISTRY))}"
        )

    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    embedder_class = getattr(module, class_name)
    return cast("Embedder", embedder_class(config))


__all__ = ["create_embedder"]
