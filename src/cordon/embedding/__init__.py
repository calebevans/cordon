"""Embedding module for log analysis."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cordon.core.config import AnalysisConfig
    from cordon.core.types import Embedder


def create_embedder(config: "AnalysisConfig") -> "Embedder":
    """Factory function to create the appropriate embedder for a config.

    Args:
        config: Analysis configuration with backend selection.

    Returns:
        Embedder instance matching the configured backend.

    Raises:
        ValueError: If the backend is not recognized.
    """
    if config.backend == "remote":
        from cordon.embedding.remote import RemoteEmbedder

        return RemoteEmbedder(config)

    if config.backend == "llama-cpp":
        from cordon.embedding.llama_cpp import LlamaCppEmbedder

        return LlamaCppEmbedder(config)

    if config.backend == "sentence-transformers":
        from cordon.embedding.transformer import TransformerEmbedder

        return TransformerEmbedder(config)

    raise ValueError(f"Unknown backend: {config.backend}")


__all__ = ["create_embedder"]
