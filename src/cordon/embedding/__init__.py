"""Embedding module for log analysis."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cordon.core.config import AnalysisConfig
    from cordon.core.types import Embedder


def create_vectorizer(config: "AnalysisConfig") -> "Embedder":
    """Factory function to create appropriate vectorizer based on config.

    Args:
        config: Analysis configuration with backend selection

    Returns:
        Vectorizer instance implementing the Embedder protocol
    """
    if config.backend == "remote":
        from cordon.embedding.remote import RemoteVectorizer

        return RemoteVectorizer(config)

    if config.backend == "llama-cpp":
        from cordon.embedding.llama_cpp import LlamaCppVectorizer

        return LlamaCppVectorizer(config)

    from cordon.embedding.transformer import TransformerVectorizer

    return TransformerVectorizer(config)


__all__ = ["create_vectorizer"]
