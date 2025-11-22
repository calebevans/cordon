"""Embedding module providing LlamaIndex-based backend implementation.

This module supports multiple embedding backends via LlamaIndex:
- sentence-transformers: Default backend using HuggingFace models
- llama-cpp: Alternative backend using GGUF models for better container performance
"""

from typing import TYPE_CHECKING

from cordon.embedding.llamaindex_vectorizer import LlamaIndexVectorizer

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
    return LlamaIndexVectorizer(config)


__all__ = ["LlamaIndexVectorizer", "create_vectorizer"]
