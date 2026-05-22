"""Shared embedding normalization utilities."""

from typing import Any

import numpy as np
import numpy.typing as npt


def normalize_embeddings(
    embeddings: npt.NDArray[np.floating[Any]],
) -> npt.NDArray[np.floating[Any]]:
    """L2-normalize embedding vectors.

    Handles both single vectors (1D) and batches (2D). Zero vectors
    are left as-is to avoid division by zero.

    Args:
        embeddings: Array of shape (dim,) or (batch, dim).

    Returns:
        L2-normalized array of the same shape.
    """
    result: npt.NDArray[np.floating[Any]]

    if embeddings.ndim == 1:
        norm: np.floating[Any] = np.linalg.norm(embeddings)
        if norm > 0:
            result = embeddings / norm
            return result
        return embeddings

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    safe_norms = np.maximum(norms, 1e-10)
    result = embeddings / safe_norms
    return result
