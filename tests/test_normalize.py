"""Tests for shared normalization utility."""

import numpy as np

from cordon.embedding.normalize import normalize_embeddings


class TestNormalizeEmbeddings:
    """Tests for L2 normalization of embedding vectors."""

    def test_single_vector(self) -> None:
        """Test normalization of a single 1D vector."""
        vec = np.array([3.0, 4.0], dtype=np.float32)
        result = normalize_embeddings(vec)
        assert np.isclose(np.linalg.norm(result), 1.0)

    def test_batch(self) -> None:
        """Test normalization of a 2D batch of vectors."""
        batch = np.array([[3.0, 4.0], [1.0, 0.0]], dtype=np.float32)
        result = normalize_embeddings(batch)
        for row in result:
            assert np.isclose(np.linalg.norm(row), 1.0)

    def test_zero_vector(self) -> None:
        """Test that a zero vector is left as-is."""
        vec = np.zeros(3, dtype=np.float32)
        result = normalize_embeddings(vec)
        assert np.allclose(result, 0.0)

    def test_batch_with_zero(self) -> None:
        """Test that zero vectors in a batch are handled safely."""
        batch = np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)
        result = normalize_embeddings(batch)
        assert np.isclose(np.linalg.norm(result[0]), 1.0)
        assert np.allclose(result[1], 0.0, atol=1e-9)

    def test_already_normalized(self) -> None:
        """Test that an already-normalized vector is unchanged."""
        vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        result = normalize_embeddings(vec)
        assert np.allclose(result, vec)
